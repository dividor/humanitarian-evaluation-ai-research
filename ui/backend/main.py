"""
Search API Backend
FastAPI server that provides semantic search over indexed documents in Qdrant.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from pipeline.db import db  # noqa: E402
from pipeline.search import search_chunks  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Humanitarian Evaluation Search API")


def infer_paragraphs_from_bboxes(text: str, bboxes: List[tuple]) -> str:
    """
    Infer paragraph breaks in text based on vertical gaps in bounding boxes.

    Args:
        text: The chunk text
        bboxes: List of bbox tuples (l, t, r, b) for each text element

    Returns:
        Text with inferred line breaks based on bbox gaps
    """
    if not bboxes or len(bboxes) < 3:
        return text

    try:
        # Sort bboxes by vertical position (top to bottom)
        # bbox format: (l, t, r, b) where smaller t = higher on page
        sorted_bboxes = sorted(bboxes, key=lambda b: b[1] if len(b) >= 2 else 0)

        # Calculate vertical gaps between consecutive bboxes
        gaps = []
        for i in range(len(sorted_bboxes) - 1):
            if len(sorted_bboxes[i]) >= 4 and len(sorted_bboxes[i + 1]) >= 2:
                curr_bottom = sorted_bboxes[i][3]  # bottom of current
                next_top = sorted_bboxes[i + 1][1]  # top of next
                gap = abs(curr_bottom - next_top)
                gaps.append(gap)

        if not gaps or len(gaps) < 2:
            return text

        # Find threshold: gaps significantly larger than normal line spacing
        sorted_gaps = sorted(gaps)
        # Use 75th percentile as baseline for normal spacing
        baseline_idx = int(len(sorted_gaps) * 0.75)
        baseline_gap = sorted_gaps[baseline_idx]

        # Threshold: 2.5x the baseline (paragraph breaks are much larger)
        threshold = baseline_gap * 2.5

        # Count significant gaps
        large_gaps = [g for g in gaps if g > threshold]

        logger.info(
            f"Bbox analysis: {len(bboxes)} boxes, "
            f"{len(large_gaps)} large gaps (threshold: {threshold:.2f})"
        )

        # If we found significant gaps, look for sentence breaks in the text
        if large_gaps:
            # Look for patterns that indicate paragraph breaks:
            # 1. Period followed by space and capital letter
            # 2. Period at end followed by newline
            import re

            # Find sentence boundaries
            sentences = re.split(r"(\.\s+(?=[A-ZÁÉÍÓÚÑ]))", text)

            if len(sentences) > 2:
                # Rejoin with double newlines at major boundaries
                # Heuristic: insert breaks roughly proportional to number of large gaps
                break_frequency = max(2, len(sentences) // (len(large_gaps) + 1))

                result_parts = []
                for i, part in enumerate(sentences):
                    result_parts.append(part)
                    # Add paragraph break at sentence boundaries if we're past break frequency
                    if i > 0 and i % (break_frequency * 2) == 0 and ". " in part:
                        result_parts.append("\n\n")

                return "".join(result_parts)

        return text

    except Exception as e:
        logger.warning(f"Error inferring paragraphs from bboxes: {e}")
        return text


# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    page_num: int
    headings: List[str]
    score: float
    # Document metadata (from join)
    title: str
    organization: Optional[str] = None
    year: Optional[str] = None
    year_published: Optional[str] = None
    evaluation_type: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    theme: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query: str
    filters: Optional[Dict[str, List[str]]] = None


class FacetValue(BaseModel):
    value: str
    count: int


class Facets(BaseModel):
    agencies: List[FacetValue]
    titles: List[FacetValue]
    years: List[FacetValue]
    evaluation_types: List[FacetValue]
    countries: List[FacetValue]
    regions: List[FacetValue]
    themes: List[FacetValue]


class HighlightBox(BaseModel):
    page: int
    bbox: Dict[str, float]  # {l, r, t, b}
    text: str


class HighlightResponse(BaseModel):
    highlights: List[HighlightBox]
    total: int


@app.get("/")
def root():
    """API root"""
    return {
        "name": "Humanitarian Evaluation Search API",
        "version": "1.0.0",
        "endpoints": {
            "/search": "Semantic search",
            "/facets": "Get filter facets",
            "/document/{doc_id}": "Get document metadata",
            "/pdf/{doc_id}": "Serve PDF file",
            "/highlight/{doc_id}": "Get highlight bounding boxes",
        },
    }


@app.get("/health")
def health():
    """Health check"""
    return {"status": "healthy"}


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(50, description="Maximum results"),
    agency: Optional[str] = Query(None, description="Filter by agency"),
    title: Optional[str] = Query(None, description="Filter by title (partial match)"),
    year: Optional[str] = Query(None, description="Filter by year"),
    evaluation_type: Optional[str] = Query(
        None, description="Filter by evaluation type"
    ),
    country: Optional[str] = Query(None, description="Filter by country"),
    region: Optional[str] = Query(None, description="Filter by region"),
    theme: Optional[str] = Query(None, description="Filter by theme"),
):
    """
    Perform semantic search over document chunks.
    Returns chunks with document metadata joined.
    """
    try:
        # Get search results from Qdrant
        results = search_chunks(q, limit=limit * 2)  # Get extra for filtering

        # Group by doc_id and get document metadata
        doc_cache = {}
        filtered_results = []

        for result in results:
            doc_id = result.payload.get("doc_id")

            # Fetch document metadata if not cached
            if doc_id not in doc_cache:
                doc_metadata = db.get_document(doc_id)
                if not doc_metadata:
                    continue
                doc_cache[doc_id] = doc_metadata

            doc = doc_cache[doc_id]

            # Apply filters
            if agency and doc.get("agency") != agency:
                continue
            if title and title.lower() not in doc.get("title", "").lower():
                continue
            if year and str(doc.get("year")) != year:
                continue
            if evaluation_type and doc.get("evaluation_type") != evaluation_type:
                continue
            if (
                country
                and doc.get("country")
                and country.lower() not in doc.get("country", "").lower()
            ):
                continue
            if region and doc.get("region") != region:
                continue
            if (
                theme
                and doc.get("theme")
                and theme.lower() not in doc.get("theme", "").lower()
            ):
                continue

            # Build result with inferred paragraphs
            chunk_text = result.payload.get("text", "")
            chunk_bboxes = result.payload.get("bbox", [])

            # Ensure paragraph breaks are visible by replacing single \n with \n\n
            # This creates blank lines between paragraphs
            formatted_text = chunk_text.replace("\n", "\n\n")

            # Also try to infer additional breaks from bboxes
            formatted_text = infer_paragraphs_from_bboxes(formatted_text, chunk_bboxes)

            search_result = SearchResult(
                chunk_id=str(result.id),
                doc_id=doc_id,
                text=formatted_text,
                page_num=result.payload.get("page_num", 0),
                headings=result.payload.get("headings") or [],  # Handle None
                score=result.score,
                title=doc.get("title", "Unknown"),
                organization=doc.get(
                    "agency"
                ),  # Map agency to organization for frontend
                year=str(doc.get("year")) if doc.get("year") else None,
                year_published=(
                    str(doc.get("year_published"))
                    if doc.get("year_published")
                    else None
                ),
                evaluation_type=doc.get("evaluation_type"),
                country=doc.get("country"),
                region=doc.get("region"),
                theme=doc.get("theme"),
            )

            filtered_results.append(search_result)

            if len(filtered_results) >= limit:
                break

        return SearchResponse(
            results=filtered_results,
            total=len(filtered_results),
            query=q,
            filters={
                "agency": [agency] if agency else [],
                "title": [title] if title else [],
                "year": [year] if year else [],
                "evaluation_type": [evaluation_type] if evaluation_type else [],
                "country": [country] if country else [],
                "region": [region] if region else [],
                "theme": [theme] if theme else [],
            },
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/facets", response_model=Facets)
async def get_facets():
    """
    Get facet counts for all filterable fields.
    Used to populate filter UI.
    """
    try:
        # Get all documents
        docs = list(db.get_all_documents())

        # Count facets
        from collections import Counter

        agencies = Counter()
        titles = Counter()
        years = Counter()
        evaluation_types = Counter()
        countries = Counter()
        regions = Counter()
        themes = Counter()

        for doc in docs:
            if doc.get("agency"):
                agencies[doc.get("agency")] += 1
            if doc.get("title"):
                titles[doc.get("title")] += 1
            if doc.get("year"):
                years[str(doc.get("year"))] += 1
            if doc.get("evaluation_type"):
                evaluation_types[doc.get("evaluation_type")] += 1
            if doc.get("country"):
                # Split multi-country entries
                for country in doc.get("country", "").split(","):
                    country = country.strip()
                    if country:
                        countries[country] += 1
            if doc.get("region"):
                regions[doc.get("region")] += 1
            if doc.get("theme"):
                themes[doc.get("theme")] += 1

        return Facets(
            agencies=[FacetValue(value=k, count=v) for k, v in agencies.most_common()],
            titles=[FacetValue(value=k, count=v) for k, v in titles.most_common(20)],
            years=[
                FacetValue(value=k, count=v)
                for k, v in sorted(years.items(), reverse=True)
            ],
            evaluation_types=[
                FacetValue(value=k, count=v) for k, v in evaluation_types.most_common()
            ],
            countries=[
                FacetValue(value=k, count=v) for k, v in countries.most_common(20)
            ],
            regions=[FacetValue(value=k, count=v) for k, v in regions.most_common()],
            themes=[FacetValue(value=k, count=v) for k, v in themes.most_common()],
        )

    except Exception as e:
        logger.error(f"Facets error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    """Get full document metadata"""
    try:
        doc = db.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except Exception as e:
        logger.error(f"Document fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pdf/{doc_id}")
async def serve_pdf(doc_id: str):
    """Serve PDF file for viewing"""
    try:
        doc = db.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        filepath = doc.get("filepath")
        if not filepath:
            raise HTTPException(
                status_code=404, detail="PDF filepath not found in metadata"
            )

        # Convert relative path to absolute path
        pdf_path = Path(filepath)
        if not pdf_path.is_absolute():
            # Assume paths are relative to /app (Docker container working directory)
            pdf_path = Path("/app") / filepath

        if not pdf_path.exists():
            raise HTTPException(
                status_code=404, detail=f"PDF file not found at {pdf_path}"
            )

        return FileResponse(
            str(pdf_path), media_type="application/pdf", filename=pdf_path.name
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF serve error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/highlight/chunk/{chunk_id}", response_model=HighlightResponse)
async def get_chunk_highlights(chunk_id: str):
    """
    Get bounding boxes for a specific chunk.
    This is the simplest approach - just return the bbox from the clicked chunk.
    """
    try:
        # Get the specific chunk by ID
        chunk = db.client.retrieve(
            collection_name="chunks", ids=[chunk_id], with_payload=True
        )

        if not chunk:
            return HighlightResponse(highlights=[], total=0)

        chunk_point = chunk[0]
        payload = chunk_point.payload
        chunk_page = payload.get("page_num")
        chunk_bboxes = payload.get("bbox", [])
        chunk_text = payload.get("text", "")

        highlights = []

        # Convert bboxes to highlight format
        for bbox_data in chunk_bboxes:
            if not bbox_data:
                continue

            # Handle different bbox formats
            if isinstance(bbox_data, dict):
                bbox = bbox_data
            elif isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
                # Convert tuple (l, b, r, t) from docling to dict {l, t, r, b}
                bbox = {
                    "l": bbox_data[0],  # left
                    "b": bbox_data[1],  # bottom
                    "r": bbox_data[2],  # right
                    "t": bbox_data[3],  # top
                }
            else:
                continue

            # Ensure all required keys are present
            if all(k in bbox for k in ["l", "t", "r", "b"]):
                highlights.append(
                    HighlightBox(
                        page=chunk_page,
                        bbox=bbox,
                        text=chunk_text[:100],
                    )
                )

        return HighlightResponse(highlights=highlights, total=len(highlights))

    except Exception as e:
        logger.error(f"Chunk highlight error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/highlight/{doc_id}", response_model=HighlightResponse)
async def get_highlights(
    doc_id: str,
    page: Optional[int] = Query(None, description="Filter by page number"),
    text: Optional[str] = Query(
        None,
        description="Filter by text content (not recommended - use page filter instead)",
    ),
):
    """
    Get bounding box data for highlighting chunks on pages.

    Note: Text filtering may miss results due to semantic vs literal matching.
    For best results, filter by page only and let all chunks on that page be highlighted.
    """
    try:
        # Query chunks from Qdrant for this document
        from qdrant_client.http import models as qmodels

        # Filter chunks by doc_id
        results, _ = db.client.scroll(
            collection_name="chunks",
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_id", match=qmodels.MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=10000,  # Get all chunks for the document
            with_payload=True,
        )

        highlights = []

        for chunk_point in results:
            payload = chunk_point.payload
            chunk_text = payload.get("text", "")
            chunk_page = payload.get("page_num")
            chunk_bboxes = payload.get("bbox", [])

            # Filter by page if provided (RECOMMENDED)
            if page and chunk_page != page:
                continue

            # OPTIONAL: Filter by text if explicitly requested
            # Note: This may filter out semantically similar matches
            # (e.g., "évaluation" vs "evaluation")
            # so it's generally better to rely on page filtering only
            if text and text.lower() not in chunk_text.lower():
                continue

            # Convert bboxes to highlight format
            # Each bbox is stored as a tuple from docling: (l, b, r, t)
            # Where coordinate origin is BOTTOMLEFT
            for bbox_data in chunk_bboxes:
                if not bbox_data:
                    continue

                # Handle different bbox formats
                if isinstance(bbox_data, dict):
                    bbox = bbox_data
                elif isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
                    # Convert tuple (l, b, r, t) from docling to dict {l, t, r, b}
                    bbox = {
                        "l": bbox_data[0],  # left
                        "b": bbox_data[1],  # bottom
                        "r": bbox_data[2],  # right
                        "t": bbox_data[3],  # top
                    }
                else:
                    continue

                # Ensure all required keys are present
                if all(k in bbox for k in ["l", "t", "r", "b"]):
                    highlights.append(
                        HighlightBox(
                            page=chunk_page,
                            bbox=bbox,
                            text=chunk_text[:100],  # Truncate for performance
                        )
                    )

        return HighlightResponse(highlights=highlights, total=len(highlights))

    except Exception as e:
        logger.error(f"Highlight error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
