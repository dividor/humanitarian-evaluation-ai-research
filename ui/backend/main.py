"""
Search API Backend
FastAPI server that provides semantic search over indexed documents in Qdrant.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from qdrant_client.http import models

from pipeline.db import db
from pipeline.search import search_chunks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Humanitarian Evaluation Search API")

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
    organizations: List[FacetValue]
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
    organization: Optional[str] = Query(None, description="Filter by organization"),
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
            if organization and doc.get("organization") != organization:
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

            # Build result
            search_result = SearchResult(
                chunk_id=str(result.id),
                doc_id=doc_id,
                text=result.payload.get("text", ""),
                page_num=result.payload.get("page_num", 0),
                headings=result.payload.get("headings") or [],  # Handle None
                score=result.score,
                title=doc.get("title", "Unknown"),
                organization=doc.get("organization"),
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
                "organization": [organization] if organization else [],
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

        organizations = Counter()
        years = Counter()
        evaluation_types = Counter()
        countries = Counter()
        regions = Counter()
        themes = Counter()

        for doc in docs:
            if doc.get("organization"):
                organizations[doc.get("organization")] += 1
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
            organizations=[
                FacetValue(value=k, count=v) for k, v in organizations.most_common()
            ],
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


@app.get("/highlight/{doc_id}", response_model=HighlightResponse)
async def get_highlights(
    doc_id: str,
    page: Optional[int] = Query(None, description="Filter by page number"),
    text: Optional[str] = Query(None, description="Filter by text content"),
):
    """
    Get bounding box data for highlighting specific chunks on pages.
    Uses bbox metadata stored in Qdrant chunks collection.
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

            # Filter by text if provided
            if text and text.lower() not in chunk_text.lower():
                continue

            # Filter by page if provided
            if page and chunk_page != page:
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
