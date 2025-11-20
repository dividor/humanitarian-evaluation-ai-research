import argparse
import logging
import os
import sys
from typing import Any, Dict, List

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client.http import models

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.db import CHUNKS_COLLECTION, db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache models for reuse
_dense_model = None
_sparse_model = None


def get_models():
    """Get or initialize embedding models (cached)"""
    global _dense_model, _sparse_model
    if _dense_model is None:
        logger.info("Loading embedding models...")
        _dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("✓ Models loaded")
    return _dense_model, _sparse_model


def search_chunks(query: str, limit: int = 10) -> List[Any]:
    """
    Search chunks and return raw Qdrant results.
    Used by API to get chunks with scores.

    Args:
        query: Search query string
        limit: Maximum results to return

    Returns:
        List of Qdrant ScoredPoint objects with payload and score
    """
    dense_model, sparse_model = get_models()

    # Embed Query
    dense_vec = list(dense_model.embed([query]))[0]
    sparse_vec = list(sparse_model.embed([query]))[0]

    # Search Chunks (dense vector search)
    search_result = db.client.search(
        collection_name=CHUNKS_COLLECTION,
        query_vector=models.NamedVector(name="dense", vector=dense_vec.tolist()),
        query_filter=None,
        limit=limit,
        with_payload=True,
    )

    return search_result


def search(query: str, limit: int = 5):
    """
    Search the index for the query.
    CLI interface for semantic search.
    """
    # Search using shared function
    search_result = search_chunks(query, limit)

    # Aggregate results by Document
    logger.info(f"Found {len(search_result)} hits.")

    for hit in search_result:
        payload = hit.payload
        doc_id = payload.get("doc_id")
        text = payload.get("text")
        page = payload.get("page_num")
        score = hit.score

        # Fetch Document Metadata
        doc_meta = db.get_document(doc_id)
        agency = doc_meta.get("agency", "Unknown") if doc_meta else "Unknown"
        year = doc_meta.get("year", "Unknown") if doc_meta else "Unknown"
        title = doc_meta.get("title", "Unknown") if doc_meta else "Unknown"

        print(f"\n--- Score: {score:.4f} ---")
        print(f"Document: {title} ({agency}, {year})")
        print(f"Page: {page}")
        print(f"Snippet: {text[:200]}...")
        print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search the humanitarian evaluation index."
    )
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    args = parser.parse_args()
    search(args.query, args.limit)
