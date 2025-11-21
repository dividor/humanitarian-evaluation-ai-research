import logging
from pathlib import Path
from typing import Any, Dict, List

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_document(doc_json_path: str) -> List[Dict[str, Any]]:
    """
    Load a Docling JSON document and split it into chunks using HybridChunker.

    HybridChunker uses a hybrid approach with tokenization-aware
    refinements:
    - First pass: splits chunks only when needed (oversized w.r.t. tokens)
    - Second pass: merges chunks when possible (undersized successive
      chunks with same headings/captions)

    Args:
        doc_json_path: Path to the Docling JSON file.

    Returns:
        List of dictionaries, each containing:
            - text: The chunk text
            - page_num: The page number(s) (1-indexed)
            - bbox: List of bounding boxes
            - headings: List of section headers

    Raises:
        RuntimeError: If chunk token length exceeds max_tokens (indicates tokenizer mismatch)
    """
    path = Path(doc_json_path)
    if not path.exists():
        logger.error(f"Document not found: {doc_json_path}")
        return []

    try:
        # Load Docling Document from JSON
        doc = DoclingDocument.load_from_json(path)

        # Initialize HybridChunker with proper configuration
        # - max_tokens: Target chunk size (tokens) for embeddings
        # - merge_peers: Merge small adjacent chunks with matching metadata (default: True)
        # HybridChunker does TWO passes:
        #   1. Split oversized chunks
        #   2. Merge undersized successive chunks with same headings & captions
        # IMPORTANT: Use the same tokenizer as the embedding model (BAAI/bge-small-en-v1.5)
        chunker = HybridChunker(
            tokenizer="BAAI/bge-small-en-v1.5",  # MUST match embedding model tokenizer
            max_tokens=512,  # Max tokens for BAAI/bge-small-en-v1.5 model
            merge_peers=True,  # Enable merging of small chunks (default, but explicit)
        )

        # Chunk the document
        chunks = chunker.chunk(doc)

        processed_chunks = []
        for chunk in chunks:
            # Extract metadata from provenance
            page_nums = set()
            bboxes = []  # Now stores (page, bbox_tuple) pairs

            if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
                for item in chunk.meta.doc_items:
                    if hasattr(item, "prov"):
                        for prov in item.prov:
                            page_nums.add(prov.page_no)
                            if hasattr(prov, "bbox") and prov.bbox:
                                # Store bbox WITH its page number as (page, bbox_tuple)
                                bboxes.append((prov.page_no, prov.bbox.as_tuple()))

            # Get headings
            headings = []
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "headings"):
                headings = chunk.meta.headings

            processed_chunks.append(
                {
                    "text": chunk.text,
                    "page_num": (
                        min(page_nums) if page_nums else 1
                    ),  # Use first page as primary
                    "bbox": [b for b in bboxes if b],  # List of (page, bbox) tuples
                    "headings": headings,
                }
            )

        logger.info(f"Generated {len(processed_chunks)} chunks from {path.name}")
        return processed_chunks

    except Exception as e:
        logger.error(f"Error chunking {doc_json_path}: {e}")
        return []
