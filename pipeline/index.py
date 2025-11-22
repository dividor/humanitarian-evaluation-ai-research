import logging
import os
import sys
import uuid
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client.http import models

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.chunk import chunk_document  # noqa: E402
from pipeline.db import db  # noqa: E402
from pipeline.utils import generate_doc_id  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize Embedding Models
logger.info("Loading embedding models...")
dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
logger.info("Models loaded.")


def index_documents(limit: int = 10, force: bool = False):
    """
    Index parsed documents into Qdrant with dual-mode architecture:
    1. Chunks: Text chunks with embeddings + document metadata (for detailed search)
    2. Documents: Document-level with embeddings from title+summary (for overview search)

    Args:
        limit: Maximum number of documents to process.
        force: If True, re-index documents that are already indexed.
    """
    logger.info("Fetching documents to index from Qdrant...")

    # Get documents based on force flag
    if force:
        # Re-index all parsed documents regardless of status
        logger.info("Force mode: Re-indexing all parsed documents...")
        docs_to_index = db.get_documents_by_status("parsed")

        # Also get already indexed documents
        indexed_docs = db.get_documents_by_status("indexed")
        if indexed_docs:
            docs_to_index.extend(indexed_docs)

        # Also get summarized documents
        summarized_docs = db.get_documents_by_status("summarized")
        if summarized_docs:
            docs_to_index.extend(summarized_docs)
    else:
        # Only index documents that haven't been indexed yet
        docs_to_index = db.get_documents_by_status("summarized")
        if not docs_to_index:
            logger.info("No summarized documents found, trying parsed documents...")
            docs_to_index = db.get_documents_by_status("parsed")

    if not docs_to_index:
        logger.info("No documents found to index.")
        return

    logger.info(f"Found {len(docs_to_index)} documents to index.")

    processed = 0
    for doc in docs_to_index:
        if processed >= limit:
            break

        # Construct expected JSON path
        parsed_folder = doc.get("parsed_folder")
        if not parsed_folder:
            continue

        # Fix path if it's relative/absolute mismatch
        if parsed_folder.startswith("/"):
            parsed_folder = "." + parsed_folder
        elif not parsed_folder.startswith("./"):
            parsed_folder = "./" + parsed_folder

        # Look for .json file in that folder
        json_path = Path(parsed_folder) / f"{Path(parsed_folder).name}.json"

        if not json_path.exists():
            logger.warning(f"JSON not found for {doc.get('title')}: {json_path}")
            continue

        logger.info(f"Indexing {doc.get('title')}...")
        doc_id = generate_doc_id(doc.get("url") or doc.get("filepath"))

        # 1. CHUNK INDEXING: Chunk document text
        chunks = chunk_document(str(json_path))
        if not chunks:
            logger.warning(f"No chunks generated for {doc.get('title')}")
            continue

        # 2. Embed chunks and validate token lengths
        texts = [c["text"] for c in chunks]

        # CRITICAL: Validate chunk token lengths before embedding
        # This catches tokenizer mismatches between chunking and embedding
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        max_allowed_tokens = 512

        for idx, text in enumerate(texts):
            token_count = len(tokenizer.encode(text, add_special_tokens=True))
            if token_count > max_allowed_tokens:
                error_msg = (
                    f"CRITICAL ERROR: Chunk {idx} has {token_count} tokens, "
                    f"exceeding max of {max_allowed_tokens}. "
                    f"This indicates a tokenizer mismatch between chunking and embedding. "
                    f"Document: {doc.get('title')} "
                    f"Chunk preview: {text[:200]}..."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        dense_embeddings = list(dense_model.embed(texts))
        sparse_embeddings = list(sparse_model.embed(texts))

        # 3. Prepare chunk points WITH document metadata (denormalized)
        chunk_points = []
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{i}"))
            sparse_vec = sparse_embeddings[i]

            chunk_points.append(
                models.PointStruct(
                    id=chunk_id,
                    vector={
                        "dense": dense_embeddings[i].tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload={
                        # Chunk-specific data
                        "text": chunk["text"],
                        "doc_id": doc_id,
                        "page_num": chunk["page_num"],
                        # Convert bbox tuples to nested lists for JSON compatibility
                        # Format: [[page, [l, b, r, t]], [page, [l, b, r, t]], ...]
                        "bbox": [
                            (
                                [
                                    bbox[0],
                                    (
                                        list(bbox[1])
                                        if isinstance(bbox[1], tuple)
                                        else bbox[1]
                                    ),
                                ]
                                if isinstance(bbox, tuple) and len(bbox) == 2
                                else bbox
                            )
                            for bbox in chunk["bbox"]
                        ],
                        "headings": chunk["headings"],
                        # Document metadata (denormalized for faceting)
                        "title": doc.get("title"),
                        "organization": doc.get("organization"),
                        "year": int(doc.get("year")) if doc.get("year") else None,
                        "country": doc.get("country"),
                        "region": doc.get("region"),
                        "evaluation_type": doc.get("evaluation_type"),
                        "theme": doc.get("theme"),
                    },
                )
            )

        # 4. Upsert chunks
        db.upsert_chunks(chunk_points)
        logger.info(f"Upserted {len(chunk_points)} chunks for {doc.get('title')}")

        # 5. DOCUMENT INDEXING: Create document-level embedding from title + summary
        doc_text = doc.get("title", "")

        # Add summary if available (check various summary fields)
        summary_fields = [
            "abstractive_summary",
            "key_content_sections",
            "centroid_summary",
        ]
        for field in summary_fields:
            if doc.get(field):
                doc_text += "\n\n" + str(doc.get(field))
                break  # Use first available summary

        # Generate document-level embedding
        doc_embedding = list(dense_model.embed([doc_text]))[0].tolist()

        # 6. Update document in documents collection with embedding and status
        try:
            # Update with embedding and status
            doc["status"] = "indexed"
            db.upsert_document(doc_id, doc, vector=doc_embedding)
            logger.info(
                f"Updated document embedding and status to 'indexed' for {doc.get('title')}"
            )
        except Exception as e:
            logger.warning(f"Could not update document embedding for {doc_id}: {e}")

        processed += 1

    logger.info(f"Indexing complete. Processed {processed} documents.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Index documents into Qdrant for semantic search"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of documents to process"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index all documents, including already indexed ones",
    )

    args = parser.parse_args()

    index_documents(limit=args.limit, force=args.force)
