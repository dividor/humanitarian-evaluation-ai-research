import logging
import os
from typing import Any, Dict, Generator, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
DOCUMENTS_COLLECTION = "documents"
CHUNKS_COLLECTION = "chunks"

# Vector Configuration
DENSE_VECTOR_SIZE = 384  # bge-small-en-v1.5
SPARSE_VECTOR_NAME = "sparse"


class Database:
    def __init__(self):
        # Try localhost first (for running on host), then qdrant (for Docker)
        host = os.getenv("QDRANT_HOST")
        if not host:
            # Auto-detect: try localhost, fall back to qdrant
            try:
                test_client = QdrantClient(host="localhost", port=6333, timeout=2)
                test_client.get_collections()
                host = "localhost"
            except:
                host = "qdrant"

        port = int(os.getenv("QDRANT_PORT", "6333"))

        self.client = QdrantClient(host=host, port=port)
        self.init_collections()
        self.create_payload_indexes()  # Automatically create indexes for faceting

    def init_collections(self):
        """Initialize collections if they don't exist."""
        # Get existing collections
        existing_collections = {
            c.name for c in self.client.get_collections().collections
        }

        # 1. Documents Collection (with vectors for document-level search)
        if DOCUMENTS_COLLECTION not in existing_collections:
            logger.info(f"Creating collection: {DOCUMENTS_COLLECTION}")
            self.client.create_collection(
                collection_name=DOCUMENTS_COLLECTION,
                vectors_config={
                    "dense": VectorParams(
                        size=DENSE_VECTOR_SIZE, distance=Distance.COSINE
                    )
                },
            )
        else:
            # Update existing collection to add vectors if needed
            try:
                collection_info = self.client.get_collection(DOCUMENTS_COLLECTION)
                if not collection_info.config.params.vectors:
                    logger.info(
                        f"Updating {DOCUMENTS_COLLECTION} to add vector support"
                    )
                    self.client.update_collection(
                        collection_name=DOCUMENTS_COLLECTION,
                        vectors_config={
                            "dense": VectorParams(
                                size=DENSE_VECTOR_SIZE, distance=Distance.COSINE
                            )
                        },
                    )
            except Exception as e:
                logger.warning(f"Could not update collection config: {e}")

        # 2. Chunks Collection (Hybrid Search with document metadata)
        if CHUNKS_COLLECTION not in existing_collections:
            logger.info(f"Creating collection: {CHUNKS_COLLECTION}")
            self.client.create_collection(
                collection_name=CHUNKS_COLLECTION,
                vectors_config={
                    "dense": VectorParams(
                        size=DENSE_VECTOR_SIZE, distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )

    def create_payload_indexes(self):
        """Create payload indexes for faceting on both collections.
        Only creates indexes if collections have data (indexes created during first indexing).
        """

        # Check if collections have data - skip if empty
        try:
            docs_count = self.client.count(collection_name=DOCUMENTS_COLLECTION).count
            chunks_count = self.client.count(collection_name=CHUNKS_COLLECTION).count

            if docs_count == 0 and chunks_count == 0:
                logger.debug(
                    "Collections empty - indexes will be created during indexing"
                )
                return
        except Exception as e:
            logger.debug(
                f"Could not check collection counts, skipping index creation: {e}"
            )
            return

        logger.info("Creating payload indexes for faceting...")

        # Fields to index for faceting
        facet_fields = [
            ("organization", models.PayloadSchemaType.KEYWORD),
            ("year", models.PayloadSchemaType.INTEGER),
            ("country", models.PayloadSchemaType.KEYWORD),
            ("region", models.PayloadSchemaType.KEYWORD),
            ("evaluation_type", models.PayloadSchemaType.KEYWORD),
            ("theme", models.PayloadSchemaType.KEYWORD),
        ]

        # Create indexes on documents collection
        for field_name, field_type in facet_fields:
            try:
                self.client.create_payload_index(
                    collection_name=DOCUMENTS_COLLECTION,
                    field_name=field_name,
                    field_schema=field_type,
                )
                logger.debug(f"Created index on documents.{field_name}")
            except Exception as e:
                # Silently skip errors (index may already exist)
                pass

        # Create indexes on chunks collection (for denormalized metadata)
        for field_name, field_type in facet_fields:
            try:
                self.client.create_payload_index(
                    collection_name=CHUNKS_COLLECTION,
                    field_name=field_name,
                    field_schema=field_type,
                )
                logger.debug(f"Created index on chunks.{field_name}")
            except Exception as e:
                # Silently skip errors (index may already exist)
                pass

        logger.info("Payload indexes ready")

    def upsert_document(
        self,
        doc_id: str,
        metadata: Dict[str, Any],
        vector: Optional[List[float]] = None,
    ):
        """Upsert a document metadata record with optional embedding vector."""
        point = models.PointStruct(
            id=doc_id, vector={"dense": vector} if vector else {}, payload=metadata
        )
        self.client.upsert(collection_name=DOCUMENTS_COLLECTION, points=[point])

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document metadata by ID."""
        results = self.client.retrieve(
            collection_name=DOCUMENTS_COLLECTION, ids=[doc_id]
        )
        if results:
            return results[0].payload
        return None

    def document_exists(self, url: str) -> bool:
        """Check if a document with the given URL exists."""
        # Filter by URL in payload
        results = self.client.scroll(
            collection_name=DOCUMENTS_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="url", match=models.MatchValue(value=url))
                ]
            ),
            limit=1,
        )
        return len(results[0]) > 0

    def get_documents_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Retrieve documents with a specific status."""
        results, _ = self.client.scroll(
            collection_name=DOCUMENTS_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="status", match=models.MatchValue(value=status)
                    )
                ]
            ),
            limit=10000,  # Fetch all for now
            with_payload=True,
        )
        return [point.payload for point in results]

    def update_document(self, doc_id: str, updates: Dict[str, Any]):
        """Update specific fields of a document."""
        # Retrieve current payload to merge (Qdrant set_payload is partial update)
        self.client.set_payload(
            collection_name=DOCUMENTS_COLLECTION, payload=updates, points=[doc_id]
        )

    def get_all_documents(self) -> Generator[Dict[str, Any], None, None]:
        """Yield all documents from the collection."""
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=DOCUMENTS_COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for point in results:
                yield point.payload

            if offset is None:
                break

    def upsert_chunks(self, points: List[models.PointStruct]):
        """Upsert a batch of chunks."""
        self.client.upsert(collection_name=CHUNKS_COLLECTION, points=points)

    def facet(
        self,
        collection_name: str,
        key: str,
        filter_conditions: Optional[models.Filter] = None,
        limit: int = 100,
        exact: bool = False,
    ):
        """
        Get facet counts for a field using Qdrant's native faceting.

        Args:
            collection_name: Name of the collection (documents or chunks)
            key: Field name to facet on
            filter_conditions: Optional filters to apply
            limit: Maximum number of facet values to return
            exact: Whether to use exact counting (slower but accurate)
        """
        from qdrant_client.http.models import FacetRequest

        request = FacetRequest(
            key=key, limit=limit, exact=exact, filter=filter_conditions
        )

        return self.client.facet(collection_name=collection_name, facet_request=request)


# Singleton instance
db = Database()
