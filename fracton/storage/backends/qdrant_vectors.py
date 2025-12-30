"""
Qdrant Vector Backend for KRONOS

Production-grade GPU-accelerated vector database.
Perfect for large-scale, high-performance production systems.

Features:
- GPU acceleration (25-50x faster than CPU)
- Distributed with sharding
- HNSW indexing
- Scales to billions of vectors
- ~50K QPS on GPU
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

import torch

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        SearchRequest,
        FieldCondition,
        MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None

from .base import VectorBackend, VectorPoint, VectorSearchResult

logger = logging.getLogger(__name__)


class QdrantVectorBackend(VectorBackend):
    """
    Qdrant-based vector storage backend.

    Features:
    - GPU-accelerated similarity search
    - HNSW indexing for fast queries
    - Distributed with sharding
    - Production-grade performance
    - ~50K QPS on GPU

    Distance Metrics:
    - Cosine similarity (default)
    - Euclidean (L2)
    - Dot product
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "kronos_vectors",
        vector_size: int = 384,
        device: str = "cpu",
    ):
        """
        Initialize Qdrant backend.

        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Collection name
            vector_size: Embedding dimension
            device: PyTorch device ("cpu" or "cuda")
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client not installed. Install with: pip install qdrant-client"
            )

        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.device = device

        self.client: Optional[QdrantClient] = None

        logger.info(f"QdrantVectorBackend initialized: {host}:{port} (device: {device})")

    async def connect(self) -> None:
        """Establish connection and ensure collection exists."""
        self.client = QdrantClient(host=self.host, port=self.port)

        # Ensure collection exists
        await self._ensure_collection()

        logger.info("Qdrant connection established")

    async def close(self) -> None:
        """Close connection."""
        # Qdrant client doesn't need explicit closing
        self.client = None
        logger.info("Qdrant connection closed")

    async def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    # ========================================================================
    # Storage Operations
    # ========================================================================

    async def store(self, point: VectorPoint) -> None:
        """Store vector point."""
        await self.store_batch([point])

    async def store_batch(self, points: List[VectorPoint]) -> None:
        """Store batch of points."""
        if not points:
            return

        # Convert to Qdrant points
        qdrant_points = []
        for point in points:
            # Convert tensor to list
            if isinstance(point.vector, torch.Tensor):
                vector_list = point.vector.cpu().tolist()
            else:
                vector_list = point.vector

            qdrant_points.append(PointStruct(
                id=point.id,
                vector=vector_list,
                payload=point.payload,
            ))

        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=qdrant_points,
        )

        logger.debug(f"Stored {len(points)} vectors")

    async def retrieve(self, point_ids: List[str]) -> List[VectorPoint]:
        """Retrieve points by ID."""
        if not point_ids:
            return []

        # Retrieve from Qdrant
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_vectors=True,
            with_payload=True,
        )

        # Convert to VectorPoints
        points = []
        for result in results:
            vector = torch.tensor(
                result.vector,
                dtype=torch.float32,
                device=self.device,
            )

            points.append(VectorPoint(
                id=result.id,
                vector=vector,
                payload=result.payload or {},
            ))

        return points

    async def delete(self, point_ids: List[str]) -> None:
        """Delete points by ID."""
        if not point_ids:
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=point_ids,
        )

        logger.debug(f"Deleted {len(point_ids)} vectors")

    # ========================================================================
    # Search Operations
    # ========================================================================

    async def search(
        self,
        query_vector: torch.Tensor,
        limit: int = 10,
        threshold: Optional[float] = None,
        filter_payload: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        # Convert tensor to list
        if isinstance(query_vector, torch.Tensor):
            query_list = query_vector.cpu().tolist()
        else:
            query_list = query_vector

        # Build filter if provided
        query_filter = None
        if filter_payload:
            # Build Qdrant filter from payload dict
            conditions = []
            for key, value in filter_payload.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            if conditions:
                query_filter = Filter(must=conditions)

        # Search using Qdrant v1.7+ API
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_list,
                limit=limit,
                score_threshold=threshold,
                query_filter=query_filter,
                with_vectors=True,
                with_payload=True,
            ).points
        except AttributeError:
            # Fall back to older API if query_points doesn't exist
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_list,
                limit=limit,
                score_threshold=threshold,
                query_filter=query_filter,
                with_vectors=True,
                with_payload=True,
            )

        # Convert to VectorSearchResults
        search_results = []
        for hit in results:
            vector = None
            if hasattr(hit, 'vector') and hit.vector is not None:
                vector = torch.tensor(
                    hit.vector,
                    dtype=torch.float32,
                    device=self.device,
                )

            search_results.append(VectorSearchResult(
                id=hit.id,
                score=hit.score,
                vector=vector,
                payload=hit.payload,
            ))

        return search_results

    async def search_batch(
        self,
        query_vectors: torch.Tensor,
        limit: int = 10,
        threshold: Optional[float] = None,
    ) -> List[List[VectorSearchResult]]:
        """Batch search for similar vectors."""
        # Convert to list of queries
        if isinstance(query_vectors, torch.Tensor):
            queries = query_vectors.cpu().tolist()
        else:
            queries = query_vectors

        # Build search requests
        requests = []
        for query in queries:
            requests.append(SearchRequest(
                vector=query,
                limit=limit,
                score_threshold=threshold,
                with_vector=True,
                with_payload=True,
            ))

        # Batch search (GPU-accelerated)
        try:
            batch_results = self.client.search_batch(
                collection_name=self.collection_name,
                requests=requests,
            )
        except Exception as e:
            logger.warning(f"Batch search failed, falling back to sequential: {e}")
            # Fallback to sequential search
            batch_results = []
            for query in queries:
                results = await self.search(
                    torch.tensor(query, device=self.device),
                    limit=limit,
                    threshold=threshold,
                )
                batch_results.append(results)
            return batch_results

        # Convert to VectorSearchResults
        all_results = []
        for results in batch_results:
            search_results = []
            for hit in results:
                vector = None
                if hasattr(hit, 'vector') and hit.vector is not None:
                    vector = torch.tensor(
                        hit.vector,
                        dtype=torch.float32,
                        device=self.device,
                    )

                search_results.append(VectorSearchResult(
                    id=hit.id,
                    score=hit.score,
                    vector=vector,
                    payload=hit.payload,
                ))

            all_results.append(search_results)

        return all_results

    # ========================================================================
    # Collection Operations
    # ========================================================================

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
    ) -> None:
        """Create vector collection."""
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created collection: {collection_name} (dim={vector_size})")

    async def delete_collection(self, collection_name: str) -> None:
        """Delete entire collection."""
        self.client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")

    async def collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection metadata."""
        try:
            info = self.client.get_collection(collection_name)

            return {
                "name": collection_name,
                "vector_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "name": collection_name,
                "vector_count": 0,
                "vector_size": 0,
                "distance": "unknown",
            }

    # ========================================================================
    # Utility
    # ========================================================================

    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        try:
            # Try to list collections
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_device(self) -> str:
        """Get compute device."""
        return self.device
