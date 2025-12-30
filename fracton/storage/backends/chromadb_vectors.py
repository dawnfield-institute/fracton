"""
ChromaDB Vector Backend for KRONOS

Lightweight, embedded vector database using ChromaDB.
Perfect for development and CPU-only systems.

Features:
- No external server required
- Embedded database (file-based)
- Efficient similarity search
- Metadata filtering
- Up to ~1M vectors (384D)
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import numpy as np

import torch

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

from .base import VectorBackend, VectorPoint, VectorSearchResult

logger = logging.getLogger(__name__)


class ChromaVectorBackend(VectorBackend):
    """
    ChromaDB-based vector storage backend.

    Features:
    - Embedded database (no server)
    - Cosine similarity search
    - Metadata filtering
    - Persistent storage

    Limitations:
    - CPU-only (no GPU acceleration)
    - ~100 QPS for similarity search
    - Up to ~1M vectors
    """

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "kronos_vectors",
        device: str = "cpu",
    ):
        """
        Initialize ChromaDB backend.

        Args:
            persist_directory: Directory for ChromaDB data
            collection_name: Collection name
            device: Device for tensor operations ("cpu" or "cuda")
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.device = device

        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None

        logger.info(f"ChromaVectorBackend initialized: {persist_directory}")

    async def connect(self) -> None:
        """Establish connection and create collection."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Create ChromaDB persistent client (avoids singleton issues)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, will be created on first use
            logger.info(f"Collection will be created on first use: {self.collection_name}")

        logger.info("ChromaDB connection established")

    async def close(self) -> None:
        """Close connection."""
        # Clear references to allow cleanup
        self.collection = None
        if self.client:
            # ChromaDB v0.4+ has clear_system_cache
            try:
                import chromadb
                chromadb.api.shared_system_client.SharedSystemClient._identifier_to_system.clear()
            except:
                pass
        self.client = None
        logger.info("ChromaDB connection closed")

    # ========================================================================
    # Storage Operations
    # ========================================================================

    async def store(self, point: VectorPoint) -> None:
        """Store vector point."""
        await self.store_batch([point])

    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten metadata for ChromaDB compatibility.

        ChromaDB only supports str, int, float, bool values.
        Nested dicts are flattened with dot notation.
        """
        if not metadata:
            # ChromaDB requires at least one metadata field
            return {"_empty": "true"}

        flattened = {}

        def flatten_dict(d: Dict[str, Any], prefix: str = ""):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    # Recursively flatten nested dicts
                    flatten_dict(value, full_key)
                elif isinstance(value, (str, int, float, bool)):
                    # Keep primitive types as-is
                    flattened[full_key] = value
                elif value is None:
                    # Convert None to string
                    flattened[full_key] = "null"
                else:
                    # Convert other types to string
                    flattened[full_key] = str(value)

        flatten_dict(metadata)

        # Ensure at least one field
        if not flattened:
            flattened["_empty"] = "true"

        return flattened

    async def store_batch(self, points: List[VectorPoint]) -> None:
        """Store batch of points."""
        if not points:
            return

        # Ensure collection exists
        if self.collection is None:
            vector_size = points[0].vector.shape[0]
            await self._ensure_collection(vector_size)

        # Convert tensors to numpy and flatten metadata
        ids = [p.id for p in points]
        embeddings = [p.vector.cpu().numpy().tolist() for p in points]
        metadatas = [self._flatten_metadata(p.payload) for p in points]

        # Store in ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.debug(f"Stored {len(points)} vectors")

    async def retrieve(self, point_ids: List[str]) -> List[VectorPoint]:
        """Retrieve points by ID."""
        if not point_ids or self.collection is None:
            return []

        # Get from ChromaDB
        results = self.collection.get(
            ids=point_ids,
            include=["embeddings", "metadatas"],
        )

        # Convert to VectorPoints
        points = []
        for i, point_id in enumerate(results["ids"]):
            vector = torch.tensor(
                results["embeddings"][i],
                dtype=torch.float32,
                device=self.device,
            )
            payload = results["metadatas"][i] if results["metadatas"] else {}

            points.append(VectorPoint(
                id=point_id,
                vector=vector,
                payload=payload,
            ))

        return points

    async def delete(self, point_ids: List[str]) -> None:
        """Delete points by ID."""
        if not point_ids or self.collection is None:
            return

        self.collection.delete(ids=point_ids)
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
        if self.collection is None:
            return []

        # Convert query to numpy
        query_np = query_vector.cpu().numpy().tolist()

        # Build where filter
        where = None
        if filter_payload:
            # ChromaDB uses simple key-value filtering
            where = filter_payload

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_np],
            n_results=limit,
            where=where,
            include=["embeddings", "metadatas", "distances"],
        )

        # Convert to VectorSearchResults
        search_results = []

        if not results["ids"] or not results["ids"][0]:
            return []

        for i, point_id in enumerate(results["ids"][0]):
            # ChromaDB returns L2 distance, convert to cosine similarity
            # similarity = 1 - (distance / 2)  # Approximate for normalized vectors
            distance = results["distances"][0][i]
            similarity = 1.0 - (distance / 2.0)

            # Apply threshold
            if threshold and similarity < threshold:
                continue

            vector = None
            if results["embeddings"] is not None and len(results["embeddings"]) > 0 and results["embeddings"][0] is not None:
                vector = torch.tensor(
                    results["embeddings"][0][i],
                    dtype=torch.float32,
                    device=self.device,
                )

            payload = None
            if results["metadatas"] is not None and len(results["metadatas"]) > 0 and results["metadatas"][0] is not None:
                payload = results["metadatas"][0][i]

            search_results.append(VectorSearchResult(
                id=point_id,
                score=similarity,
                vector=vector,
                payload=payload,
            ))

        # Sort by score descending
        search_results.sort(key=lambda x: x.score, reverse=True)

        return search_results

    async def search_batch(
        self,
        query_vectors: torch.Tensor,
        limit: int = 10,
        threshold: Optional[float] = None,
    ) -> List[List[VectorSearchResult]]:
        """Batch search for similar vectors."""
        if self.collection is None:
            return [[] for _ in range(len(query_vectors))]

        # Convert queries to numpy
        queries_np = query_vectors.cpu().numpy().tolist()

        # Query ChromaDB (batch)
        results = self.collection.query(
            query_embeddings=queries_np,
            n_results=limit,
            include=["embeddings", "metadatas", "distances"],
        )

        # Convert to list of VectorSearchResults
        all_results = []

        for batch_idx in range(len(queries_np)):
            batch_results = []

            if not results["ids"] or batch_idx >= len(results["ids"]):
                all_results.append([])
                continue

            for i, point_id in enumerate(results["ids"][batch_idx]):
                distance = results["distances"][batch_idx][i]
                similarity = 1.0 - (distance / 2.0)

                # Apply threshold
                if threshold and similarity < threshold:
                    continue

                vector = None
                if results["embeddings"] is not None and len(results["embeddings"]) > batch_idx and results["embeddings"][batch_idx] is not None:
                    vector = torch.tensor(
                        results["embeddings"][batch_idx][i],
                        dtype=torch.float32,
                        device=self.device,
                    )

                payload = None
                if results["metadatas"] is not None and len(results["metadatas"]) > batch_idx and results["metadatas"][batch_idx] is not None:
                    payload = results["metadatas"][batch_idx][i]

                batch_results.append(VectorSearchResult(
                    id=point_id,
                    score=similarity,
                    vector=vector,
                    payload=payload,
                ))

            # Sort by score descending
            batch_results.sort(key=lambda x: x.score, reverse=True)
            all_results.append(batch_results)

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
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"vector_size": vector_size},
            )
            self.collection_name = collection_name
            logger.info(f"Created collection: {collection_name} (dim={vector_size})")
        except Exception as e:
            logger.warning(f"Collection may already exist: {e}")
            self.collection = self.client.get_collection(collection_name)

    async def delete_collection(self, collection_name: str) -> None:
        """Delete entire collection."""
        self.client.delete_collection(collection_name)
        if collection_name == self.collection_name:
            self.collection = None
        logger.info(f"Deleted collection: {collection_name}")

    async def collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection metadata."""
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()

            return {
                "name": collection_name,
                "vector_count": count,
                "metadata": collection.metadata or {},
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "name": collection_name,
                "vector_count": 0,
                "metadata": {},
            }

    # ========================================================================
    # Utility
    # ========================================================================

    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        try:
            # Try to list collections
            self.client.list_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_device(self) -> str:
        """Get compute device."""
        return self.device

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _ensure_collection(self, vector_size: int) -> None:
        """Ensure collection exists."""
        if self.collection is None:
            await self.create_collection(self.collection_name, vector_size)
