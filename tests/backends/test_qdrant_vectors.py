"""
Test Qdrant Vector Backend

Comprehensive tests for QdrantVectorBackend implementation.
Requires Qdrant server running on localhost:6333.
"""

import pytest
import pytest_asyncio
from pathlib import Path
import torch
import os

from fracton.storage.backends.base import VectorPoint, VectorSearchResult

# Try to import Qdrant backend
try:
    from fracton.storage.backends import QdrantVectorBackend
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantVectorBackend = None


# Check if Qdrant is actually running
QDRANT_RUNNING = False
if QDRANT_AVAILABLE:
    import asyncio
    try:
        async def check_qdrant():
            backend = QdrantVectorBackend(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
                collection_name="test_check",
            )
            try:
                await backend.connect()
                await backend.close()
                return True
            except:
                return False
        QDRANT_RUNNING = asyncio.run(check_qdrant())
    except:
        pass


@pytest_asyncio.fixture
async def backend():
    """Create Qdrant backend."""
    if not QDRANT_AVAILABLE:
        pytest.skip("Qdrant client not installed")
    if not QDRANT_RUNNING:
        pytest.skip("Qdrant server not running on localhost:6333")

    backend = QdrantVectorBackend(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        collection_name="test_collection",
        vector_size=384,
        device="cpu",
    )
    await backend.connect()

    # Clean up test collection
    try:
        await backend.delete_collection("test_collection")
    except:
        pass

    # Recreate
    await backend.create_collection("test_collection", 384)
    backend.collection_name = "test_collection"

    yield backend

    # Clean up after tests
    try:
        await backend.delete_collection("test_collection")
    except:
        pass

    await backend.close()


@pytest.mark.skipif(not QDRANT_AVAILABLE or not QDRANT_RUNNING, reason="Qdrant not available")
class TestQdrantVectorBackend:
    """Test Qdrant vector backend."""

    @pytest.mark.asyncio
    async def test_connection(self):
        """Test connect and close."""
        backend = QdrantVectorBackend(collection_name="test_conn")
        await backend.connect()
        assert backend.client is not None
        await backend.close()
        assert backend.client is None

    @pytest.mark.asyncio
    async def test_health_check(self, backend):
        """Test health check."""
        is_healthy = await backend.health_check()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_get_device(self, backend):
        """Test device getter."""
        device = backend.get_device()
        assert device == "cpu"

    @pytest.mark.asyncio
    async def test_store_single(self, backend):
        """Test storing single vector."""
        vector = torch.randn(384)
        point = VectorPoint(
            id="vec1",
            vector=vector,
            payload={"content": "test", "metadata": {"key": "value"}},
        )

        await backend.store(point)

        # Retrieve and verify
        retrieved = await backend.retrieve(["vec1"])
        assert len(retrieved) == 1
        assert retrieved[0].id == "vec1"
        assert retrieved[0].payload["content"] == "test"
        assert torch.allclose(retrieved[0].vector, vector, atol=1e-5)

    @pytest.mark.asyncio
    async def test_store_batch(self, backend):
        """Test batch storage."""
        points = []
        for i in range(10):
            vector = torch.randn(384)
            point = VectorPoint(
                id=f"vec{i}",
                vector=vector,
                payload={"content": f"content_{i}", "index": i},
            )
            points.append(point)

        await backend.store_batch(points)

        # Retrieve all
        ids = [f"vec{i}" for i in range(10)]
        retrieved = await backend.retrieve(ids)

        assert len(retrieved) == 10
        retrieved_ids = {p.id for p in retrieved}
        assert retrieved_ids == set(ids)

    @pytest.mark.asyncio
    async def test_search(self, backend):
        """Test similarity search."""
        # Create and store reference vector
        ref_vector = torch.randn(384)
        ref_point = VectorPoint(
            id="ref",
            vector=ref_vector,
            payload={"type": "reference"},
        )
        await backend.store(ref_point)

        # Create and store similar vectors
        for i in range(5):
            # Add small noise to reference
            similar_vector = ref_vector + torch.randn(384) * 0.1
            point = VectorPoint(
                id=f"similar{i}",
                vector=similar_vector,
                payload={"type": "similar", "index": i},
            )
            await backend.store(point)

        # Create and store dissimilar vectors
        for i in range(5):
            dissimilar_vector = torch.randn(384)
            point = VectorPoint(
                id=f"dissimilar{i}",
                vector=dissimilar_vector,
                payload={"type": "dissimilar", "index": i},
            )
            await backend.store(point)

        # Search
        results = await backend.search(ref_vector, limit=3)

        assert len(results) <= 3
        assert len(results) > 0
        # Reference should be first (exact match)
        assert results[0].id == "ref"
        # Check scores are descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_with_threshold(self, backend):
        """Test search with similarity threshold."""
        # Store vectors
        for i in range(5):
            vector = torch.randn(384)
            point = VectorPoint(
                id=f"vec{i}",
                vector=vector,
                payload={"index": i},
            )
            await backend.store(point)

        # Search with threshold
        query = torch.randn(384)
        results = await backend.search(query, limit=10, threshold=0.5)

        # All results should meet threshold
        for result in results:
            assert result.score >= 0.5

    @pytest.mark.asyncio
    async def test_search_with_filter(self, backend):
        """Test search with payload filtering."""
        # Store vectors with different types
        for i in range(10):
            vector = torch.randn(384)
            point = VectorPoint(
                id=f"vec{i}",
                vector=vector,
                payload={"type": "even" if i % 2 == 0 else "odd", "index": i},
            )
            await backend.store(point)

        # Search with filter
        query = torch.randn(384)
        results = await backend.search(
            query,
            limit=10,
            filter_payload={"type": "even"}
        )

        # All results should have type=even
        for result in results:
            assert result.payload["type"] == "even"

    @pytest.mark.asyncio
    async def test_search_batch(self, backend):
        """Test batch search."""
        # Store vectors
        for i in range(20):
            vector = torch.randn(384)
            point = VectorPoint(
                id=f"vec{i}",
                vector=vector,
                payload={"index": i},
            )
            await backend.store(point)

        # Batch search
        queries = torch.randn(3, 384)
        results = await backend.search_batch(queries, limit=5)

        assert len(results) == 3
        for batch_results in results:
            assert len(batch_results) <= 5
            # Check results are sorted by score
            scores = [r.score for r in batch_results]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_delete(self, backend):
        """Test vector deletion."""
        # Store vectors
        for i in range(5):
            vector = torch.randn(384)
            point = VectorPoint(
                id=f"vec{i}",
                vector=vector,
                payload={"index": i},
            )
            await backend.store(point)

        # Verify stored
        retrieved = await backend.retrieve(["vec0", "vec1", "vec2"])
        assert len(retrieved) == 3

        # Delete
        await backend.delete(["vec0", "vec2"])

        # Verify deleted
        retrieved = await backend.retrieve(["vec0", "vec1", "vec2"])
        ids = {p.id for p in retrieved}
        assert "vec0" not in ids
        assert "vec1" in ids
        assert "vec2" not in ids

    @pytest.mark.asyncio
    async def test_create_collection(self):
        """Test collection creation."""
        if not QDRANT_AVAILABLE or not QDRANT_RUNNING:
            pytest.skip("Qdrant not available")

        backend = QdrantVectorBackend(collection_name="custom_collection")
        await backend.connect()

        try:
            await backend.create_collection("custom_collection", vector_size=512)

            # Verify collection exists
            info = await backend.collection_info("custom_collection")
            assert info["name"] == "custom_collection"
            assert info["vector_size"] == 512
        finally:
            try:
                await backend.delete_collection("custom_collection")
            except:
                pass
            await backend.close()

    @pytest.mark.asyncio
    async def test_collection_info(self, backend):
        """Test collection metadata."""
        # Store some vectors
        for i in range(5):
            vector = torch.randn(384)
            point = VectorPoint(
                id=f"vec{i}",
                vector=vector,
                payload={"index": i},
            )
            await backend.store(point)

        # Get info
        info = await backend.collection_info("test_collection")

        assert info["name"] == "test_collection"
        assert info["vector_count"] >= 5
        assert info["vector_size"] == 384

    @pytest.mark.asyncio
    async def test_delete_collection(self):
        """Test collection deletion."""
        if not QDRANT_AVAILABLE or not QDRANT_RUNNING:
            pytest.skip("Qdrant not available")

        backend = QdrantVectorBackend(collection_name="to_delete")
        await backend.connect()

        try:
            # Create and populate
            await backend.create_collection("to_delete", 384)
            vector = torch.randn(384)
            point = VectorPoint(id="vec1", vector=vector, payload={})
            await backend.store(point)

            # Delete
            await backend.delete_collection("to_delete")

            # Verify deleted - collection_info should return minimal info
            info = await backend.collection_info("to_delete")
            assert info["vector_count"] == 0
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_empty_search(self, backend):
        """Test search on empty collection returns empty results."""
        query = torch.randn(384)
        results = await backend.search(query, limit=10)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_retrieve_missing_ids(self, backend):
        """Test retrieving non-existent IDs."""
        # Store one vector
        vector = torch.randn(384)
        point = VectorPoint(id="exists", vector=vector, payload={})
        await backend.store(point)

        # Try to retrieve mix of existing and non-existing
        retrieved = await backend.retrieve(["exists", "missing1", "missing2"])

        # Should only return existing
        assert len(retrieved) == 1
        assert retrieved[0].id == "exists"

    @pytest.mark.asyncio
    async def test_large_batch(self, backend):
        """Test large batch operations."""
        # Store 100 vectors
        points = []
        for i in range(100):
            vector = torch.randn(384)
            point = VectorPoint(
                id=f"vec{i}",
                vector=vector,
                payload={"index": i},
            )
            points.append(point)

        await backend.store_batch(points)

        # Retrieve all
        ids = [f"vec{i}" for i in range(100)]
        retrieved = await backend.retrieve(ids)

        assert len(retrieved) == 100

    @pytest.mark.asyncio
    async def test_gpu_device(self):
        """Test GPU device configuration."""
        if not QDRANT_AVAILABLE or not QDRANT_RUNNING:
            pytest.skip("Qdrant not available")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        backend = QdrantVectorBackend(
            collection_name="gpu_test",
            device="cuda",
        )
        await backend.connect()

        try:
            # Store vector
            vector = torch.randn(384, device="cuda")
            point = VectorPoint(id="gpu_vec", vector=vector, payload={})
            await backend.store(point)

            # Retrieve
            retrieved = await backend.retrieve(["gpu_vec"])
            assert len(retrieved) == 1
            assert retrieved[0].vector.device.type == "cuda"
        finally:
            try:
                await backend.delete_collection("gpu_test")
            except:
                pass
            await backend.close()

    @pytest.mark.asyncio
    async def test_cosine_similarity(self, backend):
        """Test that cosine similarity works correctly."""
        # Create identical vectors (should have similarity ~1.0)
        vec1 = torch.randn(384)
        vec1 = vec1 / vec1.norm()  # Normalize

        point1 = VectorPoint(id="vec1", vector=vec1, payload={})
        await backend.store(point1)

        # Search with same vector
        results = await backend.search(vec1, limit=1)

        assert len(results) == 1
        assert results[0].id == "vec1"
        # Cosine similarity should be very close to 1.0
        assert results[0].score > 0.99

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, backend):
        """Test concurrent store operations."""
        import asyncio

        # Create multiple coroutines that store vectors
        async def store_vector(i):
            vector = torch.randn(384)
            point = VectorPoint(
                id=f"concurrent{i}",
                vector=vector,
                payload={"index": i},
            )
            await backend.store(point)

        # Run 10 concurrent stores
        await asyncio.gather(*[store_vector(i) for i in range(10)])

        # Verify all were stored
        ids = [f"concurrent{i}" for i in range(10)]
        retrieved = await backend.retrieve(ids)
        assert len(retrieved) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
