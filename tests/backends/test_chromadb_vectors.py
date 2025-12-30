"""
Test ChromaDB Vector Backend

Comprehensive tests for ChromaVectorBackend implementation.
"""

import pytest
import pytest_asyncio
from pathlib import Path
import tempfile
import torch

from fracton.storage.backends.base import VectorPoint, VectorSearchResult

# Try to import ChromaDB
try:
    from fracton.storage.backends import ChromaVectorBackend
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    ChromaVectorBackend = None


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest_asyncio.fixture
async def backend(temp_dir):
    """Create ChromaDB backend."""
    if not CHROMADB_AVAILABLE:
        pytest.skip("ChromaDB not installed")

    backend = ChromaVectorBackend(
        persist_directory=temp_dir / "chroma",
        collection_name="test_collection",
        device="cpu",
    )
    await backend.connect()
    yield backend
    await backend.close()


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
class TestChromaVectorBackend:
    """Test ChromaDB vector backend."""

    @pytest.mark.asyncio
    async def test_connection(self, temp_dir):
        """Test connect and close."""
        backend = ChromaVectorBackend(
            persist_directory=temp_dir / "chroma",
            collection_name="test",
        )

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
        # Reference should be first (exact match)
        assert results[0].id == "ref"
        # Similar vectors should rank higher
        similar_count = sum(1 for r in results if "similar" in r.id)
        assert similar_count >= 2

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

        # Search with high threshold
        query = torch.randn(384)
        results = await backend.search(query, limit=10, threshold=0.9)

        # With random vectors, very few should match high threshold
        # (This is probabilistic but should hold)
        assert len(results) < 5

    @pytest.mark.asyncio
    async def test_search_batch(self, backend):
        """Test batch search."""
        # Store vectors
        for i in range(10):
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
    async def test_create_collection(self, temp_dir):
        """Test collection creation."""
        backend = ChromaVectorBackend(
            persist_directory=temp_dir / "chroma2",
            collection_name="new_collection",
        )
        await backend.connect()

        await backend.create_collection("custom_collection", vector_size=512)

        # Verify collection exists
        info = await backend.collection_info("custom_collection")
        assert info["name"] == "custom_collection"
        assert info["vector_count"] == 0

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
        assert info["vector_count"] == 5

    @pytest.mark.asyncio
    async def test_delete_collection(self, temp_dir):
        """Test collection deletion."""
        backend = ChromaVectorBackend(
            persist_directory=temp_dir / "chroma3",
            collection_name="to_delete",
        )
        await backend.connect()

        # Create and populate
        await backend.create_collection("to_delete", 384)
        vector = torch.randn(384)
        point = VectorPoint(id="vec1", vector=vector, payload={})
        await backend.store(point)

        # Delete
        await backend.delete_collection("to_delete")

        # Verify deleted (should return empty info)
        info = await backend.collection_info("to_delete")
        assert info["vector_count"] == 0

        await backend.close()

    @pytest.mark.asyncio
    async def test_persistence(self, temp_dir):
        """Test that data persists across connections."""
        persist_path = temp_dir / "chroma_persist"

        # Create and store data
        backend1 = ChromaVectorBackend(
            persist_directory=persist_path,
            collection_name="persist_test",
        )
        await backend1.connect()

        vector = torch.randn(384)
        point = VectorPoint(
            id="persistent",
            vector=vector,
            payload={"persisted": True},
        )
        await backend1.store(point)
        await backend1.close()

        # Reconnect and verify
        backend2 = ChromaVectorBackend(
            persist_directory=persist_path,
            collection_name="persist_test",
        )
        await backend2.connect()

        retrieved = await backend2.retrieve(["persistent"])
        assert len(retrieved) == 1
        assert retrieved[0].id == "persistent"
        assert retrieved[0].payload["persisted"] is True
        assert torch.allclose(retrieved[0].vector, vector, atol=1e-5)

        await backend2.close()

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
