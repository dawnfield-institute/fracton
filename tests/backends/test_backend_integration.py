"""
Backend Integration Tests

Tests that verify all backends work together and are compatible with each other.
These tests ensure:
1. All backends implement the same interface correctly
2. Data can be migrated between backends
3. Performance characteristics are as expected
4. Edge cases are handled consistently
"""

import pytest
import pytest_asyncio
from datetime import datetime
from pathlib import Path
import tempfile
import torch
import os
import time

from fracton.storage.backends import (
    SQLiteGraphBackend,
    ChromaVectorBackend,
    BackendConfig,
)
from fracton.storage.backends.base import GraphNode, GraphEdge, VectorPoint

# Optional production backends
try:
    from fracton.storage.backends import Neo4jGraphBackend
    NEO4J_AVAILABLE = True
except (ImportError, TypeError):
    NEO4J_AVAILABLE = False
    Neo4jGraphBackend = None

try:
    from fracton.storage.backends import QdrantVectorBackend
    QDRANT_AVAILABLE = True
except (ImportError, TypeError):
    QDRANT_AVAILABLE = False
    QdrantVectorBackend = None


# Check if services are running
def check_neo4j():
    if not NEO4J_AVAILABLE:
        return False
    import asyncio
    try:
        async def test():
            b = Neo4jGraphBackend()
            await b.connect()
            await b.close()
            return True
        return asyncio.run(test())
    except:
        return False


def check_qdrant():
    if not QDRANT_AVAILABLE:
        return False
    import asyncio
    try:
        async def test():
            b = QdrantVectorBackend(collection_name="test")
            await b.connect()
            await b.close()
            return True
        return asyncio.run(test())
    except:
        return False


NEO4J_RUNNING = check_neo4j()
QDRANT_RUNNING = check_qdrant()


class TestGraphBackendCompatibility:
    """Test that all graph backends implement the same interface."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest_asyncio.fixture
    async def sqlite_backend(self, temp_dir):
        """SQLite backend."""
        backend = SQLiteGraphBackend(temp_dir / "test.db")
        await backend.connect()
        yield backend
        await backend.close()

    @pytest_asyncio.fixture
    async def neo4j_backend(self):
        """Neo4j backend."""
        if not NEO4J_RUNNING:
            pytest.skip("Neo4j not running")

        backend = Neo4jGraphBackend()
        await backend.connect()

        # Clean up
        async with backend.driver.session() as session:
            await session.run("MATCH (m:Memory {graph_name: 'integration_test'}) DETACH DELETE m")

        yield backend

        # Clean up
        async with backend.driver.session() as session:
            await session.run("MATCH (m:Memory {graph_name: 'integration_test'}) DETACH DELETE m")

        await backend.close()

    @pytest.mark.asyncio
    async def test_create_retrieve_node_sqlite(self, sqlite_backend):
        """Test node creation and retrieval on SQLite."""
        await self._test_create_retrieve_node(sqlite_backend)

    @pytest.mark.asyncio
    async def test_create_retrieve_node_neo4j(self, neo4j_backend):
        """Test node creation and retrieval on Neo4j."""
        await self._test_create_retrieve_node(neo4j_backend)

    async def _test_create_retrieve_node(self, backend):
        """Common test for node creation."""
        await backend.create_graph("integration_test")

        node = GraphNode(
            id="test_node",
            content="Test content",
            timestamp=datetime.now(),
            fractal_signature="sig123",
            metadata={"key": "value"},
            parent_id="-1",
            children_ids=[],
            potential=0.8,
            entropy=0.3,
            coherence=0.6,
            phase="STABLE",
        )

        await backend.create_node(node, "integration_test")

        retrieved = await backend.get_node("test_node", "integration_test")
        assert retrieved is not None
        assert retrieved.id == "test_node"
        assert retrieved.content == "Test content"
        assert retrieved.metadata["key"] == "value"

    @pytest.mark.asyncio
    async def test_lineage_tracing_sqlite(self, sqlite_backend):
        """Test lineage tracing on SQLite."""
        await self._test_lineage_tracing(sqlite_backend)

    @pytest.mark.asyncio
    async def test_lineage_tracing_neo4j(self, neo4j_backend):
        """Test lineage tracing on Neo4j."""
        await self._test_lineage_tracing(neo4j_backend)

    async def _test_lineage_tracing(self, backend):
        """Common test for lineage tracing."""
        await backend.create_graph("integration_test")

        # Create chain: root -> child -> grandchild
        root = GraphNode(
            id="root",
            content="Root",
            timestamp=datetime.now(),
            fractal_signature="sig_root",
            parent_id="-1",
            children_ids=["child"],
            entropy=0.1,
        )
        await backend.create_node(root, "integration_test")

        child = GraphNode(
            id="child",
            content="Child",
            timestamp=datetime.now(),
            fractal_signature="sig_child",
            parent_id="root",
            children_ids=["grandchild"],
            entropy=0.2,
        )
        await backend.create_node(child, "integration_test")

        grandchild = GraphNode(
            id="grandchild",
            content="Grandchild",
            timestamp=datetime.now(),
            fractal_signature="sig_grand",
            parent_id="child",
            children_ids=[],
            entropy=0.3,
        )
        await backend.create_node(grandchild, "integration_test")

        # Trace backward
        path = await backend.trace_lineage("grandchild", "backward", graph_name="integration_test")

        assert "root" in path.path
        assert "child" in path.path
        assert "grandchild" in path.path


class TestVectorBackendCompatibility:
    """Test that all vector backends implement the same interface."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest_asyncio.fixture
    async def chroma_backend(self, temp_dir):
        """ChromaDB backend."""
        backend = ChromaVectorBackend(
            persist_directory=temp_dir / "chroma",
            collection_name="integration_test",
        )
        await backend.connect()
        yield backend
        await backend.close()

    @pytest_asyncio.fixture
    async def qdrant_backend(self):
        """Qdrant backend."""
        if not QDRANT_RUNNING:
            pytest.skip("Qdrant not running")

        backend = QdrantVectorBackend(collection_name="integration_test")
        await backend.connect()

        # Clean up
        try:
            await backend.delete_collection("integration_test")
        except:
            pass
        await backend.create_collection("integration_test", 384)
        backend.collection_name = "integration_test"

        yield backend

        # Clean up
        try:
            await backend.delete_collection("integration_test")
        except:
            pass
        await backend.close()

    @pytest.mark.asyncio
    async def test_store_search_chroma(self, chroma_backend):
        """Test vector storage and search on ChromaDB."""
        await self._test_store_search(chroma_backend)

    @pytest.mark.asyncio
    async def test_store_search_qdrant(self, qdrant_backend):
        """Test vector storage and search on Qdrant."""
        await self._test_store_search(qdrant_backend)

    async def _test_store_search(self, backend):
        """Common test for vector storage and search."""
        # Store reference vector
        ref_vector = torch.randn(384)
        ref_point = VectorPoint(
            id="ref",
            vector=ref_vector,
            payload={"type": "reference"},
        )
        await backend.store(ref_point)

        # Store similar vectors
        for i in range(5):
            similar_vector = ref_vector + torch.randn(384) * 0.1
            point = VectorPoint(
                id=f"similar{i}",
                vector=similar_vector,
                payload={"type": "similar", "index": i},
            )
            await backend.store(point)

        # Search
        results = await backend.search(ref_vector, limit=3)

        assert len(results) > 0
        assert results[0].id == "ref"  # Exact match should be first


class TestBackendMigration:
    """Test data migration between backends."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_migrate_graph_sqlite_to_neo4j(self, temp_dir):
        """Test migrating graph data from SQLite to Neo4j."""
        if not NEO4J_RUNNING:
            pytest.skip("Neo4j not running")

        # Create SQLite backend and add data
        sqlite = SQLiteGraphBackend(temp_dir / "source.db")
        await sqlite.connect()
        await sqlite.create_graph("migration_test")

        nodes = []
        for i in range(10):
            node = GraphNode(
                id=f"node{i}",
                content=f"Content {i}",
                timestamp=datetime.now(),
                fractal_signature=f"sig{i}",
                parent_id="-1",
                children_ids=[],
            )
            await sqlite.create_node(node, "migration_test")
            nodes.append(node)

        # Migrate to Neo4j
        neo4j = Neo4jGraphBackend()
        await neo4j.connect()

        # Clean up destination
        async with neo4j.driver.session() as session:
            await session.run("MATCH (m:Memory {graph_name: 'migration_test'}) DETACH DELETE m")

        # Copy all nodes
        for node in nodes:
            await neo4j.create_node(node, "migration_test")

        # Verify migration
        for node in nodes:
            retrieved = await neo4j.get_node(node.id, "migration_test")
            assert retrieved is not None
            assert retrieved.content == node.content

        # Clean up
        await sqlite.close()
        async with neo4j.driver.session() as session:
            await session.run("MATCH (m:Memory {graph_name: 'migration_test'}) DETACH DELETE m")
        await neo4j.close()

    @pytest.mark.asyncio
    async def test_migrate_vectors_chroma_to_qdrant(self, temp_dir):
        """Test migrating vector data from ChromaDB to Qdrant."""
        if not QDRANT_RUNNING:
            pytest.skip("Qdrant not running")

        # Create ChromaDB backend and add data
        chroma = ChromaVectorBackend(
            persist_directory=temp_dir / "chroma",
            collection_name="migration_source",
        )
        await chroma.connect()

        vectors = []
        for i in range(10):
            vector = torch.randn(384)
            point = VectorPoint(
                id=f"vec{i}",
                vector=vector,
                payload={"index": i, "content": f"Vector {i}"},
            )
            await chroma.store(point)
            vectors.append((point.id, vector, point.payload))

        # Migrate to Qdrant
        qdrant = QdrantVectorBackend(collection_name="migration_target")
        await qdrant.connect()

        # Clean up and create collection
        try:
            await qdrant.delete_collection("migration_target")
        except:
            pass
        await qdrant.create_collection("migration_target", 384)
        qdrant.collection_name = "migration_target"

        # Copy all vectors
        for vec_id, vector, payload in vectors:
            point = VectorPoint(id=vec_id, vector=vector, payload=payload)
            await qdrant.store(point)

        # Verify migration
        ids = [v[0] for v in vectors]
        retrieved = await qdrant.retrieve(ids)
        assert len(retrieved) == 10

        # Clean up
        await chroma.close()
        try:
            await qdrant.delete_collection("migration_target")
        except:
            pass
        await qdrant.close()


class TestPerformanceCharacteristics:
    """Test performance characteristics of different backends."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_sqlite_bulk_insert_performance(self, temp_dir):
        """Test SQLite bulk insert performance."""
        backend = SQLiteGraphBackend(temp_dir / "perf.db")
        await backend.connect()
        await backend.create_graph("perf_test")

        # Insert 100 nodes
        start = time.time()
        for i in range(100):
            node = GraphNode(
                id=f"node{i}",
                content=f"Content {i}",
                timestamp=datetime.now(),
                fractal_signature=f"sig{i}",
            )
            await backend.create_node(node, "perf_test")
        duration = time.time() - start

        # Should complete in reasonable time (< 1 second)
        assert duration < 1.0
        print(f"SQLite: Inserted 100 nodes in {duration:.3f}s ({100/duration:.0f} ops/sec)")

        await backend.close()

    @pytest.mark.asyncio
    async def test_chroma_bulk_insert_performance(self, temp_dir):
        """Test ChromaDB bulk insert performance."""
        backend = ChromaVectorBackend(
            persist_directory=temp_dir / "chroma",
            collection_name="perf_test",
        )
        await backend.connect()

        # Insert 100 vectors
        points = []
        for i in range(100):
            vector = torch.randn(384)
            point = VectorPoint(
                id=f"vec{i}",
                vector=vector,
                payload={"index": i},
            )
            points.append(point)

        start = time.time()
        await backend.store_batch(points)
        duration = time.time() - start

        # Should complete in reasonable time (< 2 seconds)
        assert duration < 2.0
        print(f"ChromaDB: Inserted 100 vectors in {duration:.3f}s ({100/duration:.0f} ops/sec)")

        await backend.close()

    @pytest.mark.asyncio
    async def test_search_performance_comparison(self, temp_dir):
        """Compare search performance across vector backends."""
        if not QDRANT_RUNNING:
            pytest.skip("Qdrant not running - can't compare")

        # Setup ChromaDB
        chroma = ChromaVectorBackend(
            persist_directory=temp_dir / "chroma",
            collection_name="search_perf",
        )
        await chroma.connect()

        # Setup Qdrant
        qdrant = QdrantVectorBackend(collection_name="search_perf")
        await qdrant.connect()
        try:
            await qdrant.delete_collection("search_perf")
        except:
            pass
        await qdrant.create_collection("search_perf", 384)
        qdrant.collection_name = "search_perf"

        # Insert same 1000 vectors into both
        points = []
        for i in range(1000):
            vector = torch.randn(384)
            point = VectorPoint(
                id=f"vec{i}",
                vector=vector,
                payload={"index": i},
            )
            points.append(point)

        await chroma.store_batch(points)
        await qdrant.store_batch(points)

        # Test search performance
        query = torch.randn(384)

        # ChromaDB
        start = time.time()
        chroma_results = await chroma.search(query, limit=10)
        chroma_time = time.time() - start

        # Qdrant
        start = time.time()
        qdrant_results = await qdrant.search(query, limit=10)
        qdrant_time = time.time() - start

        print(f"ChromaDB search: {chroma_time*1000:.1f}ms")
        print(f"Qdrant search: {qdrant_time*1000:.1f}ms")
        print(f"Speedup: {chroma_time/qdrant_time:.1f}x")

        # Both should return results
        assert len(chroma_results) == 10
        assert len(qdrant_results) == 10

        # Qdrant should be faster (typically 2-10x on CPU)
        # But we won't assert this as it depends on hardware

        # Clean up
        await chroma.close()
        try:
            await qdrant.delete_collection("search_perf")
        except:
            pass
        await qdrant.close()


class TestEdgeCases:
    """Test edge cases and error handling across backends."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_empty_content(self, temp_dir):
        """Test nodes with empty content."""
        backend = SQLiteGraphBackend(temp_dir / "edge.db")
        await backend.connect()
        await backend.create_graph("edge_test")

        node = GraphNode(
            id="empty",
            content="",  # Empty content
            timestamp=datetime.now(),
            fractal_signature="sig",
        )
        await backend.create_node(node, "edge_test")

        retrieved = await backend.get_node("empty", "edge_test")
        assert retrieved is not None
        assert retrieved.content == ""

        await backend.close()

    @pytest.mark.asyncio
    async def test_unicode_content(self, temp_dir):
        """Test nodes with unicode content."""
        backend = SQLiteGraphBackend(temp_dir / "unicode.db")
        await backend.connect()
        await backend.create_graph("unicode_test")

        unicode_text = "Hello 世界 🌍 Привет मस्ते"
        node = GraphNode(
            id="unicode",
            content=unicode_text,
            timestamp=datetime.now(),
            fractal_signature="sig",
        )
        await backend.create_node(node, "unicode_test")

        retrieved = await backend.get_node("unicode", "unicode_test")
        assert retrieved is not None
        assert retrieved.content == unicode_text

        await backend.close()

    @pytest.mark.asyncio
    async def test_zero_vector(self, temp_dir):
        """Test storing zero vectors."""
        backend = ChromaVectorBackend(
            persist_directory=temp_dir / "chroma",
            collection_name="zero_test",
        )
        await backend.connect()

        zero_vector = torch.zeros(384)
        point = VectorPoint(
            id="zero",
            vector=zero_vector,
            payload={"type": "zero"},
        )
        await backend.store(point)

        retrieved = await backend.retrieve(["zero"])
        assert len(retrieved) == 1
        assert torch.allclose(retrieved[0].vector, zero_vector)

        await backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
