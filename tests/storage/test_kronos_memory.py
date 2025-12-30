"""
Test KronosMemory with Pluggable Backends

Tests the KRONOS implementation with:
- Backend factory and fallback mechanisms
- PAC delta encoding with backends
- SEC resonance ranking
- Cross-graph linking
- Temporal lineage tracing
- Backend switching and migration
"""

import pytest
import pytest_asyncio
from pathlib import Path
import tempfile
import shutil
import torch

from fracton.storage import (
    KronosMemory,
    BackendFactory,
    BackendConfig,
    NodeType,
    RelationType,
)


@pytest_asyncio.fixture
async def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    try:
        shutil.rmtree(temp_path)
    except:
        pass


@pytest_asyncio.fixture
async def unified_memory(temp_dir):
    """Create KronosMemory with lightweight backends."""
    memory = KronosMemory(
        storage_path=temp_dir,
        namespace="test",
        backend_config=BackendConfig(
            graph_type="sqlite",
            vector_type="chromadb",
            device="cpu",
            embedding_dim=384,
        ),
        enable_auto_fallback=True,
    )
    await memory.connect()
    yield memory
    await memory.close()


class TestKronosMemoryBasics:
    """Test basic KronosMemory functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self, temp_dir):
        """Test initialization and connection."""
        memory = KronosMemory(
            storage_path=temp_dir,
            namespace="test_init",
            device="cpu",
        )

        await memory.connect()

        assert memory.graph_backend is not None
        assert memory.vector_backend is not None
        assert memory.namespace == "test_init"

        await memory.close()

    @pytest.mark.asyncio
    async def test_auto_fallback(self, temp_dir):
        """Test automatic fallback to lightweight backends."""
        # Request production backends that may not be available
        config = BackendConfig(
            graph_type="neo4j",
            vector_type="qdrant",
            graph_uri="bolt://nonexistent:7687",
            vector_host="nonexistent",
            vector_port=6333,
        )

        memory = KronosMemory(
            storage_path=temp_dir,
            namespace="test_fallback",
            backend_config=config,
            enable_auto_fallback=True,
        )

        await memory.connect()

        # Should fall back to SQLite/ChromaDB
        assert memory.graph_backend is not None
        assert memory.vector_backend is not None

        await memory.close()

    @pytest.mark.asyncio
    async def test_create_graph(self, unified_memory):
        """Test graph creation."""
        await unified_memory.create_graph("test_graph", "Test description")

        graphs = await unified_memory.list_graphs()
        graph_names = [g["name"] for g in graphs]

        # May be empty if no nodes yet, but should not error
        assert isinstance(graphs, list)

    @pytest.mark.asyncio
    async def test_health_check(self, unified_memory):
        """Test backend health checks."""
        health = await unified_memory.health_check()

        assert "graph" in health
        assert "vector" in health
        assert health["graph"] is True
        assert health["vector"] is True


class TestPACStorage:
    """Test PAC delta encoding with backends."""

    @pytest.mark.asyncio
    async def test_store_root_node(self, unified_memory):
        """Test storing root node (delta == full embedding)."""
        await unified_memory.create_graph("pac_test")

        # Use simple embedding for testing
        embedding = torch.randn(384)

        node_id = await unified_memory.store(
            content="Root node content",
            graph="pac_test",
            node_type=NodeType.CONCEPT,
            embedding=embedding,
            importance=1.0,
        )

        assert node_id is not None

        # Retrieve and verify
        node = await unified_memory.retrieve("pac_test", node_id)
        assert node is not None
        assert node.content == "Root node content"
        assert node.is_root is True
        assert node.parent_id == "-1"
        assert node.potential == 1.0

    @pytest.mark.asyncio
    async def test_store_child_with_delta(self, unified_memory):
        """Test storing child node with PAC delta encoding."""
        await unified_memory.create_graph("pac_test")

        # Create parent
        parent_emb = torch.randn(384)
        parent_id = await unified_memory.store(
            content="Parent content",
            graph="pac_test",
            node_type=NodeType.CONCEPT,
            embedding=parent_emb,
        )

        # Create child with modified embedding
        child_emb = parent_emb + torch.randn(384) * 0.1  # Small delta
        child_id = await unified_memory.store(
            content="Child content",
            graph="pac_test",
            node_type=NodeType.CONCEPT,
            parent_id=parent_id,
            embedding=child_emb,
        )

        # Verify child stored correctly
        child = await unified_memory.retrieve("pac_test", child_id)
        assert child.parent_id == parent_id
        assert child.is_root is False

        # Verify parent tracks child
        parent = await unified_memory.retrieve("pac_test", parent_id)
        assert child_id in parent.children_ids

    @pytest.mark.asyncio
    async def test_pac_reconstruction(self, unified_memory):
        """Test PAC embedding reconstruction from deltas."""
        await unified_memory.create_graph("pac_test")

        # Create chain: root -> child -> grandchild
        root_emb = torch.randn(384)
        root_id = await unified_memory.store(
            content="Root",
            graph="pac_test",
            node_type=NodeType.CONCEPT,
            embedding=root_emb,
        )

        child_emb = root_emb + torch.randn(384) * 0.1
        child_id = await unified_memory.store(
            content="Child",
            graph="pac_test",
            node_type=NodeType.CONCEPT,
            parent_id=root_id,
            embedding=child_emb,
        )

        grandchild_emb = child_emb + torch.randn(384) * 0.1
        grandchild_id = await unified_memory.store(
            content="Grandchild",
            graph="pac_test",
            node_type=NodeType.CONCEPT,
            parent_id=child_id,
            embedding=grandchild_emb,
        )

        # Reconstruct embedding for grandchild
        reconstructed = await unified_memory._reconstruct_embedding("pac_test", grandchild_id)

        # Should match original grandchild embedding closely
        assert torch.allclose(reconstructed, grandchild_emb, atol=1e-4)

    @pytest.mark.asyncio
    async def test_potential_decay(self, unified_memory):
        """Test potential decay in PAC hierarchy."""
        await unified_memory.create_graph("pac_test")

        # Create parent
        parent_id = await unified_memory.store(
            content="Parent",
            graph="pac_test",
            node_type=NodeType.CONCEPT,
            embedding=torch.randn(384),
        )

        parent = await unified_memory.retrieve("pac_test", parent_id)
        parent_potential = parent.potential

        # Create child
        child_id = await unified_memory.store(
            content="Child",
            graph="pac_test",
            node_type=NodeType.CONCEPT,
            parent_id=parent_id,
            embedding=torch.randn(384),
        )

        child = await unified_memory.retrieve("pac_test", child_id)

        # Child potential should be parent_potential * LAMBDA_STAR
        # LAMBDA_STAR ≈ 0.9816 (optimal decay constant)
        assert child.potential < parent_potential
        assert abs(child.potential / parent_potential - 0.9816) < 0.01


class TestSECQuerying:
    """Test SEC resonance ranking and querying."""

    @pytest.mark.asyncio
    async def test_query_empty_graph(self, unified_memory):
        """Test querying empty graph returns empty results."""
        await unified_memory.create_graph("empty_graph")

        results = await unified_memory.query(
            query_text="test query",
            graphs=["empty_graph"],
            limit=10,
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_query_with_results(self, unified_memory):
        """Test SEC resonance ranking."""
        await unified_memory.create_graph("query_test")

        # Store several nodes with different content
        embeddings = []
        node_ids = []

        for i in range(5):
            emb = torch.randn(384)
            embeddings.append(emb)

            node_id = await unified_memory.store(
                content=f"Test content {i}",
                graph="query_test",
                node_type=NodeType.CONCEPT,
                embedding=emb,
            )
            node_ids.append(node_id)

        # Query with one of the embeddings
        results = await unified_memory.query(
            query_text="Test content 0",
            graphs=["query_test"],
            limit=3,
        )

        # Should return results
        assert len(results) > 0
        # Results should be sorted by score
        if len(results) > 1:
            assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_multi_graph_query(self, unified_memory):
        """Test querying across multiple graphs."""
        # Create two graphs
        await unified_memory.create_graph("graph1")
        await unified_memory.create_graph("graph2")

        # Store nodes in each
        emb = torch.randn(384)

        await unified_memory.store(
            content="Content in graph 1",
            graph="graph1",
            node_type=NodeType.CONCEPT,
            embedding=emb,
        )

        await unified_memory.store(
            content="Content in graph 2",
            graph="graph2",
            node_type=NodeType.CONCEPT,
            embedding=emb + torch.randn(384) * 0.1,
        )

        # Query both graphs
        results = await unified_memory.query(
            query_text="Content",
            graphs=["graph1", "graph2"],
            limit=10,
        )

        # Should return results from both graphs
        assert len(results) >= 2
        graph_sources = {r.node.graph for r in results}
        assert "graph1" in graph_sources
        assert "graph2" in graph_sources


class TestCrossGraphLinking:
    """Test linking nodes across different graphs."""

    @pytest.mark.asyncio
    async def test_create_cross_graph_link(self, unified_memory):
        """Test creating links between nodes in different graphs."""
        # Create two graphs
        await unified_memory.create_graph("code")
        await unified_memory.create_graph("research")

        # Create nodes
        commit_id = await unified_memory.store(
            content="Implemented transformer",
            graph="code",
            node_type=NodeType.COMMIT,
            embedding=torch.randn(384),
        )

        paper_id = await unified_memory.store(
            content="Attention Is All You Need",
            graph="research",
            node_type=NodeType.PAPER,
            embedding=torch.randn(384),
        )

        # Link them
        await unified_memory.link_across_graphs(
            from_graph="code",
            from_id=commit_id,
            to_graph="research",
            to_id=paper_id,
            relation=RelationType.IMPLEMENTS,
        )

        # Verify link exists
        neighborhood = await unified_memory.get_neighborhood(
            graph="code",
            node_id=commit_id,
            max_hops=1,
        )

        # Should include edges
        assert len(neighborhood.edges) >= 1

    @pytest.mark.asyncio
    async def test_expand_across_graphs(self, unified_memory):
        """Test expanding query across linked graphs."""
        # Create two graphs
        await unified_memory.create_graph("graph1")
        await unified_memory.create_graph("graph2")

        # Create and link nodes
        emb1 = torch.randn(384)
        node1_id = await unified_memory.store(
            content="Node in graph 1",
            graph="graph1",
            node_type=NodeType.CONCEPT,
            embedding=emb1,
        )

        emb2 = torch.randn(384)
        node2_id = await unified_memory.store(
            content="Node in graph 2",
            graph="graph2",
            node_type=NodeType.CONCEPT,
            embedding=emb2,
        )

        await unified_memory.link_across_graphs(
            from_graph="graph1",
            from_id=node1_id,
            to_graph="graph2",
            to_id=node2_id,
            relation=RelationType.RELATES_TO,
        )

        # Query with graph expansion
        results = await unified_memory.query(
            query_text="Node in graph 1",
            graphs=["graph1"],
            limit=10,
            expand_graph=True,
        )

        # Should find results
        assert len(results) > 0


class TestTemporalLineage:
    """Test temporal lineage tracing."""

    @pytest.mark.asyncio
    async def test_trace_forward(self, unified_memory):
        """Test forward lineage tracing."""
        await unified_memory.create_graph("lineage_test")

        # Create chain: parent -> child -> grandchild
        parent_id = await unified_memory.store(
            content="Parent",
            graph="lineage_test",
            node_type=NodeType.CONCEPT,
            embedding=torch.randn(384),
        )

        child_id = await unified_memory.store(
            content="Child",
            graph="lineage_test",
            node_type=NodeType.CONCEPT,
            parent_id=parent_id,
            embedding=torch.randn(384),
        )

        grandchild_id = await unified_memory.store(
            content="Grandchild",
            graph="lineage_test",
            node_type=NodeType.CONCEPT,
            parent_id=child_id,
            embedding=torch.randn(384),
        )

        # Trace forward from parent
        trace = await unified_memory.trace_evolution(
            graph="lineage_test",
            node_id=parent_id,
            direction="forward",
        )

        assert trace["root_id"] == parent_id
        assert parent_id in trace["path"]
        assert child_id in trace["path"]

    @pytest.mark.asyncio
    async def test_trace_backward(self, unified_memory):
        """Test backward lineage tracing."""
        await unified_memory.create_graph("lineage_test")

        # Create chain
        parent_id = await unified_memory.store(
            content="Parent",
            graph="lineage_test",
            node_type=NodeType.CONCEPT,
            embedding=torch.randn(384),
        )

        child_id = await unified_memory.store(
            content="Child",
            graph="lineage_test",
            node_type=NodeType.CONCEPT,
            parent_id=parent_id,
            embedding=torch.randn(384),
        )

        grandchild_id = await unified_memory.store(
            content="Grandchild",
            graph="lineage_test",
            node_type=NodeType.CONCEPT,
            parent_id=child_id,
            embedding=torch.randn(384),
        )

        # Trace backward from grandchild
        trace = await unified_memory.trace_evolution(
            graph="lineage_test",
            node_id=grandchild_id,
            direction="backward",
        )

        assert trace["root_id"] == parent_id
        assert parent_id in trace["path"]
        assert child_id in trace["path"]
        assert grandchild_id in trace["path"]

    @pytest.mark.asyncio
    async def test_trace_both_directions(self, unified_memory):
        """Test bidirectional lineage tracing."""
        await unified_memory.create_graph("lineage_test")

        # Create chain: gp -> parent -> child -> grandchild
        grandparent_id = await unified_memory.store(
            content="Grandparent",
            graph="lineage_test",
            node_type=NodeType.CONCEPT,
            embedding=torch.randn(384),
        )

        parent_id = await unified_memory.store(
            content="Parent",
            graph="lineage_test",
            node_type=NodeType.CONCEPT,
            parent_id=grandparent_id,
            embedding=torch.randn(384),
        )

        child_id = await unified_memory.store(
            content="Child",
            graph="lineage_test",
            node_type=NodeType.CONCEPT,
            parent_id=parent_id,
            embedding=torch.randn(384),
        )

        grandchild_id = await unified_memory.store(
            content="Grandchild",
            graph="lineage_test",
            node_type=NodeType.CONCEPT,
            parent_id=child_id,
            embedding=torch.randn(384),
        )

        # Trace both directions from parent
        trace = await unified_memory.trace_evolution(
            graph="lineage_test",
            node_id=parent_id,
            direction="both",
        )

        # Should include ancestors and descendants
        assert grandparent_id in trace["path"]
        assert parent_id in trace["path"]
        assert child_id in trace["path"]


class TestBackendSwitching:
    """Test switching between different backend configurations."""

    @pytest.mark.asyncio
    async def test_sqlite_backends(self, temp_dir):
        """Test with SQLite + ChromaDB backends."""
        config = BackendConfig(
            graph_type="sqlite",
            vector_type="chromadb",
            device="cpu",
            embedding_dim=384,
        )

        memory = KronosMemory(
            storage_path=temp_dir,
            namespace="sqlite_test",
            backend_config=config,
            enable_auto_fallback=False,
        )

        await memory.connect()
        await memory.create_graph("test")

        # Store node
        node_id = await memory.store(
            content="Test content",
            graph="test",
            node_type=NodeType.CONCEPT,
            embedding=torch.randn(384),
        )

        # Retrieve
        node = await memory.retrieve("test", node_id)
        assert node.content == "Test content"

        await memory.close()


class TestCaching:
    """Test in-memory caching."""

    @pytest.mark.asyncio
    async def test_node_caching(self, unified_memory):
        """Test nodes are cached after retrieval."""
        await unified_memory.create_graph("cache_test")

        # Store node
        node_id = await unified_memory.store(
            content="Cached content",
            graph="cache_test",
            node_type=NodeType.CONCEPT,
            embedding=torch.randn(384),
        )

        # First retrieval - hits backend
        node1 = await unified_memory.retrieve("cache_test", node_id)

        # Check cache
        cache_key = ("cache_test", node_id)
        assert cache_key in unified_memory._node_cache

        # Second retrieval - should hit cache
        node2 = await unified_memory.retrieve("cache_test", node_id)

        assert node1.id == node2.id
        assert node1.content == node2.content

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_update(self, unified_memory):
        """Test cache is invalidated on node updates."""
        await unified_memory.create_graph("cache_test")

        # Store and retrieve to cache
        node_id = await unified_memory.store(
            content="Original content",
            graph="cache_test",
            node_type=NodeType.CONCEPT,
            embedding=torch.randn(384),
        )

        await unified_memory.retrieve("cache_test", node_id)
        cache_key = ("cache_test", node_id)
        assert cache_key in unified_memory._node_cache

        # Update node
        await unified_memory.update_node(
            graph="cache_test",
            node_id=node_id,
            updates={"content": "Updated content"},
        )

        # Cache should be invalidated
        assert cache_key not in unified_memory._node_cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
