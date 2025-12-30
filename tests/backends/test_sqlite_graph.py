"""
Test SQLite Graph Backend

Comprehensive tests for SQLiteGraphBackend implementation.
"""

import pytest
import pytest_asyncio
from datetime import datetime
from pathlib import Path
import tempfile

from fracton.storage.backends import SQLiteGraphBackend
from fracton.storage.backends.base import GraphNode, GraphEdge


@pytest.fixture
def temp_db():
    """Create temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest_asyncio.fixture
async def backend(temp_db):
    """Create SQLite backend."""
    backend = SQLiteGraphBackend(temp_db)
    await backend.connect()
    yield backend
    await backend.close()


class TestSQLiteGraphBackend:
    """Test SQLite graph backend."""

    @pytest.mark.asyncio
    async def test_connection(self, temp_db):
        """Test connect and close."""
        backend = SQLiteGraphBackend(temp_db)

        await backend.connect()
        assert backend.conn is not None

        await backend.close()
        assert backend.conn is None

    @pytest.mark.asyncio
    async def test_health_check(self, backend):
        """Test health check."""
        is_healthy = await backend.health_check()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_create_graph(self, backend):
        """Test graph creation."""
        await backend.create_graph("test_graph", "Test description")

        graphs = await backend.list_graphs()
        assert len(graphs) == 1
        assert graphs[0]["name"] == "test_graph"
        assert graphs[0]["description"] == "Test description"
        assert graphs[0]["node_count"] == 0
        assert graphs[0]["edge_count"] == 0

    @pytest.mark.asyncio
    async def test_create_node(self, backend):
        """Test node creation."""
        await backend.create_graph("test_graph")

        node = GraphNode(
            id="node1",
            content="Test content",
            timestamp=datetime.now(),
            fractal_signature="sig123",
            metadata={"key": "value"},
            parent_id="-1",
            children_ids=[],
            potential=0.9,
            entropy=0.4,
            coherence=0.7,
            phase="STABLE",
        )

        await backend.create_node(node, "test_graph")

        # Verify node was created
        retrieved = await backend.get_node("node1", "test_graph")
        assert retrieved is not None
        assert retrieved.id == "node1"
        assert retrieved.content == "Test content"
        assert retrieved.fractal_signature == "sig123"
        assert retrieved.metadata["key"] == "value"
        assert retrieved.potential == 0.9
        assert retrieved.entropy == 0.4
        assert retrieved.coherence == 0.7
        assert retrieved.phase == "STABLE"

    @pytest.mark.asyncio
    async def test_parent_child_relationship(self, backend):
        """Test parent-child tracking."""
        await backend.create_graph("test_graph")

        # Create parent
        parent = GraphNode(
            id="parent",
            content="Parent node",
            timestamp=datetime.now(),
            fractal_signature="sig_parent",
            parent_id="-1",
            children_ids=[],
        )
        await backend.create_node(parent, "test_graph")

        # Create child
        child = GraphNode(
            id="child",
            content="Child node",
            timestamp=datetime.now(),
            fractal_signature="sig_child",
            parent_id="parent",
            children_ids=[],
        )
        await backend.create_node(child, "test_graph")

        # Verify parent's children_ids was updated
        retrieved_parent = await backend.get_node("parent", "test_graph")
        assert "child" in retrieved_parent.children_ids

    @pytest.mark.asyncio
    async def test_update_node(self, backend):
        """Test node updates."""
        await backend.create_graph("test_graph")

        node = GraphNode(
            id="node1",
            content="Original",
            timestamp=datetime.now(),
            fractal_signature="sig123",
        )
        await backend.create_node(node, "test_graph")

        # Update node
        await backend.update_node(
            "node1",
            {"content": "Updated", "entropy": 0.2},
            "test_graph"
        )

        # Verify update
        retrieved = await backend.get_node("node1", "test_graph")
        assert retrieved.content == "Updated"
        assert retrieved.entropy == 0.2

    @pytest.mark.asyncio
    async def test_delete_node(self, backend):
        """Test node deletion."""
        await backend.create_graph("test_graph")

        node = GraphNode(
            id="node1",
            content="To delete",
            timestamp=datetime.now(),
            fractal_signature="sig123",
        )
        await backend.create_node(node, "test_graph")

        # Verify created
        assert await backend.get_node("node1", "test_graph") is not None

        # Delete
        await backend.delete_node("node1", "test_graph")

        # Verify deleted
        assert await backend.get_node("node1", "test_graph") is None

    @pytest.mark.asyncio
    async def test_create_edge(self, backend):
        """Test edge creation."""
        await backend.create_graph("test_graph")

        # Create nodes
        for i in range(2):
            node = GraphNode(
                id=f"node{i}",
                content=f"Node {i}",
                timestamp=datetime.now(),
                fractal_signature=f"sig{i}",
            )
            await backend.create_node(node, "test_graph")

        # Create edge
        edge = GraphEdge(
            from_id="node0",
            to_id="node1",
            relation_type="CONNECTS_TO",
            weight=0.9,
            metadata={"label": "test"},
        )
        await backend.create_edge(edge, "test_graph")

        # Verify edge
        edges = await backend.get_edges("node0", "out", graph_name="test_graph")
        assert len(edges) == 1
        assert edges[0].from_id == "node0"
        assert edges[0].to_id == "node1"
        assert edges[0].relation_type == "CONNECTS_TO"
        assert edges[0].weight == 0.9

    @pytest.mark.asyncio
    async def test_get_edges_directions(self, backend):
        """Test edge queries with different directions."""
        await backend.create_graph("test_graph")

        # Create nodes
        for i in range(3):
            node = GraphNode(
                id=f"node{i}",
                content=f"Node {i}",
                timestamp=datetime.now(),
                fractal_signature=f"sig{i}",
            )
            await backend.create_node(node, "test_graph")

        # Create edges: node1 -> node0, node0 -> node2
        edge1 = GraphEdge(from_id="node1", to_id="node0", relation_type="POINTS_TO")
        edge2 = GraphEdge(from_id="node0", to_id="node2", relation_type="POINTS_TO")
        await backend.create_edge(edge1, "test_graph")
        await backend.create_edge(edge2, "test_graph")

        # Test outgoing
        out_edges = await backend.get_edges("node0", "out", graph_name="test_graph")
        assert len(out_edges) == 1
        assert out_edges[0].to_id == "node2"

        # Test incoming
        in_edges = await backend.get_edges("node0", "in", graph_name="test_graph")
        assert len(in_edges) == 1
        assert in_edges[0].from_id == "node1"

        # Test both
        both_edges = await backend.get_edges("node0", "both", graph_name="test_graph")
        assert len(both_edges) == 2

    @pytest.mark.asyncio
    async def test_delete_edge(self, backend):
        """Test edge deletion."""
        await backend.create_graph("test_graph")

        # Create nodes and edge
        for i in range(2):
            node = GraphNode(
                id=f"node{i}",
                content=f"Node {i}",
                timestamp=datetime.now(),
                fractal_signature=f"sig{i}",
            )
            await backend.create_node(node, "test_graph")

        edge = GraphEdge(from_id="node0", to_id="node1", relation_type="TEST")
        await backend.create_edge(edge, "test_graph")

        # Verify created
        edges = await backend.get_edges("node0", graph_name="test_graph")
        assert len(edges) == 1

        # Delete
        await backend.delete_edge("node0", "node1", "TEST", "test_graph")

        # Verify deleted
        edges = await backend.get_edges("node0", graph_name="test_graph")
        assert len(edges) == 0

    @pytest.mark.asyncio
    async def test_get_neighbors(self, backend):
        """Test neighborhood expansion."""
        await backend.create_graph("test_graph")

        # Create linear chain: node0 -> node1 -> node2
        for i in range(3):
            node = GraphNode(
                id=f"node{i}",
                content=f"Node {i}",
                timestamp=datetime.now(),
                fractal_signature=f"sig{i}",
            )
            await backend.create_node(node, "test_graph")

        edge1 = GraphEdge(from_id="node0", to_id="node1", relation_type="NEXT")
        edge2 = GraphEdge(from_id="node1", to_id="node2", relation_type="NEXT")
        await backend.create_edge(edge1, "test_graph")
        await backend.create_edge(edge2, "test_graph")

        # Get 1-hop neighbors
        neighborhood = await backend.get_neighbors("node0", max_hops=1, graph_name="test_graph")
        assert len(neighborhood.nodes) >= 1
        assert neighborhood.center_id == "node0"
        assert neighborhood.depth_map["node0"] == 0

        # Get 2-hop neighbors
        neighborhood = await backend.get_neighbors("node0", max_hops=2, graph_name="test_graph")
        node_ids = [n.id for n in neighborhood.nodes]
        assert "node0" in node_ids
        assert "node1" in node_ids
        # node2 should be included (2 hops away)

    @pytest.mark.asyncio
    async def test_trace_lineage_backward(self, backend):
        """Test backward lineage tracing."""
        await backend.create_graph("test_graph")

        # Create parent -> child -> grandchild
        parent = GraphNode(
            id="parent",
            content="Parent",
            timestamp=datetime.now(),
            fractal_signature="sig_p",
            parent_id="-1",
            children_ids=[],
            entropy=0.3,
        )
        await backend.create_node(parent, "test_graph")

        child = GraphNode(
            id="child",
            content="Child",
            timestamp=datetime.now(),
            fractal_signature="sig_c",
            parent_id="parent",
            children_ids=[],
            entropy=0.4,
        )
        await backend.create_node(child, "test_graph")

        grandchild = GraphNode(
            id="grandchild",
            content="Grandchild",
            timestamp=datetime.now(),
            fractal_signature="sig_g",
            parent_id="child",
            children_ids=[],
            entropy=0.5,
        )
        await backend.create_node(grandchild, "test_graph")

        # Trace backward from grandchild
        path = await backend.trace_lineage("grandchild", "backward", graph_name="test_graph")

        assert path.root_id == "parent"
        assert path.path == ["parent", "child", "grandchild"]
        assert len(path.nodes) == 3
        assert path.entropy_evolution == [0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_trace_lineage_forward(self, backend):
        """Test forward lineage tracing."""
        await backend.create_graph("test_graph")

        # Create chain
        parent = GraphNode(
            id="parent",
            content="Parent",
            timestamp=datetime.now(),
            fractal_signature="sig_p",
            parent_id="-1",
            children_ids=[],
        )
        await backend.create_node(parent, "test_graph")

        child = GraphNode(
            id="child",
            content="Child",
            timestamp=datetime.now(),
            fractal_signature="sig_c",
            parent_id="parent",
            children_ids=[],
        )
        await backend.create_node(child, "test_graph")

        # Trace forward from parent
        path = await backend.trace_lineage("parent", "forward", graph_name="test_graph")

        assert "parent" in path.path
        assert "child" in path.path

    @pytest.mark.asyncio
    async def test_find_contradictions(self, backend):
        """Test contradiction detection."""
        await backend.create_graph("test_graph")

        # Create nodes
        for i in range(3):
            node = GraphNode(
                id=f"node{i}",
                content=f"Node {i}",
                timestamp=datetime.now(),
                fractal_signature=f"sig{i}",
            )
            await backend.create_node(node, "test_graph")

        # Create CONTRADICTS edges
        edge1 = GraphEdge(from_id="node0", to_id="node1", relation_type="CONTRADICTS")
        edge2 = GraphEdge(from_id="node2", to_id="node0", relation_type="CONTRADICTS")
        await backend.create_edge(edge1, "test_graph")
        await backend.create_edge(edge2, "test_graph")

        # Find contradictions
        contradictions = await backend.find_contradictions("node0", "test_graph")

        assert len(contradictions) == 2
        assert "node1" in contradictions
        assert "node2" in contradictions

    @pytest.mark.asyncio
    async def test_persistence(self, temp_db):
        """Test that data persists across connections."""
        # Create and store data
        backend1 = SQLiteGraphBackend(temp_db)
        await backend1.connect()
        await backend1.create_graph("test_graph")

        node = GraphNode(
            id="persistent",
            content="Persisted data",
            timestamp=datetime.now(),
            fractal_signature="sig_persist",
        )
        await backend1.create_node(node, "test_graph")
        await backend1.close()

        # Reconnect and verify
        backend2 = SQLiteGraphBackend(temp_db)
        await backend2.connect()

        retrieved = await backend2.get_node("persistent", "test_graph")
        assert retrieved is not None
        assert retrieved.content == "Persisted data"

        await backend2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
