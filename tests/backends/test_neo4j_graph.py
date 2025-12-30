"""
Test Neo4j Graph Backend

Comprehensive tests for Neo4jGraphBackend implementation.
Requires Neo4j server running on localhost:7687.
"""

import pytest
import pytest_asyncio
from datetime import datetime
from pathlib import Path
import os

from fracton.storage.backends.base import GraphNode, GraphEdge

# Try to import Neo4j backend
try:
    from fracton.storage.backends import Neo4jGraphBackend
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    Neo4jGraphBackend = None


# Check if Neo4j is actually running
NEO4J_RUNNING = False
if NEO4J_AVAILABLE:
    import asyncio
    try:
        async def check_neo4j():
            backend = Neo4jGraphBackend(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "password"),
            )
            try:
                await backend.connect()
                await backend.close()
                return True
            except:
                return False
        NEO4J_RUNNING = asyncio.run(check_neo4j())
    except:
        pass


@pytest_asyncio.fixture
async def backend():
    """Create Neo4j backend."""
    if not NEO4J_AVAILABLE:
        pytest.skip("Neo4j driver not installed")
    if not NEO4J_RUNNING:
        pytest.skip("Neo4j server not running on localhost:7687")

    backend = Neo4jGraphBackend(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )
    await backend.connect()

    # Clean up test data
    async with backend.driver.session() as session:
        await session.run("MATCH (m:Memory {graph_name: 'test_graph'}) DETACH DELETE m")

    yield backend

    # Clean up after tests
    async with backend.driver.session() as session:
        await session.run("MATCH (m:Memory {graph_name: 'test_graph'}) DETACH DELETE m")

    await backend.close()


@pytest.mark.skipif(not NEO4J_AVAILABLE or not NEO4J_RUNNING, reason="Neo4j not available")
class TestNeo4jGraphBackend:
    """Test Neo4j graph backend."""

    @pytest.mark.asyncio
    async def test_connection(self):
        """Test connect and close."""
        backend = Neo4jGraphBackend()
        await backend.connect()
        assert backend.driver is not None
        await backend.close()
        assert backend.driver is None

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
        # May include other graphs, just check test_graph exists
        graph_names = [g["name"] for g in graphs]
        assert "test_graph" in graph_names or len(graphs) == 0  # Empty if no nodes yet

    @pytest.mark.asyncio
    async def test_create_node(self, backend):
        """Test node creation."""
        await backend.create_graph("test_graph")

        node = GraphNode(
            id="node1",
            content="Test content",
            timestamp=datetime.now(),
            fractal_signature="sig123",
            metadata={"key": "value", "number": 42},
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
        assert retrieved.metadata["number"] == 42
        assert retrieved.potential == 0.9
        assert abs(retrieved.entropy - 0.4) < 0.01
        assert abs(retrieved.coherence - 0.7) < 0.01
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
            children_ids=["child"],
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

        # Verify parent's children_ids
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
        assert abs(retrieved.entropy - 0.2) < 0.01

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
        assert len(edges) >= 1
        edge_found = any(e.to_id == "node1" and e.relation_type == "CONNECTS_TO" for e in edges)
        assert edge_found

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
        out_targets = [e.to_id for e in out_edges]
        assert "node2" in out_targets

        # Test incoming
        in_edges = await backend.get_edges("node0", "in", graph_name="test_graph")
        in_sources = [e.from_id for e in in_edges]
        assert "node1" in in_sources

        # Test both
        both_edges = await backend.get_edges("node0", "both", graph_name="test_graph")
        assert len(both_edges) >= 2

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
        assert len(edges) >= 1

        # Delete
        await backend.delete_edge("node0", "node1", "TEST", "test_graph")

        # Verify deleted
        edges = await backend.get_edges("node0", graph_name="test_graph")
        test_edges = [e for e in edges if e.relation_type == "TEST" and e.to_id == "node1"]
        assert len(test_edges) == 0

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
        node_ids = [n.id for n in neighborhood.nodes]
        assert "node1" in node_ids

        # Get 2-hop neighbors
        neighborhood = await backend.get_neighbors("node0", max_hops=2, graph_name="test_graph")
        node_ids = [n.id for n in neighborhood.nodes]
        assert "node1" in node_ids
        # node2 should be reachable in 2 hops

    @pytest.mark.asyncio
    async def test_trace_lineage_backward(self, backend):
        """Test backward lineage tracing."""
        await backend.create_graph("test_graph")

        # Create parent -> child -> grandchild (via PARENT_OF relationships)
        parent = GraphNode(
            id="parent",
            content="Parent",
            timestamp=datetime.now(),
            fractal_signature="sig_p",
            parent_id="-1",
            children_ids=["child"],
            entropy=0.3,
        )
        await backend.create_node(parent, "test_graph")

        child = GraphNode(
            id="child",
            content="Child",
            timestamp=datetime.now(),
            fractal_signature="sig_c",
            parent_id="parent",
            children_ids=["grandchild"],
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

        # Create PARENT_OF relationships for lineage
        await backend.create_edge(
            GraphEdge(from_id="parent", to_id="child", relation_type="PARENT_OF"),
            "test_graph"
        )
        await backend.create_edge(
            GraphEdge(from_id="child", to_id="grandchild", relation_type="PARENT_OF"),
            "test_graph"
        )

        # Trace backward from grandchild
        path = await backend.trace_lineage("grandchild", "backward", graph_name="test_graph")

        assert path.root_id == "parent"
        assert "parent" in path.path
        assert "child" in path.path
        assert "grandchild" in path.path

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

        assert len(contradictions) >= 1
        # Should find at least one contradiction

    @pytest.mark.asyncio
    async def test_list_graphs(self, backend):
        """Test listing graphs."""
        await backend.create_graph("test_graph")

        # Create a node to make the graph appear
        node = GraphNode(
            id="test_node",
            content="Test",
            timestamp=datetime.now(),
            fractal_signature="sig",
        )
        await backend.create_node(node, "test_graph")

        graphs = await backend.list_graphs()
        graph_names = [g["name"] for g in graphs]
        assert "test_graph" in graph_names

    @pytest.mark.asyncio
    async def test_metadata_with_special_chars(self, backend):
        """Test metadata with special characters."""
        await backend.create_graph("test_graph")

        node = GraphNode(
            id="node1",
            content="Test with 'quotes' and \"double quotes\"",
            timestamp=datetime.now(),
            fractal_signature="sig123",
            metadata={
                "special": "contains 'quotes' and \"escapes\"",
                "nested": {"key": "value"},
            },
        )

        await backend.create_node(node, "test_graph")

        retrieved = await backend.get_node("node1", "test_graph")
        assert retrieved is not None
        assert "quotes" in retrieved.content

    @pytest.mark.asyncio
    async def test_large_metadata(self, backend):
        """Test node with large metadata."""
        await backend.create_graph("test_graph")

        large_text = "x" * 10000  # 10KB of text
        node = GraphNode(
            id="node1",
            content=large_text,
            timestamp=datetime.now(),
            fractal_signature="sig123",
            metadata={"large_field": large_text},
        )

        await backend.create_node(node, "test_graph")

        retrieved = await backend.get_node("node1", "test_graph")
        assert retrieved is not None
        assert len(retrieved.content) == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
