"""
Test Backend Abstract Base Classes

Verifies that the abstract interfaces are properly defined.
"""

import pytest
from datetime import datetime
import torch

from fracton.storage.backends import (
    GraphBackend,
    VectorBackend,
    BackendConfig,
)
from fracton.storage.backends.base import (
    GraphNode,
    GraphEdge,
    GraphNeighborhood,
    TemporalPath,
    VectorPoint,
    VectorSearchResult,
)


class TestBackendConfig:
    """Test BackendConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = BackendConfig()

        assert config.graph_type == "sqlite"
        assert config.vector_type == "chromadb"
        assert config.embedding_dim == 384
        assert config.device == "cpu"

    def test_custom_config(self):
        """Test custom configuration."""
        config = BackendConfig(
            graph_type="neo4j",
            graph_uri="bolt://localhost:7687",
            graph_user="neo4j",
            graph_password="test",
            vector_type="qdrant",
            vector_host="localhost",
            vector_port=6333,
            embedding_dim=512,
            device="cuda",
        )

        assert config.graph_type == "neo4j"
        assert config.graph_uri == "bolt://localhost:7687"
        assert config.vector_type == "qdrant"
        assert config.vector_host == "localhost"
        assert config.embedding_dim == 512
        assert config.device == "cuda"


class TestGraphDataStructures:
    """Test graph data structures."""

    def test_graph_node(self):
        """Test GraphNode creation."""
        node = GraphNode(
            id="node123",
            content="Test content",
            timestamp=datetime.now(),
            fractal_signature="abc123",
            metadata={"key": "value"},
            parent_id="parent123",
            potential=0.8,
            entropy=0.3,
        )

        assert node.id == "node123"
        assert node.content == "Test content"
        assert node.fractal_signature == "abc123"
        assert node.parent_id == "parent123"
        assert node.potential == 0.8
        assert node.entropy == 0.3
        assert node.metadata["key"] == "value"

    def test_graph_edge(self):
        """Test GraphEdge creation."""
        edge = GraphEdge(
            from_id="node1",
            to_id="node2",
            relation_type="EVOLVES_FROM",
            weight=1.0,
        )

        assert edge.from_id == "node1"
        assert edge.to_id == "node2"
        assert edge.relation_type == "EVOLVES_FROM"
        assert edge.weight == 1.0

    def test_graph_neighborhood(self):
        """Test GraphNeighborhood creation."""
        nodes = [
            GraphNode(
                id=f"node{i}",
                content=f"Content {i}",
                timestamp=datetime.now(),
                fractal_signature=f"sig{i}",
            )
            for i in range(3)
        ]

        edges = [
            GraphEdge(from_id="node0", to_id="node1", relation_type="CONNECTS"),
            GraphEdge(from_id="node0", to_id="node2", relation_type="CONNECTS"),
        ]

        neighborhood = GraphNeighborhood(
            center_id="node0",
            nodes=nodes,
            edges=edges,
            depth_map={"node0": 0, "node1": 1, "node2": 1},
        )

        assert neighborhood.center_id == "node0"
        assert len(neighborhood.nodes) == 3
        assert len(neighborhood.edges) == 2
        assert neighborhood.depth_map["node1"] == 1

    def test_temporal_path(self):
        """Test TemporalPath creation."""
        nodes = [
            GraphNode(
                id=f"node{i}",
                content=f"Content {i}",
                timestamp=datetime.now(),
                fractal_signature=f"sig{i}",
            )
            for i in range(3)
        ]

        path = TemporalPath(
            root_id="node0",
            path=["node0", "node1", "node2"],
            nodes=nodes,
            edges=[],
            total_potential=2.5,
            entropy_evolution=[0.5, 0.4, 0.3],
        )

        assert path.root_id == "node0"
        assert len(path.path) == 3
        assert path.total_potential == 2.5
        assert len(path.entropy_evolution) == 3


class TestVectorDataStructures:
    """Test vector data structures."""

    def test_vector_point(self):
        """Test VectorPoint creation."""
        vec = torch.randn(384)
        point = VectorPoint(
            id="vec123",
            vector=vec,
            payload={"content": "test", "metadata": {"key": "value"}},
        )

        assert point.id == "vec123"
        assert point.vector.shape == (384,)
        assert point.payload["content"] == "test"

    def test_vector_point_to_dict(self):
        """Test VectorPoint serialization."""
        vec = torch.randn(384)
        point = VectorPoint(
            id="vec123",
            vector=vec,
            payload={"content": "test"},
        )

        d = point.to_dict()

        assert d["id"] == "vec123"
        assert len(d["vector"]) == 384
        assert d["payload"]["content"] == "test"

    def test_vector_search_result(self):
        """Test VectorSearchResult creation."""
        vec = torch.randn(384)
        result = VectorSearchResult(
            id="vec123",
            score=0.95,
            vector=vec,
            payload={"content": "result"},
        )

        assert result.id == "vec123"
        assert result.score == 0.95
        assert result.vector.shape == (384,)
        assert result.payload["content"] == "result"


class TestAbstractBackends:
    """Test abstract backend classes."""

    def test_graph_backend_is_abstract(self):
        """Test that GraphBackend cannot be instantiated."""
        with pytest.raises(TypeError):
            GraphBackend()

    def test_vector_backend_is_abstract(self):
        """Test that VectorBackend cannot be instantiated."""
        with pytest.raises(TypeError):
            VectorBackend()

    def test_graph_backend_methods(self):
        """Test that GraphBackend defines required methods."""
        required_methods = [
            'connect', 'close',
            'create_node', 'get_node', 'update_node', 'delete_node',
            'create_edge', 'get_edges', 'delete_edge',
            'create_graph', 'list_graphs', 'get_neighbors',
            'trace_lineage', 'find_contradictions',
            'health_check',
        ]

        for method in required_methods:
            assert hasattr(GraphBackend, method)
            assert callable(getattr(GraphBackend, method))

    def test_vector_backend_methods(self):
        """Test that VectorBackend defines required methods."""
        required_methods = [
            'connect', 'close',
            'store', 'store_batch', 'retrieve', 'delete',
            'search', 'search_batch',
            'create_collection', 'delete_collection', 'collection_info',
            'health_check', 'get_device',
        ]

        for method in required_methods:
            assert hasattr(VectorBackend, method)
            assert callable(getattr(VectorBackend, method))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
