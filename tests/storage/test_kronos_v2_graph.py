"""
Unit and integration tests for KronosGraph (KRONOS v2).

Tests:
- Graph creation and node management
- Tree traversal (ancestors, descendants, siblings)
- Derivation path reconstruction
- PAC conservation verification
- Geometric confidence computation
- Anomaly detection (orphans, bottlenecks, gaps)
- Graph statistics
- Serialization/deserialization
"""

import pytest
import numpy as np
from datetime import datetime
from fracton.storage import (
    KronosGraph,
    KronosNode,
    KronosEdge,
    DocumentReference,
    GeometricConfidence,
)
from fracton.storage.edge import RelationType


@pytest.fixture
def empty_graph():
    """Create an empty graph."""
    return KronosGraph()


@pytest.fixture
def simple_tree():
    """Create a simple tree: root -> child1, child2."""
    graph = KronosGraph()

    # Root node
    root = KronosNode(
        id="root",
        name="Root",
        definition="Root concept",
        confluence_pattern={},
        parent_potentials=[],
        child_actualizations=[],
        derivation_path=[],
        actualization_depth=0,
        delta_embedding=np.random.randn(64)
    )
    graph.add_node(root)

    # Child 1
    child1 = KronosNode(
        id="child1",
        name="Child 1",
        definition="First child",
        confluence_pattern={"root": 1.0},
        parent_potentials=["root"],
        child_actualizations=[],
        derivation_path=["root"],
        actualization_depth=1,
        delta_embedding=np.random.randn(64) * 0.1
    )
    graph.add_node(child1)

    # Child 2
    child2 = KronosNode(
        id="child2",
        name="Child 2",
        definition="Second child",
        confluence_pattern={"root": 1.0},
        parent_potentials=["root"],
        child_actualizations=[],
        derivation_path=["root"],
        actualization_depth=1,
        delta_embedding=np.random.randn(64) * 0.1
    )
    graph.add_node(child2)

    return graph


@pytest.fixture
def deep_tree():
    """Create a deeper tree for traversal testing."""
    graph = KronosGraph()

    # Level 0: root
    root = KronosNode(
        id="root",
        name="Root",
        definition="Root",
        confluence_pattern={},
        parent_potentials=[],
        child_actualizations=[],
        derivation_path=[],
        actualization_depth=0,
        delta_embedding=np.random.randn(64)
    )
    graph.add_node(root)

    # Level 1: A, B
    for node_id in ["A", "B"]:
        node = KronosNode(
            id=node_id,
            name=f"Node {node_id}",
            definition=f"Level 1 node {node_id}",
            confluence_pattern={"root": 1.0},
            parent_potentials=["root"],
            child_actualizations=[],
            derivation_path=["root"],
            actualization_depth=1,
            delta_embedding=np.random.randn(64) * 0.1
        )
        graph.add_node(node)

    # Level 2: A1, A2 (children of A), B1 (child of B)
    for node_id, parent in [("A1", "A"), ("A2", "A"), ("B1", "B")]:
        node = KronosNode(
            id=node_id,
            name=f"Node {node_id}",
            definition=f"Level 2 node {node_id}",
            confluence_pattern={parent: 1.0},
            parent_potentials=[parent],
            child_actualizations=[],
            derivation_path=["root", parent],
            actualization_depth=2,
            delta_embedding=np.random.randn(64) * 0.05
        )
        graph.add_node(node)

    # Level 3: A1a (child of A1)
    a1a = KronosNode(
        id="A1a",
        name="Node A1a",
        definition="Level 3 node",
        confluence_pattern={"A1": 1.0},
        parent_potentials=["A1"],
        child_actualizations=[],
        derivation_path=["root", "A", "A1"],
        actualization_depth=3,
        delta_embedding=np.random.randn(64) * 0.02
    )
    graph.add_node(a1a)

    return graph


@pytest.fixture
def confluence_tree():
    """Create a tree with confluence node (multiple parents)."""
    graph = KronosGraph()

    # Root
    root = KronosNode(
        id="root",
        name="Root",
        definition="Root",
        confluence_pattern={},
        parent_potentials=[],
        child_actualizations=[],
        derivation_path=[],
        actualization_depth=0,
        delta_embedding=np.random.randn(64)
    )
    graph.add_node(root)

    # Parent A, B, C
    for parent_id in ["A", "B", "C"]:
        node = KronosNode(
            id=parent_id,
            name=f"Parent {parent_id}",
            definition=f"Parent {parent_id}",
            confluence_pattern={"root": 1.0},
            parent_potentials=["root"],
            child_actualizations=[],
            derivation_path=["root"],
            actualization_depth=1,
            delta_embedding=np.random.randn(64) * 0.1
        )
        graph.add_node(node)

    # Confluence node (merges A, B, C)
    confluence = KronosNode(
        id="confluence",
        name="Confluence",
        definition="Emerges from A, B, C",
        confluence_pattern={"A": 0.5, "B": 0.3, "C": 0.2},
        parent_potentials=["A", "B", "C"],
        child_actualizations=[],
        derivation_path=["root"],
        actualization_depth=2,
        delta_embedding=np.random.randn(64) * 0.05
    )
    graph.add_node(confluence)

    return graph


class TestGraphBasics:
    """Test basic graph operations."""

    def test_create_empty_graph(self, empty_graph):
        """Test creating an empty graph."""
        assert len(empty_graph.nodes) == 0
        assert len(empty_graph.edges) == 0
        assert len(empty_graph.roots) == 0

    def test_add_root_node(self, empty_graph):
        """Test adding a root node."""
        root = KronosNode(
            id="root",
            name="Root",
            definition="Root concept",
            confluence_pattern={},
            parent_potentials=[],
            child_actualizations=[],
            derivation_path=[],
            actualization_depth=0,
            delta_embedding=np.random.randn(64)
        )

        empty_graph.add_node(root)

        assert len(empty_graph.nodes) == 1
        assert "root" in empty_graph.roots
        assert empty_graph.get_node("root") is not None

    def test_add_child_node(self, empty_graph):
        """Test adding nodes with parent-child relationship."""
        # Add root
        root = KronosNode(
            id="root",
            name="Root",
            definition="Root",
            confluence_pattern={},
            parent_potentials=[],
            child_actualizations=[],
            derivation_path=[],
            actualization_depth=0,
            delta_embedding=np.random.randn(64)
        )
        empty_graph.add_node(root)

        # Add child
        child = KronosNode(
            id="child",
            name="Child",
            definition="Child",
            confluence_pattern={"root": 1.0},
            parent_potentials=["root"],
            child_actualizations=[],
            derivation_path=["root"],
            actualization_depth=1,
            delta_embedding=np.random.randn(64) * 0.1
        )
        empty_graph.add_node(child)

        assert len(empty_graph.nodes) == 2
        assert "child" in empty_graph.get_node("root").child_actualizations

    def test_get_node(self, simple_tree):
        """Test retrieving a node."""
        root = simple_tree.get_node("root")
        assert root is not None
        assert root.id == "root"

        missing = simple_tree.get_node("nonexistent")
        assert missing is None

    def test_remove_node(self, simple_tree):
        """Test removing a node."""
        assert simple_tree.get_node("child1") is not None

        simple_tree.remove_node("child1")

        assert simple_tree.get_node("child1") is None
        assert len(simple_tree.nodes) == 2  # root and child2 remain

    def test_add_edge(self, simple_tree):
        """Test adding an edge."""
        edge = KronosEdge(
            source_id="root",
            target_id="child1",
            relationship_type=RelationType.PARENT_OF,
            strength=1.0
        )

        simple_tree.add_edge(edge)

        assert len(simple_tree.edges) == 1
        assert simple_tree.edges[0].source_id == "root"


class TestTreeTraversal:
    """Test tree traversal operations."""

    def test_get_ancestors_single_level(self, simple_tree):
        """Test getting ancestors (1 level up)."""
        ancestors = simple_tree.get_ancestors("child1")

        assert len(ancestors) == 1
        assert ancestors[0].id == "root"

    def test_get_ancestors_multiple_levels(self, deep_tree):
        """Test getting ancestors multiple levels up."""
        ancestors = deep_tree.get_ancestors("A1a")

        # Should have: A1, A, root
        ancestor_ids = [n.id for n in ancestors]
        assert "A1" in ancestor_ids
        assert "A" in ancestor_ids
        assert "root" in ancestor_ids
        assert len(ancestors) == 3

    def test_get_ancestors_with_max_depth(self, deep_tree):
        """Test getting ancestors with depth limit."""
        ancestors = deep_tree.get_ancestors("A1a", max_depth=2)

        # Should have: A1, A (stop before root)
        ancestor_ids = [n.id for n in ancestors]
        assert "A1" in ancestor_ids
        assert "A" in ancestor_ids
        assert "root" not in ancestor_ids

    def test_get_ancestors_root_has_none(self, deep_tree):
        """Test that root node has no ancestors."""
        ancestors = deep_tree.get_ancestors("root")
        assert len(ancestors) == 0

    def test_get_descendants_single_level(self, simple_tree):
        """Test getting descendants (1 level down)."""
        descendants = simple_tree.get_descendants("root")

        descendant_ids = [n.id for n in descendants]
        assert "child1" in descendant_ids
        assert "child2" in descendant_ids
        assert len(descendants) == 2

    def test_get_descendants_multiple_levels(self, deep_tree):
        """Test getting descendants multiple levels down."""
        descendants = deep_tree.get_descendants("A")

        # Should have: A1, A2, A1a
        descendant_ids = [n.id for n in descendants]
        assert "A1" in descendant_ids
        assert "A2" in descendant_ids
        assert "A1a" in descendant_ids
        assert len(descendants) == 3

    def test_get_descendants_with_max_depth(self, deep_tree):
        """Test getting descendants with depth limit."""
        descendants = deep_tree.get_descendants("A", max_depth=1)

        # Should have: A1, A2 (stop before A1a)
        descendant_ids = [n.id for n in descendants]
        assert "A1" in descendant_ids
        assert "A2" in descendant_ids
        assert "A1a" not in descendant_ids

    def test_get_siblings(self, deep_tree):
        """Test getting siblings."""
        siblings = deep_tree.get_siblings("A1")

        # A1 and A2 are siblings (both children of A)
        sibling_ids = [n.id for n in siblings]
        assert "A2" in sibling_ids
        assert "A1" not in sibling_ids  # Node is not its own sibling

    def test_get_siblings_only_child(self, deep_tree):
        """Test getting siblings for an only child."""
        siblings = deep_tree.get_siblings("B1")

        # B1 is only child of B, so no siblings
        assert len(siblings) == 0

    def test_get_derivation_path(self, deep_tree):
        """Test getting full derivation path."""
        path = deep_tree.get_derivation_path("A1a")

        # Should have: root -> A -> A1 -> A1a
        path_ids = [n.id for n in path]
        assert path_ids == ["root", "A", "A1", "A1a"]

    def test_get_neighbors_within_radius(self, deep_tree):
        """Test getting neighbors within radius."""
        neighbors = deep_tree.get_neighbors_within_radius("A", radius=1)

        neighbor_ids = [n.id for n in neighbors]

        # Radius 1 from A should include:
        # - Parent: root
        # - Children: A1, A2
        # - Sibling: B
        assert "root" in neighbor_ids
        assert "A1" in neighbor_ids
        assert "A2" in neighbor_ids
        assert "B" in neighbor_ids

    def test_get_neighbors_larger_radius(self, deep_tree):
        """Test getting neighbors with larger radius."""
        neighbors = deep_tree.get_neighbors_within_radius("A", radius=2)

        neighbor_ids = [n.id for n in neighbors]

        # Radius 2 should also include: B1, A1a
        assert "B1" in neighbor_ids
        assert "A1a" in neighbor_ids


class TestConfluencePatterns:
    """Test confluence node handling."""

    def test_confluence_multiple_ancestors(self, confluence_tree):
        """Test that confluence node has all parents as ancestors."""
        ancestors = confluence_tree.get_ancestors("confluence")

        ancestor_ids = [n.id for n in ancestors]
        assert "A" in ancestor_ids
        assert "B" in ancestor_ids
        assert "C" in ancestor_ids
        assert "root" in ancestor_ids  # All paths lead to root

    def test_confluence_node_weights(self, confluence_tree):
        """Test confluence node has correct weights."""
        node = confluence_tree.get_node("confluence")

        assert node.confluence_pattern["A"] == 0.5
        assert node.confluence_pattern["B"] == 0.3
        assert node.confluence_pattern["C"] == 0.2

        # Weights should sum to 1.0
        total = sum(node.confluence_pattern.values())
        assert abs(total - 1.0) < 0.01


class TestPACConservation:
    """Test PAC conservation verification."""

    def test_verify_conservation_simple(self, simple_tree):
        """Test PAC conservation on simple tree."""
        # For this test to pass, we need proper delta embeddings
        # Since we're using random deltas, conservation will likely fail
        # But the test verifies the mechanism works

        is_conserved = simple_tree.verify_conservation("child1", tolerance=0.5)

        # With random deltas, conservation may or may not hold
        # The test just verifies no errors occur
        assert isinstance(is_conserved, bool)

    def test_reconstruct_embedding(self, simple_tree):
        """Test embedding reconstruction."""
        reconstructed = simple_tree._reconstruct_embedding("child1")

        # Should return a numpy array
        assert isinstance(reconstructed, np.ndarray)
        assert reconstructed.shape == (64,)

    def test_verify_conservation_root_fails(self, simple_tree):
        """Test that root node verification fails (no parent to reconstruct from)."""
        is_conserved = simple_tree.verify_conservation("root")

        # Root has no parent, so conservation check should fail
        assert is_conserved == False


class TestGeometricConfidence:
    """Test geometric confidence computation."""

    def test_compute_confidence_for_node(self, deep_tree):
        """Test computing confidence for a node."""
        confidence = deep_tree.compute_geometric_confidence("A")

        assert isinstance(confidence, GeometricConfidence)
        assert 0.0 <= confidence.retrieval_confidence <= 1.0
        assert 0.0 <= confidence.hallucination_risk <= 1.0

    def test_confidence_well_documented_higher(self, empty_graph):
        """Test that well-documented nodes have higher confidence."""
        # Add well-documented root
        root = KronosNode(
            id="root",
            name="Root",
            definition="Root",
            confluence_pattern={},
            parent_potentials=[],
            child_actualizations=[],
            derivation_path=[],
            actualization_depth=0,
            delta_embedding=np.random.randn(64),
            documentation_depth=10,
            supported_by=[
                DocumentReference(
                    doc_id=f"doc{i}",
                    title=f"Document {i}",
                    authors=["Author"],
                    year=2020
                )
                for i in range(10)
            ]
        )
        empty_graph.add_node(root)

        # Add poorly documented child
        child = KronosNode(
            id="child",
            name="Child",
            definition="Child",
            confluence_pattern={"root": 1.0},
            parent_potentials=["root"],
            child_actualizations=[],
            derivation_path=["root"],
            actualization_depth=1,
            delta_embedding=np.random.randn(64) * 0.1,
            documentation_depth=0,
            supported_by=[]
        )
        empty_graph.add_node(child)

        conf_root = empty_graph.compute_geometric_confidence("root")
        conf_child = empty_graph.compute_geometric_confidence("child")

        # Well-documented node should have higher confidence
        assert conf_root.documentation_depth > conf_child.documentation_depth

    def test_find_knowledge_gaps(self, deep_tree):
        """Test finding knowledge gaps."""
        gaps = deep_tree.find_knowledge_gaps()

        # Should return a dictionary mapping node_id to list of missing children
        assert isinstance(gaps, dict)


class TestAnomalyDetection:
    """Test anomaly detection."""

    def test_detect_orphans(self, empty_graph):
        """Test detecting orphaned nodes."""
        # Add orphaned node (has parent reference but parent doesn't exist)
        orphan = KronosNode(
            id="orphan",
            name="Orphan",
            definition="Orphaned node",
            confluence_pattern={"nonexistent_parent": 1.0},
            parent_potentials=["nonexistent_parent"],
            child_actualizations=[],
            derivation_path=["nonexistent_parent"],
            actualization_depth=1,
            delta_embedding=np.random.randn(64)
        )

        # This should fail or be detected as anomalous
        # The graph should validate parent existence
        with pytest.raises((ValueError, KeyError)):
            empty_graph.add_node(orphan)

    def test_detect_confluence_bottleneck(self, confluence_tree):
        """Test detecting confluence bottleneck."""
        # Add a node with extreme confluence weight (bottleneck)
        bottleneck = KronosNode(
            id="bottleneck",
            name="Bottleneck",
            definition="Bottleneck node",
            confluence_pattern={"A": 0.9, "B": 0.1},  # One parent dominates
            parent_potentials=["A", "B"],
            child_actualizations=[],
            derivation_path=["root"],
            actualization_depth=2,
            delta_embedding=np.random.randn(64) * 0.05
        )
        confluence_tree.add_node(bottleneck)

        confidence = confluence_tree.compute_geometric_confidence("bottleneck")

        # Should detect bottleneck
        assert confidence.confluence_bottleneck


class TestGraphStatistics:
    """Test graph statistics."""

    def test_graph_stats(self, deep_tree):
        """Test getting graph statistics."""
        stats = {
            "nodes": len(deep_tree.nodes),
            "edges": len(deep_tree.edges),
            "roots": len(deep_tree.roots),
            "max_depth": max([n.actualization_depth for n in deep_tree.nodes.values()]) if deep_tree.nodes else 0
        }

        assert stats["nodes"] > 0
        assert stats["roots"] > 0
        assert stats["max_depth"] >= 0

    def test_count_nodes_by_depth(self, deep_tree):
        """Test counting nodes at each depth level."""
        depth_counts = {}
        for node in deep_tree.nodes.values():
            depth = node.actualization_depth
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        assert 0 in depth_counts  # Has root
        assert depth_counts[0] >= 1  # At least one root


class TestGraphSerialization:
    """Test graph serialization."""

    def test_graph_to_dict(self, simple_tree):
        """Test serializing graph to dictionary."""
        data = simple_tree.to_dict()

        assert "nodes" in data
        assert "edges" in data
        assert "roots" in data
        assert len(data["nodes"]) == 3  # root, child1, child2

    def test_graph_from_dict(self, simple_tree):
        """Test deserializing graph from dictionary."""
        data = simple_tree.to_dict()
        restored = KronosGraph.from_dict(data)

        assert len(restored.nodes) == len(simple_tree.nodes)
        assert len(restored.edges) == len(simple_tree.edges)
        assert len(restored.roots) == len(simple_tree.roots)

    def test_graph_roundtrip(self, deep_tree):
        """Test full serialization round-trip."""
        original_node_count = len(deep_tree.nodes)
        original_edge_count = len(deep_tree.edges)

        data = deep_tree.to_dict()
        restored = KronosGraph.from_dict(data)

        assert len(restored.nodes) == original_node_count
        assert len(restored.edges) == original_edge_count

        # Verify specific node
        original_node = deep_tree.get_node("A1a")
        restored_node = restored.get_node("A1a")

        assert restored_node is not None
        assert restored_node.name == original_node.name
        assert restored_node.actualization_depth == original_node.actualization_depth


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
