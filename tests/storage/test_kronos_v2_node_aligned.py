"""
Unit tests for KronosNode (KRONOS v2) - Aligned with Implementation.

Tests aligned with the actual flexible, auto-correcting implementation behavior:
- Auto-correction of depth mismatches
- Auto-population of missing parents
- Computed properties (documentation_depth)
- Flexible validation philosophy
"""

import pytest
import numpy as np
from datetime import datetime
from fracton.storage import (
    KronosNode,
    DocumentReference,
    CrystallizationEvent,
)


class TestDocumentReference:
    """Test DocumentReference validation and behavior."""

    def test_valid_document_reference(self):
        """Test creating a valid document reference."""
        doc = DocumentReference(
            doc_id="paper_001",
            title="Test Paper",
            authors=["Author A", "Author B"],
            year=2020,
            doi="10.1234/test",
            uri="https://example.com/paper",
            excerpt="Key insight from the paper..."
        )

        assert doc.doc_id == "paper_001"
        assert doc.title == "Test Paper"
        assert len(doc.authors) == 2
        assert doc.year == 2020
        assert doc.doi == "10.1234/test"

    def test_document_requires_title(self):
        """Test that document reference requires a title."""
        with pytest.raises(ValueError, match="Document must have a title"):
            DocumentReference(
                doc_id="paper_002",
                title="",
                authors=["Author C"],
                year=2021
            )

    def test_document_year_validation_historical(self):
        """Test that historical years (1600+) are accepted."""
        doc = DocumentReference(
            doc_id="principia",
            title="Philosophiæ Naturalis Principia Mathematica",
            authors=["Isaac Newton"],
            year=1687
        )

        assert doc.year == 1687

    def test_document_year_validation_range(self):
        """Test year validation range (1600 to current+1)."""
        # Too early (before 1600)
        with pytest.raises(ValueError, match="Invalid year"):
            DocumentReference(
                doc_id="ancient",
                title="Ancient Text",
                authors=["Ancient Author"],
                year=1599
            )

        # Current year (should pass)
        current_year = datetime.now().year
        doc = DocumentReference(
            doc_id="current",
            title="Current Paper",
            authors=["Author"],
            year=current_year
        )
        assert doc.year == current_year

        # Next year (should pass)
        next_year = current_year + 1
        doc = DocumentReference(
            doc_id="next_year",
            title="Next Year Paper",
            authors=["Author"],
            year=next_year
        )
        assert doc.year == next_year

        # Too far future
        with pytest.raises(ValueError, match="Invalid year"):
            DocumentReference(
                doc_id="far_future",
                title="Far Future Paper",
                authors=["Author"],
                year=current_year + 2
            )


class TestCrystallizationEvent:
    """Test CrystallizationEvent tracking."""

    def test_valid_crystallization_event(self):
        """Test creating a valid crystallization event."""
        doc = DocumentReference(
            doc_id="paper_006",
            title="Crystallization Paper",
            authors=["Author E"],
            year=2022
        )

        event = CrystallizationEvent(
            timestamp=datetime(2022, 6, 15, 10, 30),
            document=doc,
            context="First mention of concept in literature",
            confidence=0.95
        )

        assert event.timestamp == datetime(2022, 6, 15, 10, 30)
        assert event.document.doc_id == "paper_006"
        assert event.context == "First mention of concept in literature"
        assert event.confidence == 0.95


class TestKronosNodeBasics:
    """Test basic KronosNode creation and properties."""

    def test_create_root_node(self):
        """Test creating a root node (no parents)."""
        node = KronosNode(
            id="root",
            name="Root Concept",
            definition="The foundational concept",
            confluence_pattern={},
            parent_potentials=[],
            child_actualizations=[],
            derivation_path=[],
            actualization_depth=0,
            delta_embedding=np.random.randn(64)
        )

        assert node.id == "root"
        assert node.name == "Root Concept"
        assert node.actualization_depth == 0
        assert node.is_root
        assert len(node.parent_potentials) == 0
        assert len(node.confluence_pattern) == 0

    def test_create_single_parent_node(self):
        """Test creating a node with single parent."""
        node = KronosNode(
            id="child",
            name="Child Concept",
            definition="Derived from root",
            confluence_pattern={"root": 1.0},
            parent_potentials=["root"],
            child_actualizations=[],
            derivation_path=["root"],
            actualization_depth=1,
            delta_embedding=np.random.randn(64)
        )

        assert node.id == "child"
        assert node.actualization_depth == 1
        assert not node.is_root
        assert len(node.parent_potentials) == 1
        assert node.primary_parent == "root"

    def test_create_confluence_node(self):
        """Test creating a confluence node (multiple parents)."""
        node = KronosNode(
            id="confluence",
            name="Confluence Concept",
            definition="Emerges from multiple parents",
            confluence_pattern={
                "parent_a": 0.4,
                "parent_b": 0.35,
                "parent_c": 0.25
            },
            parent_potentials=["parent_a", "parent_b", "parent_c"],
            child_actualizations=[],
            derivation_path=["root", "branch"],
            actualization_depth=2,
            delta_embedding=np.random.randn(64)
        )

        assert node.id == "confluence"
        assert len(node.parent_potentials) == 3
        assert len(node.confluence_pattern) == 3

        # Check confluence weights sum to ~1.0
        total_weight = sum(node.confluence_pattern.values())
        assert abs(total_weight - 1.0) < 0.01

        # Parents should be sorted by weight (highest first)
        assert node.parent_potentials[0] == "parent_a"  # 0.4
        assert node.primary_parent == "parent_a"

    def test_confluence_pattern_validation_sum(self):
        """Test that confluence pattern must sum to ~1.0."""
        with pytest.raises(ValueError, match="Confluence pattern must sum to ~1.0"):
            KronosNode(
                id="invalid_confluence",
                name="Invalid Confluence",
                definition="Bad weights",
                confluence_pattern={
                    "parent_a": 0.5,
                    "parent_b": 0.3  # Sum = 0.8, should fail
                },
                parent_potentials=["parent_a", "parent_b"],
                child_actualizations=[],
                derivation_path=["root"],
                actualization_depth=1,
                delta_embedding=np.random.randn(64)
            )


class TestKronosNodeAutoCorrection:
    """Test auto-correction behaviors in KronosNode."""

    def test_auto_add_missing_parents(self):
        """Test that missing parents in parent_potentials get auto-added."""
        node = KronosNode(
            id="node",
            name="Node",
            definition="Test",
            confluence_pattern={
                "parent_a": 0.6,
                "parent_b": 0.4
            },
            parent_potentials=["parent_a"],  # Missing parent_b
            child_actualizations=[],
            derivation_path=["parent_a"],
            actualization_depth=1,
            delta_embedding=np.random.randn(64)
        )

        # parent_b should be auto-added
        assert "parent_b" in node.parent_potentials
        assert len(node.parent_potentials) == 2

    def test_auto_correct_depth_mismatch(self):
        """Test that depth is auto-corrected to match derivation_path."""
        node = KronosNode(
            id="node",
            name="Node",
            definition="Test",
            confluence_pattern={"parent": 1.0},
            parent_potentials=["parent"],
            child_actualizations=[],
            derivation_path=["root", "parent"],  # Length 2
            actualization_depth=1,  # Says 1, but should be 2
            delta_embedding=np.random.randn(64)
        )

        # Depth should be auto-corrected to 2
        assert node.actualization_depth == 2

    def test_auto_sort_parents_by_weight(self):
        """Test that parents are sorted by confluence weight."""
        node = KronosNode(
            id="node",
            name="Node",
            definition="Test",
            confluence_pattern={
                "parent_a": 0.2,  # Lowest
                "parent_b": 0.5,  # Highest
                "parent_c": 0.3   # Middle
            },
            parent_potentials=["parent_a", "parent_b", "parent_c"],
            child_actualizations=[],
            derivation_path=["root"],
            actualization_depth=1,
            delta_embedding=np.random.randn(64)
        )

        # Should be sorted: parent_b (0.5), parent_c (0.3), parent_a (0.2)
        assert node.parent_potentials[0] == "parent_b"
        assert node.parent_potentials[1] == "parent_c"
        assert node.parent_potentials[2] == "parent_a"


class TestKronosNodeMethods:
    """Test KronosNode methods."""

    def test_add_child(self):
        """Test adding children."""
        node = KronosNode(
            id="parent",
            name="Parent",
            definition="Parent node",
            confluence_pattern={},
            parent_potentials=[],
            child_actualizations=[],
            derivation_path=[],
            actualization_depth=0,
            delta_embedding=np.random.randn(64)
        )

        node.add_child("child_1")
        assert "child_1" in node.child_actualizations
        assert len(node.child_actualizations) == 1

        node.add_child("child_2")
        assert len(node.child_actualizations) == 2

        # Adding duplicate should not increase count
        node.add_child("child_1")
        assert len(node.child_actualizations) == 2

    def test_add_sibling(self):
        """Test adding siblings."""
        node = KronosNode(
            id="node_a",
            name="Node A",
            definition="Node A",
            confluence_pattern={"parent": 1.0},
            parent_potentials=["parent"],
            child_actualizations=[],
            derivation_path=["parent"],
            actualization_depth=1,
            delta_embedding=np.random.randn(64)
        )

        node.add_sibling("node_b")
        assert "node_b" in node.sibling_nodes
        assert len(node.sibling_nodes) == 1

        # Can't be your own sibling
        node.add_sibling("node_a")
        assert "node_a" not in node.sibling_nodes

    def test_add_supporting_document(self):
        """Test adding supporting documents."""
        node = KronosNode(
            id="node",
            name="Node",
            definition="Test",
            confluence_pattern={},
            parent_potentials=[],
            child_actualizations=[],
            derivation_path=[],
            actualization_depth=0,
            delta_embedding=np.random.randn(64)
        )

        doc1 = DocumentReference(
            doc_id="doc1",
            title="Document 1",
            authors=["Author"],
            year=2020
        )

        node.add_supporting_document(doc1)
        assert len(node.supported_by) == 1
        assert node.documentation_depth == 1

        doc2 = DocumentReference(
            doc_id="doc2",
            title="Document 2",
            authors=["Author"],
            year=2021
        )

        node.add_supporting_document(doc2)
        assert len(node.supported_by) == 2
        assert node.documentation_depth == 2

    def test_record_crystallization(self):
        """Test recording crystallization events."""
        node = KronosNode(
            id="concept",
            name="Concept",
            definition="Test concept",
            confluence_pattern={"parent": 1.0},
            parent_potentials=["parent"],
            child_actualizations=[],
            derivation_path=["parent"],
            actualization_depth=1,
            delta_embedding=np.random.randn(64)
        )

        doc = DocumentReference(
            doc_id="paper_007",
            title="Crystallization Paper",
            authors=["Author F"],
            year=2023
        )

        event = CrystallizationEvent(
            timestamp=datetime(2023, 1, 1),
            document=doc,
            context="First crystallization",
            confidence=0.9
        )

        node.record_crystallization(event)
        assert len(node.crystallization_events) == 1
        assert node.crystallization_events[0].context == "First crystallization"
        assert node.first_crystallization == datetime(2023, 1, 1)

        # Add earlier event
        earlier_event = CrystallizationEvent(
            timestamp=datetime(2022, 6, 1),
            document=doc,
            context="Earlier crystallization",
            confidence=0.8
        )

        node.record_crystallization(earlier_event)
        assert len(node.crystallization_events) == 2
        assert node.first_crystallization == datetime(2022, 6, 1)  # Should update

    def test_record_access(self):
        """Test access tracking."""
        node = KronosNode(
            id="node",
            name="Node",
            definition="Test",
            confluence_pattern={},
            parent_potentials=[],
            child_actualizations=[],
            derivation_path=[],
            actualization_depth=0,
            delta_embedding=np.random.randn(64)
        )

        assert node.access_count == 0

        node.record_access()
        assert node.access_count == 1

        node.record_access()
        assert node.access_count == 2


class TestKronosNodeProperties:
    """Test KronosNode computed properties."""

    def test_is_root_property(self):
        """Test is_root property."""
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

        child = KronosNode(
            id="child",
            name="Child",
            definition="Child",
            confluence_pattern={"root": 1.0},
            parent_potentials=["root"],
            child_actualizations=[],
            derivation_path=["root"],
            actualization_depth=1,
            delta_embedding=np.random.randn(64)
        )

        assert root.is_root
        assert not child.is_root

    def test_is_leaf_property(self):
        """Test is_leaf property."""
        leaf = KronosNode(
            id="leaf",
            name="Leaf",
            definition="Leaf",
            confluence_pattern={"parent": 1.0},
            parent_potentials=["parent"],
            child_actualizations=[],  # No children
            derivation_path=["parent"],
            actualization_depth=1,
            delta_embedding=np.random.randn(64)
        )

        parent = KronosNode(
            id="parent",
            name="Parent",
            definition="Parent",
            confluence_pattern={},
            parent_potentials=[],
            child_actualizations=["child"],  # Has children
            derivation_path=[],
            actualization_depth=0,
            delta_embedding=np.random.randn(64)
        )

        assert leaf.is_leaf
        assert not parent.is_leaf

    def test_primary_parent_property(self):
        """Test primary_parent property."""
        # Single parent
        node1 = KronosNode(
            id="node1",
            name="Node 1",
            definition="Test",
            confluence_pattern={"parent": 1.0},
            parent_potentials=["parent"],
            child_actualizations=[],
            derivation_path=["parent"],
            actualization_depth=1,
            delta_embedding=np.random.randn(64)
        )

        assert node1.primary_parent == "parent"

        # Multiple parents (highest weight)
        node2 = KronosNode(
            id="node2",
            name="Node 2",
            definition="Test",
            confluence_pattern={
                "parent_a": 0.6,
                "parent_b": 0.4
            },
            parent_potentials=["parent_a", "parent_b"],
            child_actualizations=[],
            derivation_path=["root"],
            actualization_depth=1,
            delta_embedding=np.random.randn(64)
        )

        assert node2.primary_parent == "parent_a"

    def test_documentation_depth_property(self):
        """Test documentation_depth computed property."""
        node = KronosNode(
            id="node",
            name="Node",
            definition="Test",
            confluence_pattern={},
            parent_potentials=[],
            child_actualizations=[],
            derivation_path=[],
            actualization_depth=0,
            delta_embedding=np.random.randn(64),
            supported_by=[]
        )

        assert node.documentation_depth == 0

        # Add documents
        node.supported_by.append(DocumentReference(
            doc_id="doc1",
            title="Doc 1",
            authors=["Author"],
            year=2020
        ))
        assert node.documentation_depth == 1

        node.supported_by.append(DocumentReference(
            doc_id="doc2",
            title="Doc 2",
            authors=["Author"],
            year=2021
        ))
        assert node.documentation_depth == 2

    def test_get_full_path_str(self):
        """Test derivation path string representation."""
        node = KronosNode(
            id="leaf",
            name="Leaf",
            definition="Test",
            confluence_pattern={"parent": 1.0},
            parent_potentials=["parent"],
            child_actualizations=[],
            derivation_path=["root", "branch", "parent"],
            actualization_depth=3,
            delta_embedding=np.random.randn(64)
        )

        path_str = node.get_full_path_str()
        assert path_str == "root → branch → parent"


class TestKronosNodeSerialization:
    """Test node serialization and deserialization."""

    def test_node_to_dict(self):
        """Test serializing node to dictionary."""
        node = KronosNode(
            id="test_node",
            name="Test Node",
            definition="Test definition",
            confluence_pattern={"parent_a": 0.6, "parent_b": 0.4},
            parent_potentials=["parent_a", "parent_b"],
            child_actualizations=["child_1", "child_2"],
            sibling_nodes=["sibling_1"],
            derivation_path=["root", "parent_a"],
            actualization_depth=2,
            delta_embedding=np.array([1.0, 2.0, 3.0]),
            delta_structural=np.array([0.1, 0.2, 0.3]),
            supported_by=[
                DocumentReference(
                    doc_id="doc1",
                    title="Doc 1",
                    authors=["Author"],
                    year=2020
                )
            ]
        )

        data = node.to_dict()

        assert data["id"] == "test_node"
        assert data["name"] == "Test Node"
        assert len(data["parent_potentials"]) == 2
        assert len(data["child_actualizations"]) == 2
        assert data["actualization_depth"] == 2

    def test_node_from_dict(self):
        """Test deserializing node from dictionary."""
        # First create and serialize
        original = KronosNode(
            id="test_node",
            name="Test Node",
            definition="Test",
            confluence_pattern={"parent": 1.0},
            parent_potentials=["parent"],
            child_actualizations=[],
            derivation_path=["parent"],
            actualization_depth=1,
            delta_embedding=np.array([1.0, 2.0, 3.0])
        )

        data = original.to_dict()

        # Deserialize
        restored = KronosNode.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.actualization_depth == original.actualization_depth
        assert np.array_equal(restored.delta_embedding, original.delta_embedding)


class TestKronosNodeValidation:
    """Test node validation (actual behavior)."""

    def test_invalid_node_id_format(self):
        """Test that invalid ID format is rejected."""
        with pytest.raises(ValueError, match="Invalid node ID"):
            KronosNode(
                id="",  # Empty
                name="Node",
                definition="Test",
                confluence_pattern={},
                parent_potentials=[],
                child_actualizations=[],
                derivation_path=[],
                actualization_depth=0,
                delta_embedding=np.random.randn(64)
            )

        with pytest.raises(ValueError, match="Invalid node ID"):
            KronosNode(
                id="node with spaces",  # Spaces not allowed
                name="Node",
                definition="Test",
                confluence_pattern={},
                parent_potentials=[],
                child_actualizations=[],
                derivation_path=[],
                actualization_depth=0,
                delta_embedding=np.random.randn(64)
            )

    def test_valid_node_id_formats(self):
        """Test valid ID formats."""
        # Underscore
        node1 = KronosNode(
            id="node_with_underscore",
            name="Node",
            definition="Test",
            confluence_pattern={},
            parent_potentials=[],
            child_actualizations=[],
            derivation_path=[],
            actualization_depth=0,
            delta_embedding=np.random.randn(64)
        )
        assert node1.id == "node_with_underscore"

        # Hyphen
        node2 = KronosNode(
            id="node-with-hyphen",
            name="Node",
            definition="Test",
            confluence_pattern={},
            parent_potentials=[],
            child_actualizations=[],
            derivation_path=[],
            actualization_depth=0,
            delta_embedding=np.random.randn(64)
        )
        assert node2.id == "node-with-hyphen"

        # Mixed
        node3 = KronosNode(
            id="node_123-test",
            name="Node",
            definition="Test",
            confluence_pattern={},
            parent_potentials=[],
            child_actualizations=[],
            derivation_path=[],
            actualization_depth=0,
            delta_embedding=np.random.randn(64)
        )
        assert node3.id == "node_123-test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
