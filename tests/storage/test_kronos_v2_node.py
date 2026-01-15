"""
Unit tests for KronosNode (KRONOS v2).

Tests:
- Node creation and validation
- Confluence pattern validation
- DocumentReference validation
- CrystallizationEvent tracking
- Node serialization/deserialization
- Edge cases and error handling
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

    def test_document_year_validation_too_early(self):
        """Test year validation rejects years before 1600."""
        with pytest.raises(ValueError, match="Invalid year: 1599"):
            DocumentReference(
                doc_id="paper_003",
                title="Ancient Text",
                authors=["Ancient Author"],
                year=1599
            )

    def test_document_year_validation_too_late(self):
        """Test year validation rejects future years."""
        future_year = datetime.now().year + 2
        with pytest.raises(ValueError, match=f"Invalid year: {future_year}"):
            DocumentReference(
                doc_id="paper_004",
                title="Future Paper",
                authors=["Future Author"],
                year=future_year
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

    def test_document_year_validation_current(self):
        """Test that current year is accepted."""
        current_year = datetime.now().year
        doc = DocumentReference(
            doc_id="paper_005",
            title="Current Paper",
            authors=["Author D"],
            year=current_year
        )

        assert doc.year == current_year


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


class TestKronosNode:
    """Test KronosNode creation, validation, and behavior."""

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
            delta_embedding=np.random.randn(64),
            delta_structural=None,
            supported_by=[]
        )

        assert node.id == "root"
        assert node.name == "Root Concept"
        assert node.actualization_depth == 0
        assert len(node.parent_potentials) == 0
        assert len(node.confluence_pattern) == 0
        assert len(node.derivation_path) == 0

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
            delta_embedding=np.random.randn(64),
            delta_structural=None,
            supported_by=[]
        )

        assert node.id == "child"
        assert node.actualization_depth == 1
        assert len(node.parent_potentials) == 1
        assert node.parent_potentials[0] == "root"
        assert node.confluence_pattern["root"] == 1.0
        assert len(node.derivation_path) == 1

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
            delta_embedding=np.random.randn(64),
            delta_structural=None,
            supported_by=[]
        )

        assert node.id == "confluence"
        assert node.actualization_depth == 2
        assert len(node.parent_potentials) == 3
        assert len(node.confluence_pattern) == 3

        # Check confluence weights sum to ~1.0
        total_weight = sum(node.confluence_pattern.values())
        assert abs(total_weight - 1.0) < 0.01

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

    def test_confluence_pattern_validation_parents_match(self):
        """Test that confluence pattern keys must match parent_potentials."""
        with pytest.raises(ValueError, match="Confluence pattern keys must match parent_potentials"):
            KronosNode(
                id="mismatched_confluence",
                name="Mismatched Confluence",
                definition="Mismatched keys",
                confluence_pattern={
                    "parent_a": 0.6,
                    "parent_b": 0.4
                },
                parent_potentials=["parent_a", "parent_c"],  # parent_c != parent_b
                child_actualizations=[],
                derivation_path=["root"],
                actualization_depth=1,
                delta_embedding=np.random.randn(64)
            )

    def test_derivation_path_depth_validation(self):
        """Test that derivation_path length must equal actualization_depth."""
        with pytest.raises(ValueError, match="actualization_depth must equal len"):
            KronosNode(
                id="bad_depth",
                name="Bad Depth",
                definition="Depth mismatch",
                confluence_pattern={"parent": 1.0},
                parent_potentials=["parent"],
                child_actualizations=[],
                derivation_path=["root", "parent"],  # Length 2
                actualization_depth=1,  # But depth says 1
                delta_embedding=np.random.randn(64)
            )

    def test_add_child(self):
        """Test adding a child to a node."""
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
        """Test adding a sibling to a node."""
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
        assert len(node.crystallization_history) == 1
        assert node.crystallization_history[0].context == "First crystallization"

    def test_node_serialization(self):
        """Test node to_dict and from_dict."""
        original = KronosNode(
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
            documentation_depth=3,
            supported_by=[
                DocumentReference(
                    doc_id="doc1",
                    title="Doc 1",
                    authors=["Author"],
                    year=2020
                )
            ]
        )

        # Serialize
        data = original.to_dict()

        # Check serialized data
        assert data["id"] == "test_node"
        assert data["name"] == "Test Node"
        assert len(data["parent_potentials"]) == 2
        assert len(data["child_actualizations"]) == 2
        assert data["actualization_depth"] == 2
        assert "delta_embedding" in data
        assert "delta_structural" in data

        # Deserialize
        restored = KronosNode.from_dict(data)

        # Verify restoration
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.actualization_depth == original.actualization_depth
        assert np.array_equal(restored.delta_embedding, original.delta_embedding)
        assert np.array_equal(restored.delta_structural, original.delta_structural)
        assert len(restored.parent_potentials) == len(original.parent_potentials)
        assert len(restored.child_actualizations) == len(original.child_actualizations)


class TestKronosNodeEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_node_id_fails(self):
        """Test that empty node ID is rejected."""
        with pytest.raises(ValueError, match="Node ID cannot be empty"):
            KronosNode(
                id="",
                name="Empty ID",
                definition="Test",
                confluence_pattern={},
                parent_potentials=[],
                child_actualizations=[],
                derivation_path=[],
                actualization_depth=0,
                delta_embedding=np.random.randn(64)
            )

    def test_empty_node_name_fails(self):
        """Test that empty node name is rejected."""
        with pytest.raises(ValueError, match="Node name cannot be empty"):
            KronosNode(
                id="test",
                name="",
                definition="Test",
                confluence_pattern={},
                parent_potentials=[],
                child_actualizations=[],
                derivation_path=[],
                actualization_depth=0,
                delta_embedding=np.random.randn(64)
            )

    def test_negative_depth_fails(self):
        """Test that negative depth is rejected."""
        with pytest.raises(ValueError, match="Depth cannot be negative"):
            KronosNode(
                id="test",
                name="Test",
                definition="Test",
                confluence_pattern={},
                parent_potentials=[],
                child_actualizations=[],
                derivation_path=[],
                actualization_depth=-1,
                delta_embedding=np.random.randn(64)
            )

    def test_confluence_with_negative_weight_fails(self):
        """Test that negative confluence weights are rejected."""
        with pytest.raises(ValueError, match="Confluence weights must be positive"):
            KronosNode(
                id="test",
                name="Test",
                definition="Test",
                confluence_pattern={"parent_a": -0.5, "parent_b": 1.5},
                parent_potentials=["parent_a", "parent_b"],
                child_actualizations=[],
                derivation_path=["parent_a"],
                actualization_depth=1,
                delta_embedding=np.random.randn(64)
            )

    def test_root_with_parents_fails(self):
        """Test that root node (depth=0) cannot have parents."""
        with pytest.raises(ValueError, match="Root node .* cannot have parents"):
            KronosNode(
                id="root",
                name="Root",
                definition="Test",
                confluence_pattern={"parent": 1.0},
                parent_potentials=["parent"],  # Root shouldn't have parents
                child_actualizations=[],
                derivation_path=[],
                actualization_depth=0,
                delta_embedding=np.random.randn(64)
            )

    def test_non_root_without_parents_fails(self):
        """Test that non-root node must have parents."""
        with pytest.raises(ValueError, match="Non-root node must have at least one parent"):
            KronosNode(
                id="child",
                name="Child",
                definition="Test",
                confluence_pattern={},
                parent_potentials=[],  # Non-root needs parents
                child_actualizations=[],
                derivation_path=[],
                actualization_depth=1,
                delta_embedding=np.random.randn(64)
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
