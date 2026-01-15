"""
Unit tests for KronosEdge and RelationType (KRONOS v2).

Tests:
- Edge creation and validation
- RelationType semantics
- Inverse edge generation
- Edge strengthening/weakening
- Evidence tracking
- Edge serialization/deserialization
"""

import pytest
from datetime import datetime, timedelta
from fracton.storage import KronosEdge
from fracton.storage.edge import RelationType


class TestRelationType:
    """Test RelationType enum and its properties."""

    def test_relation_type_values(self):
        """Test that all relation types exist."""
        assert RelationType.PRECEDES.value == "precedes"
        assert RelationType.ADVANCES_FROM.value == "advances_from"
        assert RelationType.GENERALIZES.value == "generalizes"
        assert RelationType.SPECIALIZES.value == "specializes"
        assert RelationType.ENABLES.value == "enables"
        assert RelationType.IS_ENABLED_BY.value == "is_enabled_by"
        assert RelationType.SUPPORTS.value == "supports"
        assert RelationType.CONTRADICTS.value == "contradicts"
        assert RelationType.EXTENDS.value == "extends"
        assert RelationType.CONFLUENCE.value == "confluence"
        assert RelationType.PARENT_OF.value == "parent_of"
        assert RelationType.CHILD_OF.value == "child_of"
        assert RelationType.SIBLING_OF.value == "sibling_of"

    def test_inverse_relationships(self):
        """Test that inverse relationships are correct."""
        assert RelationType.PRECEDES.inverse() == RelationType.ADVANCES_FROM
        assert RelationType.ADVANCES_FROM.inverse() == RelationType.PRECEDES
        assert RelationType.GENERALIZES.inverse() == RelationType.SPECIALIZES
        assert RelationType.SPECIALIZES.inverse() == RelationType.GENERALIZES
        assert RelationType.ENABLES.inverse() == RelationType.IS_ENABLED_BY
        assert RelationType.IS_ENABLED_BY.inverse() == RelationType.ENABLES
        assert RelationType.PARENT_OF.inverse() == RelationType.CHILD_OF
        assert RelationType.CHILD_OF.inverse() == RelationType.PARENT_OF

    def test_self_inverse_relationships(self):
        """Test relationships that are their own inverse."""
        assert RelationType.SUPPORTS.inverse() == RelationType.SUPPORTS
        assert RelationType.CONTRADICTS.inverse() == RelationType.CONTRADICTS
        assert RelationType.EXTENDS.inverse() == RelationType.EXTENDS
        assert RelationType.CONFLUENCE.inverse() == RelationType.CONFLUENCE
        assert RelationType.SIBLING_OF.inverse() == RelationType.SIBLING_OF

    def test_hierarchical_relationships(self):
        """Test hierarchical relationship detection."""
        assert RelationType.GENERALIZES.is_hierarchical
        assert RelationType.SPECIALIZES.is_hierarchical
        assert RelationType.PARENT_OF.is_hierarchical
        assert RelationType.CHILD_OF.is_hierarchical
        assert not RelationType.PRECEDES.is_hierarchical
        assert not RelationType.SUPPORTS.is_hierarchical

    def test_temporal_relationships(self):
        """Test temporal relationship detection."""
        assert RelationType.PRECEDES.is_temporal
        assert RelationType.ADVANCES_FROM.is_temporal
        assert not RelationType.GENERALIZES.is_temporal
        assert not RelationType.ENABLES.is_temporal

    def test_functional_relationships(self):
        """Test functional relationship detection."""
        assert RelationType.ENABLES.is_functional
        assert RelationType.IS_ENABLED_BY.is_functional
        assert not RelationType.PRECEDES.is_functional
        assert not RelationType.GENERALIZES.is_functional

    def test_epistemic_relationships(self):
        """Test epistemic relationship detection."""
        assert RelationType.SUPPORTS.is_epistemic
        assert RelationType.CONTRADICTS.is_epistemic
        assert RelationType.EXTENDS.is_epistemic
        assert not RelationType.ENABLES.is_epistemic
        assert not RelationType.PRECEDES.is_epistemic


class TestKronosEdge:
    """Test KronosEdge creation and validation."""

    def test_create_simple_edge(self):
        """Test creating a simple edge."""
        edge = KronosEdge(
            source_id="node_a",
            target_id="node_b",
            relationship_type=RelationType.PARENT_OF,
            strength=1.0
        )

        assert edge.source_id == "node_a"
        assert edge.target_id == "node_b"
        assert edge.relationship_type == RelationType.PARENT_OF
        assert edge.strength == 1.0
        assert edge.evidence_count == 0

    def test_create_edge_with_evidence(self):
        """Test creating edge with supporting documents."""
        edge = KronosEdge(
            source_id="concept_a",
            target_id="concept_b",
            relationship_type=RelationType.SUPPORTS,
            strength=0.8,
            supporting_documents=["doc1", "doc2", "doc3"]
        )

        assert edge.evidence_count == 3
        assert len(edge.supporting_documents) == 3
        assert "doc1" in edge.supporting_documents

    def test_edge_validation_empty_source_fails(self):
        """Test that empty source ID is rejected."""
        with pytest.raises(ValueError, match="Source and target IDs are required"):
            KronosEdge(
                source_id="",
                target_id="node_b",
                relationship_type=RelationType.PARENT_OF
            )

    def test_edge_validation_empty_target_fails(self):
        """Test that empty target ID is rejected."""
        with pytest.raises(ValueError, match="Source and target IDs are required"):
            KronosEdge(
                source_id="node_a",
                target_id="",
                relationship_type=RelationType.PARENT_OF
            )

    def test_edge_validation_self_loop_fails(self):
        """Test that self-loops are rejected."""
        with pytest.raises(ValueError, match="Self-loops not allowed"):
            KronosEdge(
                source_id="node_a",
                target_id="node_a",
                relationship_type=RelationType.SIBLING_OF
            )

    def test_edge_validation_strength_range(self):
        """Test that strength must be in [0, 1]."""
        # Too low
        with pytest.raises(ValueError, match="Strength must be 0-1"):
            KronosEdge(
                source_id="node_a",
                target_id="node_b",
                relationship_type=RelationType.SUPPORTS,
                strength=-0.1
            )

        # Too high
        with pytest.raises(ValueError, match="Strength must be 0-1"):
            KronosEdge(
                source_id="node_a",
                target_id="node_b",
                relationship_type=RelationType.SUPPORTS,
                strength=1.5
            )

    def test_inverse_edge_creation(self):
        """Test creating inverse edge."""
        original = KronosEdge(
            source_id="parent",
            target_id="child",
            relationship_type=RelationType.PARENT_OF,
            strength=0.9,
            supporting_documents=["doc1"]
        )

        inverse = original.inverse_edge()

        assert inverse.source_id == "child"
        assert inverse.target_id == "parent"
        assert inverse.relationship_type == RelationType.CHILD_OF
        assert inverse.strength == 0.9
        assert len(inverse.supporting_documents) == 1

    def test_add_supporting_document(self):
        """Test adding supporting documents."""
        edge = KronosEdge(
            source_id="node_a",
            target_id="node_b",
            relationship_type=RelationType.SUPPORTS,
            strength=0.5
        )

        assert edge.evidence_count == 0

        edge.add_supporting_document("doc1")
        assert edge.evidence_count == 1
        assert "doc1" in edge.supporting_documents

        edge.add_supporting_document("doc2")
        assert edge.evidence_count == 2

        # Adding duplicate should not increase count
        edge.add_supporting_document("doc1")
        assert edge.evidence_count == 2

    def test_strengthen_edge(self):
        """Test strengthening an edge."""
        edge = KronosEdge(
            source_id="node_a",
            target_id="node_b",
            relationship_type=RelationType.SUPPORTS,
            strength=0.5
        )

        edge.strengthen(0.2)
        assert edge.strength == 0.7

        # Strengthening should cap at 1.0
        edge.strengthen(0.5)
        assert edge.strength == 1.0

    def test_weaken_edge(self):
        """Test weakening an edge."""
        edge = KronosEdge(
            source_id="node_a",
            target_id="node_b",
            relationship_type=RelationType.SUPPORTS,
            strength=0.7
        )

        edge.weaken(0.2)
        assert edge.strength == 0.5

        # Weakening should floor at 0.0
        edge.weaken(0.8)
        assert edge.strength == 0.0

    def test_validate_edge(self):
        """Test edge validation updates timestamp."""
        edge = KronosEdge(
            source_id="node_a",
            target_id="node_b",
            relationship_type=RelationType.SUPPORTS,
            strength=0.8
        )

        original_validation = edge.last_validated
        edge.validate()
        new_validation = edge.last_validated

        assert new_validation >= original_validation

    def test_edge_strength_properties(self):
        """Test is_strong and is_weak properties."""
        strong_edge = KronosEdge(
            source_id="a",
            target_id="b",
            relationship_type=RelationType.SUPPORTS,
            strength=0.8
        )

        weak_edge = KronosEdge(
            source_id="c",
            target_id="d",
            relationship_type=RelationType.SUPPORTS,
            strength=0.2
        )

        medium_edge = KronosEdge(
            source_id="e",
            target_id="f",
            relationship_type=RelationType.SUPPORTS,
            strength=0.5
        )

        assert strong_edge.is_strong
        assert not strong_edge.is_weak

        assert weak_edge.is_weak
        assert not weak_edge.is_strong

        assert not medium_edge.is_strong
        assert not medium_edge.is_weak

    def test_edge_support_property(self):
        """Test is_well_supported property."""
        edge = KronosEdge(
            source_id="a",
            target_id="b",
            relationship_type=RelationType.SUPPORTS,
            supporting_documents=["doc1", "doc2"]
        )

        assert not edge.is_well_supported

        edge.add_supporting_document("doc3")
        assert edge.is_well_supported

    def test_edge_age_calculation(self):
        """Test age_days calculation."""
        past_time = datetime.now() - timedelta(days=10)
        edge = KronosEdge(
            source_id="a",
            target_id="b",
            relationship_type=RelationType.SUPPORTS,
            established_at=past_time
        )

        assert edge.age_days >= 9.9  # Allow small timing variance

    def test_edge_revalidation_check(self):
        """Test needs_revalidation."""
        recent_time = datetime.now() - timedelta(days=30)
        old_time = datetime.now() - timedelta(days=100)

        recent_edge = KronosEdge(
            source_id="a",
            target_id="b",
            relationship_type=RelationType.SUPPORTS,
            last_validated=recent_time
        )

        old_edge = KronosEdge(
            source_id="c",
            target_id="d",
            relationship_type=RelationType.SUPPORTS,
            last_validated=old_time
        )

        assert not recent_edge.needs_revalidation(threshold_days=90)
        assert old_edge.needs_revalidation(threshold_days=90)

    def test_edge_serialization(self):
        """Test edge to_dict and from_dict."""
        original = KronosEdge(
            source_id="source",
            target_id="target",
            relationship_type=RelationType.GENERALIZES,
            strength=0.75,
            supporting_documents=["doc1", "doc2"],
            notes="Test edge"
        )

        # Serialize
        data = original.to_dict()

        # Check serialized data
        assert data["source_id"] == "source"
        assert data["target_id"] == "target"
        assert data["relationship_type"] == "generalizes"
        assert data["strength"] == 0.75
        assert len(data["supporting_documents"]) == 2

        # Deserialize
        restored = KronosEdge.from_dict(data)

        # Verify restoration
        assert restored.source_id == original.source_id
        assert restored.target_id == original.target_id
        assert restored.relationship_type == original.relationship_type
        assert restored.strength == original.strength
        assert len(restored.supporting_documents) == len(original.supporting_documents)
        assert restored.notes == original.notes

    def test_edge_equality(self):
        """Test edge equality comparison."""
        edge1 = KronosEdge(
            source_id="a",
            target_id="b",
            relationship_type=RelationType.PARENT_OF,
            strength=0.8
        )

        edge2 = KronosEdge(
            source_id="a",
            target_id="b",
            relationship_type=RelationType.PARENT_OF,
            strength=0.9  # Different strength, but same structure
        )

        edge3 = KronosEdge(
            source_id="a",
            target_id="c",  # Different target
            relationship_type=RelationType.PARENT_OF,
            strength=0.8
        )

        # Same structure (source, target, type) = equal
        assert edge1 == edge2

        # Different target = not equal
        assert edge1 != edge3

    def test_edge_hash(self):
        """Test edge hashing for use in sets/dicts."""
        edge1 = KronosEdge(
            source_id="a",
            target_id="b",
            relationship_type=RelationType.PARENT_OF
        )

        edge2 = KronosEdge(
            source_id="a",
            target_id="b",
            relationship_type=RelationType.PARENT_OF
        )

        edge3 = KronosEdge(
            source_id="a",
            target_id="c",
            relationship_type=RelationType.PARENT_OF
        )

        # Same structure should have same hash
        assert hash(edge1) == hash(edge2)

        # Different structure should (usually) have different hash
        assert hash(edge1) != hash(edge3)

        # Should be usable in sets
        edge_set = {edge1, edge2, edge3}
        assert len(edge_set) == 2  # edge1 and edge2 are the same


class TestKronosEdgeSemantics:
    """Test semantic meaning of different edge types."""

    def test_temporal_edge_semantics(self):
        """Test temporal relationship semantics."""
        precedes = KronosEdge(
            source_id="event_a",
            target_id="event_b",
            relationship_type=RelationType.PRECEDES
        )

        assert precedes.relationship_type.is_temporal
        assert precedes.relationship_type.inverse() == RelationType.ADVANCES_FROM

    def test_hierarchical_edge_semantics(self):
        """Test hierarchical relationship semantics."""
        generalizes = KronosEdge(
            source_id="general",
            target_id="specific",
            relationship_type=RelationType.GENERALIZES
        )

        assert generalizes.relationship_type.is_hierarchical
        assert generalizes.relationship_type.inverse() == RelationType.SPECIALIZES

    def test_functional_edge_semantics(self):
        """Test functional relationship semantics."""
        enables = KronosEdge(
            source_id="prerequisite",
            target_id="capability",
            relationship_type=RelationType.ENABLES
        )

        assert enables.relationship_type.is_functional
        assert enables.relationship_type.inverse() == RelationType.IS_ENABLED_BY

    def test_epistemic_edge_semantics(self):
        """Test epistemic relationship semantics."""
        supports = KronosEdge(
            source_id="evidence",
            target_id="claim",
            relationship_type=RelationType.SUPPORTS
        )

        assert supports.relationship_type.is_epistemic
        assert supports.relationship_type.inverse() == RelationType.SUPPORTS  # Self-inverse


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
