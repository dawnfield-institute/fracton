"""
Unit tests for GeometricConfidence (KRONOS v2).

Tests:
- Confidence metric calculation
- Anomaly detection
- Confidence interpretation
- Action recommendations
- Serialization/deserialization
"""

import pytest
from fracton.storage import GeometricConfidence


class TestGeometricConfidenceCreation:
    """Test creating GeometricConfidence instances."""

    def test_create_default_confidence(self):
        """Test creating confidence with default values."""
        conf = GeometricConfidence()

        assert conf.local_density == 0.0
        assert conf.branch_symmetry == 0.0
        assert conf.traversal_distance == 0.0
        assert conf.documentation_depth == 0
        assert conf.orphan_score == 0.0
        assert conf.confluence_bottleneck == False
        assert len(conf.missing_expected_children) == 0
        assert conf.retrieval_confidence == 0.0
        assert conf.hallucination_risk == 1.0

    def test_create_custom_confidence(self):
        """Test creating confidence with custom values."""
        conf = GeometricConfidence(
            local_density=0.8,
            branch_symmetry=0.7,
            traversal_distance=2.0,
            documentation_depth=5,
            orphan_score=0.1,
            confluence_bottleneck=False,
            missing_expected_children=[]
        )

        assert conf.local_density == 0.8
        assert conf.branch_symmetry == 0.7
        assert conf.traversal_distance == 2.0
        assert conf.documentation_depth == 5


class TestGeometricConfidenceComputation:
    """Test confidence score computation."""

    def test_high_confidence_scenario(self):
        """Test scenario that should yield high confidence."""
        conf = GeometricConfidence(
            local_density=0.9,          # Dense neighborhood
            branch_symmetry=0.8,        # Balanced tree
            traversal_distance=1.0,     # Close to well-documented nodes
            documentation_depth=10,     # Well-documented
            orphan_score=0.0,           # Not orphaned
            confluence_bottleneck=False, # No bottleneck
            missing_expected_children=[]
        )

        score = conf.compute()

        assert score > 0.7  # High confidence threshold
        assert conf.retrieval_confidence > 0.7
        assert conf.hallucination_risk < 0.3
        assert conf.is_trustworthy

    def test_low_confidence_scenario(self):
        """Test scenario that should yield low confidence."""
        conf = GeometricConfidence(
            local_density=0.2,          # Sparse neighborhood
            branch_symmetry=0.1,        # Asymmetric tree
            traversal_distance=10.0,    # Far from documented nodes
            documentation_depth=0,      # No documentation
            orphan_score=0.8,           # Highly orphaned
            confluence_bottleneck=True,  # Has bottleneck
            missing_expected_children=["child1", "child2", "child3"]
        )

        score = conf.compute()

        assert score < 0.3  # Low confidence threshold
        assert conf.retrieval_confidence < 0.3
        assert conf.hallucination_risk > 0.7
        assert conf.is_suspicious

    def test_moderate_confidence_scenario(self):
        """Test scenario with moderate confidence."""
        conf = GeometricConfidence(
            local_density=0.6,
            branch_symmetry=0.5,
            traversal_distance=3.0,
            documentation_depth=3,
            orphan_score=0.2,
            confluence_bottleneck=False,
            missing_expected_children=["child1"]
        )

        score = conf.compute()

        assert 0.3 <= score <= 0.7  # Moderate range
        assert not conf.is_trustworthy
        assert not conf.is_suspicious

    def test_confidence_metric_weights(self):
        """Test that confidence metrics have correct weights."""
        # Test local_density weight (30%)
        conf1 = GeometricConfidence(
            local_density=1.0,
            branch_symmetry=0.0,
            traversal_distance=0.0,
            documentation_depth=0
        )
        score1 = conf1.compute()

        # Should contribute ~0.3 to score
        assert 0.25 <= score1 <= 0.35

        # Test branch_symmetry weight (20%)
        conf2 = GeometricConfidence(
            local_density=0.0,
            branch_symmetry=1.0,
            traversal_distance=0.0,
            documentation_depth=0
        )
        score2 = conf2.compute()

        # Should contribute ~0.2 to score
        assert 0.15 <= score2 <= 0.25

    def test_orphan_penalty(self):
        """Test that orphan score heavily penalizes confidence."""
        # High metrics but orphaned
        conf = GeometricConfidence(
            local_density=1.0,
            branch_symmetry=1.0,
            traversal_distance=1.0,
            documentation_depth=10,
            orphan_score=0.9  # Highly orphaned
        )

        score = conf.compute()

        # Orphan penalty should significantly reduce score
        assert score < 0.7  # Should not be high confidence

    def test_bottleneck_penalty(self):
        """Test that confluence bottleneck penalizes confidence."""
        conf = GeometricConfidence(
            local_density=0.8,
            branch_symmetry=0.8,
            traversal_distance=2.0,
            documentation_depth=5,
            confluence_bottleneck=True  # Has bottleneck
        )

        score = conf.compute()

        # Bottleneck should apply -0.1 penalty
        conf_no_bottleneck = GeometricConfidence(
            local_density=0.8,
            branch_symmetry=0.8,
            traversal_distance=2.0,
            documentation_depth=5,
            confluence_bottleneck=False
        )

        score_no_bottleneck = conf_no_bottleneck.compute()

        assert score < score_no_bottleneck
        assert abs(score - score_no_bottleneck) >= 0.09  # ~0.1 penalty

    def test_missing_children_penalty(self):
        """Test that missing children penalize confidence."""
        conf_no_gaps = GeometricConfidence(
            local_density=0.7,
            branch_symmetry=0.7,
            traversal_distance=2.0,
            documentation_depth=5,
            missing_expected_children=[]
        )

        conf_with_gaps = GeometricConfidence(
            local_density=0.7,
            branch_symmetry=0.7,
            traversal_distance=2.0,
            documentation_depth=5,
            missing_expected_children=["child1", "child2"]
        )

        score_no_gaps = conf_no_gaps.compute()
        score_with_gaps = conf_with_gaps.compute()

        # Each missing child applies -0.05 penalty
        assert score_with_gaps < score_no_gaps
        assert abs(score_with_gaps - score_no_gaps) >= 0.09  # ~0.1 penalty for 2 gaps

    def test_confidence_clamping(self):
        """Test that confidence is clamped to [0, 1]."""
        # Extreme positive values
        conf_high = GeometricConfidence(
            local_density=1.0,
            branch_symmetry=1.0,
            traversal_distance=0.0,
            documentation_depth=100
        )

        score_high = conf_high.compute()
        assert 0.0 <= score_high <= 1.0

        # Extreme negative values
        conf_low = GeometricConfidence(
            local_density=0.0,
            branch_symmetry=0.0,
            traversal_distance=100.0,
            documentation_depth=0,
            orphan_score=1.0,
            confluence_bottleneck=True,
            missing_expected_children=["c1", "c2", "c3", "c4", "c5", "c6"]
        )

        score_low = conf_low.compute()
        assert 0.0 <= score_low <= 1.0


class TestGeometricConfidenceInterpretation:
    """Test confidence interpretation and recommendations."""

    def test_interpretation_high_confidence(self):
        """Test interpretation for high confidence."""
        conf = GeometricConfidence(retrieval_confidence=0.85)
        assert conf.interpretation == "High confidence - well-trodden territory"

    def test_interpretation_moderate_confidence(self):
        """Test interpretation for moderate confidence."""
        conf = GeometricConfidence(retrieval_confidence=0.6)
        assert conf.interpretation == "Moderate confidence - reasonable extrapolation"

    def test_interpretation_low_confidence(self):
        """Test interpretation for low confidence."""
        conf = GeometricConfidence(retrieval_confidence=0.4)
        assert conf.interpretation == "Low confidence - weak support"

    def test_interpretation_very_low_confidence(self):
        """Test interpretation for very low confidence."""
        conf = GeometricConfidence(retrieval_confidence=0.15)
        assert conf.interpretation == "Very low confidence - potential hallucination"

    def test_interpretation_no_confidence(self):
        """Test interpretation for no confidence."""
        conf = GeometricConfidence(retrieval_confidence=0.05)
        assert conf.interpretation == "No confidence - likely fabrication"

    def test_action_recommendation_trust(self):
        """Test action recommendation for trustworthy knowledge."""
        conf = GeometricConfidence(retrieval_confidence=0.85)
        assert conf.action_recommendation == "Trust retrieval"

    def test_action_recommendation_use_with_context(self):
        """Test action recommendation for moderate confidence."""
        conf = GeometricConfidence(retrieval_confidence=0.6)
        assert conf.action_recommendation == "Use with context"

    def test_action_recommendation_flag(self):
        """Test action recommendation for low confidence."""
        conf = GeometricConfidence(retrieval_confidence=0.4)
        assert conf.action_recommendation == "Flag uncertainty"

    def test_action_recommendation_investigate(self):
        """Test action recommendation for very low confidence."""
        conf = GeometricConfidence(retrieval_confidence=0.15)
        assert conf.action_recommendation == "Investigate/verify"

    def test_action_recommendation_reject(self):
        """Test action recommendation for no confidence."""
        conf = GeometricConfidence(retrieval_confidence=0.05)
        assert conf.action_recommendation == "Reject"


class TestGeometricConfidenceAnomalies:
    """Test anomaly detection."""

    def test_has_anomalies_orphan(self):
        """Test anomaly detection for orphaned nodes."""
        conf = GeometricConfidence(orphan_score=0.6)
        assert conf.has_anomalies

    def test_has_anomalies_bottleneck(self):
        """Test anomaly detection for confluence bottleneck."""
        conf = GeometricConfidence(confluence_bottleneck=True)
        assert conf.has_anomalies

    def test_has_anomalies_missing_children(self):
        """Test anomaly detection for missing children."""
        conf = GeometricConfidence(missing_expected_children=["child1"])
        assert conf.has_anomalies

    def test_no_anomalies(self):
        """Test no anomalies detected."""
        conf = GeometricConfidence(
            orphan_score=0.3,
            confluence_bottleneck=False,
            missing_expected_children=[]
        )
        assert not conf.has_anomalies

    def test_anomaly_report_orphan(self):
        """Test anomaly report for high orphan score."""
        conf = GeometricConfidence(orphan_score=0.8)
        report = conf.get_anomaly_report()

        assert len(report) > 0
        assert any("orphan" in r.lower() for r in report)

    def test_anomaly_report_bottleneck(self):
        """Test anomaly report for confluence bottleneck."""
        conf = GeometricConfidence(confluence_bottleneck=True)
        report = conf.get_anomaly_report()

        assert len(report) > 0
        assert any("bottleneck" in r.lower() for r in report)

    def test_anomaly_report_missing_children(self):
        """Test anomaly report for missing children."""
        conf = GeometricConfidence(
            missing_expected_children=["child1", "child2", "child3"]
        )
        report = conf.get_anomaly_report()

        assert len(report) > 0
        assert any("missing" in r.lower() for r in report)

    def test_anomaly_report_sparse_region(self):
        """Test anomaly report for sparse regions."""
        conf = GeometricConfidence(local_density=0.1)
        report = conf.get_anomaly_report()

        assert any("sparse" in r.lower() for r in report)

    def test_anomaly_report_asymmetric_tree(self):
        """Test anomaly report for asymmetric tree."""
        conf = GeometricConfidence(branch_symmetry=0.2)
        report = conf.get_anomaly_report()

        assert any("asymmetric" in r.lower() for r in report)

    def test_anomaly_report_no_documentation(self):
        """Test anomaly report for no documentation."""
        conf = GeometricConfidence(documentation_depth=0)
        report = conf.get_anomaly_report()

        assert any("documentation" in r.lower() for r in report)

    def test_anomaly_report_far_traversal(self):
        """Test anomaly report for far traversal distance."""
        conf = GeometricConfidence(traversal_distance=8.0)
        report = conf.get_anomaly_report()

        assert any("distance" in r.lower() or "far" in r.lower() for r in report)

    def test_anomaly_report_multiple(self):
        """Test anomaly report with multiple anomalies."""
        conf = GeometricConfidence(
            orphan_score=0.8,
            confluence_bottleneck=True,
            missing_expected_children=["child1", "child2"],
            local_density=0.1,
            documentation_depth=0
        )

        report = conf.get_anomaly_report()

        # Should have multiple anomalies reported
        assert len(report) >= 3


class TestGeometricConfidenceSerialization:
    """Test confidence serialization."""

    def test_confidence_to_dict(self):
        """Test serializing confidence to dictionary."""
        conf = GeometricConfidence(
            local_density=0.7,
            branch_symmetry=0.6,
            traversal_distance=3.0,
            documentation_depth=4,
            orphan_score=0.2,
            confluence_bottleneck=False,
            missing_expected_children=["child1"]
        )

        conf.compute()  # Compute scores
        data = conf.to_dict()

        assert data["local_density"] == 0.7
        assert data["branch_symmetry"] == 0.6
        assert data["traversal_distance"] == 3.0
        assert data["documentation_depth"] == 4
        assert data["orphan_score"] == 0.2
        assert data["confluence_bottleneck"] == False
        assert len(data["missing_expected_children"]) == 1
        assert "retrieval_confidence" in data
        assert "hallucination_risk" in data
        assert "interpretation" in data
        assert "action_recommendation" in data

    def test_confidence_from_dict(self):
        """Test deserializing confidence from dictionary."""
        data = {
            "local_density": 0.7,
            "branch_symmetry": 0.6,
            "traversal_distance": 3.0,
            "documentation_depth": 4,
            "orphan_score": 0.2,
            "confluence_bottleneck": False,
            "missing_expected_children": ["child1"],
            "retrieval_confidence": 0.5,
            "hallucination_risk": 0.5
        }

        conf = GeometricConfidence.from_dict(data)

        assert conf.local_density == 0.7
        assert conf.branch_symmetry == 0.6
        assert conf.traversal_distance == 3.0
        assert conf.documentation_depth == 4
        assert conf.orphan_score == 0.2
        assert conf.confluence_bottleneck == False
        assert len(conf.missing_expected_children) == 1

    def test_confidence_roundtrip(self):
        """Test serialization round-trip."""
        original = GeometricConfidence(
            local_density=0.8,
            branch_symmetry=0.7,
            traversal_distance=2.5,
            documentation_depth=6,
            orphan_score=0.1,
            confluence_bottleneck=True,
            missing_expected_children=["child1", "child2"]
        )

        original.compute()

        # Serialize and deserialize
        data = original.to_dict()
        restored = GeometricConfidence.from_dict(data)

        # Verify all fields match
        assert restored.local_density == original.local_density
        assert restored.branch_symmetry == original.branch_symmetry
        assert restored.traversal_distance == original.traversal_distance
        assert restored.documentation_depth == original.documentation_depth
        assert restored.orphan_score == original.orphan_score
        assert restored.confluence_bottleneck == original.confluence_bottleneck
        assert restored.missing_expected_children == original.missing_expected_children


class TestGeometricConfidenceRepr:
    """Test string representation."""

    def test_repr_format(self):
        """Test __repr__ format."""
        conf = GeometricConfidence(
            local_density=0.7,
            branch_symmetry=0.6,
            traversal_distance=3.0,
            documentation_depth=4
        )

        conf.compute()
        repr_str = repr(conf)

        assert "GeometricConfidence" in repr_str
        assert "confidence=" in repr_str
        assert "risk=" in repr_str
        assert "interpretation=" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
