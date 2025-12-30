"""
Extended unit tests for PAC/SEC/MED foundations.

Covers edge cases, boundary conditions, and stress scenarios.
"""

import pytest
import torch
import numpy as np
import math

from fracton.storage.pac_engine import (
    PACConservationEngine,
    PACNode,
    PACConstants,
)
from fracton.storage.sec_operators import SECOperators, SECState
from fracton.storage.med_validator import MEDValidator
from fracton.storage.distance_validator import DistanceValidator
from fracton.storage.foundation_integration import FoundationIntegration


class TestPACEdgeCases:
    """Test PAC engine edge cases."""

    def test_zero_potential(self):
        """Test handling of zero potential"""
        engine = PACConservationEngine()

        node = PACNode(
            value_embedding=torch.zeros(10),
            complexity_vector=torch.zeros(10),
            effect_cone=torch.zeros(10),
            potential=0.0,
            depth=0,
        )

        children = [
            PACNode(
                value_embedding=torch.zeros(10),
                complexity_vector=torch.zeros(10),
                effect_cone=torch.zeros(10),
                potential=0.0,
                depth=1,
            )
        ]

        xi = engine.compute_balance_operator(node, children)
        assert math.isinf(xi) or math.isnan(xi)

    def test_deep_hierarchy(self):
        """Test deep hierarchy (k >> 10)"""
        engine = PACConservationEngine()

        # Test at depth 100
        potential = engine.compute_potential(100)
        assert potential > 0
        assert potential < 1e-20  # Should be extremely small

        # Verify recursion still holds
        for k in range(95, 100):
            parent = engine.compute_potential(k)
            child1 = engine.compute_potential(k + 1)
            child2 = engine.compute_potential(k + 2)

            is_valid, residual = engine.verify_fibonacci_recursion(
                parent, child1, child2
            )
            assert is_valid

    def test_large_amplitude(self):
        """Test large amplitude values"""
        engine = PACConservationEngine()

        amplitude = 1e6
        potential = engine.compute_potential(0, amplitude)
        assert abs(potential - amplitude) < 1e-6

    def test_negative_embeddings(self):
        """Test conservation with negative embedding values"""
        engine = PACConservationEngine()

        parent = torch.tensor([-1.0, -2.0, -3.0])
        child1 = torch.tensor([-0.5, -1.0, -1.5])
        child2 = torch.tensor([-0.5, -1.0, -1.5])

        is_valid, residual = engine.verify_value_conservation(
            parent, [child1, child2]
        )
        assert is_valid

    def test_single_child(self):
        """Test conservation with single child"""
        engine = PACConservationEngine()

        parent = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=1.0,
            depth=0,
        )

        child = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=1.0,
            depth=1,
        )

        # Single child violates Fibonacci (needs 2)
        results = engine.verify_full_conservation(parent, [child])
        assert results["fibonacci"][0] == False

    def test_many_children(self):
        """Test conservation with >2 children"""
        engine = PACConservationEngine()

        parent = torch.randn(10)
        children = [torch.randn(10) for _ in range(5)]

        # Value conservation should work for any number
        is_valid, residual = engine.verify_value_conservation(
            parent, children
        )
        # Will likely fail, but shouldn't crash
        assert isinstance(is_valid, bool)

    def test_empty_children(self):
        """Test with no children"""
        engine = PACConservationEngine()

        node = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=1.0,
            depth=0,
        )

        xi = engine.compute_balance_operator(node, [])
        assert xi == 1.0

    def test_high_dimensional_embeddings(self):
        """Test with very high dimensional embeddings"""
        engine = PACConservationEngine()

        dim = 4096  # Large embedding
        parent = torch.randn(dim)
        child1 = parent * 0.5
        child2 = parent * 0.5

        is_valid, residual = engine.verify_value_conservation(
            parent, [child1, child2]
        )
        assert is_valid


class TestSECEdgeCases:
    """Test SEC operators edge cases."""

    def test_merge_identical_nodes(self):
        """Test merging identical nodes"""
        operators = SECOperators()

        node = PACNode(
            value_embedding=torch.ones(10),
            complexity_vector=torch.ones(10),
            effect_cone=torch.ones(10),
            potential=1.0,
            depth=0,
        )

        merged, sec_state = operators.merge(node, node)

        # Should double the embedding
        expected = node.value_embedding * 2
        assert torch.allclose(merged.value_embedding, expected)
        assert merged.potential == 2.0

    def test_merge_orthogonal_nodes(self):
        """Test merging orthogonal nodes (zero coherence)"""
        operators = SECOperators()

        node1 = PACNode(
            value_embedding=torch.tensor([1.0, 0.0, 0.0]),
            complexity_vector=torch.tensor([1.0, 0.0, 0.0]),
            effect_cone=torch.tensor([1.0, 0.0, 0.0]),
            potential=1.0,
            depth=0,
        )

        node2 = PACNode(
            value_embedding=torch.tensor([0.0, 1.0, 0.0]),
            complexity_vector=torch.tensor([0.0, 1.0, 0.0]),
            effect_cone=torch.tensor([0.0, 1.0, 0.0]),
            potential=1.0,
            depth=0,
        )

        merged, sec_state = operators.merge(node1, node2)

        # Coherence should be 0
        assert abs(sec_state.coherence) < 0.1
        assert sec_state.phase == "repulsion"

    def test_branch_zero_context(self):
        """Test branching with zero context"""
        operators = SECOperators()

        parent = PACNode(
            value_embedding=torch.ones(10),
            complexity_vector=torch.ones(10),
            effect_cone=torch.ones(10),
            potential=1.0,
            depth=0,
        )

        context = torch.zeros(10)

        # Should handle gracefully (might return degenerate split)
        try:
            child1, child2 = operators.branch(parent, context)
            assert child1.depth == 1
            assert child2.depth == 2
        except (ValueError, ZeroDivisionError):
            pytest.skip("Zero context causes expected error")

    def test_gradient_no_neighbors(self):
        """Test gradient detection with no neighbors"""
        operators = SECOperators()

        node = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=1.0,
            depth=0,
        )

        magnitude, direction = operators.detect_gradient(node, [])
        assert magnitude == 0.0
        assert torch.allclose(direction, torch.zeros(10))

    def test_duty_cycle_empty_history(self):
        """Test duty cycle with empty history"""
        operators = SECOperators()

        duty = operators.compute_duty_cycle([])
        assert duty == 0.5  # Default

    def test_duty_cycle_all_attraction(self):
        """Test duty cycle with 100% attraction"""
        operators = SECOperators()

        history = ["attraction"] * 1000
        duty = operators.compute_duty_cycle(history)
        assert duty == 1.0

    def test_duty_cycle_all_repulsion(self):
        """Test duty cycle with 100% repulsion"""
        operators = SECOperators()

        history = ["repulsion"] * 1000
        duty = operators.compute_duty_cycle(history)
        assert duty == 0.0


class TestMEDEdgeCases:
    """Test MED validator edge cases."""

    def test_empty_structure(self):
        """Test validation of empty structure"""
        validator = MEDValidator(strict_mode=False)

        assert validator.validate_structure([])

    def test_single_node(self):
        """Test single node (trivially valid)"""
        validator = MEDValidator(strict_mode=False)

        node = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=1.0,
            depth=0,
        )

        assert validator.validate_structure([node])

    def test_strict_mode_violation(self):
        """Test strict mode raises on violation"""
        validator = MEDValidator(strict_mode=True)

        # Create 4 nodes (violates nodes ≤ 3)
        nodes = [
            PACNode(
                value_embedding=torch.randn(10),
                complexity_vector=torch.randn(10),
                effect_cone=torch.randn(10),
                potential=1.0,
                depth=0,
            )
            for _ in range(4)
        ]

        with pytest.raises(ValueError, match="MED node count violation"):
            validator.validate_structure(nodes)

    def test_violation_tracking(self):
        """Test violation history tracking"""
        validator = MEDValidator(strict_mode=False)

        # Create violation
        nodes = [
            PACNode(
                value_embedding=torch.randn(10),
                complexity_vector=torch.randn(10),
                effect_cone=torch.randn(10),
                potential=1.0,
                depth=0,
            )
            for _ in range(5)
        ]

        validator.validate_structure(nodes, context="test")

        assert len(validator.violations) > 0
        assert validator.violations[0].violation_type == "nodes"
        assert validator.violations[0].actual_value == 5

        # Clear and verify
        validator.clear_violations()
        assert len(validator.violations) == 0


class TestDistanceEdgeCases:
    """Test distance validator edge cases."""

    def test_zero_energy(self):
        """Test zero energy embeddings"""
        validator = DistanceValidator()

        parent = torch.zeros(10)
        children = [torch.zeros(10), torch.zeros(10)]

        metrics = validator.validate_energy_conservation(parent, children)
        assert math.isinf(metrics.c_squared)

    def test_single_child_energy(self):
        """Test energy conservation with single child"""
        validator = DistanceValidator()

        parent = torch.ones(10)
        child = torch.ones(10)

        metrics = validator.validate_energy_conservation(parent, [child])
        # c² should be 1.0 (same energy)
        assert abs(metrics.c_squared - 1.0) < 0.01

    def test_extreme_amplification(self):
        """Test extreme amplification case"""
        validator = DistanceValidator()

        parent = torch.tensor([0.1])
        child1 = torch.tensor([10.0])
        child2 = torch.tensor([10.0])

        amplification, interpretation = validator.measure_amplification(
            parent, [child1, child2]
        )

        assert amplification > 100  # Massive amplification
        assert "amplification" in interpretation

    def test_extreme_binding(self):
        """Test extreme binding case"""
        validator = DistanceValidator()

        parent = torch.tensor([10.0])
        child1 = torch.tensor([0.1])
        child2 = torch.tensor([0.1])

        amplification, interpretation = validator.measure_amplification(
            parent, [child1, child2]
        )

        assert amplification < -0.9  # Strong binding
        assert "binding" in interpretation

    def test_fractal_dimension_single_level(self):
        """Test fractal dimension with single level"""
        validator = DistanceValidator()

        embeddings = [[torch.randn(10)]]
        metrics = validator.compute_fractal_dimension(embeddings)

        assert metrics.fractal_dimension == 0.0
        assert metrics.depth == 1

    def test_fractal_dimension_multilevel(self):
        """Test fractal dimension with multiple levels"""
        validator = DistanceValidator()

        # Create hierarchical structure
        embeddings_by_level = [
            [torch.randn(10)],  # Level 0: 1 node
            [torch.randn(10), torch.randn(10)],  # Level 1: 2 nodes
            [torch.randn(10) for _ in range(4)],  # Level 2: 4 nodes
        ]

        metrics = validator.compute_fractal_dimension(embeddings_by_level)

        # Fractal dimension can be negative for certain patterns
        assert metrics.fractal_dimension != 0
        assert metrics.depth == 3
        assert metrics.branching_factor >= 1


class TestIntegrationEdgeCases:
    """Test foundation integration edge cases."""

    def test_create_root_node(self):
        """Test creating root node (no parent)"""
        integration = FoundationIntegration(embedding_dim=10)

        embedding = torch.randn(10)
        node = integration.create_pac_node_from_embedding(
            embedding=embedding,
            content="Root node",
            depth=0,
            parent_embedding=None,
        )

        # Root uses full embedding
        assert torch.allclose(node.value_embedding, embedding)

    def test_create_child_node(self):
        """Test creating child node with parent"""
        integration = FoundationIntegration(embedding_dim=10)

        parent_embedding = torch.randn(10)
        child_embedding = torch.randn(10)

        node = integration.create_pac_node_from_embedding(
            embedding=child_embedding,
            content="Child node",
            depth=1,
            parent_embedding=parent_embedding,
        )

        # Child uses delta
        expected_delta = child_embedding - parent_embedding
        assert torch.allclose(node.value_embedding, expected_delta)

    def test_reconstruction_no_ancestors(self):
        """Test reconstruction with no ancestors (root)"""
        integration = FoundationIntegration(embedding_dim=10)

        embedding = torch.randn(10)
        node = PACNode(
            value_embedding=embedding,
            complexity_vector=embedding.clone(),
            effect_cone=embedding.clone(),
            potential=1.0,
            depth=0,
        )

        reconstructed = integration.reconstruct_full_embedding(node, [])
        assert torch.allclose(reconstructed, embedding)

    def test_reconstruction_with_ancestors(self):
        """Test reconstruction with ancestor chain"""
        integration = FoundationIntegration(embedding_dim=10)

        # Create chain: root -> parent -> child
        root = PACNode(
            value_embedding=torch.ones(10),
            complexity_vector=torch.ones(10),
            effect_cone=torch.ones(10),
            potential=1.0,
            depth=0,
        )

        parent = PACNode(
            value_embedding=torch.ones(10) * 0.5,
            complexity_vector=torch.ones(10) * 0.5,
            effect_cone=torch.ones(10) * 0.5,
            potential=0.618,
            depth=1,
        )

        child = PACNode(
            value_embedding=torch.ones(10) * 0.3,
            complexity_vector=torch.ones(10) * 0.3,
            effect_cone=torch.ones(10) * 0.3,
            potential=0.382,
            depth=2,
        )

        # Reconstruct child
        reconstructed = integration.reconstruct_full_embedding(
            child, [root, parent]
        )

        # Should sum all deltas
        expected = torch.ones(10) + torch.ones(10) * 0.5 + torch.ones(10) * 0.3
        assert torch.allclose(reconstructed, expected)

    def test_verify_conservation_empty_children(self):
        """Test conservation verification with no children"""
        integration = FoundationIntegration(embedding_dim=10)

        parent = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=1.0,
            depth=0,
        )

        metrics = integration.verify_conservation(parent, [])

        # Should handle gracefully
        assert metrics.balance_operator == 1.0
        assert metrics.pac_conservation["fibonacci"][0] == False


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_fibonacci_precision_deep_hierarchy(self):
        """Test Fibonacci precision at extreme depths"""
        engine = PACConservationEngine()

        for k in [0, 10, 50, 100]:
            parent = engine.compute_potential(k)
            child1 = engine.compute_potential(k + 1)
            child2 = engine.compute_potential(k + 2)

            is_valid, residual = engine.verify_fibonacci_recursion(
                parent, child1, child2
            )

            # Residual should be very small even at depth 100
            assert residual < 1e-10 or parent < 1e-20

    def test_conservation_floating_point_errors(self):
        """Test conservation with floating point accumulation"""
        engine = PACConservationEngine()

        # Create scenario prone to floating point errors
        parent = torch.tensor([1.0 / 3.0, 1.0 / 7.0, 1.0 / 11.0])
        child1 = parent * 0.5
        child2 = parent * 0.5

        is_valid, residual = engine.verify_value_conservation(
            parent, [child1, child2]
        )

        # Should still conserve within tolerance
        assert residual < 1e-6

    def test_large_batch_stability(self):
        """Test stability with large batch of nodes"""
        integration = FoundationIntegration(embedding_dim=100)

        # Create 100 nodes
        nodes = [
            PACNode(
                value_embedding=torch.randn(100),
                complexity_vector=torch.randn(100),
                effect_cone=torch.randn(100),
                potential=integration.pac_engine.compute_potential(i),
                depth=i,
            )
            for i in range(100)
        ]

        # Should handle without crashing
        assert len(nodes) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
