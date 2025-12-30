"""
Tests for PAC/SEC/MED theoretical foundations.

Validates:
1. PAC conservation (Fibonacci recursion, 3D conservation)
2. SEC operators (⊕, ⊗, δ) and duty cycle
3. MED universal bounds (depth≤1, nodes≤3)
4. Distance validation (E=mc²)
5. Integration layer
"""

import pytest
import torch
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


class TestPACConstants:
    """Test PAC constants match theoretical values."""

    def test_golden_ratio(self):
        """Test φ = (1 + √5) / 2"""
        constants = PACConstants()
        expected = (1 + math.sqrt(5)) / 2
        assert abs(constants.PHI - expected) < 1e-10

    def test_balance_operator(self):
        """Test Ξ = 1 + π/F₁₀ where F₁₀=55"""
        constants = PACConstants()
        expected = 1 + math.pi / 55
        assert abs(constants.XI - expected) < 1e-10

    def test_duty_cycle(self):
        """Test duty = φ/(φ+1) ≈ 0.618"""
        constants = PACConstants()
        phi = constants.PHI
        expected = phi / (phi + 1)
        assert abs(constants.DUTY_CYCLE - expected) < 1e-10
        assert abs(constants.DUTY_CYCLE - 0.618) < 0.001

    def test_balance_ratio(self):
        """Test 4:1 attraction/repulsion ratio"""
        constants = PACConstants()
        assert constants.BALANCE_RATIO == 4.0


class TestPACConservationEngine:
    """Test PAC conservation engine."""

    def test_potential_computation(self):
        """Test Ψ(k) = A · φ^(-k)"""
        engine = PACConservationEngine()
        phi = engine.constants.PHI

        # k=0: Ψ(0) = 1.0
        assert abs(engine.compute_potential(0, amplitude=1.0) - 1.0) < 1e-10

        # k=1: Ψ(1) = φ^(-1)
        expected_k1 = phi ** (-1)
        assert abs(engine.compute_potential(1) - expected_k1) < 1e-10

        # k=2: Ψ(2) = φ^(-2)
        expected_k2 = phi ** (-2)
        assert abs(engine.compute_potential(2) - expected_k2) < 1e-10

    def test_fibonacci_recursion(self):
        """Test Ψ(k) = Ψ(k+1) + Ψ(k+2)"""
        engine = PACConservationEngine()

        for k in range(10):
            parent = engine.compute_potential(k)
            child1 = engine.compute_potential(k + 1)
            child2 = engine.compute_potential(k + 2)

            is_valid, residual = engine.verify_fibonacci_recursion(
                parent, child1, child2
            )

            assert is_valid, f"Failed at k={k}, residual={residual}"
            assert residual < 1e-10

    def test_value_conservation(self):
        """Test f(parent) = Σ f(children)"""
        engine = PACConservationEngine()

        # Create parent and children
        parent = torch.tensor([1.0, 2.0, 3.0])
        child1 = torch.tensor([0.5, 1.0, 1.5])
        child2 = torch.tensor([0.5, 1.0, 1.5])

        is_valid, residual = engine.verify_value_conservation(
            parent, [child1, child2]
        )

        assert is_valid
        assert residual < 1e-6

    def test_complexity_conservation(self):
        """Test ||C(parent)||² = ||Σ C(children)||²"""
        engine = PACConservationEngine()

        # Create vectors where sum conservation holds
        parent = torch.tensor([3.0, 4.0])  # ||parent|| = 5
        child1 = torch.tensor([1.5, 2.0])
        child2 = torch.tensor([1.5, 2.0])
        # ||child1 + child2|| = ||(3,4)|| = 5 ✓

        is_valid, residual = engine.verify_complexity_conservation(
            parent, [child1, child2]
        )

        assert is_valid
        assert residual < 1e-6

    def test_balance_operator_computation(self):
        """Test balance operator Ξ computation"""
        engine = PACConservationEngine()

        # Create parent and children with known potentials
        parent = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=1.0,
            depth=0,
        )

        child1 = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=0.6,
            depth=1,
        )

        child2 = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=0.4,
            depth=2,
        )

        xi_local = engine.compute_balance_operator(parent, [child1, child2])

        # children_sum = 1.0, parent = 1.0, excess = 0, xi = 1.0
        assert abs(xi_local - 1.0) < 0.01

    def test_collapse_trigger(self):
        """Test collapse trigger detection"""
        engine = PACConservationEngine()

        # Stable: xi ≈ XI
        assert engine.check_collapse_trigger(1.057) == "STABLE"

        # Collapse: xi > XI
        assert engine.check_collapse_trigger(1.1) == "COLLAPSE"

        # Decay: xi < 0.9*XI
        assert engine.check_collapse_trigger(0.9) == "DECAY"


class TestSECOperators:
    """Test SEC operators."""

    def test_merge_operator(self):
        """Test ⊕ (merge) operator"""
        operators = SECOperators()

        node1 = PACNode(
            value_embedding=torch.tensor([1.0, 0.0, 0.0]),
            complexity_vector=torch.tensor([1.0, 0.0, 0.0]),
            effect_cone=torch.tensor([1.0, 0.0, 0.0]),
            potential=0.6,
            depth=1,
        )

        node2 = PACNode(
            value_embedding=torch.tensor([0.0, 1.0, 0.0]),
            complexity_vector=torch.tensor([0.0, 1.0, 0.0]),
            effect_cone=torch.tensor([0.0, 1.0, 0.0]),
            potential=0.4,
            depth=1,
        )

        merged, sec_state = operators.merge(node1, node2)

        # Check value conservation
        expected_value = node1.value_embedding + node2.value_embedding
        assert torch.allclose(merged.value_embedding, expected_value)

        # Check potential conservation (Fibonacci)
        assert abs(merged.potential - (node1.potential + node2.potential)) < 1e-6

        # Check SEC state
        assert isinstance(sec_state, SECState)
        assert sec_state.phase in ["attraction", "repulsion"]

    def test_branch_operator(self):
        """Test ⊗ (branch) operator"""
        operators = SECOperators()

        parent = PACNode(
            value_embedding=torch.tensor([1.0, 1.0, 1.0]),
            complexity_vector=torch.tensor([1.0, 1.0, 1.0]),
            effect_cone=torch.tensor([1.0, 1.0, 1.0]),
            potential=1.0,
            depth=0,
        )

        context = torch.tensor([1.0, 0.0, 0.0])

        child1, child2 = operators.branch(parent, context)

        # Check depths (Fibonacci sequence)
        assert child1.depth == parent.depth + 1
        assert child2.depth == parent.depth + 2

        # Check potentials (golden ratio scaling)
        phi = operators.constants.PHI
        assert abs(child1.potential - parent.potential * (phi ** (-1))) < 1e-6
        assert abs(child2.potential - parent.potential * (phi ** (-2))) < 1e-6

    def test_gradient_detection(self):
        """Test δ (gradient) operator"""
        operators = SECOperators()

        node = PACNode(
            value_embedding=torch.tensor([0.0, 0.0, 0.0]),
            complexity_vector=torch.tensor([0.0, 0.0, 0.0]),
            effect_cone=torch.tensor([0.0, 0.0, 0.0]),
            potential=1.0,
            depth=0,
        )

        neighbor1 = PACNode(
            value_embedding=torch.tensor([1.0, 0.0, 0.0]),
            complexity_vector=torch.tensor([1.0, 0.0, 0.0]),
            effect_cone=torch.tensor([1.0, 0.0, 0.0]),
            potential=1.0,
            depth=0,
        )

        neighbor2 = PACNode(
            value_embedding=torch.tensor([0.0, 1.0, 0.0]),
            complexity_vector=torch.tensor([0.0, 1.0, 0.0]),
            effect_cone=torch.tensor([0.0, 1.0, 0.0]),
            potential=1.0,
            depth=0,
        )

        magnitude, direction = operators.detect_gradient(
            node, [neighbor1, neighbor2]
        )

        # Should detect gradient
        assert magnitude >= 0
        assert direction.shape == node.value_embedding.shape

    def test_duty_cycle_computation(self):
        """Test duty cycle = φ/(φ+1)"""
        operators = SECOperators()

        # Perfect 61.8% attraction
        history = ["attraction"] * 618 + ["repulsion"] * 382
        duty = operators.compute_duty_cycle(history)

        assert abs(duty - 0.618) < 0.01


class TestMEDValidator:
    """Test MED universal bounds."""

    def test_depth_validation(self):
        """Test depth(S) ≤ 1"""
        validator = MEDValidator(strict_mode=False)

        # Valid: depth difference = 1
        node1 = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=1.0,
            depth=0,
        )
        node2 = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=0.6,
            depth=1,
        )

        assert validator.validate_depth([node1, node2])

        # Invalid: depth difference = 2
        node3 = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=0.4,
            depth=2,
        )

        assert not validator.validate_depth([node1, node3])

    def test_node_count_validation(self):
        """Test nodes(S) ≤ 3"""
        validator = MEDValidator(strict_mode=False)

        # Valid: 3 nodes
        nodes = [
            PACNode(
                value_embedding=torch.randn(10),
                complexity_vector=torch.randn(10),
                effect_cone=torch.randn(10),
                potential=1.0,
                depth=0,
            )
            for _ in range(3)
        ]

        assert validator.validate_node_count(nodes)

        # Invalid: 4 nodes
        nodes.append(
            PACNode(
                value_embedding=torch.randn(10),
                complexity_vector=torch.randn(10),
                effect_cone=torch.randn(10),
                potential=1.0,
                depth=0,
            )
        )

        assert not validator.validate_node_count(nodes)


class TestDistanceValidator:
    """Test E=mc² distance validation."""

    def test_synthetic_conservation(self):
        """Test c² ≈ 1.0 for synthetic embeddings"""
        validator = DistanceValidator()

        # Perfect conservation
        parent = torch.tensor([3.0, 4.0])  # E = 25
        child1 = torch.tensor([1.5, 2.0])  # E = 6.25
        child2 = torch.tensor([1.5, 2.0])  # E = 6.25
        # Total children E = 12.5, c² = 12.5/25 = 0.5 (binding)

        metrics = validator.validate_energy_conservation(
            parent, [child1, child2], embedding_type="synthetic"
        )

        # Should detect binding energy
        assert metrics.c_squared < 1.0
        assert metrics.binding_energy > 0  # Parent > children

    def test_amplification_measurement(self):
        """Test semantic amplification measurement"""
        validator = DistanceValidator()

        # Create amplification case (children > parent)
        parent = torch.tensor([1.0, 0.0])  # E = 1
        child1 = torch.tensor([2.0, 0.0])  # E = 4
        child2 = torch.tensor([1.0, 0.0])  # E = 1
        # Total = 5, amplification = +400%

        amplification, interpretation = validator.measure_amplification(
            parent, [child1, child2]
        )

        assert amplification > 0  # Amplification detected
        assert "+400%" in interpretation


class TestFoundationIntegration:
    """Test integration layer."""

    def test_create_pac_node(self):
        """Test PAC node creation"""
        integration = FoundationIntegration(embedding_dim=384)

        embedding = torch.randn(384)
        node = integration.create_pac_node_from_embedding(
            embedding=embedding,
            content="Test node",
            depth=0,
            metadata={"test": True},
        )

        assert isinstance(node, PACNode)
        assert node.depth == 0
        assert node.value_embedding.shape == (384,)

    def test_conservation_verification(self):
        """Test full conservation verification"""
        integration = FoundationIntegration(embedding_dim=10)

        # Create parent and children
        parent = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=1.0,
            depth=0,
        )

        child1 = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=0.6,
            depth=1,
        )

        child2 = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=0.4,
            depth=2,
        )

        metrics = integration.verify_conservation(parent, [child1, child2])

        # Check all components present
        assert "fibonacci" in metrics.pac_conservation
        assert "value" in metrics.pac_conservation
        assert "complexity" in metrics.pac_conservation
        assert "effect" in metrics.pac_conservation
        assert metrics.balance_operator > 0
        assert metrics.collapse_status in ["STABLE", "COLLAPSE", "DECAY"]
        assert isinstance(metrics.sec_state, SECState)
        assert 0 <= metrics.duty_cycle <= 1
        assert isinstance(metrics.med_valid, bool)

    def test_resonance_scoring(self):
        """Test resonance score computation"""
        integration = FoundationIntegration()

        node = PACNode(
            value_embedding=torch.randn(10),
            complexity_vector=torch.randn(10),
            effect_cone=torch.randn(10),
            potential=1.0,
            depth=2,
        )

        score = integration.compute_resonance_score(node, query_depth=0)

        assert score > 0
        assert isinstance(score, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
