"""
PAC (Potential-Actualization Conservation) Engine

Implements the foundational conservation principle from Dawn Field Theory:
    Ψ(k) = Ψ(k+1) + Ψ(k+2)  (Fibonacci recursion)

Conservation holds across three dimensions:
1. Value conservation: f(v) = Σ f(children)
2. Complexity conservation: ||C(v)||² = ||Σ C(children)||²
3. Effect conservation: Effect(v) = Σ Effect(children)

Key Constants:
- φ (PHI): Golden ratio = (1 + √5) / 2 ≈ 1.618
- Ξ (XI): Balance operator = 1 + π/F₁₀ ≈ 1.0571
- LAMBDA_STAR: Optimal decay = 0.9816

References:
- foundational/arithmetic/unified_pac_framework_comprehensive.md
- foundational/arithmetic/euclidean_distance_validation/
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# Fibonacci sequence for reference
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]


@dataclass
class PACConstants:
    """Fundamental constants from Dawn Field Theory."""

    # Golden ratio: φ = (1 + √5) / 2
    PHI: float = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895

    # Balance operator: Ξ = 1 + π/F₁₀ where F₁₀ = 55
    XI: float = 1 + math.pi / 55  # ≈ 1.0571238898

    # Optimal decay constant (experimentally determined)
    LAMBDA_STAR: float = 0.9816

    # Duty cycle at equilibrium: φ/(φ+1)
    DUTY_CYCLE: float = PHI / (PHI + 1)  # ≈ 0.618 (61.8%)

    # Attraction/repulsion balance ratio
    BALANCE_RATIO: float = 4.0  # 4:1 (PAC:SEC)

    # MED universal bounds
    MAX_DEPTH: int = 1
    MAX_NODES: int = 3

    # Distance validation tolerances
    SYNTHETIC_C_SQUARED: float = 1.0  # For synthetic embeddings
    REAL_C_SQUARED_MIN: float = 100.0  # For real LLM embeddings
    REAL_C_SQUARED_MAX: float = 1000.0

    # Conservation tolerance
    CONSERVATION_TOLERANCE: float = 1e-3


@dataclass
class PACNode:
    """
    Node in PAC hierarchy with three-dimensional conservation.

    Attributes:
        value_embedding: Value representation (semantic embedding)
        complexity_vector: Complexity representation
        effect_cone: Causal effect propagation
        potential: Potential energy (Ψ)
        depth: Depth in hierarchy (k in Ψ(k))
        metadata: Additional properties
    """

    value_embedding: torch.Tensor  # Semantic embedding
    complexity_vector: torch.Tensor  # Complexity representation
    effect_cone: torch.Tensor  # Effect propagation
    potential: float  # Ψ(k)
    depth: int  # k in Ψ(k)
    metadata: Optional[Dict] = None

    def __post_init__(self):
        """Validate dimensions match."""
        if self.value_embedding.shape != self.complexity_vector.shape:
            raise ValueError(
                f"Value and complexity dimensions must match: "
                f"{self.value_embedding.shape} != {self.complexity_vector.shape}"
            )


class PACConservationEngine:
    """
    Core PAC conservation engine implementing Fibonacci recursion.

    The fundamental principle:
        Ψ(k) = Ψ(k+1) + Ψ(k+2)

    Conservation across three dimensions:
        1. Value: f(parent) = Σ f(children)
        2. Complexity: ||C(parent)||² = ||Σ C(children)||²
        3. Effect: Effect(parent) = Σ Effect(children)
    """

    def __init__(
        self,
        device: str = "cpu",
        constants: Optional[PACConstants] = None,
    ):
        """
        Initialize PAC conservation engine.

        Args:
            device: Computation device ('cpu' or 'cuda')
            constants: PAC constants (uses defaults if None)
        """
        self.device = device
        self.constants = constants or PACConstants()

    def compute_potential(self, depth: int, amplitude: float = 1.0) -> float:
        """
        Compute potential energy at given depth.

        Uses golden ratio scaling:
            Ψ(k) = A · φ^(-k)

        Args:
            depth: Depth k in hierarchy
            amplitude: Amplitude A (default 1.0)

        Returns:
            Potential energy Ψ(k)
        """
        return amplitude * (self.constants.PHI ** (-depth))

    def verify_fibonacci_recursion(
        self,
        parent_potential: float,
        child1_potential: float,
        child2_potential: float,
    ) -> Tuple[bool, float]:
        """
        Verify Fibonacci recursion: Ψ(k) = Ψ(k+1) + Ψ(k+2)

        Args:
            parent_potential: Ψ(k)
            child1_potential: Ψ(k+1)
            child2_potential: Ψ(k+2)

        Returns:
            (is_valid, residual) where residual = |Ψ(k) - (Ψ(k+1) + Ψ(k+2))|
        """
        expected = child1_potential + child2_potential
        residual = abs(parent_potential - expected)
        is_valid = residual < self.constants.CONSERVATION_TOLERANCE

        return is_valid, residual

    def verify_value_conservation(
        self,
        parent: torch.Tensor,
        children: List[torch.Tensor],
    ) -> Tuple[bool, float]:
        """
        Verify value conservation: f(parent) ≈ Σ f(children)

        Args:
            parent: Parent value embedding
            children: List of children value embeddings

        Returns:
            (is_valid, residual) where residual = ||parent - Σ children||
        """
        if not children:
            # No children: cannot verify conservation
            return False, float("inf")

        children_sum = torch.stack(children).sum(dim=0)
        residual = torch.norm(parent - children_sum).item()
        is_valid = residual < self.constants.CONSERVATION_TOLERANCE

        return is_valid, residual

    def verify_complexity_conservation(
        self,
        parent_complexity: torch.Tensor,
        children_complexity: List[torch.Tensor],
    ) -> Tuple[bool, float]:
        """
        Verify complexity conservation: ||C(parent)||² ≈ ||Σ C(children)||²

        This allows surface complexity to amplify while total complexity conserves.

        Args:
            parent_complexity: Parent complexity vector
            children_complexity: List of children complexity vectors

        Returns:
            (is_valid, residual) where residual = |parent² - children_sum²|
        """
        if not children_complexity:
            # No children: cannot verify conservation
            return False, float("inf")

        parent_norm_sq = torch.norm(parent_complexity).item() ** 2
        children_sum = torch.stack(children_complexity).sum(dim=0)
        children_norm_sq = torch.norm(children_sum).item() ** 2

        residual = abs(parent_norm_sq - children_norm_sq)
        is_valid = residual < self.constants.CONSERVATION_TOLERANCE

        return is_valid, residual

    def verify_effect_conservation(
        self,
        parent_effect: torch.Tensor,
        children_effects: List[torch.Tensor],
    ) -> Tuple[bool, float]:
        """
        Verify effect conservation: Effect(parent) ≈ Σ Effect(children)

        Args:
            parent_effect: Parent effect cone
            children_effects: List of children effect cones

        Returns:
            (is_valid, residual)
        """
        if not children_effects:
            # No children: cannot verify conservation
            return False, float("inf")

        children_sum = torch.stack(children_effects).sum(dim=0)
        residual = torch.norm(parent_effect - children_sum).item()
        is_valid = residual < self.constants.CONSERVATION_TOLERANCE

        return is_valid, residual

    def verify_full_conservation(
        self,
        parent: PACNode,
        children: List[PACNode],
    ) -> Dict[str, Tuple[bool, float]]:
        """
        Verify full PAC conservation across all three dimensions.

        Args:
            parent: Parent PAC node
            children: List of children PAC nodes

        Returns:
            Dictionary with results for each conservation law:
            {
                'fibonacci': (is_valid, residual),
                'value': (is_valid, residual),
                'complexity': (is_valid, residual),
                'effect': (is_valid, residual),
            }
        """
        results = {}

        # Fibonacci recursion (requires exactly 2 children)
        if len(children) == 2:
            results["fibonacci"] = self.verify_fibonacci_recursion(
                parent.potential,
                children[0].potential,
                children[1].potential,
            )
        else:
            results["fibonacci"] = (False, float("inf"))

        # Value conservation
        results["value"] = self.verify_value_conservation(
            parent.value_embedding,
            [c.value_embedding for c in children],
        )

        # Complexity conservation
        results["complexity"] = self.verify_complexity_conservation(
            parent.complexity_vector,
            [c.complexity_vector for c in children],
        )

        # Effect conservation
        results["effect"] = self.verify_effect_conservation(
            parent.effect_cone,
            [c.effect_cone for c in children],
        )

        return results

    def compute_balance_operator(
        self,
        node: PACNode,
        children: List[PACNode],
    ) -> float:
        """
        Compute local balance operator Ξ.

        When Ξ > XI_THRESHOLD: trigger collapse
        When Ξ ≈ XI_THRESHOLD: stable recursion
        When Ξ < XI_THRESHOLD: field decay

        Args:
            node: Current node
            children: Node's children

        Returns:
            Local balance operator value
        """
        if not children:
            return 1.0

        # Compute symbolic pressure from children
        children_potential = sum(c.potential for c in children)
        parent_potential = node.potential

        if parent_potential == 0:
            return float("inf")

        # Balance = 1 + (excess_pressure / parent_potential)
        excess_pressure = children_potential - parent_potential
        xi_local = 1.0 + (excess_pressure / parent_potential)

        return xi_local

    def check_collapse_trigger(self, xi_local: float) -> str:
        """
        Check if collapse should be triggered based on balance operator.

        Args:
            xi_local: Local balance operator value

        Returns:
            Status: 'COLLAPSE', 'STABLE', or 'DECAY'
        """
        # Note: Values near DUTY_CYCLE (0.618) are normal for golden ratio recursion
        # Only flag as decay if significantly below normal range
        if xi_local > self.constants.XI:
            return "COLLAPSE"  # Excess symbolic pressure
        elif xi_local < 0.5:  # Well below duty cycle threshold
            return "DECAY"  # Symbolic decay
        else:
            return "STABLE"  # Stable recursion (includes duty cycle range)

    def validate_distance_conservation(
        self,
        parent_embedding: torch.Tensor,
        children_embeddings: List[torch.Tensor],
    ) -> Tuple[bool, float]:
        """
        Validate PAC via Euclidean distance (E=mc² framework).

        For synthetic embeddings: c² ≈ 1.0
        For real LLM embeddings: c² ≈ 100-1000

        Args:
            parent_embedding: Parent embedding vector
            children_embeddings: List of children embedding vectors

        Returns:
            (is_valid, c_squared) where c² = E_children / E_parent
        """
        # E = ||embedding||² (semantic energy)
        parent_energy = torch.norm(parent_embedding).item() ** 2
        children_energy = sum(
            torch.norm(c).item() ** 2 for c in children_embeddings
        )

        if parent_energy == 0:
            return False, float("inf")

        # Compute c² (model-specific constant)
        c_squared = children_energy / parent_energy

        # Validate range (synthetic or real LLM)
        is_synthetic = (
            0.8 < c_squared < 1.2
        )  # ±20% tolerance for synthetic
        is_real_llm = (
            self.constants.REAL_C_SQUARED_MIN
            < c_squared
            < self.constants.REAL_C_SQUARED_MAX
        )

        is_valid = is_synthetic or is_real_llm

        return is_valid, c_squared

    def create_pac_node(
        self,
        value_embedding: torch.Tensor,
        depth: int,
        complexity_vector: Optional[torch.Tensor] = None,
        effect_cone: Optional[torch.Tensor] = None,
        amplitude: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> PACNode:
        """
        Create a PAC node with proper initialization.

        Args:
            value_embedding: Semantic embedding
            depth: Depth k in hierarchy
            complexity_vector: Complexity representation (defaults to value)
            effect_cone: Effect propagation (defaults to value)
            amplitude: Potential amplitude
            metadata: Additional properties

        Returns:
            Initialized PACNode
        """
        # Default complexity and effect to value if not provided
        if complexity_vector is None:
            complexity_vector = value_embedding.clone()
        if effect_cone is None:
            effect_cone = value_embedding.clone()

        # Compute potential using golden ratio scaling
        potential = self.compute_potential(depth, amplitude)

        return PACNode(
            value_embedding=value_embedding,
            complexity_vector=complexity_vector,
            effect_cone=effect_cone,
            potential=potential,
            depth=depth,
            metadata=metadata,
        )
