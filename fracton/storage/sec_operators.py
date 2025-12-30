"""
SEC (Symbolic Entropy Collapse) Operators

Implements the geometric collapse dynamics from Dawn Field Theory:
- Attraction/repulsion balance: 4:1 ratio
- Duty cycle equilibrium: φ/(φ+1) ≈ 61.8%
- Resonance ranking through harmonic relationships

Operators:
- ⊕ (merge): Symbolic merge with coherence
- ⊗ (branch): Memory retention branching
- δ (detect): Entropy gradient detection

References:
- foundational/docs/[id][F][v1.0][C5][I5][E]_symbolic_entropy_collapse_geometry_foundation.md
- foundational/docs/[id][F][v1.0][C6][I6][E]_pac_sec_as_information_dynamics.md
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .pac_engine import PACConstants, PACNode


@dataclass
class SECState:
    """
    State of SEC collapse dynamics.

    Attributes:
        entropy: Current structural entropy S
        coherence: Coherence measure (0-1)
        duty_cycle: Fraction of time in attraction (target: 0.618)
        resonance_rank: Position in resonance hierarchy
        phase: Current phase state
    """

    entropy: float
    coherence: float
    duty_cycle: float
    resonance_rank: int
    phase: str  # 'attraction', 'repulsion', 'equilibrium'


class SECOperators:
    """
    SEC operators implementing symbolic entropy collapse dynamics.

    The fundamental equation:
        ∂S/∂t = α∇I - β∇H

    Where:
        S = Structural Entropy
        ∇I = Information Gradient (actualization pressure)
        ∇H = Entropy Gradient (conservation constraint)
        α, β = Coupling constants (ratio 4:1)
    """

    def __init__(
        self,
        device: str = "cpu",
        constants: Optional[PACConstants] = None,
        alpha: float = 0.005857,  # Actualization rate (MED validated)
        beta: Optional[float] = None,  # Auto-computed for 4:1 ratio
    ):
        """
        Initialize SEC operators.

        Args:
            device: Computation device
            constants: PAC constants
            alpha: Actualization rate (default from MED experiments)
            beta: Conservation constraint (auto-computed if None)
        """
        self.device = device
        self.constants = constants or PACConstants()

        # Actualization and conservation rates
        self.alpha = alpha
        self.beta = beta or (alpha / self.constants.BALANCE_RATIO)  # 4:1 ratio

    def merge(
        self,
        node1: PACNode,
        node2: PACNode,
        coherence_threshold: float = 0.5,
    ) -> Tuple[PACNode, SECState]:
        """
        ⊕ operator: Symbolic merge with coherence preservation.

        Merges two nodes while maintaining PAC conservation and
        tracking SEC collapse state.

        Args:
            node1: First node
            node2: Second node
            coherence_threshold: Minimum coherence for stable merge

        Returns:
            (merged_node, sec_state)
        """
        # Value: sum for conservation
        merged_value = node1.value_embedding + node2.value_embedding

        # Complexity: combine and normalize
        merged_complexity = node1.complexity_vector + node2.complexity_vector

        # Effect: union of effect cones
        merged_effect = node1.effect_cone + node2.effect_cone

        # Potential: sum (Fibonacci)
        merged_potential = node1.potential + node2.potential

        # Depth: min (merged node is at higher level)
        merged_depth = min(node1.depth, node2.depth)

        # Compute coherence (cosine similarity)
        coherence = torch.nn.functional.cosine_similarity(
            node1.value_embedding.unsqueeze(0),
            node2.value_embedding.unsqueeze(0),
        ).item()

        # Compute entropy (normalized)
        entropy = self._compute_entropy(merged_value)

        # Determine phase
        if coherence > coherence_threshold:
            phase = "attraction"
            duty_cycle = self.constants.DUTY_CYCLE
        else:
            phase = "repulsion"
            duty_cycle = 1 - self.constants.DUTY_CYCLE

        # Create merged node
        merged = PACNode(
            value_embedding=merged_value,
            complexity_vector=merged_complexity,
            effect_cone=merged_effect,
            potential=merged_potential,
            depth=merged_depth,
            metadata={
                "merged_from": [
                    node1.metadata.get("id") if node1.metadata else None,
                    node2.metadata.get("id") if node2.metadata else None,
                ],
                "coherence": coherence,
            },
        )

        # SEC state
        sec_state = SECState(
            entropy=entropy,
            coherence=coherence,
            duty_cycle=duty_cycle,
            resonance_rank=self._compute_resonance_rank(merged_depth),
            phase=phase,
        )

        return merged, sec_state

    def branch(
        self,
        parent: PACNode,
        context: torch.Tensor,
        retention_factor: float = 0.618,
    ) -> Tuple[PACNode, PACNode]:
        """
        ⊗ operator: Memory retention branching.

        Creates two children from parent, preserving PAC conservation.

        Args:
            parent: Parent node
            context: Context vector for branching direction
            retention_factor: Memory retention (default: φ/(φ+1))

        Returns:
            (child1, child2) satisfying Ψ(k) = Ψ(k+1) + Ψ(k+2)
        """
        # Compute branching direction from context
        direction = context / torch.norm(context)

        # Project parent onto direction and orthogonal
        projection = torch.dot(parent.value_embedding, direction) * direction
        orthogonal = parent.value_embedding - projection

        # Split using retention factor (golden ratio)
        child1_value = retention_factor * projection + (
            1 - retention_factor
        ) * orthogonal
        child2_value = (1 - retention_factor) * projection + retention_factor * orthogonal

        # Normalize to conserve total value
        total_norm = torch.norm(parent.value_embedding)
        child1_value = child1_value * (
            retention_factor * total_norm / torch.norm(child1_value)
        )
        child2_value = child2_value * (
            (1 - retention_factor) * total_norm / torch.norm(child2_value)
        )

        # Depths (Fibonacci sequence)
        child1_depth = parent.depth + 1  # k+1
        child2_depth = parent.depth + 2  # k+2

        # Potentials (golden ratio scaling)
        child1_potential = parent.potential * (self.constants.PHI ** (-1))
        child2_potential = parent.potential * (self.constants.PHI ** (-2))

        # Create children
        child1 = PACNode(
            value_embedding=child1_value,
            complexity_vector=child1_value.clone(),
            effect_cone=child1_value.clone(),
            potential=child1_potential,
            depth=child1_depth,
            metadata={"parent_id": parent.metadata.get("id") if parent.metadata else None},
        )

        child2 = PACNode(
            value_embedding=child2_value,
            complexity_vector=child2_value.clone(),
            effect_cone=child2_value.clone(),
            potential=child2_potential,
            depth=child2_depth,
            metadata={"parent_id": parent.metadata.get("id") if parent.metadata else None},
        )

        return child1, child2

    def detect_gradient(
        self,
        node: PACNode,
        neighbors: List[PACNode],
    ) -> Tuple[float, torch.Tensor]:
        """
        δ operator: Entropy gradient detection.

        Detects local entropy gradients that trigger collapse.

        Args:
            node: Central node
            neighbors: Neighboring nodes

        Returns:
            (gradient_magnitude, gradient_direction)
        """
        if not neighbors:
            return 0.0, torch.zeros_like(node.value_embedding)

        # Compute entropy for node and neighbors
        node_entropy = self._compute_entropy(node.value_embedding)
        neighbor_entropies = [
            self._compute_entropy(n.value_embedding) for n in neighbors
        ]

        # Gradient magnitude (average difference)
        gradient_magnitude = sum(
            abs(node_entropy - ne) for ne in neighbor_entropies
        ) / len(neighbors)

        # Gradient direction (weighted by entropy difference)
        gradient_direction = torch.zeros_like(node.value_embedding)
        for neighbor, ne in zip(neighbors, neighbor_entropies):
            weight = node_entropy - ne
            direction = neighbor.value_embedding - node.value_embedding
            gradient_direction += weight * direction

        # Normalize
        norm = torch.norm(gradient_direction)
        if norm > 0:
            gradient_direction = gradient_direction / norm

        return gradient_magnitude, gradient_direction

    def compute_duty_cycle(
        self,
        history: List[str],
    ) -> float:
        """
        Compute duty cycle from phase history.

        Duty cycle = fraction of time in attraction phase.
        Target equilibrium: φ/(φ+1) ≈ 0.618

        Args:
            history: List of phase states ('attraction' or 'repulsion')

        Returns:
            Duty cycle (0-1)
        """
        if not history:
            return 0.5

        attraction_count = sum(1 for phase in history if phase == "attraction")
        return attraction_count / len(history)

    def compute_resonance_rank(
        self,
        node: PACNode,
        reference_depth: int = 0,
    ) -> Tuple[int, float]:
        """
        Compute resonance ranking from harmonic relationship to φ.

        The resonance function:
            R(k) = φ^(1 + (k_eq - k)/2)

        Where k_eq is equilibrium depth.

        Args:
            node: Node to rank
            reference_depth: Equilibrium depth (default: 0)

        Returns:
            (rank, resonance_value)
        """
        k_diff = reference_depth - node.depth
        resonance = self.constants.PHI ** (1 + k_diff / 2)
        rank = int(resonance * 100)  # Scale for integer ranking

        return rank, resonance

    def evolve_structure(
        self,
        entropy: float,
        information_gradient: float,
        entropy_gradient: float,
        dt: float = 0.01,
    ) -> float:
        """
        Evolve structural entropy via SEC dynamics.

        ∂S/∂t = α∇I - β∇H

        Args:
            entropy: Current entropy S
            information_gradient: ∇I (actualization pressure)
            entropy_gradient: ∇H (conservation constraint)
            dt: Time step

        Returns:
            New entropy value
        """
        dS_dt = self.alpha * information_gradient - self.beta * entropy_gradient
        new_entropy = entropy + dS_dt * dt

        # Bound entropy to [0, ∞)
        return max(0.0, new_entropy)

    def _compute_entropy(self, embedding: torch.Tensor) -> float:
        """
        Compute normalized entropy of embedding.

        Uses normalized variance as proxy for entropy.

        Args:
            embedding: Embedding vector

        Returns:
            Entropy (0-1 normalized)
        """
        # Normalize embedding
        if torch.norm(embedding) == 0:
            return 0.0

        normalized = embedding / torch.norm(embedding)

        # Compute variance (proxy for entropy)
        variance = torch.var(normalized).item()

        # Normalize to [0, 1]
        entropy = min(1.0, variance / 0.25)  # 0.25 ≈ max variance for normalized

        return entropy

    def _compute_resonance_rank(self, depth: int) -> int:
        """
        Compute resonance rank from depth.

        Args:
            depth: Node depth

        Returns:
            Resonance rank (integer)
        """
        rank, _ = self.compute_resonance_rank(
            PACNode(
                value_embedding=torch.zeros(1),
                complexity_vector=torch.zeros(1),
                effect_cone=torch.zeros(1),
                potential=1.0,
                depth=depth,
            )
        )
        return rank
