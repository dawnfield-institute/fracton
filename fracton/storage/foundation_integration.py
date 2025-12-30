"""
Foundation Integration Layer

Integrates PAC/SEC/MED theoretical foundations with existing KronosMemory architecture.

This module bridges the new theoretical foundations with the existing codebase:
- Wraps PACMemoryNode to include proper PAC conservation
- Adds SEC collapse dynamics to store/retrieve operations
- Enforces MED universal bounds
- Validates via distance metrics

The integration is designed to be backward-compatible while adding theoretical rigor.
"""

from typing import List, Optional, Tuple
import torch
from dataclasses import dataclass

from .pac_engine import PACConservationEngine, PACNode, PACConstants
from .sec_operators import SECOperators, SECState
from .med_validator import MEDValidator
from .distance_validator import DistanceValidator, DistanceMetrics


@dataclass
class FoundationMetrics:
    """Combined metrics from all foundation components."""

    # PAC
    pac_conservation: dict  # Results from verify_full_conservation
    balance_operator: float  # Ξ value
    collapse_status: str  # 'STABLE', 'COLLAPSE', 'DECAY'

    # SEC
    sec_state: SECState
    duty_cycle: float
    resonance_rank: int

    # MED
    med_valid: bool
    med_quality: float

    # Distance
    distance_metrics: DistanceMetrics


class FoundationIntegration:
    """
    Integration layer for PAC/SEC/MED foundations.

    Wraps the theoretical engines and provides high-level operations
    for KronosMemory to use.
    """

    def __init__(
        self,
        device: str = "cpu",
        embedding_dim: int = 384,
        constants: Optional[PACConstants] = None,
        enable_strict_med: bool = False,
    ):
        """
        Initialize foundation integration.

        Args:
            device: Computation device
            embedding_dim: Embedding dimension
            constants: PAC constants (uses defaults if None)
            enable_strict_med: If True, raises on MED violations
        """
        self.device = device
        self.embedding_dim = embedding_dim
        self.constants = constants or PACConstants()

        # Initialize engines
        self.pac_engine = PACConservationEngine(
            device=device, constants=self.constants
        )
        self.sec_operators = SECOperators(
            device=device, constants=self.constants
        )
        self.med_validator = MEDValidator(
            constants=self.constants, strict_mode=enable_strict_med
        )
        self.distance_validator = DistanceValidator(device=device)

        # State tracking
        self.phase_history: List[str] = []

    def create_pac_node_from_embedding(
        self,
        embedding: torch.Tensor,
        content: str,
        depth: int,
        parent_embedding: Optional[torch.Tensor] = None,
        metadata: Optional[dict] = None,
    ) -> PACNode:
        """
        Create PAC node with proper conservation setup.

        Args:
            embedding: Full semantic embedding
            content: Text content
            depth: Depth in hierarchy
            parent_embedding: Parent embedding (if not root)
            metadata: Additional metadata

        Returns:
            PACNode with proper initialization
        """
        # For non-root nodes, compute delta
        if parent_embedding is not None:
            # Delta encoding (will be replaced by proper conservation)
            value_embedding = embedding - parent_embedding
        else:
            # Root node uses full embedding
            value_embedding = embedding

        # Complexity and effect default to value
        complexity_vector = value_embedding.clone()
        effect_cone = value_embedding.clone()

        return self.pac_engine.create_pac_node(
            value_embedding=value_embedding,
            depth=depth,
            complexity_vector=complexity_vector,
            effect_cone=effect_cone,
            amplitude=1.0,
            metadata=metadata or {},
        )

    def verify_conservation(
        self,
        parent: PACNode,
        children: List[PACNode],
    ) -> FoundationMetrics:
        """
        Verify full PAC/SEC/MED conservation and return metrics.

        Args:
            parent: Parent node
            children: Children nodes

        Returns:
            FoundationMetrics with all validation results
        """
        # PAC conservation
        pac_results = self.pac_engine.verify_full_conservation(parent, children)

        # Balance operator
        xi_local = self.pac_engine.compute_balance_operator(parent, children)
        collapse_status = self.pac_engine.check_collapse_trigger(xi_local)

        # SEC state (use first child if available, or merge if multiple)
        if len(children) == 1:
            sec_state = SECState(
                entropy=self.sec_operators._compute_entropy(
                    children[0].value_embedding
                ),
                coherence=0.5,
                duty_cycle=self.constants.DUTY_CYCLE,
                resonance_rank=self.sec_operators._compute_resonance_rank(
                    children[0].depth
                ),
                phase="stable",
            )
        elif len(children) >= 2:
            _, sec_state = self.sec_operators.merge(
                children[0], children[1]
            )
        else:
            sec_state = SECState(
                entropy=0.0,
                coherence=1.0,
                duty_cycle=0.5,
                resonance_rank=0,
                phase="stable",
            )

        # Update phase history
        self.phase_history.append(sec_state.phase)
        if len(self.phase_history) > 100:
            self.phase_history.pop(0)

        # Duty cycle from history
        duty_cycle = self.sec_operators.compute_duty_cycle(
            self.phase_history
        )

        # MED validation
        all_nodes = [parent] + children
        med_valid = self.med_validator.validate_structure(
            all_nodes, context=f"depth_{parent.depth}"
        )
        med_quality = self.med_validator.compute_quality_score()

        # Distance validation
        parent_full = self.reconstruct_full_embedding(parent, [])
        children_full = [
            self.reconstruct_full_embedding(child, [])
            for child in children
        ]
        distance_metrics = self.distance_validator.validate_energy_conservation(
            parent_full, children_full
        )

        return FoundationMetrics(
            pac_conservation=pac_results,
            balance_operator=xi_local,
            collapse_status=collapse_status,
            sec_state=sec_state,
            duty_cycle=duty_cycle,
            resonance_rank=sec_state.resonance_rank,
            med_valid=med_valid,
            med_quality=med_quality,
            distance_metrics=distance_metrics,
        )

    def reconstruct_full_embedding(
        self,
        node: PACNode,
        ancestors: List[PACNode],
    ) -> torch.Tensor:
        """
        Reconstruct full embedding from PAC deltas.

        Args:
            node: Target node
            ancestors: List of ancestors from root to parent

        Returns:
            Full reconstructed embedding
        """
        # Start with node's value (which is delta if not root)
        full_embedding = node.value_embedding.clone()

        # Add all ancestor deltas
        for ancestor in ancestors:
            full_embedding = full_embedding + ancestor.value_embedding

        return full_embedding

    def apply_sec_merge(
        self,
        node1: PACNode,
        node2: PACNode,
    ) -> Tuple[PACNode, SECState]:
        """
        Apply SEC merge operator.

        Args:
            node1: First node
            node2: Second node

        Returns:
            (merged_node, sec_state)
        """
        return self.sec_operators.merge(node1, node2)

    def apply_sec_branch(
        self,
        parent: PACNode,
        context: torch.Tensor,
    ) -> Tuple[PACNode, PACNode]:
        """
        Apply SEC branch operator.

        Args:
            parent: Parent node
            context: Context for branching

        Returns:
            (child1, child2) satisfying Fibonacci recursion
        """
        return self.sec_operators.branch(parent, context)

    def detect_entropy_gradient(
        self,
        node: PACNode,
        neighbors: List[PACNode],
    ) -> Tuple[float, torch.Tensor]:
        """
        Detect entropy gradient (SEC δ operator).

        Args:
            node: Central node
            neighbors: Neighbor nodes

        Returns:
            (magnitude, direction)
        """
        return self.sec_operators.detect_gradient(node, neighbors)

    def should_trigger_collapse(
        self,
        node: PACNode,
        children: List[PACNode],
    ) -> bool:
        """
        Check if collapse should be triggered.

        Args:
            node: Parent node
            children: Children nodes

        Returns:
            True if collapse should occur
        """
        xi_local = self.pac_engine.compute_balance_operator(node, children)
        status = self.pac_engine.check_collapse_trigger(xi_local)
        return status == "COLLAPSE"

    def compute_resonance_score(
        self,
        node: PACNode,
        query_depth: int = 0,
    ) -> float:
        """
        Compute resonance score for ranking.

        Args:
            node: Node to score
            query_depth: Query depth for relative resonance

        Returns:
            Resonance score (higher = more resonant)
        """
        rank, resonance = self.sec_operators.compute_resonance_rank(
            node, reference_depth=query_depth
        )
        return resonance

    def get_metrics_summary(
        self, metrics: FoundationMetrics
    ) -> str:
        """
        Get human-readable summary of metrics.

        Args:
            metrics: Foundation metrics

        Returns:
            Formatted summary string
        """
        summary = []
        summary.append("Foundation Metrics:")
        summary.append(f"  PAC Conservation:")
        for key, (valid, residual) in metrics.pac_conservation.items():
            status = "✓" if valid else "✗"
            summary.append(
                f"    {status} {key}: residual={residual:.6f}"
            )
        summary.append(
            f"  Balance Operator: Ξ={metrics.balance_operator:.4f} ({metrics.collapse_status})"
        )
        summary.append(
            f"  SEC: {metrics.sec_state.phase}, duty={metrics.duty_cycle:.3f}, rank={metrics.resonance_rank}"
        )
        summary.append(
            f"  MED: valid={metrics.med_valid}, quality={metrics.med_quality:.3f}"
        )
        summary.append(
            f"  Distance: c²={metrics.distance_metrics.c_squared:.2f}, "
            f"type={metrics.distance_metrics.embedding_type}"
        )

        return "\n".join(summary)
