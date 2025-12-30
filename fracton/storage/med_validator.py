"""
MED (Macro Emergence Dynamics) Validator

Enforces universal bounded complexity constraints:
- depth(S) ≤ 1
- nodes(S) ≤ 3

These bounds emerge from PAC/SEC dynamics and are validated across
1000+ simulations in the foundational theory.

References:
- foundational/arithmetic/macro_emergence_dynamics/README.md
- foundational/arithmetic/macro_emergence_dynamics/proofs/
"""

from dataclasses import dataclass
from typing import List, Optional

from .pac_engine import PACConstants, PACNode


@dataclass
class MEDViolation:
    """Record of MED bound violation."""

    violation_type: str  # 'depth' or 'nodes'
    actual_value: int
    max_allowed: int
    context: str


class MEDValidator:
    """
    MED validator enforcing universal bounded complexity.

    Key discoveries:
    - depth(S) ≤ 1: Maximum depth in any emergent structure
    - nodes(S) ≤ 3: Maximum nodes in any emergent structure
    - Balance operator Ξ ≈ 1.0571 maintains stability

    These bounds hold across ALL scales and flow regimes.
    """

    def __init__(
        self,
        constants: Optional[PACConstants] = None,
        strict_mode: bool = True,
    ):
        """
        Initialize MED validator.

        Args:
            constants: PAC constants (uses defaults if None)
            strict_mode: If True, raises on violations; if False, logs warnings
        """
        self.constants = constants or PACConstants()
        self.strict_mode = strict_mode
        self.violations: List[MEDViolation] = []

    def validate_depth(
        self,
        structure: List[PACNode],
        context: str = "unknown",
    ) -> bool:
        """
        Validate depth constraint: depth(S) ≤ 1

        Args:
            structure: List of nodes in structure
            context: Description of structure for error reporting

        Returns:
            True if valid, False otherwise
        """
        if not structure:
            return True

        max_depth = max(node.depth for node in structure)
        min_depth = min(node.depth for node in structure)
        actual_depth = max_depth - min_depth

        if actual_depth > self.constants.MAX_DEPTH:
            violation = MEDViolation(
                violation_type="depth",
                actual_value=actual_depth,
                max_allowed=self.constants.MAX_DEPTH,
                context=context,
            )
            self.violations.append(violation)

            if self.strict_mode:
                raise ValueError(
                    f"MED depth violation: {actual_depth} > {self.constants.MAX_DEPTH} "
                    f"in {context}"
                )
            return False

        return True

    def validate_node_count(
        self,
        structure: List[PACNode],
        context: str = "unknown",
    ) -> bool:
        """
        Validate node count constraint: nodes(S) ≤ 3

        Args:
            structure: List of nodes in structure
            context: Description of structure for error reporting

        Returns:
            True if valid, False otherwise
        """
        node_count = len(structure)

        if node_count > self.constants.MAX_NODES:
            violation = MEDViolation(
                violation_type="nodes",
                actual_value=node_count,
                max_allowed=self.constants.MAX_NODES,
                context=context,
            )
            self.violations.append(violation)

            if self.strict_mode:
                raise ValueError(
                    f"MED node count violation: {node_count} > {self.constants.MAX_NODES} "
                    f"in {context}"
                )
            return False

        return True

    def validate_structure(
        self,
        structure: List[PACNode],
        context: str = "unknown",
    ) -> bool:
        """
        Validate both MED constraints.

        Args:
            structure: List of nodes in structure
            context: Description of structure

        Returns:
            True if both constraints satisfied
        """
        depth_valid = self.validate_depth(structure, context)
        nodes_valid = self.validate_node_count(structure, context)

        return depth_valid and nodes_valid

    def check_balance_operator(
        self,
        xi_local: float,
        context: str = "unknown",
    ) -> str:
        """
        Check balance operator against MED threshold.

        Args:
            xi_local: Local balance operator value
            context: Description for reporting

        Returns:
            Status: 'stable', 'collapse_warning', or 'decay_warning'
        """
        xi_threshold = self.constants.XI

        if abs(xi_local - xi_threshold) < 0.01:
            return "stable"
        elif xi_local > xi_threshold:
            return "collapse_warning"
        else:
            return "decay_warning"

    def get_violation_summary(self) -> str:
        """
        Get summary of all violations.

        Returns:
            Human-readable summary string
        """
        if not self.violations:
            return "No MED violations detected."

        summary = f"MED Violations ({len(self.violations)}):\n"
        for v in self.violations:
            summary += (
                f"  - {v.violation_type.upper()}: {v.actual_value} > {v.max_allowed} "
                f"({v.context})\n"
            )

        return summary

    def clear_violations(self):
        """Clear violation history."""
        self.violations.clear()

    def compute_quality_score(self) -> float:
        """
        Compute quality score based on MED compliance.

        From MED experiments, optimal quality ≈ 0.91

        Returns:
            Quality score (0-1)
        """
        if not self.violations:
            return 1.0

        # Penalize violations
        depth_violations = sum(
            1 for v in self.violations if v.violation_type == "depth"
        )
        node_violations = sum(
            1 for v in self.violations if v.violation_type == "nodes"
        )

        total_violations = depth_violations + node_violations

        # Exponential penalty
        quality = 1.0 / (1 + total_violations)

        return quality
