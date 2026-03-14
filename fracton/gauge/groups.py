"""
Gauge group Fibonacci structure.

Analyzes the Fibonacci closure property of the Standard Model:
adjoint dimensions of U(1), SU(2), SU(3) are 1, 3, 8 — all Fibonacci.
With the Higgs scalar (dim 1), the total is 1+3+8+1 = 13 = F₇.

Source: MAR exp_38, PACSeries Paper 4.
"""

from typing import Dict, List, Tuple

from ..fibonacci import fib, is_fibonacci


# Standard Model gauge content
STANDARD_MODEL_GAUGE: Dict[str, int] = {
    "U(1)": 1,       # F₁ = F₂ = 1
    "SU(2)": 3,      # F₄ = 3
    "SU(3)": 8,      # F₆ = 8
    "Higgs": 1,      # F₁ = 1 (scalar doublet, 1 DOF after SSB)
}


def adjoint_dim(n: int) -> int:
    """Compute the adjoint dimension of SU(N): N² - 1.

    For U(1), returns 1 by convention.

    Args:
        n: The N in SU(N), or 1 for U(1).

    Returns:
        Adjoint dimension.

    Examples:
        >>> adjoint_dim(2)
        3
        >>> adjoint_dim(3)
        8
    """
    if n <= 1:
        return 1
    return n * n - 1


def gauge_closure_sum(groups: Dict[str, int] = None) -> Tuple[int, bool]:
    """Compute total gauge DOF and check Fibonacci closure.

    Args:
        groups: Dict mapping group names to adjoint dimensions.
            Defaults to the Standard Model.

    Returns:
        Tuple of (total_dim, is_fibonacci_closed).

    Examples:
        >>> total, closed = gauge_closure_sum()
        >>> total
        13
        >>> closed
        True
    """
    if groups is None:
        groups = STANDARD_MODEL_GAUGE

    total = sum(groups.values())
    return total, is_fibonacci(total)


def is_fibonacci_gauge(groups: Dict[str, int] = None) -> Dict[str, bool]:
    """Check which gauge group dimensions are Fibonacci numbers.

    Args:
        groups: Dict mapping group names to adjoint dimensions.

    Returns:
        Dict mapping each group to whether its dimension is Fibonacci.

    Examples:
        >>> is_fibonacci_gauge()
        {'U(1)': True, 'SU(2)': True, 'SU(3)': True, 'Higgs': True}
    """
    if groups is None:
        groups = STANDARD_MODEL_GAUGE
    return {name: is_fibonacci(dim) for name, dim in groups.items()}
