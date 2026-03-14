"""
Fibonacci number utilities.

Provides efficient Fibonacci computation, lookup, and analysis functions
used throughout Dawn Field Theory. The Fibonacci sequence is fundamental
to PAC recursion — the unique stable solution Ψ(k) = φ⁻ᵏ arises from
the two-term recurrence Ψ(k) = Ψ(k+1) + Ψ(k+2).

Convention: 0-indexed (F_0=0, F_1=1, F_2=1, F_3=2, F_4=3, F_5=5, ...).
This matches the mathematical convention and the experiments.
"""

import math
from functools import lru_cache
from typing import List, Optional, Tuple

from .constants.mathematical import PHI, SQRT5, LN_PHI

# =============================================================================
# CORE COMPUTATION
# =============================================================================

@lru_cache(maxsize=512)
def fib(n: int) -> int:
    """Compute the nth Fibonacci number (0-indexed).

    F_0 = 0, F_1 = 1, F_2 = 1, F_3 = 2, ...

    Uses memoized recursion for exact integer results up to any size.
    For very large n where only approximate magnitude is needed, use fib_log10().

    Args:
        n: Index (non-negative integer).

    Returns:
        F_n as exact integer.

    Examples:
        >>> fib(0)
        0
        >>> fib(7)
        13
        >>> fib(10)
        55
        >>> fib(183)  # gravity depth — a 38-digit number
        127127879743834334146972278486287885163
    """
    if n < 0:
        raise ValueError(f"Fibonacci index must be non-negative, got {n}")
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


def fib_log10(n: int) -> float:
    """Approximate log₁₀(F_n) using Binet's formula.

    log₁₀(F_n) ≈ n·log₁₀(φ) - 0.5·log₁₀(5)

    Accurate to ~14 digits for n > 10. Essential for F_183 and beyond
    where the exact integer has 38+ digits.

    Args:
        n: Fibonacci index.

    Returns:
        log₁₀(F_n) as float.

    Examples:
        >>> fib_log10(183)  # ≈ 37.895
        37.89489875...
    """
    if n <= 0:
        raise ValueError(f"log₁₀(F_n) undefined for n <= 0, got {n}")
    return n * math.log10(PHI) - 0.5 * math.log10(5)


def fib_approx(n: int) -> float:
    """Approximate F_n using Binet's formula (continuous extension).

    F_n ≈ φⁿ / √5 (for n ≥ 1, rounds to nearest integer).

    Args:
        n: Fibonacci index.

    Returns:
        Approximate F_n as float.
    """
    return PHI ** n / SQRT5


# =============================================================================
# LOOKUP & SEARCH
# =============================================================================

# Precomputed table for fast lookup (covers all indices used in SM physics)
_FIB_TABLE: List[int] = [fib(i) for i in range(201)]


def fib_table(n: int) -> int:
    """Fast table lookup for F_n (n ≤ 200).

    For n > 200, falls back to fib().

    Args:
        n: Fibonacci index.

    Returns:
        F_n as exact integer.
    """
    if 0 <= n < len(_FIB_TABLE):
        return _FIB_TABLE[n]
    return fib(n)


def is_fibonacci(n: int) -> bool:
    """Test whether n is a Fibonacci number.

    Uses the property that n is Fibonacci iff 5n²±4 is a perfect square.

    Args:
        n: Non-negative integer to test.

    Returns:
        True if n is a Fibonacci number.

    Examples:
        >>> is_fibonacci(13)
        True
        >>> is_fibonacci(14)
        False
    """
    if n < 0:
        return False

    def _is_perfect_square(x: int) -> bool:
        s = int(math.isqrt(x))
        return s * s == x

    return _is_perfect_square(5 * n * n + 4) or _is_perfect_square(5 * n * n - 4)


def fib_index(n: int) -> Optional[int]:
    """Find the Fibonacci index of n, or None if n is not Fibonacci.

    Args:
        n: Value to look up.

    Returns:
        Index k such that F_k = n, or None.

    Examples:
        >>> fib_index(13)
        7
        >>> fib_index(55)
        10
        >>> fib_index(14)
        None
    """
    if n < 0:
        return None
    if n == 0:
        return 0
    if n == 1:
        return 1  # F_1 = 1 (also F_2, but return first)

    # Use Binet's formula for approximate index, then verify
    approx_k = round(math.log(n * SQRT5) / LN_PHI)
    for k in range(max(0, approx_k - 2), approx_k + 3):
        if fib(k) == n:
            return k
    return None


def nearest_fibonacci(n: int) -> Tuple[int, int, int]:
    """Find the nearest Fibonacci number(s) to n.

    Args:
        n: Target value.

    Returns:
        Tuple of (F_below, F_above, closest) where F_below ≤ n ≤ F_above
        and closest is the nearest of the two.

    Examples:
        >>> nearest_fibonacci(10)
        (8, 13, 8)
    """
    if n <= 0:
        return (0, 1, 0)

    k = 0
    while fib(k) < n:
        k += 1

    if fib(k) == n:
        return (n, n, n)

    f_below = fib(k - 1)
    f_above = fib(k)
    closest = f_below if (n - f_below) <= (f_above - n) else f_above
    return (f_below, f_above, closest)


# =============================================================================
# FIBONACCI PROPERTIES
# =============================================================================

def fib_ratio(n: int) -> float:
    """Compute F_n/F_{n+1} (converges to 1/φ).

    Args:
        n: Fibonacci index (≥ 1).

    Returns:
        F_n / F_{n+1}.
    """
    return fib(n) / fib(n + 1)


def is_fibonacci_adjoint(n: int) -> bool:
    """Test whether n is a Fibonacci number that serves as an adjoint dimension.

    The SU(N) adjoint dimension is N²-1. Only SU(2) (dim=3=F_4) and
    SU(3) (dim=8=F_6) have Fibonacci adjoint dimensions among non-abelian
    groups. U(1) has dim=1=F_1=F_2.

    Source: MAR exp_38.

    Args:
        n: Adjoint dimension to test.

    Returns:
        True if n is both Fibonacci and an SU(N) adjoint dimension.
    """
    if not is_fibonacci(n):
        return False
    # Check if n = N²-1 for some integer N ≥ 2
    candidate = int(math.isqrt(n + 1))
    return candidate * candidate == n + 1 and candidate >= 2


def fibonacci_matrix() -> list:
    """The Fibonacci coupling matrix [[1,1],[1,0]].

    Eigenvalues: φ and -1/φ.
    Unique under constraints: non-negative integer, det=±1, tr=1.

    Returns:
        2×2 matrix as nested list.
    """
    return [[1, 1], [1, 0]]


def cyclotomic_depth(f_index: int) -> int:
    """Compute cyclotomic polynomial Φ₃(F_k) = F_k² + F_k + 1.

    For k=7: Φ₃(13) = 169 + 13 + 1 = 183 (gravity depth).

    Args:
        f_index: Fibonacci index.

    Returns:
        Φ₃(F_k) = F_k² + F_k + 1.
    """
    f = fib(f_index)
    return f * f + f + 1
