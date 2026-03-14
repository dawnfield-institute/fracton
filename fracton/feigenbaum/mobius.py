"""
Mobius transformation approach to Feigenbaum constants.

M_10(z) = (F_11*z + F_10) / (F_10*z + F_9) = (89z + 55)/(55z + 34)

This Mobius transformation connects Fibonacci structure to Feigenbaum
universality. Key properties:
    - Fixed points: phi (stable) and -1/phi (unstable)
    - M_10(-1/phi) = -1/phi (fixed point property)
    - Eigenvalue at -1/phi: phi^20 (from derivative)

Source: SEC threshold detection exp_28, PACSeries Paper 3.
"""

import math

from ..constants.mathematical import PHI, PHI_INV
from ..fibonacci import fib


def mobius_m10(z: float) -> float:
    """Evaluate the 10th Fibonacci Mobius transformation.

    M_10(z) = (F_11*z + F_10) / (F_10*z + F_9)
            = (89z + 55) / (55z + 34)

    Fixed points: phi and -1/phi.

    Args:
        z: Parameter value.

    Returns:
        M_10(z).
    """
    f11 = fib(11)  # 89
    f10 = fib(10)  # 55
    f9 = fib(9)    # 34
    return (f11 * z + f10) / (f10 * z + f9)


def compute_delta(max_iter: int = 20, tol: float = 1e-15) -> float:
    """Compute Feigenbaum delta from the self-referential PAC formula.

    delta = phi^(20/N)
    N = sqrt(39 + 1/x)
    x = 160 + (delta-4)^2 * (1 - 1/(1371 + delta - 4))

    Converges to 13 digits in ~3 iterations.

    Args:
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Feigenbaum delta.
    """
    x = 160.0
    delta = 4.5

    for _ in range(max_iter):
        N = math.sqrt(39 + 1 / x)
        delta_new = PHI ** (20 / N)
        d4 = delta_new - 4
        correction = 1 - 1 / (1371 + d4)
        x_new = 160 + d4 ** 2 * correction

        if abs(delta_new - delta) < tol:
            return delta_new
        delta = delta_new
        x = x_new

    return delta


def compute_r_inf(delta_z: float = None) -> float:
    """Compute accumulation point r_inf from Mobius transformation.

    r_inf = pi * M_10(-1/phi + delta_z)

    Args:
        delta_z: Universal offset. If None, computed from known r_inf.

    Returns:
        r_inf for the logistic map.
    """
    if delta_z is None:
        delta_z = compute_universal_delta_z()
    z = -PHI_INV + delta_z
    return math.pi * mobius_m10(z)


def compute_universal_delta_z(target_r_inf: float = 3.5699456718695445) -> float:
    """Find delta_z that reproduces a target r_inf.

    Solves: pi * M_10(-1/phi + dz) = target_r_inf

    Args:
        target_r_inf: Target accumulation point.

    Returns:
        delta_z value.
    """
    target_m10 = target_r_inf / math.pi
    f11, f10, f9 = fib(11), fib(10), fib(9)
    z = (f9 * target_m10 - f10) / (f11 - f10 * target_m10)
    return z - (-PHI_INV)
