"""
Weinberg angle and electroweak mixing from Fibonacci arithmetic.

sin²θ_W = F₄/F₇ = 3/13 = 0.230769... (exact at M_W)
M_W/M_Z = √(10/13) (PAC-derived)

Source: PACSeries Paper 4, MAR exp_38.
"""

import math

from ..fibonacci import fib
from ..constants.mathematical import PHI


def weinberg_angle_pac() -> float:
    """Compute sin²θ_W from Fibonacci ratio.

    sin²θ_W = F₄/F₇ = 3/13 = 0.230769...

    This is exact at Q ≈ M_W (the actualization threshold energy).
    At M_Z, RG running gives 0.23121 (0.19% shift).

    Returns:
        sin²θ_W (dimensionless).
    """
    return fib(4) / fib(7)


def mw_mz_ratio_pac() -> float:
    """Compute M_W/M_Z from PAC electroweak structure.

    M_W/M_Z = √(1 - sin²θ_W) = √(10/13) = 0.87706...

    Returns:
        M_W/M_Z ratio.
    """
    sin2 = weinberg_angle_pac()
    return math.sqrt(1 - sin2)


def running_sin2_theta(q_gev: float, sin2_mw: float = None) -> float:
    """Approximate RG running of sin²θ_W from M_W to scale Q.

    Uses one-loop SM beta function:
        sin²θ(Q) ≈ sin²θ(M_W) × (1 + (19/48π²) × ln(Q/M_W))

    This is a rough approximation — proper running requires full
    two-loop RGE with threshold corrections.

    Args:
        q_gev: Energy scale in GeV.
        sin2_mw: sin²θ_W at M_W (default: 3/13).

    Returns:
        Approximate sin²θ_W at scale Q.
    """
    if sin2_mw is None:
        sin2_mw = weinberg_angle_pac()

    m_w = 80.377  # GeV
    if q_gev <= 0:
        return sin2_mw

    # One-loop running coefficient (approximate)
    beta_coeff = 19 / (48 * math.pi ** 2)
    log_ratio = math.log(q_gev / m_w)

    return sin2_mw * (1 + beta_coeff * log_ratio)
