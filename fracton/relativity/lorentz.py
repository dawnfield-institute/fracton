"""
Lorentz factor from PAC multiplicative asymmetry.

The Lorentz factor gamma = 1/sqrt(1 - beta^2) arises from the
multiplicative asymmetry between forward and reverse cascade paths.
No spacetime geometry is assumed — it's pure information theory.

At cascade fraction f = ln(2):
    deficit = f^2 = ln^2(2) = 1 - xi_floor
    This is the per-step multiplicative deficit that generates all of GR.

Source: MAR exp_28 (statistical relativity).
"""

import math

from ..constants.mathematical import XI_FLOOR, LN2_SQUARED


def lorentz_gamma(beta: float) -> float:
    """Compute Lorentz factor from velocity parameter.

    gamma = 1/sqrt(1 - beta^2)

    In PAC terms, beta is the cascade asymmetry parameter — the fraction
    of forward vs reverse path utilization.

    Args:
        beta: Velocity as fraction of c (0 <= beta < 1).

    Returns:
        Lorentz gamma factor (>= 1).

    Raises:
        ValueError: If beta >= 1 or beta < 0.
    """
    if beta < 0 or beta >= 1:
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    return 1.0 / math.sqrt(1 - beta * beta)


def time_dilation_factor(beta: float) -> float:
    """Compute time dilation: tau/t = 1/gamma = sqrt(1 - beta^2).

    This is the proper time ratio — a moving clock runs slower.

    Args:
        beta: Velocity as fraction of c.

    Returns:
        Time dilation factor (0 < factor <= 1).
    """
    if beta < 0 or beta >= 1:
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    return math.sqrt(1 - beta * beta)


def cascade_time_dilation(depth: int) -> float:
    """Compute time dilation from cascade depth.

    tau_local(d) = xi_floor^d = (1 - ln^2(2))^d

    Each cascade step accumulates a multiplicative deficit of ln^2(2).
    After d steps, the local time is xi_floor^d of the global time.

    Args:
        depth: Number of cascade steps.

    Returns:
        Time dilation factor (< 1 for depth > 0).
    """
    return XI_FLOOR ** depth
