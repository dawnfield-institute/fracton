"""
Schwarzschild metric components from PAC cascade density.

The cascade density falls as rho_c(r) ~ r_s/r near a mass,
which produces the Schwarzschild metric components:
    g_tt = -(1 - r_s/r)    (temporal compression)
    g_rr = 1/(1 - r_s/r)   (radial stretching)

The product g_tt × g_rr = -1 is enforced by local c invariance.

Source: MAR exp_30 (cascade general relativity).
"""

import math


def schwarzschild_radius(mass_kg: float, g: float = 6.674e-11, c: float = 299_792_458.0) -> float:
    """Compute the Schwarzschild radius r_s = 2GM/c².

    Args:
        mass_kg: Mass in kilograms.
        g: Gravitational constant (default: CODATA 2018).
        c: Speed of light (default: exact).

    Returns:
        Schwarzschild radius in meters.

    Examples:
        >>> schwarzschild_radius(1.989e30)  # Sun
        2953.25...
    """
    return 2 * g * mass_kg / (c * c)


def schwarzschild_g_tt(r: float, r_s: float) -> float:
    """Temporal metric component g_tt = -(1 - r_s/r).

    From PAC: the cascade time deficit at distance r from mass.
    Approaches 0 at the horizon (r → r_s) and -1 at infinity.

    Args:
        r: Radial coordinate.
        r_s: Schwarzschild radius.

    Returns:
        g_tt (negative for r > r_s).
    """
    return -(1 - r_s / r)


def schwarzschild_g_rr(r: float, r_s: float) -> float:
    """Radial metric component g_rr = 1/(1 - r_s/r).

    From PAC: local c invariance forces g_rr = -1/g_tt.
    Diverges at the horizon (r → r_s) and approaches 1 at infinity.

    Args:
        r: Radial coordinate.
        r_s: Schwarzschild radius.

    Returns:
        g_rr (positive for r > r_s).
    """
    return 1 / (1 - r_s / r)


def coordinate_speed_of_light(r: float, r_s: float) -> float:
    """Coordinate speed of light dr/dt = (1 - r_s/r).

    In PAC terms: both forward and reverse cascades slow by the same
    factor near a mass, so local speed is always c = 1 but coordinate
    speed drops.

    Args:
        r: Radial coordinate.
        r_s: Schwarzschild radius.

    Returns:
        dr/dt in units of c (1 at infinity, 0 at horizon).
    """
    return 1 - r_s / r
