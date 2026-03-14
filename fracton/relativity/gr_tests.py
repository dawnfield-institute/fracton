"""
Classical GR test predictions from the Schwarzschild metric.

All three classical tests (Mercury precession, light deflection,
Shapiro delay) follow from the PAC-derived metric.

Source: MAR exp_30 Part D.
"""

import math


def mercury_precession(
    a_m: float = 5.791e10,
    e: float = 0.2056,
    r_s: float = 2953.25,
    t_orbit_s: float = 7.6005e6,
) -> float:
    """Compute Mercury's orbital precession in arcseconds per century.

    delta_phi = 6*pi*G*M_sun / (c^2 * a * (1-e^2))
              = 3*pi*r_s / (a*(1-e^2))  per orbit

    GR prediction: 42.98 arcsec/century.

    Args:
        a_m: Semi-major axis in meters (default: Mercury).
        e: Orbital eccentricity (default: Mercury).
        r_s: Schwarzschild radius of the Sun in meters.
        t_orbit_s: Orbital period in seconds (default: Mercury).

    Returns:
        Precession in arcseconds per century.
    """
    per_orbit_rad = 3 * math.pi * r_s / (a_m * (1 - e ** 2))

    century_s = 100 * 365.25 * 86400
    orbits_per_century = century_s / t_orbit_s

    arcsec_per_rad = 180 * 3600 / math.pi
    return per_orbit_rad * orbits_per_century * arcsec_per_rad


def light_deflection(
    r_min: float = 6.957e8,
    r_s: float = 2953.25,
) -> float:
    """Compute gravitational light deflection angle.

    delta_theta = 4*G*M / (c^2 * r_min) = 2*r_s / r_min

    At the solar limb: 1.7505 arcseconds.

    Args:
        r_min: Closest approach distance in meters (default: solar radius).
        r_s: Schwarzschild radius in meters.

    Returns:
        Deflection angle in arcseconds.
    """
    delta_rad = 2 * r_s / r_min
    arcsec_per_rad = 180 * 3600 / math.pi
    return delta_rad * arcsec_per_rad


def shapiro_delay(
    r_e: float,
    r_r: float,
    r_0: float,
    r_s: float = 2953.25,
    c: float = 299_792_458.0,
) -> float:
    """Compute Shapiro time delay for a signal passing near a mass.

    delta_t = (r_s/c) * ln(4*r_e*r_r / r_0^2)

    Args:
        r_e: Distance from mass to emitter (meters).
        r_r: Distance from mass to receiver (meters).
        r_0: Closest approach distance (meters).
        r_s: Schwarzschild radius (meters).
        c: Speed of light (m/s).

    Returns:
        Time delay in seconds.
    """
    return (r_s / c) * math.log(4 * r_e * r_r / (r_0 ** 2))
