"""
Spectral analysis utilities for cascade experiments.

Provides spectral exponent fitting from energy cascade data.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple, Any


def spectral_exponent(
    wavenumbers: Sequence[float],
    energies: Sequence[float],
    trim: int = 2,
) -> Dict[str, float]:
    """Fit spectral exponent E(k) ~ k^slope via log-log linear regression.

    Trims the injection and dissipation ranges to focus on the inertial range.

    Args:
        wavenumbers: Wavenumber values (k).
        energies: Energy values E(k).
        trim: Number of points to trim from each end.

    Returns:
        Dict with slope, r_squared, intercept.

    Examples:
        >>> ks = [1, 2, 4, 8, 16, 32, 64, 128]
        >>> es = [k**(-5/3) for k in ks]
        >>> result = spectral_exponent(ks, es, trim=1)
        >>> abs(result["slope"] - (-5/3)) < 0.01
        True
    """
    # Filter positive values and apply trim
    pairs = [(k, e) for k, e in zip(wavenumbers, energies) if k > 0 and e > 0]
    if len(pairs) <= 2 * trim:
        return {"slope": 0.0, "r_squared": 0.0, "intercept": 0.0}

    pairs = pairs[trim:-trim] if trim > 0 else pairs

    log_k = [math.log(k) for k, _ in pairs]
    log_e = [math.log(e) for _, e in pairs]
    n = len(log_k)

    # Linear regression in log space
    sum_x = sum(log_k)
    sum_y = sum(log_e)
    sum_xy = sum(x * y for x, y in zip(log_k, log_e))
    sum_x2 = sum(x * x for x in log_k)

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-30:
        return {"slope": 0.0, "r_squared": 0.0, "intercept": 0.0}

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R-squared
    mean_y = sum_y / n
    ss_tot = sum((y - mean_y) ** 2 for y in log_e)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(log_k, log_e))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "slope": slope,
        "r_squared": r_squared,
        "intercept": intercept,
    }


def measure_exponent(
    cascade_results: List[Dict[str, Any]],
    trim: int = 2,
) -> Tuple[float, float, float, float]:
    """Extract spectral exponent from cascade engine output.

    Args:
        cascade_results: List of dicts with "wavenumber" and "E_organized" keys
            (as returned by the cascade engine).
        trim: Points to trim from each end.

    Returns:
        Tuple of (slope, r_squared, avg_org_fraction, std_error).
    """
    wavenumbers = [r["wavenumber"] for r in cascade_results if r.get("alive", True)]
    energies = [r["E_organized"] for r in cascade_results if r.get("alive", True)]

    result = spectral_exponent(wavenumbers, energies, trim=trim)

    org_fracs = [r.get("org_fraction", 0) for r in cascade_results if r.get("alive", True)]
    avg_org = sum(org_fracs) / len(org_fracs) if org_fracs else 0.0

    # Rough std error estimate from residuals
    n = len(wavenumbers)
    std_error = math.sqrt((1 - result["r_squared"]) / max(n - 2, 1)) if n > 2 else 0.0

    return (result["slope"], result["r_squared"], avg_org, std_error)
