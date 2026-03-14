"""
Cascade analysis utilities.

Extract spectral exponents and summary statistics from cascade results.
"""

import math
from typing import Any, Dict, List, Tuple

from ..statistics.spectral import spectral_exponent


def measure_exponent(
    cascade_results: List[Dict[str, Any]],
    trim: int = 2,
) -> Tuple[float, float, float, float]:
    """Extract spectral exponent from cascade engine output.

    Fits E(k) ~ k^slope via log-log linear regression on the inertial range.

    Args:
        cascade_results: List of dicts from energy_cascade().
        trim: Points to trim from each end.

    Returns:
        Tuple of (slope, r_squared, avg_org_fraction, std_error).
    """
    alive = [r for r in cascade_results if r.get("alive", True)]
    wavenumbers = [r["wavenumber"] for r in alive]
    energies = [r["E_organized"] for r in alive]

    result = spectral_exponent(wavenumbers, energies, trim=trim)

    org_fracs = [r.get("org_fraction", 0) for r in alive]
    avg_org = sum(org_fracs) / len(org_fracs) if org_fracs else 0.0

    n = len(wavenumbers)
    std_error = math.sqrt((1 - result["r_squared"]) / max(n - 2, 1)) if n > 2 else 0.0

    return (result["slope"], result["r_squared"], avg_org, std_error)


def cascade_summary(cascade_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics for a cascade run.

    Args:
        cascade_results: List of dicts from energy_cascade().

    Returns:
        Dict with total_scales, alive_scales, injection_energy,
        final_energy, total_dissipation, avg_org_fraction,
        max_org_fraction, min_org_fraction.
    """
    alive = [r for r in cascade_results if r.get("alive", True)]
    org_fracs = [r["org_fraction"] for r in cascade_results]

    return {
        "total_scales": len(cascade_results),
        "alive_scales": len(alive),
        "injection_energy": cascade_results[0]["P_input"] if cascade_results else 0.0,
        "final_energy": cascade_results[-1]["E_transfer"] if cascade_results else 0.0,
        "total_dissipation": (
            cascade_results[0]["P_input"] - cascade_results[-1]["E_transfer"]
            if cascade_results else 0.0
        ),
        "avg_org_fraction": sum(org_fracs) / len(org_fracs) if org_fracs else 0.0,
        "max_org_fraction": max(org_fracs) if org_fracs else 0.0,
        "min_org_fraction": min(org_fracs) if org_fracs else 0.0,
    }
