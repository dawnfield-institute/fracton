"""
Dark energy and cosmological constant from PAC/tiling.

Ω_Λ = (1/φ)(1 + F₉/(4πF₅²)) = 0.6849 (0.012% error)

log₁₀(ρ_Λ/ρ_Pl) = 2 × 183 × Ξ × log₁₀(ln²(2))
    With Ξ: -123.32 (0.38 orders gap)
    With ξ_PAC: -123.17 (0.22 orders gap)
    Measured: -122.95

Source: MAR exp_35/36, confirmed by b-hierarchy (exp_39/40).
"""

import math
from typing import Optional

from ..constants.mathematical import (
    PHI_INV, LN2_SQUARED, FIBONACCI_GRAVITY_DEPTH,
    XI_ANALYTIC, XI_PAC,
)
from ..corrections import correction_factor


def omega_lambda_pac() -> float:
    """Compute dark energy fraction from PAC correction template.

    Ω_Λ = (1/φ) × (1 + F₉/(4π·F₅²))

    Returns:
        Ω_Λ (dimensionless fraction).
    """
    correction = correction_factor(9, 5, 4, +1)
    return PHI_INV * correction


def cc_tiling_log10(tiling_factor: Optional[float] = None) -> float:
    """Compute log₁₀(ρ_Λ/ρ_Pl) from local-global tiling.

    log₁₀(ρ_Λ/ρ_Pl) = 2 × 183 × tiling_factor × log₁₀(ln²(2))

    The 183 is the gravity depth (F₇² + F₇ + 1 = 169 + 13 + 1).
    The factor of 2 comes from the forward-reverse (bifractal) structure.
    ln²(2) is the per-step multiplicative deficit.

    Args:
        tiling_factor: Balance constant. Default: Ξ = γ + ln(φ) = 1.0584.
            Use XI_PAC = 1.0571 for the cascade variant (0.22 orders gap).

    Returns:
        log₁₀(ρ_Λ/ρ_Pl), expected ≈ -123.
    """
    if tiling_factor is None:
        tiling_factor = XI_ANALYTIC

    n_eff = 2 * FIBONACCI_GRAVITY_DEPTH * tiling_factor
    return n_eff * math.log10(LN2_SQUARED)


def vacuum_energy_ratio(tiling_factor: Optional[float] = None) -> float:
    """Compute ρ_Λ/ρ_Pl directly (the cosmological constant ratio).

    Args:
        tiling_factor: Balance constant (default: Ξ_analytic).

    Returns:
        ρ_Λ/ρ_Pl as a float (extremely small, ~10⁻¹²³).
    """
    log_val = cc_tiling_log10(tiling_factor)
    return 10 ** log_val


def cc_gap_orders(
    tiling_factor: Optional[float] = None,
    measured_log10: float = -122.95,
) -> float:
    """Compute the gap between PAC prediction and measured CC in orders.

    Args:
        tiling_factor: Balance constant (default: Ξ_analytic).
        measured_log10: Measured log₁₀(ρ_Λ/ρ_Pl). Default: -122.95.

    Returns:
        Gap in orders of magnitude (positive = predicted too negative).
    """
    predicted = cc_tiling_log10(tiling_factor)
    return abs(predicted - measured_log10)
