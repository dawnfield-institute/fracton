"""
Cosmological calculations from Dawn Field Theory.

Provides PAC-derived cosmological parameters: dark energy fraction,
cosmological constant from tiling, and vacuum energy density.

Usage:
    from fracton.cosmology import omega_lambda_pac, cc_tiling_log10
    from fracton.cosmology import vacuum_energy_ratio
"""

from .dark_energy import (
    omega_lambda_pac,
    cc_tiling_log10,
    vacuum_energy_ratio,
    cc_gap_orders,
)

__all__ = [
    "omega_lambda_pac", "cc_tiling_log10",
    "vacuum_energy_ratio", "cc_gap_orders",
]
