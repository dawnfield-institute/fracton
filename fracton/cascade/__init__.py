"""
Energy cascade engine for Dawn Field Theory.

Models PAC energy cascades via eigenvalue-based partitioning.
The cascade is the fundamental mechanism: energy flows from injection
scale through organized modes, with Landauer erasure (kT·ln2) as
the dissipation floor.

Usage:
    from fracton.cascade import energy_cascade, measure_exponent
    from fracton.cascade import coupling_matrix, participation_ratio
"""

from .engine import energy_cascade, coupling_matrix, participation_ratio
from .analysis import measure_exponent, cascade_summary

__all__ = [
    "energy_cascade", "coupling_matrix", "participation_ratio",
    "measure_exponent", "cascade_summary",
]
