"""
Unified correction template engine for Dawn Field Theory.

The PAC correction template 1 ± F_a/(nπF_b²) generates all known
force couplings from Fibonacci arithmetic. This module provides the
engine for constructing, evaluating, and searching correction factors.

Usage:
    from fracton.corrections import correction, search_corrections
    from fracton.corrections import em_correction, gravity_correction
"""

from .template import (
    correction,
    correction_factor,
    build_correction,
    CorrectionTemplate,
)
from .forces import (
    em_correction,
    gravity_correction,
    dark_energy_correction,
    strong_correction_c2,
    strong_correction_c3,
)
from .search import search_corrections

__all__ = [
    "correction", "correction_factor", "build_correction",
    "CorrectionTemplate",
    "em_correction", "gravity_correction", "dark_energy_correction",
    "strong_correction_c2", "strong_correction_c3",
    "search_corrections",
]
