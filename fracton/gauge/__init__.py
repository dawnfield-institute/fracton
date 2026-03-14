"""
Gauge group structure analysis for Dawn Field Theory.

The Standard Model gauge groups U(1) × SU(2) × SU(3) have a remarkable
property: their adjoint dimensions (1, 3, 8) are ALL Fibonacci numbers,
and 1 + 3 + 8 + 1(Higgs) = 13 = F₇. This module provides tools for
analyzing this Fibonacci closure.

Usage:
    from fracton.gauge import gauge_closure_sum, adjoint_dim
    from fracton.gauge import is_fibonacci_adjoint, weinberg_angle
"""

from .groups import (
    adjoint_dim,
    gauge_closure_sum,
    is_fibonacci_gauge,
    STANDARD_MODEL_GAUGE,
)
from .weinberg import (
    weinberg_angle_pac,
    mw_mz_ratio_pac,
    running_sin2_theta,
)

__all__ = [
    "adjoint_dim", "gauge_closure_sum", "is_fibonacci_gauge",
    "STANDARD_MODEL_GAUGE",
    "weinberg_angle_pac", "mw_mz_ratio_pac", "running_sin2_theta",
]
