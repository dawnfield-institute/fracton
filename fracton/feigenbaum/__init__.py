"""
Feigenbaum universality constants from PAC/Fibonacci structure.

Provides closed-form Feigenbaum constants derived from the 4-5 pattern
(period-doubling × pentagon symmetry) discovered in PACSeries Paper 3.

Key results:
    delta = phi^(20/N) where N = sqrt(39 + 1/x) self-consistently
    r_inf = pi * M_10(-1/phi + Dz) where Dz is universal
    alpha = derived from delta via structural constants

Usage:
    from fracton.feigenbaum import DELTA, ALPHA, R_INF
    from fracton.feigenbaum import compute_delta, mobius_m10
"""

from .constants import (
    DELTA, ALPHA, R_INF,
    DELTA_LOGISTIC, ALPHA_LOGISTIC,
    R_INF_LOGISTIC, R_INF_SINE,
    UNIVERSAL_DELTA_Z,
    N_SELF_CONSISTENT,
)
from .mobius import (
    mobius_m10,
    compute_delta,
    compute_r_inf,
    compute_universal_delta_z,
)

__all__ = [
    "DELTA", "ALPHA", "R_INF",
    "DELTA_LOGISTIC", "ALPHA_LOGISTIC",
    "R_INF_LOGISTIC", "R_INF_SINE",
    "UNIVERSAL_DELTA_Z", "N_SELF_CONSISTENT",
    "mobius_m10", "compute_delta",
    "compute_r_inf", "compute_universal_delta_z",
]
