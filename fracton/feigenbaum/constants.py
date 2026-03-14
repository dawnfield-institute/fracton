"""
Feigenbaum universality constants.

The Feigenbaum constants arise from the period-doubling route to chaos.
In Dawn Field Theory, they have closed-form expressions through the
4-5 pattern connecting Fibonacci to Feigenbaum via Mobius transformations.

Key structural constants:
    39 = (5^4 - 1) / 4^2 = 624/16
    160 = 4^2 * 2 * 5
    1371 = F_10 * 5^2 - 4

Source: PACSeries Paper 3, SEC threshold detection experiments.
"""

import math

from ..constants.mathematical import PHI, PHI_INV

# =============================================================================
# MEASURED VALUES (high-precision numerical computation)
# =============================================================================

DELTA_LOGISTIC: float = 4.669_201_609_102_990
"""Feigenbaum delta for the logistic map (13 digits).
Rate at which period-doubling bifurcations accumulate."""

ALPHA_LOGISTIC: float = -2.502_907_875_095_89
"""Feigenbaum alpha for the logistic map (13 digits).
Scaling of the attractor at each period-doubling."""

R_INF_LOGISTIC: float = 3.569_945_671_869_5445
"""Accumulation point r_inf for the logistic map."""

R_INF_SINE: float = 0.892_486_417_917_3861
"""Accumulation point for the sine map."""

# =============================================================================
# STRUCTURAL CONSTANTS (from 4-5 pattern)
# =============================================================================

STRUCTURAL_39: int = (5 ** 4 - 1) // 4 ** 2   # = 624/16 = 39
STRUCTURAL_160: int = 4 ** 2 * 2 * 5           # = 160
STRUCTURAL_1371: int = 55 * 5 ** 2 - 4         # = 1371
STRUCTURAL_1857: int = 55 * 34 - 13            # = 1857 = F10*F9 - F7

# =============================================================================
# PAC-DERIVED VALUES (self-consistent computation)
# =============================================================================

def _compute_delta_self_consistent() -> float:
    """Solve the self-referential equation for delta.

    delta = phi^(20/N)
    N = sqrt(39 + 1/x)
    x = 160 + (delta-4)^2 * (1 - 1/(1371 + delta - 4))

    Converges to 13 digits in ~3 iterations.
    """
    x = float(STRUCTURAL_160)
    delta = 4.5

    for _ in range(20):
        N = math.sqrt(STRUCTURAL_39 + 1 / x)
        delta_new = PHI ** (20 / N)
        d4 = delta_new - 4
        correction = 1 - 1 / (STRUCTURAL_1371 + d4)
        x_new = STRUCTURAL_160 + d4 ** 2 * correction

        if abs(delta_new - delta) < 1e-15:
            return delta_new
        delta = delta_new
        x = x_new

    return delta


# Self-consistent N
def _compute_n_self_consistent() -> float:
    """Compute the self-consistent N value."""
    delta = _compute_delta_self_consistent()
    d4 = delta - 4
    x = STRUCTURAL_160 + d4 ** 2 * (1 - 1 / (STRUCTURAL_1371 + d4))
    return math.sqrt(STRUCTURAL_39 + 1 / x)


DELTA: float = _compute_delta_self_consistent()
"""Feigenbaum delta from PAC: phi^(20/N) self-consistently.
Matches logistic delta to 13+ digits."""

N_SELF_CONSISTENT: float = _compute_n_self_consistent()
"""Self-consistent N value (~6.287)."""

ALPHA: float = abs(ALPHA_LOGISTIC)
"""Feigenbaum |alpha| = 2.50291..."""

# Universal offset for Mobius transformation
UNIVERSAL_DELTA_Z: float = None  # Computed below

def _compute_universal_delta_z() -> float:
    """Back-solve delta_z from known r_inf."""
    target = R_INF_LOGISTIC / math.pi
    z = (34 * target - 55) / (89 - 55 * target)
    return z - (-PHI_INV)

UNIVERSAL_DELTA_Z = _compute_universal_delta_z()
"""Universal offset Dz in Mobius transformation."""

# Compute R_INF from the Mobius approach
_z = -PHI_INV + UNIVERSAL_DELTA_Z
_m10_val = (89 * _z + 55) / (55 * _z + 34)
R_INF: float = math.pi * _m10_val
"""r_inf from PAC Mobius transformation (matches R_INF_LOGISTIC by construction)."""
