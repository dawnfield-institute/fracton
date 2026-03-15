"""
Feigenbaum-Mobius validation suite.

Structural constant derivations, eigenvalue proofs, and cross-map
universality checks. Promoted from core/feigenbaum_mobius.py.
"""

from __future__ import annotations

import math
from typing import Dict, Any, Tuple

from ..constants.mathematical import PHI, PHI_INV
from ..fibonacci import fib
from .constants import (
    DELTA_LOGISTIC, R_INF_LOGISTIC, R_INF_SINE,
    STRUCTURAL_39, STRUCTURAL_160, STRUCTURAL_1371, STRUCTURAL_1857,
    UNIVERSAL_DELTA_Z,
)
from .fibonacci_mobius import FibonacciMobius, M10
from .mobius import compute_delta


def derive_structural_constants() -> Dict[str, Any]:
    """Derive 39, 160, 1371, 1857 from the 4-5 pattern.

    The 4-5 pattern:
        4 = period-doubling cascade (1->2->4->8->chaos)
        5 = pentagon/Fibonacci (F_5 = 5)
        20 = 4 * 5 = complete cycle
    """
    derivations = {}

    # 39 = (5^4 - 1) / 4^2
    derivations["39"] = {
        "formula": "(5^4 - 1) / 4^2",
        "value": (5**4 - 1) // 4**2,
        "matches": (5**4 - 1) // 4**2 == STRUCTURAL_39,
    }

    # 160 = 4^2 * 2 * 5
    derivations["160"] = {
        "formula": "4^2 * 2 * 5",
        "value": 4**2 * 2 * 5,
        "matches": 4**2 * 2 * 5 == STRUCTURAL_160,
    }

    # 1371 = F_10 * 5^2 - 4
    f10 = fib(10)
    derivations["1371"] = {
        "formula": "F_10 * 5^2 - 4",
        "value": f10 * 25 - 4,
        "matches": f10 * 25 - 4 == STRUCTURAL_1371,
    }

    # 1857 = F_10 * F_9 - F_7
    f9, f7 = fib(9), fib(7)
    derivations["1857"] = {
        "formula": "F_10 * F_9 - F_7",
        "value": f10 * f9 - f7,
        "matches": f10 * f9 - f7 == STRUCTURAL_1857,
    }

    # phi^20 ~ L_20 (20th Lucas number)
    derivations["phi_20"] = {
        "formula": "phi^20 ~ L_20",
        "value": PHI**20,
        "lucas_20": 15127,
        "difference": abs(PHI**20 - 15127),
    }

    return derivations


def prove_eigenvalue_identity() -> Dict[str, Any]:
    """Algebraically prove: 89 - 55*phi = 1/phi^10.

    Proof:
        x * phi^10 = (89 - 55*phi) * phi^10
                   = 89*(55*phi + 34) - 55*(89*phi + 55)
                   = 4895*phi + 3026 - 4895*phi - 3025
                   = 1
        Therefore x = 1/phi^10.
    """
    lhs = 89 - 55 * PHI
    rhs = PHI_INV**10

    phi_10 = fib(10) * PHI + fib(9)
    phi_11 = fib(11) * PHI + fib(10)
    product = 89 * phi_10 - 55 * phi_11

    return {
        "identity": "89 - 55*phi = 1/phi^10",
        "lhs": lhs,
        "rhs": rhs,
        "difference": abs(lhs - rhs),
        "is_exact": abs(lhs - rhs) < 1e-14,
        "proof_product": product,
        "proof_verified": abs(product - 1) < 1e-10,
    }


def validate_universality() -> Dict[str, Any]:
    """Validate that delta_z is universal across quadratic-max maps.

    Tests logistic map (scale=pi) and sine map (scale=pi/4).
    """
    results = {}

    # Logistic map
    z_logistic = -PHI_INV + UNIVERSAL_DELTA_Z
    r_inf_logistic = math.pi * M10(z_logistic)
    results["logistic"] = {
        "predicted": r_inf_logistic,
        "known": R_INF_LOGISTIC,
        "error": abs(r_inf_logistic - R_INF_LOGISTIC),
        "error_percent": abs(r_inf_logistic - R_INF_LOGISTIC) / R_INF_LOGISTIC * 100,
    }

    # Sine map
    z_sine = -PHI_INV + UNIVERSAL_DELTA_Z
    r_inf_sine = (math.pi / 4) * M10(z_sine)
    results["sine"] = {
        "predicted": r_inf_sine,
        "known": R_INF_SINE,
        "error": abs(r_inf_sine - R_INF_SINE),
        "error_percent": abs(r_inf_sine - R_INF_SINE) / R_INF_SINE * 100,
    }

    results["scale_ratio"] = {
        "observed": R_INF_LOGISTIC / R_INF_SINE,
        "expected": 4.0,
        "error_percent": abs(R_INF_LOGISTIC / R_INF_SINE - 4.0) / 4.0 * 100,
    }

    results["universal_delta_z"] = UNIVERSAL_DELTA_Z
    return results
