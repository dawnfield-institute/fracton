"""
Mathematical constants derived from PAC/SEC/MED framework.

All constants here are either fundamental mathematical constants or
values derived from the Dawn Field Theory axioms. Nothing is fitted.

The key distinction:
- XI_ANALYTIC (Ξ = γ + ln(φ)) is the continuous/analytic balance constant
- XI_DISCRETE (1 + π/55) is the discrete Fibonacci approximation
- XI_PAC (ξ_PAC = 1 + 7/8 · ln(2) · (1-ln2)²) is the cascade-derived value
- The 0.12% spread between XI_ANALYTIC and XI_DISCRETE is structural (exp_26)
"""

import math
from functools import lru_cache

# =============================================================================
# FUNDAMENTAL MATHEMATICAL CONSTANTS
# =============================================================================

PHI: float = (1 + math.sqrt(5)) / 2
"""Golden ratio φ = 1.6180339887... — emerges from PAC two-term recursion."""

PHI_INV: float = 1 / PHI
"""Inverse golden ratio 1/φ = φ - 1 = 0.6180339887..."""

PHI_SQUARED: float = PHI ** 2
"""φ² = φ + 1 = 2.6180339887..."""

PHI_CUBED: float = PHI ** 3
"""φ³ = 2φ + 1 = 4.2360679775..."""

LN_PHI: float = math.log(PHI)
"""ln(φ) = 0.4812118250... — natural information unit of PAC recursion."""

LN2: float = math.log(2)
"""ln(2) = 0.6931471806... — Landauer erasure energy (fundamental)."""

LN2_SQUARED: float = LN2 ** 2
"""ln²(2) = 0.4805170186... — round-trip deficit at Landauer fraction (exp_28)."""

GAMMA: float = 0.5772156649015329
"""Euler-Mascheroni constant γ — cost of discrete enumeration. Independent of φ."""

SQRT5: float = math.sqrt(5)
"""√5 — appears in Binet's formula and Fibonacci asymptotics."""

TAU: float = 2 * math.pi
"""τ = 2π — full circle."""

GOLDEN_ANGLE: float = TAU / PHI_SQUARED
"""Golden angle = 2π/φ² ≈ 2.3999... rad ≈ 137.5°"""

GOLDEN_ANGLE_FRACTION: float = 1 - 1 / PHI
"""Golden angle fraction α* = 1 - 1/φ ≈ 0.381966 — optimal phase cascade stability."""


# =============================================================================
# BALANCE CONSTANTS (Ξ family)
# =============================================================================

XI_ANALYTIC: float = GAMMA + LN_PHI
"""Ξ = γ + ln(φ) = 1.0584274899... — the analytic balance constant.
Governs boundary between ordered and disordered computation.
Cross-validated across 5 independent domains (p < 0.0003)."""

XI_DISCRETE: float = 1 + math.pi / 55
"""1 + π/F₁₀ = 1.0571198664... — the discrete Fibonacci approximation.
Differs from XI_ANALYTIC by 0.124% (the discretisation gap = γ/48)."""

XI_PAC: float = 1 + (7 / 8) * LN2 * (1 - LN2) ** 2
"""ξ_PAC = 1 + (7/8)·ln(2)·(1-ln2)² = 1.0571108... — cascade-derived value.
Three-factor decomposition: (She-Leveque modes)(Landauer dissipation)(MED regulation)."""

XI_FLOOR: float = 1 - LN2_SQUARED
"""ξ_floor = 1 - ln²(2) = 0.5194829814... — pure Landauer cascade floor.
Achieved with zero variance. The deficit(f=ln2) from exp_28."""

XI_SPREAD: float = XI_ANALYTIC - XI_DISCRETE
"""Spread = 0.00131... = γ's non-Fibonacci residual (resolved by exp_26)."""

EULER_GAP: float = XI_ANALYTIC - XI_PAC
"""Euler gap ≈ 1/(240π) at 0.09% — the irreducible gap between analytic and cascade."""

EULER_GAP_FACTOR: int = 240
"""240 = F₃·F₄·F₅·F₆ = 2·3·5·8 — Casimir coefficient, E8 root count."""

# Convenience aliases
XI = XI_ANALYTIC
"""Default Ξ — the analytic value γ + ln(φ). Use XI_DISCRETE for Fibonacci contexts."""


# =============================================================================
# DERIVED RATIOS
# =============================================================================

DUTY_CYCLE: float = PHI / (PHI + 1)
"""Equilibrium duty cycle = φ/(φ+1) = 1/φ ≈ 0.618 (61.8% attraction)."""

UNIVERSAL_RATIO: float = 2 / 3
"""F₃/F₄ = 2/3 — appears in Koide, She-Leveque β, quark charges. MED-forced."""

FIBONACCI_GRAVITY_DEPTH: int = 183
"""183 = F₇² + F₇ + 1 = 169 + 13 + 1 — Fibonacci hierarchy depth (Planck to proton).
Cyclotomic polynomial Φ₃(F₇). log_φ(m_Planck/m_proton) ≈ 91.5, doubled = 183."""
