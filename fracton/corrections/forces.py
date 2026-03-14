"""
Force-specific correction shortcuts.

Pre-configured correction templates for each of the five forces.
"""

import math

from ..constants.mathematical import PHI, PHI_INV
from ..fibonacci import fib
from .template import correction_factor


def em_correction() -> float:
    """EM fine structure correction factor.

    1 - F₁₀/(4π·F₇²) = 0.9741...

    Applied to base F₃/(F₄·φ·F₁₀).
    Sign: -1 (screening). a=10, b=7, n=4. Gap=3=F₄.
    """
    return correction_factor(10, 7, 4, -1)


def gravity_correction() -> float:
    """Gravity correction factor.

    1 + F₁₃/(π·F₆²) = 2.1589...

    Applied to base ℏc/(F₁₈₃·m_p²).
    Sign: +1 (anti-screening). a=13, b=6, n=1. Gap=7=F₇.
    """
    return correction_factor(13, 6, 1, +1)


def dark_energy_correction() -> float:
    """Dark energy correction factor.

    1 + F₉/(4π·F₅²) = 1.1089...

    Applied to base 1/φ.
    Sign: +1 (anti-screening). a=9, b=5, n=4. Gap=4=F₄ (not Fibonacci gap).
    """
    return correction_factor(9, 5, 4, +1)


def strong_correction_c2() -> float:
    """Strong coupling candidate C2 correction factor.

    1 + F₅/(3π·F₂²) = 1.5305...

    n=3 (colors), gap=F₄. Error: 0.29%.
    """
    return correction_factor(5, 2, 3, +1)


def strong_correction_c3() -> float:
    """Strong coupling candidate C3 correction factor.

    1 + F₇/(8π·F₂²) = 1.5172...

    n=8 (gluons), gap=F₅. Error: 0.58%.
    """
    return correction_factor(7, 2, 8, +1)
