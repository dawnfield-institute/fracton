"""
Unified correction template: 1 ± F_a / (n·π·F_b²).

This is the core pattern discovered across all five fundamental forces
in Dawn Field Theory. Each force's coupling constant follows:

    value = base × (1 ± F_a / (n·π·F_b²))

Where:
    a, b = Fibonacci indices encoding cascade depth and gauge content
    n = number of independent boundary sectors
    ± = screening (-) for EM, anti-screening (+) for gravity/strong/dark energy
    gap = |a - b| encodes the force hierarchy

Source: MAR exp_38/39/40, PACSeries Papers 4-6.
"""

import math
from dataclasses import dataclass
from typing import Optional

from ..fibonacci import fib


@dataclass
class CorrectionTemplate:
    """A PAC correction template instance.

    Attributes:
        a: Fibonacci index for numerator (F_a).
        b: Fibonacci index for denominator squared (F_b²).
        n: Boundary sector count (divisor of π).
        sign: +1 (anti-screening) or -1 (screening).
        gap: |a - b|, encodes force hierarchy.
        factor: The computed correction factor.
        description: Human-readable description.
    """
    a: int
    b: int
    n: int
    sign: int  # +1 or -1
    gap: int
    factor: float
    description: str = ""

    @property
    def term(self) -> float:
        """The correction term F_a / (n·π·F_b²)."""
        return fib(self.a) / (self.n * math.pi * fib(self.b) ** 2)


def correction_factor(a: int, b: int, n: int, sign: int) -> float:
    """Compute the correction factor 1 ± F_a/(n·π·F_b²).

    Args:
        a: Fibonacci index for the numerator.
        b: Fibonacci index for the denominator squared.
        n: Number of independent boundary sectors.
        sign: +1 for anti-screening, -1 for screening.

    Returns:
        The correction factor (dimensionless).

    Examples:
        >>> correction_factor(10, 7, 4, -1)  # EM correction
        0.9741...
        >>> correction_factor(13, 6, 1, +1)  # Gravity correction
        2.1589...
    """
    f_a = fib(a)
    f_b = fib(b)
    term = f_a / (n * math.pi * f_b ** 2)
    return 1 + sign * term


def correction(a: int, b: int, n: int, sign: int) -> CorrectionTemplate:
    """Build a full CorrectionTemplate object.

    Args:
        a: Fibonacci index for numerator.
        b: Fibonacci index for denominator.
        n: Boundary sector count.
        sign: +1 or -1.

    Returns:
        CorrectionTemplate with computed factor.
    """
    factor = correction_factor(a, b, n, sign)
    gap = abs(a - b)
    return CorrectionTemplate(
        a=a, b=b, n=n, sign=sign,
        gap=gap, factor=factor,
    )


def build_correction(
    a: int,
    b: int,
    n: int,
    sign: int,
    base: Optional[float] = None,
    description: str = "",
) -> CorrectionTemplate:
    """Build a correction template and optionally apply it to a base value.

    Args:
        a: Fibonacci index for numerator.
        b: Fibonacci index for denominator.
        n: Boundary sector count.
        sign: +1 or -1.
        base: If provided, the factor is multiplied by this base value.
        description: Human-readable label.

    Returns:
        CorrectionTemplate. If base is given, factor = base × correction.
    """
    factor = correction_factor(a, b, n, sign)
    if base is not None:
        factor = base * factor
    return CorrectionTemplate(
        a=a, b=b, n=n, sign=sign,
        gap=abs(a - b),
        factor=factor,
        description=description,
    )
