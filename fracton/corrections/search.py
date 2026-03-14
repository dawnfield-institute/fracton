"""
Correction template parameter space search.

Searches over valid (a, b, n, sign) combinations to find templates
that match a target value within a given tolerance.
"""

import math
from typing import Any, Dict, List, Optional

from ..fibonacci import fib
from .template import correction_factor, CorrectionTemplate


def search_corrections(
    target: float,
    base: float,
    a_range: range = range(2, 20),
    b_range: range = range(2, 12),
    n_range: range = range(1, 13),
    signs: tuple = (-1, +1),
    tolerance_pct: float = 1.0,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """Search correction template parameter space for matches.

    Finds (a, b, n, sign) combinations where:
        base × (1 ± F_a/(n·π·F_b²)) ≈ target

    within tolerance_pct.

    Args:
        target: Target value to match.
        base: Base value before correction.
        a_range: Range of Fibonacci indices for numerator.
        b_range: Range of Fibonacci indices for denominator.
        n_range: Range of boundary sector counts.
        signs: Tuple of signs to try.
        tolerance_pct: Maximum percent error for a match.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts sorted by error, each with:
        a, b, n, sign, gap, predicted, error_pct, correction_factor.

    Examples:
        >>> # Search for EM correction
        >>> results = search_corrections(
        ...     target=0.0072973526,
        ...     base=2/(3 * 1.618034 * 55),
        ...     tolerance_pct=0.1,
        ... )
    """
    matches = []

    for sign in signs:
        for a in a_range:
            f_a = fib(a)
            for b in b_range:
                f_b = fib(b)
                f_b_sq = f_b ** 2
                if f_b_sq == 0:
                    continue
                for n in n_range:
                    term = f_a / (n * math.pi * f_b_sq)
                    factor = 1 + sign * term
                    predicted = base * factor

                    if target == 0:
                        continue
                    error_pct = abs(predicted - target) / abs(target) * 100

                    if error_pct <= tolerance_pct:
                        matches.append({
                            "a": a,
                            "b": b,
                            "n": n,
                            "sign": sign,
                            "gap": abs(a - b),
                            "F_a": f_a,
                            "F_b": f_b,
                            "predicted": predicted,
                            "target": target,
                            "error_pct": error_pct,
                            "correction_factor": factor,
                            "term": term,
                        })

    matches.sort(key=lambda x: x["error_pct"])
    return matches[:max_results]
