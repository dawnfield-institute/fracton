"""
Structured comparison of PAC predictions against measurements.

Provides ValidationResult objects and batch comparison utilities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .metrics import percent_error, ppm_error, sigma_deviation, relative_error


@dataclass
class ValidationResult:
    """Result of validating a PAC prediction against measurement.

    Attributes:
        name: Name of the quantity (e.g., "alpha_EM").
        predicted: PAC-derived value.
        measured: Measured/CODATA value.
        uncertainty: 1-sigma measurement uncertainty (optional).
        error_pct: Percent error.
        error_ppm: Parts-per-million error.
        sigma: Sigma deviation (if uncertainty provided).
        passed: Whether the result passes the tolerance threshold.
        tolerance: The tolerance that was applied.
        formula: PAC formula string (for documentation).
        source: Source reference (e.g., "MAR exp_34").
    """

    name: str
    predicted: float
    measured: float
    uncertainty: Optional[float] = None
    error_pct: float = 0.0
    error_ppm: float = 0.0
    sigma: Optional[float] = None
    passed: Optional[bool] = None
    tolerance: Optional[str] = None
    formula: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON output."""
        d = {
            "name": self.name,
            "predicted": self.predicted,
            "measured": self.measured,
            "error_pct": self.error_pct,
            "error_ppm": self.error_ppm,
        }
        if self.uncertainty is not None:
            d["uncertainty"] = self.uncertainty
            d["sigma"] = self.sigma
        if self.passed is not None:
            d["passed"] = self.passed
            d["tolerance"] = self.tolerance
        if self.formula:
            d["formula"] = self.formula
        if self.source:
            d["source"] = self.source
        return d


def validate_result(
    name: str,
    predicted: float,
    measured: float,
    uncertainty: Optional[float] = None,
    tolerance_pct: Optional[float] = None,
    tolerance_ppm: Optional[float] = None,
    formula: str = "",
    source: str = "",
) -> ValidationResult:
    """Validate a single PAC prediction against measurement.

    Args:
        name: Quantity name.
        predicted: PAC-derived value.
        measured: Measured value.
        uncertainty: 1-sigma uncertainty.
        tolerance_pct: Pass threshold in percent.
        tolerance_ppm: Pass threshold in ppm.
        formula: PAC formula (for documentation).
        source: Source reference.

    Returns:
        ValidationResult with computed errors.

    Examples:
        >>> r = validate_result("alpha_EM", 0.007297311, 0.007297353,
        ...                     tolerance_ppm=10, formula="F3/(F4*phi*F10)*(1-...)")
        >>> r.passed
        True
        >>> r.error_ppm
        5.7...
    """
    err_pct = percent_error(predicted, measured)
    err_ppm = ppm_error(predicted, measured)

    sig = None
    if uncertainty is not None and uncertainty > 0:
        sig = sigma_deviation(predicted, measured, uncertainty)

    passed = None
    tol_str = None
    if tolerance_pct is not None:
        passed = err_pct < tolerance_pct
        tol_str = f"<{tolerance_pct}%"
    elif tolerance_ppm is not None:
        passed = err_ppm < tolerance_ppm
        tol_str = f"<{tolerance_ppm} ppm"

    return ValidationResult(
        name=name,
        predicted=predicted,
        measured=measured,
        uncertainty=uncertainty,
        error_pct=err_pct,
        error_ppm=err_ppm,
        sigma=sig,
        passed=passed,
        tolerance=tol_str,
        formula=formula,
        source=source,
    )


def compare_predictions(results: List[ValidationResult]) -> Dict[str, Any]:
    """Summarize a batch of validation results.

    Args:
        results: List of ValidationResult objects.

    Returns:
        Summary dict with pass/fail counts, best/worst errors, etc.
    """
    tracked = [r for r in results if r.passed is not None]
    n_pass = sum(1 for r in tracked if r.passed)
    n_fail = sum(1 for r in tracked if not r.passed)

    errors_ppm = [r.error_ppm for r in results if r.measured != 0]
    best = min(results, key=lambda r: r.error_ppm) if results else None
    worst = max(results, key=lambda r: r.error_ppm) if results else None

    return {
        "total": len(results),
        "passed": n_pass,
        "failed": n_fail,
        "untested": len(results) - len(tracked),
        "median_error_ppm": sorted(errors_ppm)[len(errors_ppm) // 2] if errors_ppm else None,
        "best": best.name if best else None,
        "best_error_ppm": best.error_ppm if best else None,
        "worst": worst.name if worst else None,
        "worst_error_ppm": worst.error_ppm if worst else None,
        "all_pass": n_fail == 0 and n_pass > 0,
        "results": [r.to_dict() for r in results],
    }
