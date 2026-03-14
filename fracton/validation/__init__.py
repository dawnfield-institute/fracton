"""
Validation utilities for Dawn Field Theory experiments.

Provides standardized comparison metrics for PAC-derived predictions
against measured values (CODATA/PDG).

Usage:
    from fracton.validation import percent_error, sigma_deviation
    from fracton.validation import validate_result, compare_predictions
"""

from .metrics import (
    percent_error,
    ppm_error,
    sigma_deviation,
    relative_error,
    is_within_tolerance,
)
from .compare import validate_result, compare_predictions, ValidationResult

__all__ = [
    "percent_error", "ppm_error", "sigma_deviation",
    "relative_error", "is_within_tolerance",
    "validate_result", "compare_predictions", "ValidationResult",
]
