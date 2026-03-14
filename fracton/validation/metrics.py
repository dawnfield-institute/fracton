"""
Validation metrics for comparing predictions to measurements.

All functions follow the convention: error = |predicted - measured| / |measured|.
"""

import math
from typing import Optional


def relative_error(predicted: float, measured: float) -> float:
    """Compute relative error |predicted - measured| / |measured|.

    Args:
        predicted: PAC-derived value.
        measured: CODATA/PDG measured value.

    Returns:
        Dimensionless relative error (0.01 = 1%).

    Raises:
        ValueError: If measured is zero.
    """
    if measured == 0:
        raise ValueError("Cannot compute relative error with measured=0")
    return abs(predicted - measured) / abs(measured)


def percent_error(predicted: float, measured: float) -> float:
    """Compute percent error.

    Args:
        predicted: PAC-derived value.
        measured: Measured value.

    Returns:
        Error in percent (1.0 = 1%).
    """
    return relative_error(predicted, measured) * 100


def ppm_error(predicted: float, measured: float) -> float:
    """Compute error in parts per million.

    Args:
        predicted: PAC-derived value.
        measured: Measured value.

    Returns:
        Error in ppm (1.0 = 1 ppm).
    """
    return relative_error(predicted, measured) * 1e6


def sigma_deviation(
    predicted: float,
    measured: float,
    uncertainty: float,
) -> float:
    """Compute how many sigma the prediction deviates from measurement.

    Args:
        predicted: PAC-derived value.
        measured: Central measured value.
        uncertainty: 1-sigma measurement uncertainty.

    Returns:
        Number of sigma (|predicted - measured| / uncertainty).

    Raises:
        ValueError: If uncertainty is zero or negative.
    """
    if uncertainty <= 0:
        raise ValueError(f"Uncertainty must be positive, got {uncertainty}")
    return abs(predicted - measured) / uncertainty


def is_within_tolerance(
    predicted: float,
    measured: float,
    tolerance_pct: Optional[float] = None,
    tolerance_ppm: Optional[float] = None,
    tolerance_sigma: Optional[float] = None,
    uncertainty: Optional[float] = None,
) -> bool:
    """Check if prediction is within tolerance of measurement.

    Provide exactly one tolerance type.

    Args:
        predicted: PAC-derived value.
        measured: Measured value.
        tolerance_pct: Max acceptable percent error.
        tolerance_ppm: Max acceptable ppm error.
        tolerance_sigma: Max acceptable sigma deviation.
        uncertainty: Required if using tolerance_sigma.

    Returns:
        True if within tolerance.
    """
    if tolerance_pct is not None:
        return percent_error(predicted, measured) < tolerance_pct
    if tolerance_ppm is not None:
        return ppm_error(predicted, measured) < tolerance_ppm
    if tolerance_sigma is not None:
        if uncertainty is None:
            raise ValueError("uncertainty required for sigma tolerance")
        return sigma_deviation(predicted, measured, uncertainty) < tolerance_sigma
    raise ValueError("Provide one of: tolerance_pct, tolerance_ppm, tolerance_sigma")
