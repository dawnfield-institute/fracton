"""
Statistical analysis tools for Dawn Field Theory experiments.

Provides bootstrap confidence intervals, Monte Carlo null hypothesis tests,
and related statistical utilities.

Usage:
    from fracton.statistics import bootstrap_ci, monte_carlo_null
    from fracton.statistics import effective_dof, spectral_exponent
"""

from .bootstrap import bootstrap_ci
from .nulltest import monte_carlo_null
from .spectral import spectral_exponent, measure_exponent

__all__ = [
    "bootstrap_ci",
    "monte_carlo_null",
    "effective_dof",
    "spectral_exponent",
    "measure_exponent",
]
