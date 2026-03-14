"""
Experiment toolkit for Dawn Field Theory.

Standardized utilities for running, formatting, and saving experiments.
Extracts the common patterns used across 50+ DFT experiments into
reusable infrastructure.

Usage:
    from fracton.experiment import ExperimentResult, print_header, timer
    from fracton.experiment import save_results, experiment_header
"""

from .result import ExperimentResult, PartResult, save_results
from .formatting import print_header, print_section, print_table, print_result
from .metadata import experiment_header, experiment_docstring
from .timing import timer, Timer

__all__ = [
    "ExperimentResult", "PartResult", "save_results",
    "print_header", "print_section", "print_table", "print_result",
    "experiment_header", "experiment_docstring",
    "timer", "Timer",
]
