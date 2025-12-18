"""
Fracton Physics Package

Dawn Field Theory physical constants and conservation laws.
This is the authoritative source for all physics constants used
throughout the Fracton ecosystem and GAIA.
"""

from .constants import (
    # Primary constants
    PHI, XI, PHI_XI, LAMBDA_STAR,
    # Derived thresholds
    SEC_EXPAND_THRESHOLD, SEC_COLLAPSE_THRESHOLD,
    # Validation
    validate_conservation, validate_sec_threshold
)

from .conservation import (
    PACValidator,
    validate_pac,
    compute_residual,
    enforce_conservation
)

from .phase_transitions import (
    PhaseState,
    detect_phase,
    should_expand,
    should_collapse
)

__all__ = [
    # Constants
    "PHI", "XI", "PHI_XI", "LAMBDA_STAR",
    "SEC_EXPAND_THRESHOLD", "SEC_COLLAPSE_THRESHOLD",
    # Conservation
    "PACValidator", "validate_pac", "compute_residual", "enforce_conservation",
    # Phase transitions
    "PhaseState", "detect_phase", "should_expand", "should_collapse"
]
