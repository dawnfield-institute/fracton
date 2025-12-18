"""
Dawn Field Theory Physical Constants

These are the fundamental constants governing the physics of cognition
and field dynamics in the Dawn Field Theory framework.

Values are empirically validated from POC-011 experiments:
- PHI: Golden ratio - nature's optimal growth pattern
- XI: Structural coupling constant - empirically tuned threshold
- PHI_XI: Phase transition threshold - crystallization threshold
- LAMBDA_STAR: Optimal decay - stability constant
"""

import math
from typing import Tuple

# ==============================================================================
# PRIMARY CONSTANTS
# ==============================================================================

PHI: float = (1 + math.sqrt(5)) / 2  # 1.6180339887...
"""Golden ratio - the fundamental growth constant of nature."""

XI: float = 0.0618
"""Structural coupling constant - empirically validated collapse threshold."""

PHI_XI: float = 0.1  # Phase transition threshold
"""Phase transition threshold - triggers lazy expansion."""

LAMBDA_STAR: float = 0.9816
"""Optimal decay constant - ensures stability in field evolution."""


# ==============================================================================
# DERIVED THRESHOLDS (SEC - Selective Entropy Collapse)
# ==============================================================================

SEC_EXPAND_THRESHOLD: float = PHI_XI  # 0.1 - triggers expansion
"""When potential exceeds this, pattern should expand (lazy evaluation triggers)."""

SEC_COLLAPSE_THRESHOLD: float = XI  # 0.0618 - triggers collapse  
"""When potential drops below this, pattern collapses to parent."""

# Crystallization threshold (empirically validated from POC-011)
CRYSTALLIZATION_THRESHOLD: float = 1.710  # PHI * XI scaled
"""High-importance pattern threshold for crystallization."""


# ==============================================================================
# ALTERNATIVE REPRESENTATIONS
# ==============================================================================

# Some GAIA components use different forms
TAU: float = 2 * math.pi  # Full circle
PHI_SQUARED: float = PHI ** 2  # 2.618...
PHI_CUBED: float = PHI ** 3  # 4.236...
PHI_FOURTH: float = PHI ** 4  # 6.854...

# Inverse constants
XI_INV: float = 1 / XI  # 16.18... = PHI^4
PHI_INV: float = 1 / PHI  # 0.618... = PHI - 1


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_conservation(parent: float, children_sum: float, 
                         tolerance: float = 1e-10) -> Tuple[bool, float]:
    """
    Validate PAC conservation: f(parent) = Σf(children)
    
    Args:
        parent: Parent node value
        children_sum: Sum of all children values
        tolerance: Maximum allowed residual
        
    Returns:
        Tuple of (is_valid, residual)
    """
    residual = abs(parent - children_sum)
    return residual < tolerance, residual


def validate_sec_threshold(potential: float) -> str:
    """
    Determine SEC phase based on potential.
    
    Args:
        potential: Current node potential
        
    Returns:
        "expand" | "stable" | "collapse"
    """
    if potential >= SEC_EXPAND_THRESHOLD:
        return "expand"
    elif potential <= SEC_COLLAPSE_THRESHOLD:
        return "collapse"
    else:
        return "stable"


# ==============================================================================
# CONVENIENCE EXPORTS
# ==============================================================================

# Dict form for serialization
CONSTANTS_DICT = {
    "PHI": PHI,
    "XI": XI,
    "PHI_XI": PHI_XI,
    "LAMBDA_STAR": LAMBDA_STAR,
    "SEC_EXPAND_THRESHOLD": SEC_EXPAND_THRESHOLD,
    "SEC_COLLAPSE_THRESHOLD": SEC_COLLAPSE_THRESHOLD,
}


if __name__ == "__main__":
    # Print constants for verification
    print("Dawn Field Theory Constants")
    print("=" * 40)
    print(f"PHI (φ):           {PHI:.10f}")
    print(f"XI (ξ):            {XI:.10f}")
    print(f"PHI_XI (φ×ξ):      {PHI_XI:.10f}")
    print(f"LAMBDA_STAR (λ*):  {LAMBDA_STAR:.10f}")
    print()
    print("Thresholds:")
    print(f"SEC Expand:        {SEC_EXPAND_THRESHOLD:.10f}")
    print(f"SEC Collapse:      {SEC_COLLAPSE_THRESHOLD:.10f}")
    print()
    print("Identities:")
    print(f"PHI^4 = 1/XI:      {PHI**4:.10f} = {1/XI:.10f}")
    print(f"PHI_XI = 0.1:      {PHI_XI:.10f}")
    print(f"PHI - 1 = 1/PHI:   {PHI-1:.10f} = {1/PHI:.10f}")
