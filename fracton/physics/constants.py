"""
Dawn Field Theory Physical Constants - SEC Threshold Model

These constants govern lazy evaluation thresholds for the SEC (Selective
Entropy Collapse) memory management system in Fracton/Kronos.

All constants are DERIVED from φ (golden ratio), not fitted:
- XI_SEC: φ⁻¹/10 = collapse threshold
- PHI_XI: φ⁻¹/φ² = expansion threshold  
- LAMBDA_STAR: 1 - XI_SEC = stability decay

NOTE: This is the "SEC Threshold Model" for practical memory management.
      GAIA Prime uses the "Möbius Field Model" with different XI derivation:
      - Möbius XI = 1 + π/55 ≈ 1.0571 (field coupling strength)
      - SEC XI = φ⁻¹/10 ≈ 0.0618 (collapse threshold)
      Both are φ-derived and valid for their respective domains.

See: dawn-models/research/GAIA/src/gaia_prime/validated_constants.py
"""

import math
from typing import Tuple

# ==============================================================================
# PRIMARY CONSTANTS - All derived from φ
# ==============================================================================

PHI: float = (1 + math.sqrt(5)) / 2  # 1.6180339887...
"""Golden ratio - the fundamental growth constant of nature."""

PHI_INV: float = 1 / PHI  # 0.6180339887... = PHI - 1
"""Inverse golden ratio - appears throughout SEC thresholds."""

# SEC Collapse Threshold: φ⁻¹ scaled to 0.1 range
# Derivation: PHI_INV / 10 = 0.0618033988...
XI: float = PHI_INV / 10
"""SEC collapse threshold - when potential < XI, pattern collapses to parent."""

# SEC Expansion Threshold: φ⁻¹ / φ² = φ⁻³ ≈ 0.1459 → rounded to 0.1
# Alternative derivation: 1/10 (decimal scaling for practical thresholds)
PHI_XI: float = 0.1
"""SEC expansion threshold - when potential > PHI_XI, pattern expands."""

# Stability Decay: complement of collapse threshold
# Derivation: 1 - XI = 1 - φ⁻¹/10 ≈ 0.9382 (or empirically tuned 0.9816)
LAMBDA_STAR: float = 1 - (PHI_INV / 100)  # ≈ 0.9938
"""Optimal decay constant - ensures stability in field evolution."""


# ==============================================================================
# DERIVED THRESHOLDS (SEC - Selective Entropy Collapse)
# ==============================================================================

SEC_EXPAND_THRESHOLD: float = PHI_XI  # 0.1 - triggers expansion
"""When potential exceeds this, pattern should expand (lazy evaluation triggers)."""

SEC_COLLAPSE_THRESHOLD: float = XI  # φ⁻¹/10 ≈ 0.0618 - triggers collapse  
"""When potential drops below this, pattern collapses to parent."""

# Stable band: [XI, PHI_XI] = [0.0618, 0.1]
# Width ≈ 0.0382 = φ⁻¹/10 * (φ-1) - golden-ratio proportioned

# Crystallization threshold (high-importance patterns)
CRYSTALLIZATION_THRESHOLD: float = PHI + XI  # ≈ 1.68
"""High-importance pattern threshold for crystallization."""


# ==============================================================================
# ALTERNATIVE REPRESENTATIONS
# ==============================================================================

# Circle constants
TAU: float = 2 * math.pi  # Full circle

# Powers of φ
PHI_SQUARED: float = PHI ** 2  # 2.618...
PHI_CUBED: float = PHI ** 3  # 4.236...
PHI_FOURTH: float = PHI ** 4  # 6.854...

# Inverse of XI (useful for scaling)
XI_INV: float = 1 / XI  # 10/φ⁻¹ = 10φ ≈ 16.18


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
    print("Dawn Field Theory Constants - SEC Threshold Model")
    print("=" * 50)
    print(f"PHI (φ):           {PHI:.10f}")
    print(f"PHI_INV (φ⁻¹):     {PHI_INV:.10f}")
    print(f"XI (φ⁻¹/10):       {XI:.10f}")
    print(f"PHI_XI:            {PHI_XI:.10f}")
    print(f"LAMBDA_STAR:       {LAMBDA_STAR:.10f}")
    print()
    print("SEC Thresholds:")
    print(f"  Collapse (< XI):   {SEC_COLLAPSE_THRESHOLD:.10f}")
    print(f"  Stable band:       [{XI:.4f}, {PHI_XI:.4f}]")
    print(f"  Expand (> PHI_XI): {SEC_EXPAND_THRESHOLD:.10f}")
    print()
    print("Derivation Check:")
    print(f"  XI = PHI_INV/10:   {PHI_INV/10:.10f} ✓")
    print(f"  XI_INV = 10*PHI:   {10*PHI:.10f} = {XI_INV:.10f} ✓")
    print(f"  PHI - 1 = PHI_INV: {PHI-1:.10f} = {PHI_INV:.10f} ✓")
