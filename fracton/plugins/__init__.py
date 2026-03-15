"""
Fracton physics plugins — concrete PhysicsPlugin implementations.

Bridges the runtime engine with physics sub-packages:
- CascadePlugin: applies φ⁻ᵏ decay at each recursion depth
- CorrectionPlugin: applies force-specific correction templates
- SECPlugin: gates recursion based on entropy thresholds
"""

from .cascade import CascadePlugin
from .correction import CorrectionPlugin
from .sec import SECPlugin

__all__ = ["CascadePlugin", "CorrectionPlugin", "SECPlugin"]
