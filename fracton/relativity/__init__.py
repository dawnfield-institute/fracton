"""
Relativity operators from PAC cascade structure.

The Schwarzschild metric emerges from cascade density profiles:
- g_tt = -(1 - r_s/r) from time dilation
- g_rr = 1/(1 - r_s/r) from local c invariance
- g_tt × g_rr = -1 everywhere (exact constraint)

Lorentz factor arises from multiplicative asymmetry of forward/reverse
cascade paths — no spacetime geometry assumed.

Source: MAR exp_28 (statistical relativity), exp_30 (cascade GR).
"""

from .metric import (
    schwarzschild_g_tt,
    schwarzschild_g_rr,
    coordinate_speed_of_light,
    schwarzschild_radius,
)
from .lorentz import (
    lorentz_gamma,
    time_dilation_factor,
    cascade_time_dilation,
)
from .gr_tests import (
    mercury_precession,
    light_deflection,
    shapiro_delay,
)

__all__ = [
    "schwarzschild_g_tt", "schwarzschild_g_rr",
    "coordinate_speed_of_light", "schwarzschild_radius",
    "lorentz_gamma", "time_dilation_factor", "cascade_time_dilation",
    "mercury_precession", "light_deflection", "shapiro_delay",
]
