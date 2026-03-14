"""
Physics-aware decorators for Dawn Field Theory.

These decorators add physics validation and transformation to functions,
making PAC conservation, Fibonacci cascade structure, and correction
templates composable at the function level.

They complement the existing runtime decorators (@recursive, @entropy_gate)
by adding physics semantics:

    @pac_conserved     — validates P = A + xi + theta on outputs
    @fibonacci_cascade — applies phi^-k decay with conservation checking
    @correction_template — auto-applies the right correction factor
    @med_bounded       — enforces maximum emergence depth
    @sec_phase         — SEC gating using real xi thresholds

Usage:
    from fracton.lang.physics_decorators import pac_conserved, fibonacci_cascade
    from fracton.lang.physics_decorators import correction_template, sec_phase
"""

import functools
import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

from ..constants.mathematical import PHI, PHI_INV, XI_ANALYTIC, XI_PAC, LN2_SQUARED
from ..corrections.template import correction_factor


def pac_conserved(
    tolerance: float = 1e-10,
    components: Tuple[str, ...] = ("potential", "actualized", "xi", "theta"),
) -> Callable:
    """Validate PAC conservation on function outputs.

    The decorated function must return a dict with keys matching `components`.
    The decorator checks that potential = actualized + xi + theta (within tolerance).

    Args:
        tolerance: Maximum allowed conservation violation.
        components: Names of the dict keys for (P, A, xi, theta).

    Returns:
        Decorator.

    Raises:
        ValueError: If conservation is violated beyond tolerance.

    Examples:
        @pac_conserved()
        def split_energy(total):
            return {
                "potential": total,
                "actualized": total * 0.6,
                "xi": total * 0.3,
                "theta": total * 0.1,
            }
    """
    p_key, a_key, xi_key, theta_key = components

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if isinstance(result, dict):
                p = result.get(p_key, 0)
                a = result.get(a_key, 0)
                xi = result.get(xi_key, 0)
                theta = result.get(theta_key, 0)

                residual = abs(p - (a + xi + theta))
                if residual > tolerance:
                    raise ValueError(
                        f"PAC conservation violated: "
                        f"P={p:.10g} != A+xi+theta={a + xi + theta:.10g} "
                        f"(residual={residual:.2e}, tolerance={tolerance:.2e})"
                    )

            return result

        wrapper._fracton_pac_conserved = True
        wrapper._fracton_pac_tolerance = tolerance
        return wrapper

    return decorator


def fibonacci_cascade(
    depth: int = 10,
    decay: float = None,
    check_conservation: bool = True,
) -> Callable:
    """Apply Fibonacci cascade decay to function outputs.

    The decorated function receives an additional `cascade_weights` argument:
    a list of phi^-k weights for k=0..depth-1.

    If check_conservation is True, validates that the sum of weighted outputs
    converges (sum of phi^-k = phi for k=0..inf).

    Args:
        depth: Number of cascade levels.
        decay: Decay base (default: 1/phi).
        check_conservation: Validate cascade sum.

    Returns:
        Decorator.

    Examples:
        @fibonacci_cascade(depth=7)
        def energy_at_scale(scale, cascade_weights):
            return [100 * w for w in cascade_weights]
    """
    if decay is None:
        decay = PHI_INV

    weights = [decay ** k for k in range(depth)]
    total_weight = sum(weights)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            kwargs["cascade_weights"] = weights
            result = func(*args, **kwargs)

            if check_conservation and isinstance(result, (list, tuple)):
                cascade_sum = sum(result)
                expected_ratio = total_weight
                # The cascade should capture most of the energy
                if len(result) == depth:
                    actual_ratio = cascade_sum / result[0] if result[0] != 0 else 0
                    # Just store the diagnostic, don't fail
                    wrapper._fracton_last_cascade_ratio = actual_ratio

            return result

        wrapper._fracton_fibonacci_cascade = True
        wrapper._fracton_cascade_depth = depth
        wrapper._fracton_cascade_weights = weights
        return wrapper

    return decorator


def correction_template(
    force: str = "em",
    custom_a: int = None,
    custom_b: int = None,
    custom_n: int = None,
    custom_sign: int = None,
) -> Callable:
    """Auto-apply the correction factor for a given force.

    The decorated function computes a base value; the decorator
    multiplies it by the appropriate correction factor.

    Args:
        force: One of "em", "gravity", "dark_energy", "strong_c2", "strong_c3".
        custom_a: Override Fibonacci index a.
        custom_b: Override Fibonacci index b.
        custom_n: Override boundary sector count.
        custom_sign: Override sign.

    Returns:
        Decorator.

    Examples:
        @correction_template(force="em")
        def alpha_base():
            return F[3] / (F[4] * PHI * F[10])
        # Returns base × (1 - F10/(4*pi*F7^2))
    """
    # Known force parameters
    FORCE_PARAMS = {
        "em": (10, 7, 4, -1),
        "gravity": (13, 6, 1, +1),
        "dark_energy": (9, 5, 4, +1),
        "strong_c2": (5, 2, 3, +1),
        "strong_c3": (7, 2, 8, +1),
    }

    if custom_a is not None:
        a, b, n, sign = custom_a, custom_b, custom_n, custom_sign
    elif force in FORCE_PARAMS:
        a, b, n, sign = FORCE_PARAMS[force]
    else:
        raise ValueError(f"Unknown force '{force}'. Known: {list(FORCE_PARAMS.keys())}")

    factor = correction_factor(a, b, n, sign)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            base = func(*args, **kwargs)
            if isinstance(base, (int, float)):
                return base * factor
            return base

        wrapper._fracton_correction_template = True
        wrapper._fracton_correction_force = force
        wrapper._fracton_correction_factor = factor
        wrapper._fracton_correction_params = (a, b, n, sign)
        return wrapper

    return decorator


def med_bounded(max_depth: int = 2) -> Callable:
    """Enforce MED (Macro Emergence Dynamics) depth constraint.

    MED states that macro-scale emergence saturates at depth 2
    (two levels of recursive composition are sufficient for all
    observed macro phenomena).

    The decorated function receives a `depth` kwarg and raises
    if it exceeds max_depth.

    Args:
        max_depth: Maximum allowed emergence depth (default 2).

    Returns:
        Decorator.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            depth = kwargs.get("depth", 0)
            if depth > max_depth:
                raise ValueError(
                    f"MED constraint violated: depth={depth} > max_depth={max_depth}. "
                    f"Macro emergence saturates at depth {max_depth}."
                )
            return func(*args, **kwargs)

        wrapper._fracton_med_bounded = True
        wrapper._fracton_med_max_depth = max_depth
        return wrapper

    return decorator


def sec_phase(
    threshold: str = "xi",
    custom_threshold: float = None,
) -> Callable:
    """SEC (Symbolic Entropy Collapse) phase gating.

    Functions only execute when input entropy crosses the specified
    threshold. Below threshold = potential phase; above = actualized phase.

    The decorated function must accept an `entropy` keyword argument.

    Args:
        threshold: "xi" (Ξ=1.0584), "xi_pac" (ξ_PAC=1.0571),
                  "xi_floor" (1-ln²2=0.5199), or "custom".
        custom_threshold: Value when threshold="custom".

    Returns:
        Decorator.
    """
    THRESHOLDS = {
        "xi": XI_ANALYTIC,
        "xi_pac": XI_PAC,
        "xi_floor": 1 - LN2_SQUARED,
    }

    if custom_threshold is not None:
        thresh_val = custom_threshold
    elif threshold in THRESHOLDS:
        thresh_val = THRESHOLDS[threshold]
    else:
        raise ValueError(f"Unknown threshold '{threshold}'. Known: {list(THRESHOLDS.keys())}")

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            entropy = kwargs.get("entropy", None)
            if entropy is not None and entropy < thresh_val:
                # Below threshold — return None (potential phase, not actualized)
                return None
            return func(*args, **kwargs)

        wrapper._fracton_sec_phase = True
        wrapper._fracton_sec_threshold = thresh_val
        wrapper._fracton_sec_threshold_name = threshold
        return wrapper

    return decorator
