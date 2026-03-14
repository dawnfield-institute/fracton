"""
Monte Carlo null hypothesis testing.

Tests whether an observed statistic is consistent with a null model
by generating random samples and computing empirical p-values.
"""

import math
import random
from typing import Any, Callable, Dict, Optional


def monte_carlo_null(
    observed: float,
    generator_fn: Callable[[random.Random], float],
    n_trials: int = 10_000,
    seed: int = 42,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """Monte Carlo null hypothesis test.

    Generates n_trials samples from the null distribution using generator_fn,
    then computes an empirical p-value.

    Args:
        observed: The observed test statistic.
        generator_fn: Function(rng) -> float that generates one sample
            from the null distribution.
        n_trials: Number of Monte Carlo samples.
        seed: Random seed.
        alternative: "two-sided", "greater", or "less".

    Returns:
        Dict with keys: observed, null_mean, null_std, p_value, z_score,
        n_trials, alternative, significant_at_05.

    Examples:
        >>> def null_gen(rng):
        ...     return rng.gauss(0, 1)
        >>> result = monte_carlo_null(3.0, null_gen)
        >>> result["p_value"] < 0.01
        True
    """
    rng = random.Random(seed)

    null_samples = [generator_fn(rng) for _ in range(n_trials)]
    null_mean = sum(null_samples) / len(null_samples)
    null_var = sum((x - null_mean) ** 2 for x in null_samples) / len(null_samples)
    null_std = math.sqrt(null_var) if null_var > 0 else 1e-15

    if alternative == "greater":
        count = sum(1 for x in null_samples if x >= observed)
    elif alternative == "less":
        count = sum(1 for x in null_samples if x <= observed)
    else:  # two-sided
        count = sum(1 for x in null_samples if abs(x - null_mean) >= abs(observed - null_mean))

    p_value = count / n_trials
    z_score = (observed - null_mean) / null_std

    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value": p_value,
        "z_score": z_score,
        "n_trials": n_trials,
        "alternative": alternative,
        "significant_at_05": p_value < 0.05,
    }
