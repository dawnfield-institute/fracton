"""
Bootstrap confidence interval estimation.

Standard non-parametric bootstrap for experiment statistics.
"""

import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence


def bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = None,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: Observed data points.
        statistic: Function to compute statistic from a sample.
            Defaults to mean.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (0 < ci < 1).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: estimate, ci_lower, ci_upper, std_error, n_bootstrap, ci_level.

    Examples:
        >>> result = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> result["estimate"]
        3.0
        >>> result["ci_lower"] < 3.0 < result["ci_upper"]
        True
    """
    if statistic is None:
        def statistic(x):
            return sum(x) / len(x) if x else 0.0

    data = list(data)
    n = len(data)
    if n == 0:
        return {
            "estimate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "std_error": 0.0,
            "n_bootstrap": n_bootstrap,
            "ci_level": ci,
        }

    rng = random.Random(seed)
    point_estimate = statistic(data)

    boot_stats = []
    for _ in range(n_bootstrap):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        boot_stats.append(statistic(sample))

    boot_stats.sort()
    alpha = 1 - ci
    lo_idx = int(math.floor(alpha / 2 * n_bootstrap))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))

    mean_boot = sum(boot_stats) / len(boot_stats)
    variance = sum((x - mean_boot) ** 2 for x in boot_stats) / len(boot_stats)

    return {
        "estimate": point_estimate,
        "ci_lower": boot_stats[lo_idx],
        "ci_upper": boot_stats[hi_idx],
        "std_error": math.sqrt(variance),
        "n_bootstrap": n_bootstrap,
        "ci_level": ci,
    }
