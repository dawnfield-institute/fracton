"""
Tests for fracton.statistics module.
"""

import math
import pytest


class TestBootstrapCI:
    """Test bootstrap confidence intervals."""

    def test_basic_mean(self):
        from fracton.statistics import bootstrap_ci
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_ci(data)
        assert result["estimate"] == 3.0
        assert result["ci_lower"] < 3.0
        assert result["ci_upper"] > 3.0

    def test_tight_data(self):
        from fracton.statistics import bootstrap_ci
        data = [1.0] * 100
        result = bootstrap_ci(data)
        assert result["estimate"] == 1.0
        assert abs(result["std_error"]) < 0.01

    def test_custom_statistic(self):
        from fracton.statistics import bootstrap_ci
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_ci(data, statistic=max)
        assert result["estimate"] == 5.0

    def test_empty_data(self):
        from fracton.statistics import bootstrap_ci
        result = bootstrap_ci([])
        assert result["estimate"] == 0.0

    def test_ci_level(self):
        from fracton.statistics import bootstrap_ci
        data = list(range(100))
        r90 = bootstrap_ci(data, ci=0.90)
        r99 = bootstrap_ci(data, ci=0.99)
        # 99% CI should be wider than 90% CI
        width_90 = r90["ci_upper"] - r90["ci_lower"]
        width_99 = r99["ci_upper"] - r99["ci_lower"]
        assert width_99 >= width_90

    def test_reproducible(self):
        from fracton.statistics import bootstrap_ci
        data = [1, 2, 3, 4, 5]
        r1 = bootstrap_ci(data, seed=42)
        r2 = bootstrap_ci(data, seed=42)
        assert r1["ci_lower"] == r2["ci_lower"]
        assert r1["ci_upper"] == r2["ci_upper"]


class TestMonteCarloNull:
    """Test Monte Carlo null hypothesis testing."""

    def test_significant_result(self):
        from fracton.statistics import monte_carlo_null
        # Observed value far from null (standard normal)
        result = monte_carlo_null(
            5.0,
            lambda rng: rng.gauss(0, 1),
        )
        assert result["p_value"] < 0.01
        assert result["significant_at_05"] is True
        assert result["z_score"] > 3

    def test_non_significant_result(self):
        from fracton.statistics import monte_carlo_null
        # Observed value within null distribution
        result = monte_carlo_null(
            0.1,
            lambda rng: rng.gauss(0, 1),
        )
        assert result["p_value"] > 0.05

    def test_one_sided_greater(self):
        from fracton.statistics import monte_carlo_null
        result = monte_carlo_null(
            3.0,
            lambda rng: rng.gauss(0, 1),
            alternative="greater",
        )
        assert result["p_value"] < 0.01

    def test_one_sided_less(self):
        from fracton.statistics import monte_carlo_null
        result = monte_carlo_null(
            -3.0,
            lambda rng: rng.gauss(0, 1),
            alternative="less",
        )
        assert result["p_value"] < 0.01

    def test_reproducible(self):
        from fracton.statistics import monte_carlo_null
        gen = lambda rng: rng.gauss(0, 1)
        r1 = monte_carlo_null(2.0, gen, seed=42)
        r2 = monte_carlo_null(2.0, gen, seed=42)
        assert r1["p_value"] == r2["p_value"]


class TestSpectralExponent:
    """Test spectral exponent fitting."""

    def test_known_exponent(self):
        from fracton.statistics import spectral_exponent
        # Generate k^(-5/3) data
        ks = [2 ** i for i in range(1, 10)]
        es = [k ** (-5 / 3) for k in ks]
        result = spectral_exponent(ks, es, trim=1)
        assert abs(result["slope"] - (-5 / 3)) < 0.01
        assert result["r_squared"] > 0.999

    def test_flat_spectrum(self):
        from fracton.statistics import spectral_exponent
        ks = [1, 2, 4, 8, 16]
        es = [1.0] * 5
        result = spectral_exponent(ks, es, trim=0)
        assert abs(result["slope"]) < 0.01

    def test_insufficient_data(self):
        from fracton.statistics import spectral_exponent
        result = spectral_exponent([1, 2], [1, 2], trim=2)
        assert result["slope"] == 0.0

    def test_measure_exponent(self):
        from fracton.statistics import measure_exponent
        data = [
            {"wavenumber": 2 ** i, "E_organized": (2 ** i) ** (-5 / 3),
             "org_fraction": 0.3, "alive": True}
            for i in range(1, 10)
        ]
        slope, r2, avg_org, std_err = measure_exponent(data, trim=1)
        assert abs(slope - (-5 / 3)) < 0.01
        assert r2 > 0.999
        assert abs(avg_org - 0.3) < 0.01
