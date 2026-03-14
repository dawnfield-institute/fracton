"""
Tests for fracton.relativity module.
"""

import math
import pytest


class TestSchwarzschildMetric:
    """Test Schwarzschild metric components."""

    def test_g_tt_at_infinity(self):
        from fracton.relativity import schwarzschild_g_tt
        g_tt = schwarzschild_g_tt(1e20, 1.0)
        assert abs(g_tt - (-1.0)) < 1e-10

    def test_g_rr_at_infinity(self):
        from fracton.relativity import schwarzschild_g_rr
        g_rr = schwarzschild_g_rr(1e20, 1.0)
        assert abs(g_rr - 1.0) < 1e-10

    def test_metric_product_minus_one(self):
        """g_tt × g_rr = -1 everywhere (local c invariance)."""
        from fracton.relativity import schwarzschild_g_tt, schwarzschild_g_rr
        for r in [2.0, 5.0, 10.0, 100.0, 1e6]:
            r_s = 1.0
            product = schwarzschild_g_tt(r, r_s) * schwarzschild_g_rr(r, r_s)
            assert abs(product - (-1.0)) < 1e-10

    def test_coordinate_speed_at_infinity(self):
        from fracton.relativity import coordinate_speed_of_light
        v = coordinate_speed_of_light(1e20, 1.0)
        assert abs(v - 1.0) < 1e-10

    def test_coordinate_speed_near_horizon(self):
        from fracton.relativity import coordinate_speed_of_light
        v = coordinate_speed_of_light(1.01, 1.0)
        assert v < 0.02  # Very slow near horizon

    def test_schwarzschild_radius_sun(self):
        from fracton.relativity import schwarzschild_radius
        r_s = schwarzschild_radius(1.989e30)
        assert abs(r_s - 2953.25) < 1.0  # ~3 km


class TestLorentz:
    """Test Lorentz factor and time dilation."""

    def test_gamma_at_rest(self):
        from fracton.relativity import lorentz_gamma
        assert abs(lorentz_gamma(0.0) - 1.0) < 1e-15

    def test_gamma_at_half_c(self):
        from fracton.relativity import lorentz_gamma
        gamma = lorentz_gamma(0.5)
        expected = 1 / math.sqrt(1 - 0.25)
        assert abs(gamma - expected) < 1e-10

    def test_gamma_high_v(self):
        from fracton.relativity import lorentz_gamma
        gamma = lorentz_gamma(0.999)
        assert gamma > 22

    def test_gamma_invalid_raises(self):
        from fracton.relativity import lorentz_gamma
        with pytest.raises(ValueError):
            lorentz_gamma(1.0)
        with pytest.raises(ValueError):
            lorentz_gamma(-0.1)

    def test_time_dilation(self):
        from fracton.relativity import time_dilation_factor
        tau = time_dilation_factor(0.5)
        assert abs(tau - math.sqrt(0.75)) < 1e-10

    def test_cascade_time_dilation(self):
        from fracton.relativity import cascade_time_dilation
        from fracton.constants.mathematical import XI_FLOOR
        tau_1 = cascade_time_dilation(1)
        assert abs(tau_1 - XI_FLOOR) < 1e-15
        tau_10 = cascade_time_dilation(10)
        assert abs(tau_10 - XI_FLOOR ** 10) < 1e-12


class TestGRTests:
    """Test classical GR predictions."""

    def test_mercury_precession(self):
        from fracton.relativity import mercury_precession
        precession = mercury_precession()
        # GR prediction: 42.98 arcsec/century
        assert abs(precession - 42.98) / 42.98 < 0.005  # within 0.5%

    def test_light_deflection(self):
        from fracton.relativity import light_deflection
        deflection = light_deflection()
        # GR prediction: 1.7505 arcsec at solar limb
        assert abs(deflection - 1.7505) / 1.7505 < 0.005

    def test_shapiro_delay(self):
        from fracton.relativity import shapiro_delay
        # Venus superior conjunction
        r_e = 1.082e11  # Venus orbit
        r_r = 1.496e11  # Earth orbit
        r_0 = 6.957e8   # Solar radius
        delay = shapiro_delay(r_e, r_r, r_0)
        # Should be ~200 microseconds
        assert 100e-6 < delay < 300e-6
