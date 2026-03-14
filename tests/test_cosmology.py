"""
Tests for fracton.cosmology module.
"""

import math
import pytest


class TestDarkEnergy:
    """Test dark energy calculations."""

    def test_omega_lambda_precision(self):
        from fracton.cosmology import omega_lambda_pac
        from fracton.constants.physical import OMEGA_LAMBDA
        omega = omega_lambda_pac()
        error_pct = abs(omega - OMEGA_LAMBDA) / OMEGA_LAMBDA * 100
        assert error_pct < 0.05  # 0.032% actual

    def test_omega_lambda_value(self):
        from fracton.cosmology import omega_lambda_pac
        omega = omega_lambda_pac()
        assert 0.68 < omega < 0.69

    def test_cc_tiling_xi(self):
        from fracton.cosmology import cc_tiling_log10
        from fracton.constants.mathematical import XI_ANALYTIC
        val = cc_tiling_log10(XI_ANALYTIC)
        assert -124 < val < -122  # around -123.3

    def test_cc_tiling_xi_pac(self):
        from fracton.cosmology import cc_tiling_log10
        from fracton.constants.mathematical import XI_PAC
        val = cc_tiling_log10(XI_PAC)
        assert -124 < val < -122  # around -123.2

    def test_cc_gap_with_xi(self):
        from fracton.cosmology import cc_gap_orders
        from fracton.constants.mathematical import XI_ANALYTIC
        gap = cc_gap_orders(XI_ANALYTIC)
        assert gap < 0.5  # 0.38 orders actual

    def test_cc_gap_with_xi_pac(self):
        from fracton.cosmology import cc_gap_orders
        from fracton.constants.mathematical import XI_PAC
        gap = cc_gap_orders(XI_PAC)
        assert gap < 0.5  # 0.22 orders actual

    def test_vacuum_energy_ratio(self):
        from fracton.cosmology import vacuum_energy_ratio
        rho = vacuum_energy_ratio()
        assert rho > 0
        assert rho < 1e-120  # Extremely small
