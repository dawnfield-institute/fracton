"""
Tests for fracton.gauge module.
"""

import math
import pytest


class TestGaugeGroups:
    """Test gauge group structure analysis."""

    def test_adjoint_dim_su2(self):
        from fracton.gauge import adjoint_dim
        assert adjoint_dim(2) == 3

    def test_adjoint_dim_su3(self):
        from fracton.gauge import adjoint_dim
        assert adjoint_dim(3) == 8

    def test_adjoint_dim_u1(self):
        from fracton.gauge import adjoint_dim
        assert adjoint_dim(1) == 1

    def test_gauge_closure_sum(self):
        from fracton.gauge import gauge_closure_sum
        total, closed = gauge_closure_sum()
        assert total == 13
        assert closed is True  # 13 = F7

    def test_all_fibonacci(self):
        from fracton.gauge import is_fibonacci_gauge
        result = is_fibonacci_gauge()
        assert all(result.values())

    def test_standard_model_content(self):
        from fracton.gauge import STANDARD_MODEL_GAUGE
        assert STANDARD_MODEL_GAUGE["U(1)"] == 1
        assert STANDARD_MODEL_GAUGE["SU(2)"] == 3
        assert STANDARD_MODEL_GAUGE["SU(3)"] == 8
        assert STANDARD_MODEL_GAUGE["Higgs"] == 1

    def test_custom_gauge(self):
        from fracton.gauge import gauge_closure_sum
        # SU(5) GUT: adjoint dim = 24 (not Fibonacci)
        total, closed = gauge_closure_sum({"SU(5)": 24})
        assert total == 24
        assert closed is False


class TestWeinbergAngle:
    """Test Weinberg angle from Fibonacci."""

    def test_sin2_theta_w(self):
        from fracton.gauge import weinberg_angle_pac
        sin2 = weinberg_angle_pac()
        assert abs(sin2 - 3 / 13) < 1e-15

    def test_mw_mz_ratio(self):
        from fracton.gauge import mw_mz_ratio_pac
        ratio = mw_mz_ratio_pac()
        assert abs(ratio - math.sqrt(10 / 13)) < 1e-15

    def test_running_at_mw(self):
        from fracton.gauge import running_sin2_theta
        # At M_W itself, should return the input value
        sin2 = running_sin2_theta(80.377)
        assert abs(sin2 - 3 / 13) < 1e-6

    def test_running_at_mz(self):
        from fracton.gauge import running_sin2_theta
        # At M_Z ≈ 91 GeV, should be slightly higher
        sin2_mz = running_sin2_theta(91.1876)
        assert sin2_mz > 3 / 13  # Running increases it
        # Should be within ~1% of measured 0.23121
        assert abs(sin2_mz - 0.23121) / 0.23121 < 0.01
