"""
Tests for fracton.validation module.
"""

import pytest


class TestMetrics:
    """Test validation metric functions."""

    def test_relative_error(self):
        from fracton.validation.metrics import relative_error
        assert abs(relative_error(1.01, 1.0) - 0.01) < 1e-10

    def test_percent_error(self):
        from fracton.validation import percent_error
        assert abs(percent_error(1.01, 1.0) - 1.0) < 1e-10

    def test_ppm_error(self):
        from fracton.validation import ppm_error
        assert abs(ppm_error(1.000001, 1.0) - 1.0) < 0.01

    def test_sigma_deviation(self):
        from fracton.validation import sigma_deviation
        assert abs(sigma_deviation(10.5, 10.0, 0.1) - 5.0) < 1e-10

    def test_sigma_zero_uncertainty_raises(self):
        from fracton.validation import sigma_deviation
        with pytest.raises(ValueError):
            sigma_deviation(1.0, 1.0, 0.0)

    def test_relative_error_zero_measured_raises(self):
        from fracton.validation.metrics import relative_error
        with pytest.raises(ValueError):
            relative_error(1.0, 0.0)

    def test_is_within_tolerance_pct(self):
        from fracton.validation import is_within_tolerance
        assert is_within_tolerance(1.005, 1.0, tolerance_pct=1.0) is True
        assert is_within_tolerance(1.02, 1.0, tolerance_pct=1.0) is False

    def test_is_within_tolerance_ppm(self):
        from fracton.validation import is_within_tolerance
        assert is_within_tolerance(1.0000001, 1.0, tolerance_ppm=1.0) is True

    def test_is_within_tolerance_sigma(self):
        from fracton.validation import is_within_tolerance
        assert is_within_tolerance(10.1, 10.0, tolerance_sigma=2.0, uncertainty=0.1) is True
        assert is_within_tolerance(10.3, 10.0, tolerance_sigma=2.0, uncertainty=0.1) is False

    def test_is_within_tolerance_no_type_raises(self):
        from fracton.validation import is_within_tolerance
        with pytest.raises(ValueError):
            is_within_tolerance(1.0, 1.0)


class TestValidateResult:
    """Test validation result objects."""

    def test_validate_result_ppm(self):
        from fracton.validation import validate_result
        r = validate_result("alpha_EM", 0.007297311, 0.007297353,
                            tolerance_ppm=10)
        assert r.passed is True
        assert r.error_ppm < 10
        assert r.name == "alpha_EM"

    def test_validate_result_pct(self):
        from fracton.validation import validate_result
        r = validate_result("G", 6.662e-11, 6.674e-11,
                            tolerance_pct=0.5)
        assert r.passed is True
        assert r.error_pct < 0.5

    def test_validate_result_with_uncertainty(self):
        from fracton.validation import validate_result
        r = validate_result("G", 6.662e-11, 6.674e-11,
                            uncertainty=1.5e-15,
                            tolerance_pct=0.5)
        assert r.sigma is not None
        assert r.sigma > 0

    def test_validate_result_to_dict(self):
        from fracton.validation import validate_result
        r = validate_result("test", 1.01, 1.0, tolerance_pct=5,
                            formula="test formula", source="exp_01")
        d = r.to_dict()
        assert d["name"] == "test"
        assert d["formula"] == "test formula"
        assert d["source"] == "exp_01"
        assert "error_pct" in d

    def test_compare_predictions(self):
        from fracton.validation import validate_result, compare_predictions
        results = [
            validate_result("a", 1.01, 1.0, tolerance_pct=5),
            validate_result("b", 2.01, 2.0, tolerance_pct=5),
            validate_result("c", 3.5, 3.0, tolerance_pct=5),
        ]
        summary = compare_predictions(results)
        assert summary["total"] == 3
        assert summary["passed"] == 2  # a and b pass
        assert summary["failed"] == 1  # c fails (16.7%)
        assert summary["all_pass"] is False


class TestPhysicsConstants:
    """Validate PAC predictions using the validation module."""

    def test_alpha_em_validation(self):
        from fracton.constants import ALPHA_EM_PAC, ALPHA_EM_MEASURED
        from fracton.validation import validate_result
        r = validate_result("alpha_EM", ALPHA_EM_PAC, ALPHA_EM_MEASURED,
                            tolerance_ppm=10,
                            formula="F3/(F4*phi*F10)*(1-F10/(4pi*F7^2))")
        assert r.passed is True

    def test_weinberg_validation(self):
        from fracton.constants import SIN2_THETA_W_PAC, SIN2_THETA_W_MEASURED
        from fracton.validation import validate_result
        r = validate_result("sin2_theta_W", SIN2_THETA_W_PAC, SIN2_THETA_W_MEASURED,
                            tolerance_pct=0.2)
        assert r.passed is True

    def test_full_scorecard(self):
        from fracton.constants import (
            ALPHA_EM_PAC, ALPHA_EM_MEASURED,
            SIN2_THETA_W_PAC, SIN2_THETA_W_MEASURED,
            G_PAC, OMEGA_LAMBDA_PAC,
        )
        from fracton.constants.physical import G, OMEGA_LAMBDA
        from fracton.validation import validate_result, compare_predictions

        results = [
            validate_result("alpha_EM", ALPHA_EM_PAC, ALPHA_EM_MEASURED,
                            tolerance_ppm=10),
            validate_result("sin2_theta_W", SIN2_THETA_W_PAC, SIN2_THETA_W_MEASURED,
                            tolerance_pct=0.2),
            validate_result("G", G_PAC, G, tolerance_pct=0.2),
            validate_result("Omega_Lambda", OMEGA_LAMBDA_PAC, OMEGA_LAMBDA,
                            tolerance_pct=0.05),
        ]
        summary = compare_predictions(results)
        assert summary["all_pass"] is True
