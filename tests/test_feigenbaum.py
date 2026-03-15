"""
Tests for fracton.feigenbaum module.
"""

import math
import pytest


class TestFeigenbaumConstants:
    """Test Feigenbaum constant values."""

    def test_delta_precision(self):
        from fracton.feigenbaum import DELTA, DELTA_LOGISTIC
        # PAC delta should match logistic to 13+ digits
        error = abs(DELTA - DELTA_LOGISTIC) / DELTA_LOGISTIC
        assert error < 1e-12  # sub-ppt precision

    def test_delta_value(self):
        from fracton.feigenbaum import DELTA
        assert 4.669 < DELTA < 4.670

    def test_alpha_value(self):
        from fracton.feigenbaum import ALPHA
        assert 2.502 < ALPHA < 2.503

    def test_r_inf_matches(self):
        from fracton.feigenbaum import R_INF, R_INF_LOGISTIC
        # R_INF is constructed from delta_z, should match exactly
        assert abs(R_INF - R_INF_LOGISTIC) / R_INF_LOGISTIC < 1e-10

    def test_structural_39(self):
        from fracton.feigenbaum.constants import STRUCTURAL_39
        assert STRUCTURAL_39 == 39
        assert STRUCTURAL_39 == (5**4 - 1) // 4**2

    def test_structural_1371(self):
        from fracton.feigenbaum.constants import STRUCTURAL_1371
        assert STRUCTURAL_1371 == 1371
        assert STRUCTURAL_1371 == 55 * 25 - 4

    def test_n_self_consistent(self):
        from fracton.feigenbaum import N_SELF_CONSISTENT
        assert 6.2 < N_SELF_CONSISTENT < 6.3

    def test_universal_delta_z(self):
        from fracton.feigenbaum import UNIVERSAL_DELTA_Z
        assert abs(UNIVERSAL_DELTA_Z) < 0.01  # Small perturbation


class TestMobiusTransformation:
    """Test Mobius transformation functions."""

    def test_m10_fixed_point(self):
        from fracton.feigenbaum import mobius_m10
        from fracton.constants.mathematical import PHI_INV
        # -1/phi is a fixed point: M_10(-1/phi) = -1/phi
        val = mobius_m10(-PHI_INV)
        assert abs(val - (-PHI_INV)) < 1e-10

    def test_m10_near_fixed_point(self):
        from fracton.feigenbaum import mobius_m10, UNIVERSAL_DELTA_Z
        from fracton.constants.mathematical import PHI_INV
        # Slightly off the fixed point should give a different value
        val = mobius_m10(-PHI_INV + UNIVERSAL_DELTA_Z)
        assert abs(val - (-PHI_INV)) > 1e-6

    def test_compute_delta_converges(self):
        from fracton.feigenbaum import compute_delta
        delta = compute_delta()
        assert 4.669 < delta < 4.670

    def test_compute_delta_matches_constant(self):
        from fracton.feigenbaum import compute_delta, DELTA_LOGISTIC
        delta = compute_delta()
        error = abs(delta - DELTA_LOGISTIC) / DELTA_LOGISTIC
        assert error < 1e-12

    def test_compute_r_inf(self):
        from fracton.feigenbaum import compute_r_inf, R_INF_LOGISTIC
        r_inf = compute_r_inf()
        assert abs(r_inf - R_INF_LOGISTIC) / R_INF_LOGISTIC < 1e-10

    def test_compute_universal_delta_z(self):
        from fracton.feigenbaum import compute_universal_delta_z, UNIVERSAL_DELTA_Z
        dz = compute_universal_delta_z()
        assert abs(dz - UNIVERSAL_DELTA_Z) < 1e-12


class TestFibonacciMobius:
    """Test the FibonacciMobius class (v2.1)."""

    def test_m10_callable(self):
        from fracton.feigenbaum import M10
        from fracton.constants.mathematical import PHI_INV
        val = M10(-PHI_INV)
        assert abs(val - (-PHI_INV)) < 1e-10

    def test_m10_eigenvalue(self):
        from fracton.feigenbaum import M10
        from fracton.constants.mathematical import PHI
        # Eigenvalue at unstable fixed point should be phi^20
        eigenval = M10.eigenvalue_at_unstable
        assert abs(eigenval - PHI**20) / PHI**20 < 1e-10

    def test_m10_fixed_points(self):
        from fracton.feigenbaum import M10
        from fracton.constants.mathematical import PHI, PHI_INV
        stable, unstable = M10.fixed_points
        assert abs(stable - PHI) < 1e-14
        assert abs(unstable - (-PHI_INV)) < 1e-14

    def test_verify_eigenvalue_identity(self):
        from fracton.feigenbaum import M10
        result = M10.verify_eigenvalue_identity()
        assert result["is_exact"]
        assert result["error"] < 1e-14

    def test_custom_mobius(self):
        from fracton.feigenbaum import FibonacciMobius
        from fracton.constants.mathematical import PHI_INV
        m5 = FibonacciMobius(5)
        # -1/phi is always a fixed point
        val = m5(-PHI_INV)
        assert abs(val - (-PHI_INV)) < 1e-10

    def test_derivative(self):
        from fracton.feigenbaum import M10
        from fracton.constants.mathematical import PHI_INV
        # Derivative at unstable fixed point
        deriv = M10.derivative_at(-PHI_INV)
        assert abs(deriv) > 1  # Unstable: |derivative| > 1


class TestFeigenbaumValidation:
    """Test validation functions (v2.1)."""

    def test_derive_structural_constants(self):
        from fracton.feigenbaum import derive_structural_constants
        derivations = derive_structural_constants()
        assert derivations["39"]["matches"]
        assert derivations["160"]["matches"]
        assert derivations["1371"]["matches"]
        assert derivations["1857"]["matches"]

    def test_prove_eigenvalue_identity(self):
        from fracton.feigenbaum import prove_eigenvalue_identity
        proof = prove_eigenvalue_identity()
        assert proof["is_exact"]
        assert proof["proof_verified"]
        assert abs(proof["proof_product"] - 1) < 1e-10

    def test_validate_universality(self):
        from fracton.feigenbaum import validate_universality
        results = validate_universality()
        assert results["logistic"]["error_percent"] < 1e-10
        # Scale ratio should be ~4
        assert abs(results["scale_ratio"]["observed"] - 4.0) < 0.01
