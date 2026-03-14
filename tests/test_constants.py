"""
Tests for fracton.constants and fracton.fibonacci modules.

Validates all derived constants against measured values (CODATA/PDG)
and tests Fibonacci utility functions.
"""

import math
import pytest


class TestMathematicalConstants:
    """Validate mathematical constants from PAC framework."""

    def test_phi_identity(self):
        """φ² = φ + 1 (defining property)."""
        from fracton.constants import PHI, PHI_SQUARED
        assert abs(PHI_SQUARED - (PHI + 1)) < 1e-15

    def test_phi_inverse(self):
        """1/φ = φ - 1."""
        from fracton.constants import PHI, PHI_INV
        assert abs(PHI_INV - (PHI - 1)) < 1e-15

    def test_xi_analytic_decomposition(self):
        """Ξ = γ + ln(φ) exactly."""
        from fracton.constants import XI_ANALYTIC, GAMMA, LN_PHI
        assert abs(XI_ANALYTIC - (GAMMA + LN_PHI)) < 1e-15

    def test_xi_pac_formula(self):
        """ξ_PAC = 1 + (7/8)·ln(2)·(1-ln2)²."""
        from fracton.constants import XI_PAC, LN2
        expected = 1 + (7 / 8) * LN2 * (1 - LN2) ** 2
        assert abs(XI_PAC - expected) < 1e-15

    def test_xi_floor_formula(self):
        """ξ_floor = 1 - ln²(2)."""
        from fracton.constants import XI_FLOOR, LN2_SQUARED
        assert abs(XI_FLOOR - (1 - LN2_SQUARED)) < 1e-15

    def test_xi_ordering(self):
        """ξ_PAC < Ξ_discrete < Ξ_analytic (structural ordering)."""
        from fracton.constants import XI_PAC, XI_DISCRETE, XI_ANALYTIC
        assert XI_PAC < XI_DISCRETE < XI_ANALYTIC

    def test_xi_spread_positive(self):
        """Spread between analytic and discrete is positive and small."""
        from fracton.constants import XI_SPREAD
        assert 0 < XI_SPREAD < 0.002

    def test_euler_gap_approximation(self):
        """Euler gap ≈ 1/(240π) within 1%."""
        from fracton.constants import EULER_GAP, EULER_GAP_FACTOR
        approx = 1 / (EULER_GAP_FACTOR * math.pi)
        error_pct = abs(EULER_GAP - approx) / EULER_GAP * 100
        assert error_pct < 1.0  # ~0.5%

    def test_gravity_depth(self):
        """183 = F₇² + F₇ + 1 = 169 + 13 + 1."""
        from fracton.constants import FIBONACCI_GRAVITY_DEPTH
        assert FIBONACCI_GRAVITY_DEPTH == 13 ** 2 + 13 + 1
        assert FIBONACCI_GRAVITY_DEPTH == 183

    def test_golden_angle_fraction(self):
        """α* = 1 - 1/φ."""
        from fracton.constants import GOLDEN_ANGLE_FRACTION, PHI
        assert abs(GOLDEN_ANGLE_FRACTION - (1 - 1 / PHI)) < 1e-15


class TestPhysicalConstants:
    """Validate physical constants are sensible (CODATA values)."""

    def test_speed_of_light(self):
        from fracton.constants import C
        assert C == 299_792_458.0

    def test_planck_units_self_consistent(self):
        """Planck mass² = ℏc/G."""
        from fracton.constants import HBAR, C, G, PLANCK_MASS
        expected = math.sqrt(HBAR * C / G)
        assert abs(PLANCK_MASS - expected) / expected < 1e-10

    def test_hierarchy_ratio(self):
        """m_Pl/m_p ≈ 1.3 × 10¹⁹."""
        from fracton.constants import PLANCK_PROTON_RATIO
        assert 1e19 < PLANCK_PROTON_RATIO < 2e19


class TestStandardModelConstants:
    """Validate PAC-derived SM constants against measurements."""

    def test_alpha_em_precision(self):
        """α_PAC matches α_measured within 10 ppm."""
        from fracton.constants import ALPHA_EM_PAC, ALPHA_EM_MEASURED
        error_ppm = abs(ALPHA_EM_PAC - ALPHA_EM_MEASURED) / ALPHA_EM_MEASURED * 1e6
        assert error_ppm < 10  # 5.7 ppm actual

    def test_alpha_em_formula(self):
        """α = F₃/(F₄·φ·F₁₀) × (1 - F₁₀/(4π·F₇²))."""
        from fracton.constants import ALPHA_EM_PAC, PHI, F
        base = F[3] / (F[4] * PHI * F[10])
        correction = 1 - F[10] / (4 * math.pi * F[7] ** 2)
        assert abs(ALPHA_EM_PAC - base * correction) < 1e-15

    def test_weinberg_angle(self):
        """sin²θ_W = 3/13 within 0.2% of measured."""
        from fracton.constants import SIN2_THETA_W_PAC, SIN2_THETA_W_MEASURED
        assert SIN2_THETA_W_PAC == 3 / 13
        error_pct = abs(SIN2_THETA_W_PAC - SIN2_THETA_W_MEASURED) / SIN2_THETA_W_MEASURED * 100
        assert error_pct < 0.2

    def test_g_pac_precision(self):
        """G_PAC within 0.2% of measured G."""
        from fracton.constants import G_PAC
        from fracton.constants.physical import G
        error_pct = abs(G_PAC - G) / G * 100
        assert error_pct < 0.2  # 0.18% actual

    def test_omega_lambda_precision(self):
        """Ω_Λ_PAC within 0.05% of measured."""
        from fracton.constants import OMEGA_LAMBDA_PAC
        from fracton.constants.physical import OMEGA_LAMBDA
        error_pct = abs(OMEGA_LAMBDA_PAC - OMEGA_LAMBDA) / OMEGA_LAMBDA * 100
        assert error_pct < 0.05  # 0.032% actual

    def test_koide_ratio(self):
        """Koide Q = 2/3 exactly."""
        from fracton.constants import KOIDE_Q_PAC
        assert KOIDE_Q_PAC == 2 / 3

    def test_gauge_closure(self):
        """1 + 3 + 8 + 1 = 13 = F₇."""
        from fracton.constants import GAUGE_TOTAL, SM_IS_FIBONACCI_CLOSED, F
        assert GAUGE_TOTAL == 13
        assert GAUGE_TOTAL == F[7]
        assert SM_IS_FIBONACCI_CLOSED is True

    def test_b_hierarchy_ordering(self):
        """b values decrease: 7 > 6 > 5 > 4."""
        from fracton.constants import B_HIERARCHY
        bs = [B_HIERARCHY[f]["b"] for f in ["em", "gravity", "dark_energy", "strong"]]
        assert bs == [7, 6, 5, 4]

    def test_force_scorecard_completeness(self):
        """All five forces present in scorecard."""
        from fracton.constants import FORCE_SCORECARD
        assert set(FORCE_SCORECARD.keys()) == {"em", "gravity", "weak", "strong", "dark_energy"}

    def test_cc_tiling_gap(self):
        """CC tiling formula gives gap < 0.5 orders."""
        from fracton.constants import CC_TILING_XI
        from fracton.constants.physical import VACUUM_ENERGY_DENSITY_LOG10
        gap = abs(CC_TILING_XI - VACUUM_ENERGY_DENSITY_LOG10)
        assert gap < 0.5  # 0.38 actual

    def test_mass_ratios(self):
        """Fibonacci mass ratios within 0.5% of measured."""
        from fracton.constants import MUON_ELECTRON_PAC, PROTON_ELECTRON_PAC
        from fracton.constants.physical import MUON_ELECTRON_RATIO, PROTON_ELECTRON_RATIO
        mu_err = abs(MUON_ELECTRON_PAC - MUON_ELECTRON_RATIO) / MUON_ELECTRON_RATIO * 100
        p_err = abs(PROTON_ELECTRON_PAC - PROTON_ELECTRON_RATIO) / PROTON_ELECTRON_RATIO * 100
        assert mu_err < 0.01  # 5 ppm actual
        assert p_err < 0.1  # 0.006% actual


class TestFibonacci:
    """Test Fibonacci utility functions."""

    def test_fib_base_cases(self):
        from fracton.fibonacci import fib
        assert fib(0) == 0
        assert fib(1) == 1
        assert fib(2) == 1

    def test_fib_known_values(self):
        from fracton.fibonacci import fib
        known = {3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55}
        for n, expected in known.items():
            assert fib(n) == expected, f"F({n}) should be {expected}"

    def test_fib_recurrence(self):
        """F(n) = F(n-1) + F(n-2) for n ≥ 2."""
        from fracton.fibonacci import fib
        for n in range(2, 50):
            assert fib(n) == fib(n - 1) + fib(n - 2)

    def test_fib_183(self):
        """F(183) is the gravity hierarchy number."""
        from fracton.fibonacci import fib, fib_log10
        f183 = fib(183)
        assert f183 > 0
        # Check log10 matches
        log_actual = math.log10(f183)
        log_approx = fib_log10(183)
        assert abs(log_actual - log_approx) < 0.01

    def test_fib_negative_raises(self):
        from fracton.fibonacci import fib
        with pytest.raises(ValueError):
            fib(-1)

    def test_is_fibonacci(self):
        from fracton.fibonacci import is_fibonacci
        fibs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        for f in fibs:
            assert is_fibonacci(f), f"{f} should be Fibonacci"
        non_fibs = [4, 6, 7, 9, 10, 11, 12, 14, 15]
        for n in non_fibs:
            assert not is_fibonacci(n), f"{n} should not be Fibonacci"

    def test_fib_index(self):
        from fracton.fibonacci import fib_index
        assert fib_index(13) == 7
        assert fib_index(55) == 10
        assert fib_index(14) is None
        assert fib_index(0) == 0

    def test_nearest_fibonacci(self):
        from fracton.fibonacci import nearest_fibonacci
        below, above, closest = nearest_fibonacci(10)
        assert below == 8
        assert above == 13
        assert closest == 8

    def test_nearest_fibonacci_exact(self):
        from fracton.fibonacci import nearest_fibonacci
        below, above, closest = nearest_fibonacci(13)
        assert below == 13
        assert above == 13

    def test_fib_table_matches_fib(self):
        from fracton.fibonacci import fib, fib_table
        for n in range(0, 100):
            assert fib_table(n) == fib(n)

    def test_is_fibonacci_adjoint(self):
        from fracton.fibonacci import is_fibonacci_adjoint
        assert is_fibonacci_adjoint(3)   # SU(2): 2²-1 = 3 = F₄
        assert is_fibonacci_adjoint(8)   # SU(3): 3²-1 = 8 = F₆
        assert not is_fibonacci_adjoint(13)  # 13 is Fibonacci but not N²-1
        assert not is_fibonacci_adjoint(15)  # SU(4): 15 but not Fibonacci

    def test_cyclotomic_depth(self):
        from fracton.fibonacci import cyclotomic_depth
        assert cyclotomic_depth(7) == 183  # F₇=13, 13²+13+1=183

    def test_fib_ratio_convergence(self):
        """F_n/F_{n+1} → 1/φ."""
        from fracton.fibonacci import fib_ratio
        from fracton.constants import PHI_INV
        ratio = fib_ratio(30)
        assert abs(ratio - PHI_INV) < 1e-12
