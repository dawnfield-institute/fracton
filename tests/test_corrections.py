"""
Tests for fracton.corrections module.
"""

import math
import pytest


class TestCorrectionTemplate:
    """Test the unified correction template engine."""

    def test_correction_factor_em(self):
        from fracton.corrections import correction_factor
        # EM: 1 - F10/(4*pi*F7^2) = 1 - 55/(4*pi*169)
        factor = correction_factor(10, 7, 4, -1)
        expected = 1 - 55 / (4 * math.pi * 169)
        assert abs(factor - expected) < 1e-15

    def test_correction_factor_gravity(self):
        from fracton.corrections import correction_factor
        # Gravity: 1 + F13/(pi*F6^2) = 1 + 233/(pi*64)
        factor = correction_factor(13, 6, 1, +1)
        expected = 1 + 233 / (math.pi * 64)
        assert abs(factor - expected) < 1e-15
        assert factor > 2.0  # K ≈ 2.159

    def test_correction_object(self):
        from fracton.corrections import correction
        c = correction(10, 7, 4, -1)
        assert c.a == 10
        assert c.b == 7
        assert c.n == 4
        assert c.sign == -1
        assert c.gap == 3  # |10-7|
        assert 0.97 < c.factor < 0.98

    def test_build_correction_with_base(self):
        from fracton.corrections import build_correction
        from fracton.constants.mathematical import PHI
        from fracton.fibonacci import fib
        # EM: base = F3/(F4*phi*F10)
        base = fib(3) / (fib(4) * PHI * fib(10))
        c = build_correction(10, 7, 4, -1, base=base, description="alpha_EM")
        # Should match ALPHA_EM_PAC
        from fracton.constants import ALPHA_EM_PAC
        assert abs(c.factor - ALPHA_EM_PAC) < 1e-15

    def test_correction_template_term(self):
        from fracton.corrections import correction
        c = correction(10, 7, 4, -1)
        # term = F10/(4*pi*F7^2) = 55/(4*pi*169)
        expected_term = 55 / (4 * math.pi * 169)
        assert abs(c.term - expected_term) < 1e-15


class TestForceCorrections:
    """Test force-specific correction shortcuts."""

    def test_em_correction(self):
        from fracton.corrections import em_correction
        factor = em_correction()
        assert 0.97 < factor < 0.98  # screening

    def test_gravity_correction(self):
        from fracton.corrections import gravity_correction
        factor = gravity_correction()
        assert 2.1 < factor < 2.2  # anti-screening K ≈ 2.159

    def test_dark_energy_correction(self):
        from fracton.corrections import dark_energy_correction
        factor = dark_energy_correction()
        assert 1.1 < factor < 1.12  # anti-screening

    def test_strong_corrections(self):
        from fracton.corrections import strong_correction_c2, strong_correction_c3
        c2 = strong_correction_c2()
        c3 = strong_correction_c3()
        assert c2 > 1.0  # anti-screening
        assert c3 > 1.0
        assert c2 != c3  # different candidates


class TestCorrectionSearch:
    """Test parameter space search."""

    def test_search_finds_em(self):
        from fracton.corrections import search_corrections
        from fracton.constants import ALPHA_EM_MEASURED
        from fracton.constants.mathematical import PHI
        from fracton.fibonacci import fib

        base = fib(3) / (fib(4) * PHI * fib(10))
        results = search_corrections(
            target=ALPHA_EM_MEASURED,
            base=base,
            tolerance_pct=0.01,
        )
        # Should find the (a=10, b=7, n=4, sign=-1) solution
        assert len(results) > 0
        best = results[0]
        assert best["a"] == 10
        assert best["b"] == 7
        assert best["n"] == 4
        assert best["sign"] == -1

    def test_search_empty_for_impossible(self):
        from fracton.corrections import search_corrections
        results = search_corrections(
            target=999.0,
            base=1.0,
            tolerance_pct=0.001,
        )
        assert len(results) == 0

    def test_search_max_results(self):
        from fracton.corrections import search_corrections
        results = search_corrections(
            target=1.0,
            base=1.0,
            tolerance_pct=50,
            max_results=5,
        )
        assert len(results) <= 5
