"""
Tests for fracton.cascade module.
"""

import math
import pytest


class TestCouplingMatrix:
    """Test coupling matrix construction."""

    def test_basic_matrix(self):
        from fracton.cascade import coupling_matrix
        C = coupling_matrix(4, coupling_decay=0.3)
        assert len(C) == 4
        assert len(C[0]) == 4
        # Diagonal should be 1.0 (exp(0))
        for i in range(4):
            assert abs(C[i][i] - 1.0) < 1e-10
        # Off-diagonal should be < 1
        assert C[0][1] < 1.0
        # Symmetric
        assert abs(C[0][1] - C[1][0]) < 1e-10

    def test_feedback(self):
        from fracton.cascade import coupling_matrix
        C_base = coupling_matrix(3, coupling_decay=0.3)
        C_fb = coupling_matrix(3, coupling_decay=0.3, feedback=[1.0, 0.5, 0.0])
        # Feedback should modify the matrix
        assert C_fb[0][0] > C_base[0][0]


class TestEnergyCascade:
    """Test the energy cascade engine."""

    def test_basic_cascade(self):
        from fracton.cascade import energy_cascade
        results = energy_cascade(100.0, n_scales=10)
        assert len(results) == 10
        # Energy should decrease
        assert results[-1]["P_input"] < results[0]["P_input"]

    def test_organization_fraction(self):
        from fracton.cascade import energy_cascade
        results = energy_cascade(100.0, n_scales=5)
        for r in results:
            assert 0 < r["org_fraction"] < 1
            assert r["E_organized"] > 0

    def test_landauer_floor(self):
        from fracton.cascade import energy_cascade
        # Very small injection energy — should hit Landauer floor
        results = energy_cascade(0.001, n_scales=20)
        dead = [r for r in results if not r["alive"]]
        assert len(dead) > 0  # Some scales should be dead
        for r in dead:
            assert r["E_transfer"] >= math.log(2) - 1e-10

    def test_wavenumber_doubling(self):
        from fracton.cascade import energy_cascade
        results = energy_cascade(100.0, n_scales=5)
        for i, r in enumerate(results):
            assert r["wavenumber"] == 2 ** i

    def test_custom_parameters(self):
        from fracton.cascade import energy_cascade
        results = energy_cascade(
            50.0, n_scales=8,
            n_modes=4,
            coupling_decay=0.5,
            dissipation_rate=0.05,
        )
        assert len(results) == 8


class TestParticipationRatio:
    """Test participation ratio."""

    def test_single_mode(self):
        from fracton.cascade.engine import participation_ratio
        pr = participation_ratio([1.0, 0.0, 0.0])
        assert abs(pr - 1.0) < 1e-10

    def test_equal_modes(self):
        from fracton.cascade.engine import participation_ratio
        pr = participation_ratio([1.0, 1.0, 1.0, 1.0])
        assert abs(pr - 4.0) < 1e-10


class TestCascadeAnalysis:
    """Test cascade analysis utilities."""

    def test_measure_exponent(self):
        from fracton.cascade import energy_cascade, measure_exponent
        results = energy_cascade(1000.0, n_scales=15)
        slope, r2, avg_org, std_err = measure_exponent(results, trim=2)
        # Should get a negative slope (energy decreases with wavenumber)
        assert slope < 0
        assert r2 > 0.5

    def test_cascade_summary(self):
        from fracton.cascade import energy_cascade
        from fracton.cascade.analysis import cascade_summary
        results = energy_cascade(100.0, n_scales=10)
        summary = cascade_summary(results)
        assert summary["total_scales"] == 10
        assert summary["injection_energy"] == 100.0
        assert summary["final_energy"] < 100.0
        assert summary["total_dissipation"] > 0
        assert 0 < summary["avg_org_fraction"] < 1
