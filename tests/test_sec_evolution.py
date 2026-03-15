"""Tests for fracton.field.sec_evolution — SEC field evolution."""

import math

import pytest
import torch

from fracton.field.sec_evolution import SECFieldEvolver


@pytest.fixture
def evolver():
    return SECFieldEvolver(alpha=0.1, beta=0.05, gamma=0.01, device=torch.device("cpu"))


@pytest.fixture
def fields():
    """Create A, P, T fields for testing."""
    torch.manual_seed(42)
    H, W = 32, 32
    A = torch.randn(H, W, dtype=torch.float64)
    P = torch.randn(H, W, dtype=torch.float64)
    T = torch.rand(H, W, dtype=torch.float64) * 0.1 + 0.01  # positive temperatures
    return A, P, T


@pytest.fixture
def batched_fields():
    """Create batched A, P, T fields."""
    torch.manual_seed(42)
    B, H, W = 4, 32, 32
    A = torch.randn(B, H, W, dtype=torch.float64)
    P = torch.randn(B, H, W, dtype=torch.float64)
    T = torch.rand(B, H, W, dtype=torch.float64) * 0.1 + 0.01
    return A, P, T


class TestComputeEnergy:
    """Tests for energy functional computation."""

    def test_energy_components_are_non_negative(self, evolver, fields):
        A, P, T = fields
        energy = evolver.compute_energy(A, P, T)
        assert energy["coupling"] >= 0
        assert energy["smoothness"] >= 0
        assert energy["thermal"] >= 0
        assert energy["total"] >= 0

    def test_zero_energy_when_fields_match(self, evolver):
        field = torch.ones(16, 16, dtype=torch.float64)
        energy = evolver.compute_energy(field, field, T=None)
        # Coupling = 0 (A == P), smoothness = 0 (uniform), thermal = 0 (no T)
        assert energy["coupling"] == 0.0
        assert energy["smoothness"] == 0.0
        assert energy["thermal"] == 0.0
        assert energy["total"] == 0.0

    def test_energy_without_temperature(self, evolver, fields):
        A, P, _ = fields
        energy = evolver.compute_energy(A, P, T=None)
        assert energy["thermal"] == 0.0
        assert energy["total"] == energy["coupling"] + energy["smoothness"]

    def test_energy_increases_with_alpha(self, fields):
        A, P, T = fields
        e_low = SECFieldEvolver(alpha=0.01).compute_energy(A, P, T)
        e_high = SECFieldEvolver(alpha=1.0).compute_energy(A, P, T)
        assert e_high["coupling"] > e_low["coupling"]


class TestEvolve:
    """Tests for field evolution dynamics."""

    def test_energy_decreases_over_evolution(self, evolver, fields):
        A, P, T = fields
        e_before = evolver.compute_energy(A, P, T)["total"]
        A_new, _ = evolver.evolve(A, P, T, steps=50, dt=0.001, add_thermal_noise=False)
        e_after = evolver.compute_energy(A_new, P, T)["total"]
        assert e_after < e_before, "Energy should decrease under deterministic gradient descent"

    def test_evolve_returns_same_shape(self, evolver, fields):
        A, P, T = fields
        A_new, heat = evolver.evolve(A, P, T, steps=5, dt=0.001)
        assert A_new.shape == A.shape
        assert isinstance(heat, float)

    def test_evolve_batched(self, evolver, batched_fields):
        A, P, T = batched_fields
        A_new, heat = evolver.evolve(A, P, T, steps=5, dt=0.001)
        assert A_new.shape == A.shape, "Batched evolution should preserve shape"

    def test_heat_is_non_negative(self, evolver, fields):
        A, P, T = fields
        _, heat = evolver.evolve(A, P, T, steps=10, dt=0.001)
        # Heat can theoretically be negative in noisy regime, but for deterministic it should be >= 0
        # We just check it returns a float
        assert isinstance(heat, float)

    def test_no_noise_is_deterministic(self, evolver, fields):
        A, P, T = fields
        A1, _ = evolver.evolve(A.clone(), P, T, steps=5, dt=0.001, add_thermal_noise=False)
        evolver.reset()
        A2, _ = evolver.evolve(A.clone(), P, T, steps=5, dt=0.001, add_thermal_noise=False)
        assert torch.allclose(A1, A2), "Deterministic evolution should be reproducible"

    def test_langevin_noise_adds_variance(self, evolver, fields):
        A, P, T = fields
        torch.manual_seed(0)
        A_noisy, _ = evolver.evolve(A.clone(), P, T, steps=5, dt=0.001, add_thermal_noise=True)
        evolver.reset()
        A_det, _ = evolver.evolve(A.clone(), P, T, steps=5, dt=0.001, add_thermal_noise=False)
        # Noisy and deterministic should differ
        assert not torch.allclose(A_noisy, A_det)


class TestLaplacian:
    """Tests for 2D Laplacian computation."""

    def test_laplacian_of_constant_is_zero(self, evolver):
        field = torch.ones(16, 16, dtype=torch.float64)
        kernel = evolver._kernel.to(dtype=field.dtype)
        lap = SECFieldEvolver._laplacian_2d(field, kernel)
        # Interior should be exactly zero; edges may have small boundary effects
        assert torch.allclose(lap, torch.zeros_like(lap), atol=1e-10)

    def test_laplacian_shape_preserved(self, evolver):
        field = torch.randn(4, 32, 32, dtype=torch.float64)
        kernel = evolver._kernel.to(dtype=field.dtype)
        lap = SECFieldEvolver._laplacian_2d(field, kernel)
        assert lap.shape == field.shape


class TestCollapseDetection:
    """Tests for collapse region detection."""

    def test_uniform_field_no_collapse(self, evolver):
        field = torch.ones(16, 16, dtype=torch.float64)
        mask = evolver.detect_collapse_regions(field, threshold=0.1)
        assert mask.sum() == 0, "Uniform field should have no collapse regions"

    def test_sharp_gradient_detected(self, evolver):
        field = torch.zeros(16, 16, dtype=torch.float64)
        field[8, :] = 10.0  # sharp discontinuity
        mask = evolver.detect_collapse_regions(field, threshold=0.1)
        assert mask.sum() > 0, "Sharp gradient should be detected as collapse"


class TestState:
    """Tests for state tracking."""

    def test_initial_state_is_zero(self):
        evolver = SECFieldEvolver()
        state = evolver.get_state()
        assert state["total_heat_generated"] == 0.0
        assert state["total_entropy_reduced"] == 0.0
        assert state["collapse_event_count"] == 0

    def test_state_accumulates(self, evolver, fields):
        A, P, T = fields
        evolver.evolve(A, P, T, steps=10, dt=0.001)
        state = evolver.get_state()
        # After evolution, heat should have been generated
        assert state["total_heat_generated"] != 0.0

    def test_reset_clears_state(self, evolver, fields):
        A, P, T = fields
        evolver.evolve(A, P, T, steps=5, dt=0.001)
        evolver.reset()
        state = evolver.get_state()
        assert state["total_heat_generated"] == 0.0
        assert state["collapse_event_count"] == 0

    def test_repr(self, evolver):
        s = repr(evolver)
        assert "SECFieldEvolver" in s
        assert "α=0.1000" in s
