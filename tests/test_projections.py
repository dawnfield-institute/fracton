"""
Tests for fracton.field.projections module.
"""

import math
import torch
import pytest


class TestTensorDecomposition:
    """Test symmetric/antisymmetric decomposition."""

    def test_symmetric_part(self):
        from fracton.field.projections import symmetric_part
        T = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        S = symmetric_part(T)
        # S_ij = (T_ij + T_ji) / 2
        assert torch.allclose(S, torch.tensor([[1.0, 2.5], [2.5, 4.0]]))

    def test_antisymmetric_part(self):
        from fracton.field.projections import antisymmetric_part
        T = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        A = antisymmetric_part(T)
        assert torch.allclose(A, torch.tensor([[0.0, -0.5], [0.5, 0.0]]))

    def test_decompose_roundtrip(self):
        from fracton.field.projections import decompose_tensor
        T = torch.randn(3, 3)
        S, A = decompose_tensor(T)
        # S + A should reconstruct T
        assert torch.allclose(S + A, T, atol=1e-14)

    def test_symmetric_is_symmetric(self):
        from fracton.field.projections import symmetric_part
        T = torch.randn(4, 4)
        S = symmetric_part(T)
        assert torch.allclose(S, S.T, atol=1e-14)

    def test_antisymmetric_is_antisymmetric(self):
        from fracton.field.projections import antisymmetric_part
        T = torch.randn(4, 4)
        A = antisymmetric_part(T)
        assert torch.allclose(A, -A.T, atol=1e-14)

    def test_batch_decomposition(self):
        from fracton.field.projections import decompose_tensor
        batch = torch.randn(5, 3, 3)
        S, A = decompose_tensor(batch)
        assert S.shape == (5, 3, 3)
        assert torch.allclose(S + A, batch, atol=1e-14)


class TestGradient:
    """Test 3D gradient operator."""

    def test_gradient_of_linear_field(self):
        from fracton.field.projections import gradient_3d
        # f(x, y, z) = 2*x + 3*y + 5*z (approx on grid)
        N = 16
        dx = 1.0
        x = torch.arange(N, dtype=torch.float64)
        field = (2 * x[None, None, :] + 3 * x[None, :, None] + 5 * x[:, None, None])
        gx, gy, gz = gradient_3d(field, dx)
        # Interior should be close to (2, 3, 5)
        interior = slice(2, -2)
        assert torch.allclose(gx[interior, interior, interior],
                              torch.full_like(gx[interior, interior, interior], 2.0), atol=0.01)
        assert torch.allclose(gy[interior, interior, interior],
                              torch.full_like(gy[interior, interior, interior], 3.0), atol=0.01)
        assert torch.allclose(gz[interior, interior, interior],
                              torch.full_like(gz[interior, interior, interior], 5.0), atol=0.01)

    def test_gradient_shape_preserved(self):
        from fracton.field.projections import gradient_3d
        field = torch.randn(8, 8, 8, dtype=torch.float64)
        gx, gy, gz = gradient_3d(field, 0.1)
        assert gx.shape == field.shape


class TestDivergence:
    """Test divergence operator."""

    def test_divergence_constant_field(self):
        from fracton.field.projections import divergence_3d
        N = 8
        Fx = torch.ones(N, N, N, dtype=torch.float64)
        Fy = torch.ones(N, N, N, dtype=torch.float64)
        Fz = torch.ones(N, N, N, dtype=torch.float64)
        div = divergence_3d(Fx, Fy, Fz, 1.0)
        # Divergence of constant field should be ~0
        assert div.abs().max() < 1e-10


class TestCurl:
    """Test curl operator."""

    def test_curl_of_gradient_is_zero(self):
        from fracton.field.projections import gradient_3d, curl_3d
        N = 16
        dx = 0.1
        field = torch.randn(N, N, N, dtype=torch.float64)
        gx, gy, gz = gradient_3d(field, dx)
        cx, cy, cz = curl_3d(gx, gy, gz, dx)
        # curl(grad(f)) = 0 (in interior, away from periodic edges)
        interior = slice(3, -3)
        assert cx[interior, interior, interior].abs().max() < 0.1
        assert cy[interior, interior, interior].abs().max() < 0.1
        assert cz[interior, interior, interior].abs().max() < 0.1


class TestLaplacian:
    """Test Laplacian operator."""

    def test_laplacian_shape(self):
        from fracton.field.projections import laplacian_3d
        field = torch.randn(8, 8, 8, dtype=torch.float64)
        lap = laplacian_3d(field, 0.1)
        assert lap.shape == field.shape

    def test_laplacian_batched(self):
        from fracton.field.projections import laplacian_3d
        batch = torch.randn(4, 8, 8, 8, dtype=torch.float64)
        lap = laplacian_3d(batch, 0.1)
        assert lap.shape == batch.shape

    def test_laplacian_quadratic(self):
        from fracton.field.projections import laplacian_3d
        # f = x^2 → Laplacian = 2 (in x) + 0 (y) + 0 (z)
        N = 16
        dx = 1.0
        x = torch.arange(N, dtype=torch.float64)
        field = x[None, None, :] ** 2 + torch.zeros(N, N, N, dtype=torch.float64)
        lap = laplacian_3d(field, dx)
        # Interior should be close to 2
        interior = slice(2, -2)
        assert torch.allclose(
            lap[interior, interior, interior],
            torch.full_like(lap[interior, interior, interior], 2.0),
            atol=0.5,
        )


class TestPreFieldProjections:
    """Test Maxwell/gravity pre-field projections."""

    def test_project_antisymmetric_shape(self):
        from fracton.field.projections import project_antisymmetric
        prefield = torch.randn(8, 16, 16, dtype=torch.float64)
        Fx, Fy, Fz = project_antisymmetric(prefield)
        assert Fx.shape == (16, 16)
        assert Fy.shape == (16, 16)

    def test_project_symmetric_shape(self):
        from fracton.field.projections import project_symmetric
        prefield = torch.randn(8, 16, 16, dtype=torch.float64)
        pot = project_symmetric(prefield)
        assert pot.shape == (16, 16)

    def test_depth_2_projection(self):
        from fracton.field.projections import depth_2_projection
        field_4d = torch.randn(8, 16, 16, dtype=torch.float64)
        em, grav = depth_2_projection(field_4d)
        assert len(em) == 3  # (Fx, Fy, Fz)
        assert grav.shape == (16, 16)

    def test_no_grad_tracking(self):
        from fracton.field.projections import symmetric_part
        T = torch.randn(3, 3, requires_grad=True)
        S = symmetric_part(T)
        assert not S.requires_grad
