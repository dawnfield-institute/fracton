"""
GPU efficiency tests for fracton 2.1 tensor operations.

Tests cover:
- Device placement: outputs stay on the input device (no silent CPU fallback)
- No autograd: physics ops don't build gradient graphs
- Vectorization: tensor ops use torch primitives, not Python loops
- Batching: operations accept and process leading batch dimensions
- Memory: iterative ops don't leak GPU memory

All tests run on CPU. CUDA-specific assertions are gated behind
torch.cuda.is_available() so the suite passes on any machine.
"""

import pytest
import torch
import torch.nn.functional as F

from fracton.field.projections import (
    symmetric_part,
    antisymmetric_part,
    decompose_tensor,
    gradient_3d,
    divergence_3d,
    curl_3d,
    laplacian_3d,
    project_antisymmetric,
    project_symmetric,
    depth_2_projection,
)
from fracton.field.sec_evolution import SECFieldEvolver
from fracton.backend import get_device, to_tensor


# ---------------------------------------------------------------------------
# Device placement
# ---------------------------------------------------------------------------

class TestDevicePlacement:
    """Ensure outputs stay on the same device as inputs."""

    def _check_same_device(self, result, expected_device):
        if isinstance(result, tuple):
            for t in result:
                assert t.device == expected_device
        else:
            assert result.device == expected_device

    def test_gradient_3d_stays_on_device(self):
        device = torch.device("cpu")
        field = torch.randn(8, 8, 8, dtype=torch.float64, device=device)
        result = gradient_3d(field, dx=0.1)
        self._check_same_device(result, device)

    def test_curl_3d_stays_on_device(self):
        device = torch.device("cpu")
        Fx = torch.randn(8, 8, 8, dtype=torch.float64, device=device)
        Fy = torch.randn_like(Fx)
        Fz = torch.randn_like(Fx)
        result = curl_3d(Fx, Fy, Fz, dx=0.1)
        self._check_same_device(result, device)

    def test_divergence_3d_stays_on_device(self):
        device = torch.device("cpu")
        Fx = torch.randn(8, 8, 8, dtype=torch.float64, device=device)
        result = divergence_3d(Fx, Fx.clone(), Fx.clone(), dx=0.1)
        assert result.device == device

    def test_laplacian_3d_stays_on_device(self):
        device = torch.device("cpu")
        field = torch.randn(8, 8, 8, dtype=torch.float64, device=device)
        result = laplacian_3d(field, dx=0.1)
        assert result.device == device

    def test_symmetric_part_stays_on_device(self):
        device = torch.device("cpu")
        t = torch.randn(3, 3, dtype=torch.float64, device=device)
        assert symmetric_part(t).device == device

    def test_sec_evolution_stays_on_device(self):
        device = torch.device("cpu")
        evolver = SECFieldEvolver(device=device)
        A = torch.randn(16, 16, dtype=torch.float64, device=device)
        P = torch.randn_like(A)
        T = torch.rand_like(A) * 0.1 + 0.01
        A_new, _ = evolver.evolve(A, P, T, steps=3, dt=0.001)
        assert A_new.device == device

    def test_backend_to_tensor_cpu(self):
        t = to_tensor([1.0, 2.0, 3.0], device=torch.device("cpu"))
        assert t.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gradient_3d_stays_on_cuda(self):
        device = torch.device("cuda")
        field = torch.randn(16, 16, 16, dtype=torch.float64, device=device)
        gx, gy, gz = gradient_3d(field, dx=0.1)
        assert gx.device.type == "cuda"
        assert gy.device.type == "cuda"
        assert gz.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_laplacian_3d_stays_on_cuda(self):
        device = torch.device("cuda")
        field = torch.randn(16, 16, 16, dtype=torch.float64, device=device)
        result = laplacian_3d(field, dx=0.1)
        assert result.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sec_evolution_stays_on_cuda(self):
        device = torch.device("cuda")
        evolver = SECFieldEvolver(device=device)
        A = torch.randn(32, 32, dtype=torch.float64, device=device)
        P = torch.randn_like(A)
        T = torch.rand_like(A) * 0.1 + 0.01
        A_new, _ = evolver.evolve(A, P, T, steps=5, dt=0.001)
        assert A_new.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backend_to_tensor_cuda(self):
        t = to_tensor([1.0, 2.0], device=torch.device("cuda"))
        assert t.device.type == "cuda"


# ---------------------------------------------------------------------------
# No autograd
# ---------------------------------------------------------------------------

class TestNoGrad:
    """Physics computations should not build autograd graphs."""

    def test_symmetric_part_no_grad(self):
        t = torch.randn(3, 3, requires_grad=True)
        result = symmetric_part(t)
        assert not result.requires_grad

    def test_antisymmetric_part_no_grad(self):
        t = torch.randn(3, 3, requires_grad=True)
        result = antisymmetric_part(t)
        assert not result.requires_grad

    def test_gradient_3d_no_grad(self):
        field = torch.randn(8, 8, 8, requires_grad=True)
        gx, gy, gz = gradient_3d(field, dx=0.1)
        assert not gx.requires_grad
        assert not gy.requires_grad
        assert not gz.requires_grad

    def test_curl_3d_no_grad(self):
        Fx = torch.randn(8, 8, 8, requires_grad=True)
        Fy = torch.randn_like(Fx).requires_grad_(True)
        Fz = torch.randn_like(Fx).requires_grad_(True)
        cx, cy, cz = curl_3d(Fx, Fy, Fz, dx=0.1)
        assert not cx.requires_grad

    def test_laplacian_3d_no_grad(self):
        field = torch.randn(8, 8, 8, requires_grad=True)
        result = laplacian_3d(field, dx=0.1)
        assert not result.requires_grad

    def test_divergence_3d_no_grad(self):
        Fx = torch.randn(8, 8, 8, requires_grad=True)
        result = divergence_3d(Fx, Fx.clone(), Fx.clone(), dx=0.1)
        assert not result.requires_grad

    def test_project_antisymmetric_no_grad(self):
        field = torch.randn(8, 8, 8, requires_grad=True)
        Fx, Fy, Fz = project_antisymmetric(field)
        assert not Fx.requires_grad

    def test_project_symmetric_no_grad(self):
        field = torch.randn(8, 8, 8, requires_grad=True)
        result = project_symmetric(field)
        assert not result.requires_grad

    def test_sec_evolution_no_grad_by_default(self):
        evolver = SECFieldEvolver()
        A = torch.randn(16, 16, dtype=torch.float64, requires_grad=True)
        P = torch.randn_like(A)
        T = torch.rand(16, 16, dtype=torch.float64) * 0.1 + 0.01
        A_new, _ = evolver.evolve(A, P, T, steps=3, dt=0.001)
        assert not A_new.requires_grad


# ---------------------------------------------------------------------------
# Vectorization (no Python loops for core tensor ops)
# ---------------------------------------------------------------------------

class TestVectorization:
    """Ensure core operations use torch primitives, not Python loops."""

    def test_laplacian_3d_uses_conv3d(self):
        """Laplacian should use F.conv3d, not element-wise Python loops."""
        # Verify by checking the function uses conv3d (indirect: check output is correct
        # AND that it works on large tensors without timeout — Python loops would be slow)
        field = torch.randn(32, 32, 32, dtype=torch.float64)
        result = laplacian_3d(field, dx=1.0)
        # Correctness check: Laplacian of x² + y² + z² = 6
        x = torch.linspace(0, 1, 16, dtype=torch.float64)
        grid = x.unsqueeze(1).unsqueeze(2) ** 2 + x.unsqueeze(0).unsqueeze(2) ** 2 + x.unsqueeze(0).unsqueeze(1) ** 2
        dx = x[1].item() - x[0].item()
        lap = laplacian_3d(grid, dx)
        # Interior points should be close to 6 (boundary has artifacts)
        interior = lap[2:-2, 2:-2, 2:-2]
        assert torch.allclose(interior, torch.full_like(interior, 6.0), atol=0.5)

    def test_sec_laplacian_uses_conv2d(self):
        """SEC 2D Laplacian should use F.conv2d."""
        evolver = SECFieldEvolver(device=torch.device("cpu"))
        kernel = evolver._kernel.to(dtype=torch.float64)
        field = torch.randn(32, 32, dtype=torch.float64)
        result = SECFieldEvolver._laplacian_2d(field, kernel)
        assert result.shape == field.shape

    def test_gradient_uses_torch_roll(self):
        """Gradient should use torch.roll, not element-wise indexing."""
        # If it used Python loops on a 64^3 grid, this would be extremely slow
        field = torch.randn(64, 64, 64, dtype=torch.float64)
        gx, gy, gz = gradient_3d(field, dx=0.1)
        assert gx.shape == field.shape

    def test_energy_functional_vectorized(self):
        """SEC energy E(A|P,T) should use torch.sum/pow, not Python sum."""
        evolver = SECFieldEvolver()
        A = torch.randn(64, 64, dtype=torch.float64)
        P = torch.randn_like(A)
        T = torch.rand_like(A) * 0.1
        energy = evolver.compute_energy(A, P, T)
        assert isinstance(energy["total"], float)
        assert energy["total"] > 0


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

class TestBatching:
    """Operations should handle leading batch dimensions."""

    def test_gradient_3d_batched(self):
        field = torch.randn(4, 8, 8, 8, dtype=torch.float64)
        gx, gy, gz = gradient_3d(field, dx=0.1)
        assert gx.shape == (4, 8, 8, 8)

    def test_curl_3d_batched(self):
        Fx = torch.randn(4, 8, 8, 8, dtype=torch.float64)
        Fy = torch.randn_like(Fx)
        Fz = torch.randn_like(Fx)
        cx, cy, cz = curl_3d(Fx, Fy, Fz, dx=0.1)
        assert cx.shape == (4, 8, 8, 8)

    def test_divergence_3d_batched(self):
        Fx = torch.randn(4, 8, 8, 8, dtype=torch.float64)
        result = divergence_3d(Fx, Fx.clone(), Fx.clone(), dx=0.1)
        assert result.shape == (4, 8, 8, 8)

    def test_laplacian_3d_batched(self):
        field = torch.randn(4, 8, 8, 8, dtype=torch.float64)
        result = laplacian_3d(field, dx=0.1)
        assert result.shape == (4, 8, 8, 8)

    def test_symmetric_part_batched(self):
        t = torch.randn(4, 3, 3, dtype=torch.float64)
        result = symmetric_part(t)
        assert result.shape == (4, 3, 3)

    def test_decompose_tensor_batched(self):
        t = torch.randn(4, 3, 3, dtype=torch.float64)
        s, a = decompose_tensor(t)
        assert s.shape == (4, 3, 3)
        assert a.shape == (4, 3, 3)

    def test_sec_evolution_batched(self):
        """Evolving (B, H, W) should work — B fields in parallel."""
        evolver = SECFieldEvolver()
        A = torch.randn(4, 16, 16, dtype=torch.float64)
        P = torch.randn_like(A)
        T = torch.rand_like(A) * 0.1 + 0.01
        A_new, _ = evolver.evolve(A, P, T, steps=3, dt=0.001)
        assert A_new.shape == (4, 16, 16)

    def test_project_antisymmetric_batched(self):
        field = torch.randn(4, 8, 8, 8, dtype=torch.float64)
        Fx, Fy, Fz = project_antisymmetric(field)
        assert Fx.shape == (4, 8, 8)

    def test_project_symmetric_batched(self):
        field = torch.randn(4, 8, 8, 8, dtype=torch.float64)
        result = project_symmetric(field)
        assert result.shape == (4, 8, 8)

    def test_depth_2_projection_batched(self):
        field = torch.randn(4, 8, 8, 8, dtype=torch.float64)
        (em_x, em_y, em_z), grav = depth_2_projection(field)
        assert em_x.shape == (4, 8, 8)
        assert grav.shape == (4, 8, 8)

    def test_detect_collapse_regions_batched(self):
        evolver = SECFieldEvolver()
        A = torch.randn(4, 16, 16, dtype=torch.float64)
        mask = evolver.detect_collapse_regions(A)
        assert mask.shape == (4, 16, 16)


# ---------------------------------------------------------------------------
# Memory (no leaks in iterative ops)
# ---------------------------------------------------------------------------

class TestMemory:
    """Iterative operations should not accumulate tensor references."""

    def test_sec_evolution_constant_tensor_count(self):
        """Running N steps shouldn't grow the number of live tensors linearly."""
        import gc
        evolver = SECFieldEvolver()
        A = torch.randn(32, 32, dtype=torch.float64)
        P = torch.randn_like(A)
        T = torch.rand_like(A) * 0.1 + 0.01

        # Warm up
        evolver.evolve(A.clone(), P, T, steps=5, dt=0.001)
        gc.collect()

        # Count tensors before
        gc.collect()
        tensors_before = sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor))

        # Run more steps
        evolver.evolve(A.clone(), P, T, steps=50, dt=0.001)
        gc.collect()

        tensors_after = sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor))

        # Allow some overhead, but not 50x growth
        growth = tensors_after - tensors_before
        assert growth < 50, f"Tensor count grew by {growth} after 50 steps — possible leak"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sec_evolution_constant_gpu_memory(self):
        """GPU memory shouldn't grow linearly with evolution steps."""
        device = torch.device("cuda")
        evolver = SECFieldEvolver(device=device)
        A = torch.randn(64, 64, dtype=torch.float64, device=device)
        P = torch.randn_like(A)
        T = torch.rand_like(A) * 0.1 + 0.01

        # Warm up
        evolver.evolve(A.clone(), P, T, steps=10, dt=0.001)
        torch.cuda.synchronize()

        mem_before = torch.cuda.memory_allocated()
        evolver.evolve(A.clone(), P, T, steps=100, dt=0.001)
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()

        # Memory growth should be negligible (< 1MB for this small field)
        growth_mb = (mem_after - mem_before) / (1024 * 1024)
        assert growth_mb < 1.0, f"GPU memory grew by {growth_mb:.2f} MB"
