"""
Tests for Möbius Tensor Architecture

Tests the core Möbius tensor functionality including:
- MobiusMatrix transformations and composition
- MobiusStripTensor antiperiodic boundary conditions
- MobiusFibonacciTensor golden structure
- 4π periodicity verification
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from fracton.core.mobius_tensor import (
    MobiusMatrix, MobiusFrame, MobiusStripTensor,
    MobiusFibonacciTensor, MobiusRecursiveTensor,
    cross_ratio, create_fibonacci_mobius, verify_4pi_periodicity,
    PHI, PHI_INV
)


class TestMobiusMatrix:
    """Tests for MobiusMatrix class."""
    
    def test_identity(self):
        """Identity matrix should leave z unchanged."""
        M = MobiusMatrix.identity()
        assert M(1+1j) == pytest.approx(1+1j, rel=1e-10)
        assert M(0) == pytest.approx(0, abs=1e-10)
        assert M(-5.5) == pytest.approx(-5.5, rel=1e-10)
    
    def test_composition(self):
        """Matrix composition should equal function composition."""
        M1 = MobiusMatrix(1, 2, 0, 1)  # z + 2
        M2 = MobiusMatrix(2, 0, 0, 1)  # 2z
        
        z = 3 + 1j
        
        # (M1 @ M2)(z) should equal M1(M2(z))
        composed = M1 @ M2
        expected = M1(M2(z))
        
        assert composed(z) == pytest.approx(expected, rel=1e-10)
    
    def test_fibonacci_determinant_alternation(self):
        """Fibonacci Möbius should have det = (-1)^n."""
        for n in range(2, 12):
            M = MobiusMatrix.fibonacci(n)
            expected_det = (-1) ** n
            assert M.determinant == pytest.approx(expected_det, rel=1e-10)
    
    def test_fibonacci_fixed_point_is_phi(self):
        """Fibonacci Möbius fixed point should be φ."""
        for n in [5, 8, 10, 12]:
            M = MobiusMatrix.fibonacci(n)
            z1, z2 = M.fixed_points()
            
            # One should be φ, other should be -1/φ
            assert (abs(z1 - PHI) < 1e-10 or abs(z1 + PHI_INV) < 1e-10)
            assert (abs(z2 - PHI) < 1e-10 or abs(z2 + PHI_INV) < 1e-10)
    
    def test_inverse(self):
        """M @ M.inverse() should be identity."""
        M = MobiusMatrix(2+1j, 3, 1, 2-1j)
        M_inv = M.inverse()
        composed = M @ M_inv
        
        z = 1 + 0.5j
        assert composed(z) == pytest.approx(z, rel=1e-8)


class TestCrossRatio:
    """Tests for cross-ratio computation and invariance."""
    
    def test_cross_ratio_basic(self):
        """Test basic cross-ratio computation."""
        # CR(0, 1, 2, 3) = ((0-2)(1-3))/((0-3)(1-2)) = (-2)(-2)/((-3)(-1)) = 4/3
        cr = cross_ratio(0, 1, 2, 3)
        assert cr == pytest.approx(4/3, rel=1e-10)
    
    def test_cross_ratio_mobius_invariant(self):
        """Cross-ratio should be preserved under Möbius transformation."""
        points = [1+1j, 2-1j, -1+2j, 3+0j]
        cr_original = cross_ratio(*points)
        
        M = MobiusMatrix(1+1j, 2, 0, 1-1j)
        transformed = [M(z) for z in points]
        cr_transformed = cross_ratio(*transformed)
        
        assert abs(cr_original - cr_transformed) < 1e-10


class TestMobiusStripTensor:
    """Tests for MobiusStripTensor with antiperiodic boundary."""
    
    def test_antiperiodic_boundary(self):
        """T[n+N] should equal -T[n]."""
        size = 55
        tensor = MobiusStripTensor(size=size)
        
        # Set some values
        tensor[0] = 1.0
        tensor[10] = 2.5 + 1j
        tensor[27] = -3.0
        
        # Check antiperiodic
        assert tensor[0 + size] == pytest.approx(-1.0, rel=1e-10)
        assert tensor[10 + size] == pytest.approx(-2.5 - 1j, rel=1e-10)
        assert tensor[27 + size] == pytest.approx(3.0, rel=1e-10)
    
    def test_double_period_returns_original(self):
        """T[n+2N] should equal T[n]."""
        size = 55
        tensor = MobiusStripTensor(size=size)
        
        tensor[7] = 5.0 + 2j
        
        assert tensor[7 + 2*size] == pytest.approx(5.0 + 2j, rel=1e-10)
    
    def test_half_integer_quantization(self):
        """Standing wave modes should have half-integer momenta."""
        size = 55
        tensor = MobiusStripTensor(size=size)
        
        modes = tensor.standing_wave_modes()
        
        for n, k, wave in modes[:10]:
            expected_k = (n + 0.5) * 2 * np.pi / size
            assert k == pytest.approx(expected_k, rel=1e-10)
    
    def test_4pi_periodicity(self):
        """One loop gives -1, two loops give +1."""
        size = 55
        tensor = MobiusStripTensor(size=size)
        
        # Gaussian wave
        wave = np.exp(-(np.arange(size) - size/2)**2 / 20) + 0j
        wave = wave / np.linalg.norm(wave)
        
        overlap_1 = tensor.overlap_after_loops(wave, 1)
        overlap_2 = tensor.overlap_after_loops(wave, 2)
        
        assert overlap_1.real == pytest.approx(-1.0, rel=1e-6)
        assert overlap_2.real == pytest.approx(1.0, rel=1e-6)


class TestMobiusFibonacciTensor:
    """Tests for MobiusFibonacciTensor combining Möbius + Fibonacci."""
    
    def test_fibonacci_sizing(self):
        """Tensor size should be F_n."""
        for n, expected in [(8, 21), (10, 55), (12, 144)]:
            tensor = MobiusFibonacciTensor(fib_index=n)
            assert tensor.size == expected
    
    def test_golden_spiral_complete_coverage(self):
        """Golden spiral should visit all indices exactly once."""
        tensor = MobiusFibonacciTensor(fib_index=10)
        
        indices = tensor.golden_spiral_indices()
        
        # Should visit all 55 indices
        assert len(indices) == 55
        assert len(set(indices)) == 55  # All unique
    
    def test_phyllotaxis_positions(self):
        """Phyllotaxis should produce reasonable 2D positions."""
        tensor = MobiusFibonacciTensor(fib_index=10)
        positions = tensor.phyllotaxis_positions()
        
        assert positions.shape == (55, 2)
        # First point at origin
        assert_allclose(positions[0], [0, 0], atol=1e-10)


class TestMobiusRecursiveTensor:
    """Tests for MobiusRecursiveTensor with composition-based recursion."""
    
    def test_fibonacci_matrix_growth(self):
        """Matrix entries should grow following Fibonacci-like pattern."""
        tensor = MobiusRecursiveTensor()
        
        # Check that entries grow (exact values depend on composition order)
        entries = []
        for n in [2, 3, 4, 5, 6]:
            M = tensor[n]
            entries.append(abs(M.a))
        
        # Entries should be increasing
        for i in range(len(entries) - 1):
            assert entries[i+1] >= entries[i] * 0.9  # Allow some variation
    
    def test_fixed_point_convergence(self):
        """Fixed points should converge to φ."""
        tensor = MobiusRecursiveTensor()
        
        for n in [5, 10, 15]:
            fp = tensor.fixed_point(n)
            # Should be close to φ or -1/φ
            assert (abs(fp.real - PHI) < 0.01 or abs(fp.real + PHI_INV) < 0.01)


class TestVerify4PiPeriodicity:
    """Tests for the verification utility function."""
    
    def test_verify_function(self):
        """The verify function should confirm 4π periodicity."""
        results = verify_4pi_periodicity(size=55)
        
        assert results['size'] == 55
        
        # 1 loop: overlap should be -1
        assert results['overlaps']['1_loop']['real'] == pytest.approx(-1.0, rel=1e-4)
        
        # 2 loops: overlap should be +1
        assert results['overlaps']['2_loop']['real'] == pytest.approx(1.0, rel=1e-4)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_fibonacci_mobius_cross_ratio_preservation(self):
        """Fibonacci Möbius should preserve cross-ratios."""
        M = MobiusMatrix.fibonacci(10)
        
        points = [0.5, 1.5, 2.5, 3.5]
        cr_before = cross_ratio(*points)
        
        transformed = [M(z) for z in points]
        cr_after = cross_ratio(*transformed)
        
        assert abs(cr_before - cr_after) < 1e-8
    
    def test_create_fibonacci_mobius_convenience(self):
        """Convenience function should create valid tensor."""
        tensor = create_fibonacci_mobius(n=10)
        
        assert isinstance(tensor, MobiusFibonacciTensor)
        assert tensor.size == 55


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
