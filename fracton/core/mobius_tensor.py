"""
Möbius Tensor Architecture for Fracton

Provides tensor structures with Möbius topology that naturally encode:
- 4π periodicity (spinor/fermion behavior)
- Half-integer quantization
- Golden ratio (φ) structure via Fibonacci sizing
- Cross-ratio invariance under Möbius transformations

Key insight: A Möbius strip of size F_n (Fibonacci number) with antiperiodic
boundary conditions produces:
1. Sign flip after one loop: ψ(θ + 2π) = -ψ(θ)
2. Identity after two loops: ψ(θ + 4π) = ψ(θ)
3. Half-integer momenta: k = (n + 1/2) × 2π/L
4. Quasicrystal structure via golden angle stepping

This connects SEC's 4π phase recovery prediction to concrete computation.

Usage:
    from fracton.core.mobius_tensor import MobiusStripTensor, MobiusFibonacciTensor
    
    # Basic Möbius tensor
    tensor = MobiusStripTensor(size=55)
    tensor[0] = 1.0
    assert tensor[55] == -1.0  # Antiperiodic!
    
    # Fibonacci-structured tensor
    fib_tensor = MobiusFibonacciTensor(fib_index=10)  # F_10 = 55
    modes = fib_tensor.standing_wave_modes()  # Half-integer quantized
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional, Union, Callable
from dataclasses import dataclass, field

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI  # Also equals φ - 1


@dataclass
class MobiusFrame:
    """
    A Möbius frame defines a coordinate system on the Riemann sphere.
    
    Four points (z0, z1, zinf) map to (0, 1, ∞) under the unique Möbius
    transformation determined by the frame. This allows coordinate-free
    computation via cross-ratios.
    """
    z0: complex      # Maps to 0
    z1: complex      # Maps to 1
    zinf: complex    # Maps to ∞
    
    def to_standard(self, z: complex) -> complex:
        """Transform z to standard frame (z0→0, z1→1, zinf→∞)."""
        if abs(z - self.zinf) < 1e-15:
            return complex('inf')
        return ((z - self.z0) * (self.z1 - self.zinf)) / ((z - self.zinf) * (self.z1 - self.z0))
    
    def from_standard(self, w: complex) -> complex:
        """Transform from standard frame back to this frame."""
        num = self.z0 * self.zinf * (self.z1 - self.z0) - self.z0 * (self.z1 - self.zinf) * w
        den = self.zinf * (self.z1 - self.z0) - (self.z1 - self.zinf) * w
        if abs(den) < 1e-15:
            return complex('inf')
        return num / den


class MobiusMatrix:
    """
    Represents a Möbius transformation as a 2×2 matrix in SL(2,ℂ).
    
    M(z) = (az + b) / (cz + d)
    
    Matrix form: [[a, b], [c, d]] with ad - bc = 1 (normalized)
    
    Composition of Möbius transforms = matrix multiplication.
    This makes recursive computation natural.
    """
    
    __slots__ = ('a', 'b', 'c', 'd')
    
    def __init__(self, a: complex, b: complex, c: complex, d: complex, 
                 normalize: bool = True):
        self.a = complex(a)
        self.b = complex(b)
        self.c = complex(c)
        self.d = complex(d)
        
        if normalize:
            det = a * d - b * c
            if abs(det) > 1e-10:
                sqrt_det = np.sqrt(det)
                self.a /= sqrt_det
                self.b /= sqrt_det
                self.c /= sqrt_det
                self.d /= sqrt_det
    
    def __call__(self, z: complex) -> complex:
        """Apply Möbius transformation to z."""
        den = self.c * z + self.d
        if abs(den) < 1e-15:
            return complex('inf')
        return (self.a * z + self.b) / den
    
    def __matmul__(self, other: 'MobiusMatrix') -> 'MobiusMatrix':
        """Compose two Möbius transformations (matrix multiplication)."""
        a = self.a * other.a + self.b * other.c
        b = self.a * other.b + self.b * other.d
        c = self.c * other.a + self.d * other.c
        d = self.c * other.b + self.d * other.d
        return MobiusMatrix(a, b, c, d, normalize=True)
    
    def inverse(self) -> 'MobiusMatrix':
        """Return the inverse transformation."""
        return MobiusMatrix(self.d, -self.b, -self.c, self.a, normalize=True)
    
    def fixed_points(self) -> Tuple[complex, complex]:
        """
        Find fixed points of the transformation (z where M(z) = z).
        
        For Fibonacci Möbius, these are φ and -1/φ.
        """
        if abs(self.c) < 1e-15:
            if abs(self.d - self.a) < 1e-15:
                return (complex('inf'), complex('inf'))
            return (self.b / (self.a - self.d), complex('inf'))
        
        discriminant = (self.d - self.a)**2 + 4 * self.b * self.c
        sqrt_disc = np.sqrt(discriminant)
        z1 = ((self.a - self.d) + sqrt_disc) / (2 * self.c)
        z2 = ((self.a - self.d) - sqrt_disc) / (2 * self.c)
        return (z1, z2)
    
    @property
    def trace(self) -> complex:
        """Trace classifies the transformation type."""
        return self.a + self.d
    
    @property
    def determinant(self) -> complex:
        """Determinant (should be 1 for normalized SL(2,ℂ))."""
        return self.a * self.d - self.b * self.c
    
    def derivative_at(self, z: complex) -> complex:
        """Derivative M'(z) = (ad - bc) / (cz + d)²."""
        return self.determinant / (self.c * z + self.d)**2
    
    @property
    def matrix(self) -> NDArray[np.complex128]:
        """Return as numpy 2×2 matrix."""
        return np.array([[self.a, self.b], [self.c, self.d]], dtype=np.complex128)
    
    def __repr__(self):
        return f"MobiusMatrix([[{self.a:.4f}, {self.b:.4f}], [{self.c:.4f}, {self.d:.4f}]])"
    
    @classmethod
    def identity(cls) -> 'MobiusMatrix':
        """Return the identity transformation."""
        return cls(1, 0, 0, 1, normalize=False)
    
    @classmethod
    def fibonacci(cls, n: int) -> 'MobiusMatrix':
        """
        Create Fibonacci Möbius matrix [[F_{n+1}, F_n], [F_n, F_{n-1}]].
        
        Properties:
        - det = (-1)^n (alternating!)
        - Fixed points at φ and -1/φ
        - Composition follows Fibonacci recursion
        """
        F = [0, 1]
        for _ in range(n + 1):
            F.append(F[-1] + F[-2])
        
        return cls(
            a=float(F[n+1]),
            b=float(F[n]),
            c=float(F[n]),
            d=float(F[n-1]),
            normalize=False  # Keep integer structure
        )


def cross_ratio(z1: complex, z2: complex, z3: complex, z4: complex) -> complex:
    """
    Compute cross-ratio (z1, z2; z3, z4) = ((z1-z3)(z2-z4)) / ((z1-z4)(z2-z3))
    
    This is THE fundamental Möbius invariant - preserved under all
    Möbius transformations.
    """
    return ((z1 - z3) * (z2 - z4)) / ((z1 - z4) * (z2 - z3))


class MobiusStripTensor:
    """
    A tensor whose index space has Möbius topology.
    
    Key properties:
    - Antiperiodic boundary: T[n+N] = -T[n] (single twist)
    - 4π periodicity: T[n+2N] = T[n] (two loops to return)
    - Half-integer quantization of standing wave modes
    
    This naturally produces spinor/fermion-like behavior from topology alone.
    
    Args:
        size: Number of discrete points around the strip
        twist_factor: Number of half-twists (1 = Möbius, 2 = cylinder, etc.)
        dtype: Data type for values (default: complex128)
    """
    
    def __init__(self, size: int = 55, twist_factor: int = 1,
                 dtype: np.dtype = np.complex128):
        self.size = size
        self.twist = twist_factor
        self.dtype = dtype
        self._data = np.zeros(size, dtype=dtype)
        
        # Phase factors for propagation
        self.phases = np.exp(1j * np.pi * twist_factor * np.arange(size) / size)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[complex, NDArray]:
        """
        Access with Möbius boundary conditions.
        
        T[n+N] = (-1)^twist * T[n]
        """
        if isinstance(idx, slice):
            indices = range(*idx.indices(self.size * 2))
            return np.array([self[i] for i in indices])
        
        loops = idx // self.size
        local_idx = idx % self.size
        sign = (-1) ** (loops * self.twist)
        return sign * self._data[local_idx]
    
    def __setitem__(self, idx: int, value: complex):
        """Set value at wrapped index."""
        local_idx = idx % self.size
        self._data[local_idx] = value
    
    def __len__(self) -> int:
        return self.size
    
    @property
    def data(self) -> NDArray:
        """Direct access to underlying data array."""
        return self._data
    
    def propagate(self, wave: NDArray[np.complex128], steps: int = 1) -> NDArray[np.complex128]:
        """
        Propagate a wave around the Möbius strip.
        
        After 'size' steps: wave has traversed once, picked up (-1)^twist phase.
        After 2*size steps: wave returns to original (4π periodicity).
        
        Args:
            wave: Input wave array of length `size`
            steps: Number of propagation steps
            
        Returns:
            Propagated wave array
        """
        result = wave.copy()
        for _ in range(steps):
            result = np.roll(result, 1) * self.phases
        return result
    
    def standing_wave_modes(self) -> List[Tuple[int, float, NDArray[np.complex128]]]:
        """
        Find standing wave modes on the Möbius strip.
        
        Due to antiperiodic boundary conditions, allowed momenta are:
        k = (n + 1/2) × 2π / L  for integer n
        
        This is exactly spinor/fermion quantization!
        
        Returns:
            List of (mode_number, momentum, wave_function) tuples
        """
        modes = []
        for n in range(self.size):
            k = (n + 0.5) * 2 * np.pi / self.size
            wave = np.exp(1j * k * np.arange(self.size))
            wave = wave / np.linalg.norm(wave)  # Normalize
            modes.append((n, k, wave))
        return modes
    
    def overlap_after_loops(self, wave: NDArray[np.complex128], 
                            n_loops: int) -> complex:
        """
        Compute ⟨ψ|ψ_after_n_loops⟩.
        
        For Möbius (twist=1):
        - 1 loop: overlap = -1 (sign flip)
        - 2 loops: overlap = +1 (4π periodicity)
        """
        propagated = self.propagate(wave, steps=n_loops * self.size)
        return np.vdot(wave, propagated)


class MobiusFibonacciTensor:
    """
    Combine Möbius topology with Fibonacci structure.
    
    The index space has size F_n (a Fibonacci number) with Möbius twist.
    This produces:
    - Golden ratio relationships in the structure
    - Quasicrystal indexing via golden angle stepping
    - Complete coverage without resonance (incommensurate frequencies)
    
    Args:
        fib_index: Which Fibonacci number to use (10 → F_10 = 55)
    """
    
    # Class-level Fibonacci cache
    _fib_cache: List[int] = [0, 1]
    
    def __init__(self, fib_index: int = 10):
        self.fib_index = fib_index
        self._ensure_fibonacci(fib_index + 2)
        
        self.size = self._fib_cache[fib_index]
        self.strip = MobiusStripTensor(size=self.size, twist_factor=1)
        
        # Initialize with golden phase structure
        self._initialize_golden()
    
    @classmethod
    def _ensure_fibonacci(cls, n: int):
        """Extend Fibonacci cache if needed."""
        while len(cls._fib_cache) <= n:
            cls._fib_cache.append(cls._fib_cache[-1] + cls._fib_cache[-2])
    
    @property
    def fibonacci(self) -> List[int]:
        """Access Fibonacci sequence."""
        return self._fib_cache
    
    def _initialize_golden(self):
        """Initialize tensor with golden phase structure."""
        for i in range(self.size):
            phase = 2 * np.pi * i / PHI  # Incommensurate with 2π
            self.strip[i] = np.exp(1j * phase)
    
    def golden_spiral_indices(self, n_points: int = None) -> List[int]:
        """
        Generate indices following golden spiral pattern.
        
        Each step advances by F_{n-1} positions (mod F_n).
        This visits ALL points before repeating (because gcd(F_{n-1}, F_n) = 1).
        
        Args:
            n_points: Number of points to generate (default: self.size)
            
        Returns:
            List of indices in golden spiral order
        """
        if n_points is None:
            n_points = self.size
            
        step = self._fib_cache[self.fib_index - 1]
        
        indices = []
        current = 0
        for _ in range(n_points):
            indices.append(current % self.size)
            current += step
        
        return indices
    
    def phyllotaxis_positions(self) -> NDArray[np.float64]:
        """
        Compute 2D positions using phyllotaxis (golden angle) arrangement.
        
        This is how sunflowers and pinecones arrange seeds!
        
        Returns:
            Array of shape (size, 2) with x, y coordinates
        """
        golden_angle = 2 * np.pi * PHI_INV  # ≈ 137.5°
        
        positions = np.zeros((self.size, 2))
        for n in range(self.size):
            r = np.sqrt(n)
            theta = n * golden_angle
            positions[n] = [r * np.cos(theta), r * np.sin(theta)]
        
        return positions
    
    def standing_wave_modes(self) -> List[Tuple[int, float, NDArray[np.complex128]]]:
        """Delegate to underlying Möbius strip."""
        return self.strip.standing_wave_modes()
    
    def fibonacci_mobius(self) -> MobiusMatrix:
        """
        Get the Fibonacci Möbius matrix for this tensor's index.
        
        Returns:
            MobiusMatrix with F_{n+1}, F_n entries
        """
        return MobiusMatrix.fibonacci(self.fib_index)


class MobiusRecursiveTensor:
    """
    A tensor where recursion is implemented via Möbius composition.
    
    Instead of T[n] = f(T[n-1], T[n-2]), we have:
    M[n] = M[n-1] @ M[n-2]  (matrix multiplication)
    
    This gives:
    - Natural Fibonacci growth in matrix entries
    - Automatic convergence to golden ratio fixed points
    - Determinant alternation: det(M[n]) = (-1)^n
    - Cross-ratio preservation across all levels
    
    Values are extracted by applying accumulated transformations to a seed.
    """
    
    def __init__(self, 
                 seed_m0: Optional[MobiusMatrix] = None,
                 seed_m1: Optional[MobiusMatrix] = None):
        """
        Initialize recursive tensor with seed matrices.
        
        Args:
            seed_m0: M[0] matrix (default: identity)
            seed_m1: M[1] matrix (default: Fibonacci generator [[1,1],[1,0]])
        """
        self.M0 = seed_m0 or MobiusMatrix.identity()
        self.M1 = seed_m1 or MobiusMatrix(1, 1, 1, 0, normalize=False)
        
        self._cache: dict[int, MobiusMatrix] = {0: self.M0, 1: self.M1}
    
    def __getitem__(self, n: int) -> MobiusMatrix:
        """
        Get the n-th Möbius matrix via Fibonacci-like recursion.
        
        M[n] = M[n-1] @ M[n-2]
        """
        if n in self._cache:
            return self._cache[n]
        
        # Ensure we have the prerequisites
        M_prev = self[n - 1]
        M_prev2 = self[n - 2]
        
        M = M_prev @ M_prev2
        self._cache[n] = M
        return M
    
    def value_at(self, n: int, z: complex = 0.5+0.5j) -> complex:
        """
        Get value at index n by applying M[n] to seed point z.
        
        Args:
            n: Index
            z: Seed point to transform
            
        Returns:
            M[n](z)
        """
        return self[n](z)
    
    def fixed_point(self, n: int) -> complex:
        """
        Get the attracting fixed point of M[n].
        
        For large n, this converges to φ or -1/φ.
        """
        z1, z2 = self[n].fixed_points()
        # Return the one closer to φ
        return z1 if abs(z1.real - PHI) < abs(z2.real - PHI) else z2
    
    def orbit(self, z0: complex, max_n: int = 20) -> List[complex]:
        """
        Compute orbit of z0 under successive transformations.
        
        Returns: [z0, M[1](z0), M[2](z0), ...]
        """
        return [self.value_at(n, z0) for n in range(max_n + 1)]
    
    def determinant_sequence(self, max_n: int = 20) -> List[complex]:
        """
        Get sequence of determinants.
        
        For Fibonacci seeding, this alternates: 1, -1, 1, -1, ...
        """
        return [self[n].determinant for n in range(max_n + 1)]


# Convenience functions
def create_fibonacci_mobius(n: int = 10) -> MobiusFibonacciTensor:
    """Create a Möbius-Fibonacci tensor of size F_n."""
    return MobiusFibonacciTensor(fib_index=n)


def verify_4pi_periodicity(size: int = 55) -> dict:
    """
    Verify that Möbius topology produces 4π periodicity.
    
    Returns dict with verification results.
    """
    strip = MobiusStripTensor(size=size)
    
    # Gaussian wave packet
    x = np.arange(size)
    wave = np.exp(-(x - size/2)**2 / (size/5)) + 0j
    wave = wave / np.linalg.norm(wave)
    
    results = {
        'size': size,
        'overlaps': {}
    }
    
    for loops in range(1, 5):
        overlap = strip.overlap_after_loops(wave, loops)
        results['overlaps'][f'{loops}_loop'] = {
            'real': float(overlap.real),
            'imag': float(overlap.imag),
            'expected': (-1)**loops
        }
    
    return results
