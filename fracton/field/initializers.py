"""
Field Initialization Primitives

Universal field initialization strategies for Dawn Field systems.
These initializers use PyTorch tensors and provide standardized starting
conditions for different field types.

Compatible with:
- Reality Engine (PyTorch/CUDA)
- GAIA (PyTorch)
- Any Dawn Field application
"""

import math
import torch
from typing import Tuple, Union, Optional
from abc import ABC, abstractmethod


class FieldInitializer(ABC):
    """
    Base class for field initialization strategies.

    Provides a common interface for creating initial field configurations
    that work across different Dawn Field applications.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the field initializer.

        Args:
            seed: Random seed for reproducibility (optional)
        """
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    @abstractmethod
    def initialize(self, shape: Tuple[int, ...], backend: str = 'torch') -> torch.Tensor:
        """
        Initialize a field with the given shape.

        Args:
            shape: Dimensions of the field (e.g., (64, 64, 64) for 3D)
            backend: Kept for API compatibility (always uses torch)

        Returns:
            Initialized field tensor
        """
        pass


class UniformInitializer(FieldInitializer):
    """
    Uniform random initialization.

    Creates fields with uniform random values in a specified range.
    Useful for baseline testing and simple systems.
    """

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0, seed: Optional[int] = None):
        """
        Args:
            min_val: Minimum field value
            max_val: Maximum field value
            seed: Random seed
        """
        super().__init__(seed)
        self.min_val = min_val
        self.max_val = max_val

    def initialize(self, shape: Tuple[int, ...], backend: str = 'torch') -> torch.Tensor:
        """Create uniform random field."""
        field = torch.rand(*shape) * (self.max_val - self.min_val) + self.min_val
        return field


class GaussianHotspotInitializer(FieldInitializer):
    """
    Gaussian hotspot initialization.

    Creates localized Gaussian perturbations - like seeds for structure formation.
    Used for CMB-like fluctuations and particle nucleation sites.

    Based on:
    - Cosmic microwave background patterns
    - Mobius braided strand topology
    - Structure seeds for emergence
    """

    def __init__(self, n_hotspots: int = 12, amplitude: float = 0.5,
                 width: float = 5.0, background: float = 0.05, seed: Optional[int] = None):
        """
        Args:
            n_hotspots: Number of Gaussian hotspots to create
            amplitude: Peak amplitude of each hotspot
            width: Gaussian width (sigma) in grid units
            background: Background field level
            seed: Random seed
        """
        super().__init__(seed)
        self.n_hotspots = n_hotspots
        self.amplitude = amplitude
        self.width = width
        self.background = background

    def initialize(self, shape: Tuple[int, ...], backend: str = 'torch') -> torch.Tensor:
        """Create field with Gaussian hotspots."""
        field = torch.ones(*shape) * self.background

        # Add Gaussian hotspots
        for _ in range(self.n_hotspots):
            # Random center point
            center = tuple(torch.randint(0, s, (1,)).item() for s in shape)

            # Create coordinate grids
            coords = torch.meshgrid(
                *[torch.arange(s, dtype=torch.float32) for s in shape],
                indexing='ij'
            )
            # Compute distance from center
            dist_sq = sum((coord - c)**2 for coord, c in zip(coords, center))
            # Gaussian
            gaussian = self.amplitude * torch.exp(-dist_sq / (2 * self.width**2))

            field += gaussian

        return field


class BraidedStrandInitializer(FieldInitializer):
    """
    Braided strand topology initialization.

    Creates intertwined field patterns like Mobius strips or DNA helices.
    Useful for topologically interesting starting conditions.

    Based on Mobius-Confluence topology from Dawn Field Theory.
    """

    def __init__(self, n_strands: int = 3, twist_rate: float = 0.1,
                 amplitude: float = 0.5, seed: Optional[int] = None):
        """
        Args:
            n_strands: Number of intertwined strands
            twist_rate: Rate of twisting (radians per unit length)
            amplitude: Strand amplitude
            seed: Random seed
        """
        super().__init__(seed)
        self.n_strands = n_strands
        self.twist_rate = twist_rate
        self.amplitude = amplitude

    def initialize(self, shape: Tuple[int, ...], backend: str = 'torch') -> torch.Tensor:
        """Create braided strand field."""
        field = torch.zeros(*shape)

        # Create coordinates
        if len(shape) == 3:
            # 3D braided strands
            x, y, z = torch.meshgrid(
                torch.linspace(0, 2*math.pi, shape[0]),
                torch.linspace(0, 2*math.pi, shape[1]),
                torch.linspace(0, 2*math.pi, shape[2]),
                indexing='ij'
            )

            # Create intertwined helical patterns
            for i in range(self.n_strands):
                phase = (2 * math.pi * i) / self.n_strands
                strand = self.amplitude * torch.sin(x + phase) * torch.cos(y + self.twist_rate * z)
                field += strand

        elif len(shape) == 2:
            # 2D spiral pattern
            x, y = torch.meshgrid(
                torch.linspace(-math.pi, math.pi, shape[0]),
                torch.linspace(-math.pi, math.pi, shape[1]),
                indexing='ij'
            )

            r = torch.sqrt(x**2 + y**2)
            theta = torch.atan2(y, x)

            for i in range(self.n_strands):
                phase = (2 * math.pi * i) / self.n_strands
                strand = self.amplitude * torch.sin(self.n_strands * theta + self.twist_rate * r + phase)
                field += strand

        return field


class CMBInitializer(GaussianHotspotInitializer):
    """
    Cosmic Microwave Background-like initialization.

    Alias for GaussianHotspotInitializer with CMB-appropriate parameters.
    Creates fluctuation patterns similar to the cosmic microwave background.

    Default parameters match observed CMB statistics:
    - ~10^-5 amplitude fluctuations
    - Correlation length ~5-10 units
    - Multiple scales of structure
    """

    def __init__(self, n_hotspots: int = 12, seed: Optional[int] = None):
        """
        Args:
            n_hotspots: Number of primordial fluctuations
            seed: Random seed
        """
        super().__init__(
            n_hotspots=n_hotspots,
            amplitude=0.3,     # CMB fluctuation amplitude
            width=7.0,         # CMB correlation length
            background=0.05,   # Baseline field level
            seed=seed
        )


# Convenience function for quick initialization
def initialize_field(shape: Tuple[int, ...],
                     pattern: str = 'cmb',
                     backend: str = 'torch',
                     seed: Optional[int] = None,
                     **kwargs) -> torch.Tensor:
    """
    Quick field initialization with named patterns.

    Args:
        shape: Field dimensions
        pattern: Initialization pattern ('uniform', 'cmb', 'hotspot', 'braided')
        backend: Kept for API compatibility (always uses torch)
        seed: Random seed
        **kwargs: Additional parameters for the initializer

    Returns:
        Initialized field tensor

    Example:
        >>> field = initialize_field((64, 64, 64), pattern='cmb')
    """
    patterns = {
        'uniform': UniformInitializer,
        'cmb': CMBInitializer,
        'hotspot': GaussianHotspotInitializer,
        'braided': BraidedStrandInitializer
    }

    if pattern not in patterns:
        raise ValueError(f"Unknown pattern '{pattern}'. Choose from: {list(patterns.keys())}")

    initializer = patterns[pattern](seed=seed, **kwargs)
    return initializer.initialize(shape, backend=backend)
