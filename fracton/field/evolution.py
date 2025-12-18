"""
Field Evolution - Klein-Gordon Dynamics

Implements field evolution using Klein-Gordon equation.
GPU-optimized for efficient batch processing.

The Klein-Gordon equation provides the physics substrate for
information dynamics in Dawn Field Theory.
"""

import torch
from typing import Optional, Tuple

from ..physics.constants import XI, PHI, LAMBDA_STAR


def evolve(field: torch.Tensor,
           steps: int = 5,
           dt: float = 0.1,
           mass: float = 1.0,
           damping: float = None) -> torch.Tensor:
    """
    Evolve field using Klein-Gordon dynamics.
    
    The Klein-Gordon equation: ∂²φ/∂t² = ∇²φ - m²φ
    With optional damping for stability.
    
    Args:
        field: Input field tensor (any shape)
        steps: Number of evolution steps
        dt: Time step size
        mass: Field mass parameter
        damping: Damping coefficient (default: 1-λ* = ξ)
        
    Returns:
        Evolved field tensor
    """
    if damping is None:
        damping = 1 - LAMBDA_STAR  # = XI
    
    # Initialize velocity (momentum)
    velocity = torch.zeros_like(field)
    
    # Evolution loop
    current = field.clone()
    for _ in range(steps):
        # Compute Laplacian (discrete approximation)
        laplacian = _compute_laplacian(current)
        
        # Klein-Gordon acceleration: ∇²φ - m²φ
        acceleration = laplacian - mass * mass * current
        
        # Update with damping
        velocity = LAMBDA_STAR * velocity + dt * acceleration
        current = current + dt * velocity
    
    return current


def evolve_batch(fields: torch.Tensor,
                 steps: int = 5,
                 dt: float = 0.1,
                 mass: float = 1.0) -> torch.Tensor:
    """
    Batch-optimized field evolution on GPU.
    
    Args:
        fields: Batch of fields, shape (batch, dim) or (batch, h, w)
        steps: Number of evolution steps
        dt: Time step size
        mass: Field mass parameter
        
    Returns:
        Evolved fields, same shape as input
    """
    velocity = torch.zeros_like(fields)
    current = fields.clone()
    
    for _ in range(steps):
        laplacian = _compute_laplacian_batch(current)
        acceleration = laplacian - mass * mass * current
        velocity = LAMBDA_STAR * velocity + dt * acceleration
        current = current + dt * velocity
    
    return current


def _compute_laplacian(field: torch.Tensor) -> torch.Tensor:
    """
    Compute discrete Laplacian of a field.
    
    For 1D: ∇²f[i] = f[i-1] - 2f[i] + f[i+1]
    For 2D: ∇²f[i,j] = f[i-1,j] + f[i+1,j] + f[i,j-1] + f[i,j+1] - 4f[i,j]
    
    Uses periodic boundary conditions.
    """
    if field.dim() == 1:
        # 1D Laplacian
        left = torch.roll(field, 1, dims=0)
        right = torch.roll(field, -1, dims=0)
        return left + right - 2 * field
    
    elif field.dim() == 2:
        # 2D Laplacian
        up = torch.roll(field, 1, dims=0)
        down = torch.roll(field, -1, dims=0)
        left = torch.roll(field, 1, dims=1)
        right = torch.roll(field, -1, dims=1)
        return up + down + left + right - 4 * field
    
    else:
        # General N-D Laplacian
        laplacian = -2 * field.dim() * field
        for dim in range(field.dim()):
            laplacian = laplacian + torch.roll(field, 1, dims=dim)
            laplacian = laplacian + torch.roll(field, -1, dims=dim)
        return laplacian


def _compute_laplacian_batch(fields: torch.Tensor) -> torch.Tensor:
    """Compute Laplacian for batched fields."""
    if fields.dim() == 2:
        # Shape: (batch, dim) - 1D fields
        left = torch.roll(fields, 1, dims=1)
        right = torch.roll(fields, -1, dims=1)
        return left + right - 2 * fields
    
    elif fields.dim() == 3:
        # Shape: (batch, h, w) - 2D fields
        up = torch.roll(fields, 1, dims=1)
        down = torch.roll(fields, -1, dims=1)
        left = torch.roll(fields, 1, dims=2)
        right = torch.roll(fields, -1, dims=2)
        return up + down + left + right - 4 * fields
    
    else:
        raise ValueError(f"Unsupported batch dimension: {fields.dim()}")


def evolve_with_source(field: torch.Tensor,
                       source: torch.Tensor,
                       steps: int = 5,
                       dt: float = 0.1,
                       mass: float = 1.0,
                       coupling: float = None) -> torch.Tensor:
    """
    Evolve field with external source term.
    
    Equation: ∂²φ/∂t² = ∇²φ - m²φ + g·J
    
    Args:
        field: Input field tensor
        source: Source term J
        steps: Number of evolution steps
        dt: Time step size
        mass: Field mass parameter
        coupling: Source coupling strength (default: ξ)
        
    Returns:
        Evolved field tensor
    """
    if coupling is None:
        coupling = XI
    
    velocity = torch.zeros_like(field)
    current = field.clone()
    
    for _ in range(steps):
        laplacian = _compute_laplacian(current)
        acceleration = laplacian - mass * mass * current + coupling * source
        velocity = LAMBDA_STAR * velocity + dt * acceleration
        current = current + dt * velocity
    
    return current


def compute_field_energy(field: torch.Tensor,
                         velocity: Optional[torch.Tensor] = None,
                         mass: float = 1.0) -> float:
    """
    Compute total energy of a field configuration.
    
    E = ∫ [½(∂φ/∂t)² + ½|∇φ|² + ½m²φ²] dx
    
    Args:
        field: Field tensor
        velocity: Field velocity (time derivative)
        mass: Field mass parameter
        
    Returns:
        Total energy
    """
    # Kinetic energy
    if velocity is not None:
        kinetic = 0.5 * torch.sum(velocity ** 2).item()
    else:
        kinetic = 0.0
    
    # Gradient energy (approximate via Laplacian)
    laplacian = _compute_laplacian(field)
    gradient = -0.5 * torch.sum(field * laplacian).item()
    
    # Mass energy
    mass_energy = 0.5 * mass * mass * torch.sum(field ** 2).item()
    
    return kinetic + gradient + mass_energy


def dissipate(field: torch.Tensor,
              rate: float = None,
              threshold: float = None) -> torch.Tensor:
    """
    Apply dissipation to field values below threshold.
    
    Mimics the collapse phase of SEC dynamics.
    
    Args:
        field: Input field tensor
        rate: Dissipation rate (default: 1-λ*)
        threshold: Values below this dissipate faster (default: ξ)
        
    Returns:
        Dissipated field
    """
    if rate is None:
        rate = 1 - LAMBDA_STAR  # = XI
    if threshold is None:
        threshold = XI
    
    # Create mask for below-threshold values
    mask = torch.abs(field) < threshold
    
    # Apply stronger dissipation to masked values
    result = field.clone()
    result[mask] = result[mask] * (1 - rate * PHI)  # Faster decay
    result[~mask] = result[~mask] * (1 - rate)  # Normal decay
    
    return result


def amplify(field: torch.Tensor,
            rate: float = None,
            threshold: float = None) -> torch.Tensor:
    """
    Apply amplification to field values above threshold.
    
    Mimics the expansion phase of SEC dynamics.
    
    Args:
        field: Input field tensor
        rate: Amplification rate (default: ξ)
        threshold: Values above this amplify (default: φ×ξ)
        
    Returns:
        Amplified field
    """
    if rate is None:
        rate = XI
    if threshold is None:
        threshold = PHI * XI  # PHI_XI
    
    # Create mask for above-threshold values
    mask = torch.abs(field) > threshold
    
    # Apply amplification to masked values
    result = field.clone()
    result[mask] = result[mask] * (1 + rate)
    
    return result
