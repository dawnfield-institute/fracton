"""
Recursive Balance Field (RBF) Engine

The universal field evolution equation from Dawn Field Theory.
Implements the core dynamics that generate emergent physics.

B(x,t) = nabla^2(E-I) + lambda*M*nabla^2(M) - alpha*||E-I||^2 - gamma*(E-I)

Where:
- E = Energy field (actualization)
- I = Information field (potential)
- M = Memory field (persistence)
- lambda = Memory coupling constant (0.020 from experiments)
- alpha = Collapse rate (0.964 from PAC validation)
- gamma = Dissipation coefficient (0.01 for stability & PAC conservation)

The damping term gamma(E-I) emerges from entropy production and is essential
for numerical stability and proper PAC conservation.

Torch-only backend.
"""

import math
import torch
from typing import Tuple, Union, Optional, Dict


class RBFEngine:
    """
    Recursive Balance Field evolution engine.

    The fundamental equation that generates all emergent physics
    from pure field dynamics. No hardcoded particles, forces, or
    quantum mechanics - everything emerges from RBF + PAC.

    Validated constants from Dawn Field Theory experiments:
    - lambda_mem = 0.020 Hz (universal frequency)
    - alpha_collapse = 0.964 (PAC correlation)
    """

    def __init__(self,
                 lambda_mem: float = 0.020,
                 alpha_collapse: float = 0.964,
                 gamma_damping: float = 0.1,
                 backend: str = 'torch'):
        """
        Initialize RBF engine with validated constants.

        Args:
            lambda_mem: Memory coupling constant (default: 0.020 from experiments)
            alpha_collapse: Collapse rate (default: 0.964 from PAC validation)
            gamma_damping: Dissipation coefficient (default: 0.1 for stability)
            backend: Kept for API compatibility (always uses torch)
        """
        self.lambda_mem = lambda_mem
        self.alpha_collapse = alpha_collapse
        self.gamma_damping = gamma_damping
        self.backend = 'torch'

    def compute_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian (nabla^2) using finite differences with spike protection.

        Works for 2D and 3D fields with periodic boundary conditions.
        Added clipping to prevent extreme gradient spikes in long simulations.

        Args:
            field: Input field tensor

        Returns:
            Laplacian of the field (clamped to prevent overflow)
        """
        laplacian = torch.zeros_like(field)
        ndim = len(field.shape)

        for axis in range(ndim):
            laplacian += (
                torch.roll(field, 1, dims=axis) +
                torch.roll(field, -1, dims=axis) -
                2 * field
            )

        # CRITICAL: Clamp laplacian to prevent extreme spikes
        laplacian = torch.clamp(laplacian, -1000.0, 1000.0)

        return laplacian

    def compute_balance_field(self,
                             E: torch.Tensor,
                             I: torch.Tensor,
                             M: torch.Tensor) -> torch.Tensor:
        """
        Compute the Recursive Balance Field.

        B(x,t) = nabla^2(E-I) + lambda*M*nabla^2(M) - alpha*||E-I||^2 - gamma*(E-I)

        This is the CORE equation - all physics emerges from this.
        The damping term gamma(E-I) provides natural dissipation and stability.

        Args:
            E: Energy field (actualization)
            I: Information field (potential)
            M: Memory field (persistence)

        Returns:
            Balance field B(x,t)
        """
        # Term 1: Laplacian of potential difference
        potential_diff = E - I
        term1 = self.compute_laplacian(potential_diff)

        # Term 2: Memory-modulated diffusion
        term2 = self.lambda_mem * M * self.compute_laplacian(M)

        # Term 3: Nonlinear collapse term
        term3 = -self.alpha_collapse * (potential_diff ** 2)

        # Term 4: Dissipation/damping
        term4 = -self.gamma_damping * potential_diff

        # Combine all terms
        balance_field = term1 + term2 + term3 + term4

        return balance_field

    def evolve_step(self,
                   E: torch.Tensor,
                   I: torch.Tensor,
                   M: torch.Tensor,
                   dt: float = 0.0001) -> Tuple[torch.Tensor, ...]:
        """
        Evolve fields by one time step using RBF dynamics.

        Args:
            E: Energy field
            I: Information field
            M: Memory field
            dt: Time step

        Returns:
            Tuple of (E_new, I_new, M_new)
        """
        # Compute balance field
        B = self.compute_balance_field(E, I, M)

        # Energy evolves according to balance field
        dE_dt = B

        # Information has opposing dynamics (E<->I oscillation)
        dI_dt = -B

        # Memory grows where collapse occurs (high ||E-I||)
        dM_dt = self.alpha_collapse * ((E - I) ** 2)

        # Update fields
        E_new = E + dt * dE_dt
        I_new = I + dt * dI_dt
        M_new = M + dt * dM_dt

        return E_new, I_new, M_new

    def get_field_stats(self,
                       E: torch.Tensor,
                       I: torch.Tensor,
                       M: torch.Tensor) -> Dict[str, float]:
        """
        Compute field statistics for monitoring.

        Returns:
            Dictionary with mean, std, min, max for each field
        """
        def stats(field):
            return {
                'mean': field.mean().item(),
                'std': field.std().item(),
                'min': field.min().item(),
                'max': field.max().item()
            }

        return {
            'E': stats(E),
            'I': stats(I),
            'M': stats(M),
            'E+I+M': stats(E + I + M)  # PAC conservation check
        }


def compute_rbf_balance(E: torch.Tensor,
                       I: torch.Tensor,
                       M: torch.Tensor,
                       lambda_mem: float = 0.020,
                       alpha_collapse: float = 0.964,
                       backend: str = None) -> torch.Tensor:
    """
    Convenience function for computing balance field.

    Args:
        E, I, M: Field tensors
        lambda_mem: Memory coupling
        alpha_collapse: Collapse rate
        backend: Kept for API compatibility (always uses torch)

    Returns:
        Balance field B(x,t)

    Example:
        >>> B = compute_rbf_balance(E, I, M)
    """
    engine = RBFEngine(lambda_mem=lambda_mem, alpha_collapse=alpha_collapse)
    return engine.compute_balance_field(E, I, M)
