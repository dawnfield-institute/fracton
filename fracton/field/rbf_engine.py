"""
Recursive Balance Field (RBF) Engine

The universal field evolution equation from Dawn Field Theory.
Implements the core dynamics that generate emergent physics.

B(x,t) = ∇²(E-I) + λM∇²M - α||E-I||²

Where:
- E = Energy field (actualization)
- I = Information field (potential)
- M = Memory field (persistence)
- λ = Memory coupling constant (0.020 from experiments)
- α = Collapse rate (0.964 from PAC validation)

Compatible with both NumPy and PyTorch backends.
"""

import numpy as np
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
                 backend: str = 'numpy'):
        """
        Initialize RBF engine with validated constants.
        
        Args:
            lambda_mem: Memory coupling constant (default: 0.020 from experiments)
            alpha_collapse: Collapse rate (default: 0.964 from PAC validation)
            backend: 'numpy' for CPU or 'torch' for GPU
        """
        self.lambda_mem = lambda_mem
        self.alpha_collapse = alpha_collapse
        self.backend = backend
        
        # Get backend module
        if backend == 'torch':
            try:
                import torch
                self.np = torch
                self.is_torch = True
            except ImportError:
                raise ImportError("PyTorch not available. Install with: pip install torch")
        else:
            self.np = np
            self.is_torch = False
    
    def compute_laplacian(self, field: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Compute Laplacian (∇²) using finite differences.
        
        Works for 2D and 3D fields with periodic boundary conditions.
        
        Args:
            field: Input field array
            
        Returns:
            Laplacian of the field
        """
        laplacian = self.np.zeros_like(field)
        ndim = len(field.shape)
        
        for axis in range(ndim):
            # Roll for finite differences (periodic boundaries)
            laplacian += (
                self.np.roll(field, 1, axis=axis) + 
                self.np.roll(field, -1, axis=axis) - 
                2 * field
            )
        
        return laplacian
    
    def compute_balance_field(self, 
                             E: Union[np.ndarray, 'torch.Tensor'],
                             I: Union[np.ndarray, 'torch.Tensor'],
                             M: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Compute the Recursive Balance Field.
        
        B(x,t) = ∇²(E-I) + λM∇²M - α||E-I||²
        
        This is the CORE equation - all physics emerges from this.
        
        Args:
            E: Energy field (actualization)
            I: Information field (potential)
            M: Memory field (persistence)
            
        Returns:
            Balance field B(x,t)
        """
        # Term 1: Laplacian of potential difference
        # This creates wave dynamics and drives E↔I oscillation
        potential_diff = E - I
        term1 = self.compute_laplacian(potential_diff)
        
        # Term 2: Memory-modulated diffusion
        # Memory acts as a "stiffness" - areas with M resist change
        term2 = self.lambda_mem * M * self.compute_laplacian(M)
        
        # Term 3: Nonlinear collapse term
        # When ||E-I||² is large, collapse accelerates
        if self.is_torch:
            term3 = -self.alpha_collapse * (potential_diff ** 2)
        else:
            term3 = -self.alpha_collapse * np.square(potential_diff)
        
        # Combine all terms
        balance_field = term1 + term2 + term3
        
        return balance_field
    
    def evolve_step(self,
                   E: Union[np.ndarray, 'torch.Tensor'],
                   I: Union[np.ndarray, 'torch.Tensor'],
                   M: Union[np.ndarray, 'torch.Tensor'],
                   dt: float = 0.0001) -> Tuple[Union[np.ndarray, 'torch.Tensor'], ...]:
        """
        Evolve fields by one time step using RBF dynamics.
        
        Updates:
        - dE/dt from balance field
        - dI/dt from QBE constraint (if enabled)
        - dM/dt from collapse events
        
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
        
        # Information has opposing dynamics (E↔I oscillation)
        dI_dt = -B
        
        # Memory grows where collapse occurs (high ||E-I||)
        if self.is_torch:
            dM_dt = self.alpha_collapse * ((E - I) ** 2)
        else:
            dM_dt = self.alpha_collapse * np.square(E - I)
        
        # Update fields
        E_new = E + dt * dE_dt
        I_new = I + dt * dI_dt
        M_new = M + dt * dM_dt
        
        return E_new, I_new, M_new
    
    def get_field_stats(self, 
                       E: Union[np.ndarray, 'torch.Tensor'],
                       I: Union[np.ndarray, 'torch.Tensor'],
                       M: Union[np.ndarray, 'torch.Tensor']) -> Dict[str, float]:
        """
        Compute field statistics for monitoring.
        
        Returns:
            Dictionary with mean, std, min, max for each field
        """
        def stats(field):
            if self.is_torch:
                return {
                    'mean': field.mean().item(),
                    'std': field.std().item(),
                    'min': field.min().item(),
                    'max': field.max().item()
                }
            else:
                return {
                    'mean': np.mean(field),
                    'std': np.std(field),
                    'min': np.min(field),
                    'max': np.max(field)
                }
        
        return {
            'E': stats(E),
            'I': stats(I),
            'M': stats(M),
            'E+I+M': stats(E + I + M)  # PAC conservation check
        }


def compute_rbf_balance(E: Union[np.ndarray, 'torch.Tensor'],
                       I: Union[np.ndarray, 'torch.Tensor'],
                       M: Union[np.ndarray, 'torch.Tensor'],
                       lambda_mem: float = 0.020,
                       alpha_collapse: float = 0.964,
                       backend: str = None) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Convenience function for computing balance field.
    
    Args:
        E, I, M: Field arrays
        lambda_mem: Memory coupling
        alpha_collapse: Collapse rate
        backend: 'numpy' or 'torch' (auto-detected if None)
        
    Returns:
        Balance field B(x,t)
        
    Example:
        >>> B = compute_rbf_balance(E, I, M)
    """
    if backend is None:
        backend = 'torch' if hasattr(E, 'is_cuda') else 'numpy'
    
    engine = RBFEngine(lambda_mem=lambda_mem, alpha_collapse=alpha_collapse, backend=backend)
    return engine.compute_balance_field(E, I, M)
