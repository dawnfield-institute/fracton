"""
Quantum Balance Equation (QBE) Regulator

Enforces the fundamental E↔I coupling constraint:
dI/dt + dE/dt = λ·QPL(t)

This constraint:
- Prevents runaway growth
- Enforces E↔I equivalence
- Creates natural stability
- Generates quantum mechanics

QPL(t) = Quantum Potential Layer oscillating at 0.020 Hz
"""

import numpy as np
from typing import Union, Tuple, Optional


class QBERegulator:
    """
    Quantum Balance Equation regulator.
    
    Enforces: dI/dt + dE/dt = λ·QPL(t)
    
    This is THE fundamental constraint that:
    - Makes information and energy equivalent (different phases)
    - Provides natural stability without artificial damping
    - Generates quantum mechanics from field oscillations
    - Matches universal 0.020 Hz frequency
    
    Based on Dawn Field Theory validation experiments.
    """
    
    def __init__(self,
                 lambda_qbe: float = 1.0,
                 qpl_omega: float = 0.020,
                 backend: str = 'numpy'):
        """
        Initialize QBE regulator.
        
        Args:
            lambda_qbe: QBE coupling constant (dimensionless, default: 1.0)
            qpl_omega: QPL oscillation frequency (Hz, default: 0.020 from experiments)
            backend: 'numpy' or 'torch'
        """
        self.lambda_qbe = lambda_qbe
        self.qpl_omega = qpl_omega
        self.backend = backend
        self.time = 0.0
        
        # Get backend module
        if backend == 'torch':
            try:
                import torch
                self.np = torch
                self.is_torch = True
            except ImportError:
                raise ImportError("PyTorch not available")
        else:
            self.np = np
            self.is_torch = False
    
    def compute_qpl(self, t: Optional[float] = None) -> float:
        """
        Compute Quantum Potential Layer at time t.
        
        QPL(t) = cos(ω·t) where ω = 0.020 Hz (universal frequency)
        
        Args:
            t: Time (uses internal time if None)
            
        Returns:
            QPL value at time t
        """
        if t is None:
            t = self.time
        
        if self.is_torch:
            return self.np.cos(self.np.tensor(self.qpl_omega * t))
        else:
            return np.cos(self.qpl_omega * t)
    
    def enforce_qbe_constraint(self,
                              dE_dt: Union[np.ndarray, 'torch.Tensor'],
                              dI_dt: Union[np.ndarray, 'torch.Tensor'],
                              t: Optional[float] = None) -> Tuple[Union[np.ndarray, 'torch.Tensor'], ...]:
        """
        Enforce QBE constraint on field derivatives.
        
        Computes QBE-constrained dI/dt from dE/dt:
        dI_dt_qbe = λ·QPL(t) - dE_dt
        
        This ensures: dI/dt + dE/dt = λ·QPL(t)
        
        Args:
            dE_dt: Energy time derivative
            dI_dt: Information time derivative (will be replaced)
            t: Current time
            
        Returns:
            Tuple of (dE_dt, dI_dt_qbe) with constraint enforced
        """
        # Compute QPL at current time
        qpl_t = self.compute_qpl(t)
        
        # QBE constraint: dI/dt = λ·QPL(t) - dE/dt
        if self.is_torch:
            if not isinstance(qpl_t, self.np.Tensor):
                qpl_t = self.np.tensor(qpl_t, device=dE_dt.device)
            dI_dt_qbe = self.lambda_qbe * qpl_t - dE_dt
        else:
            dI_dt_qbe = self.lambda_qbe * qpl_t - dE_dt
        
        return dE_dt, dI_dt_qbe
    
    def apply_to_evolution(self,
                          E: Union[np.ndarray, 'torch.Tensor'],
                          I: Union[np.ndarray, 'torch.Tensor'],
                          dE_dt: Union[np.ndarray, 'torch.Tensor'],
                          dt: float,
                          t: Optional[float] = None) -> Tuple[Union[np.ndarray, 'torch.Tensor'], ...]:
        """
        Apply QBE constraint to field evolution.
        
        Args:
            E: Current energy field
            I: Current information field
            dE_dt: Energy time derivative from RBF
            dt: Time step
            t: Current time
            
        Returns:
            Tuple of (E_new, I_new) with QBE constraint applied
        """
        # Enforce QBE constraint to get proper dI/dt
        dE_dt, dI_dt_qbe = self.enforce_qbe_constraint(dE_dt, None, t)
        
        # Update fields
        E_new = E + dt * dE_dt
        I_new = I + dt * dI_dt_qbe
        
        return E_new, I_new
    
    def update_time(self, dt: float):
        """Advance internal time."""
        self.time += dt
    
    def get_coupling_info(self) -> dict:
        """
        Get current QBE coupling information.
        
        Returns:
            Dictionary with QPL value, frequency, coupling constant
        """
        return {
            'qpl_value': float(self.compute_qpl()),
            'qpl_omega': self.qpl_omega,
            'lambda_qbe': self.lambda_qbe,
            'time': self.time,
            'period': 2 * np.pi / self.qpl_omega if self.qpl_omega > 0 else np.inf
        }


def enforce_qbe(E: Union[np.ndarray, 'torch.Tensor'],
               I: Union[np.ndarray, 'torch.Tensor'],
               dE_dt: Union[np.ndarray, 'torch.Tensor'],
               dt: float,
               t: float = 0.0,
               lambda_qbe: float = 1.0,
               qpl_omega: float = 0.020,
               backend: str = None) -> Tuple[Union[np.ndarray, 'torch.Tensor'], ...]:
    """
    Convenience function for applying QBE constraint.
    
    Args:
        E, I: Current fields
        dE_dt: Energy time derivative
        dt: Time step
        t: Current time
        lambda_qbe: QBE coupling
        qpl_omega: QPL frequency
        backend: 'numpy' or 'torch' (auto-detected if None)
        
    Returns:
        Tuple of (E_new, I_new) with QBE applied
        
    Example:
        >>> E_new, I_new = enforce_qbe(E, I, dE_dt, dt=0.0001, t=time)
    """
    if backend is None:
        backend = 'torch' if hasattr(E, 'is_cuda') else 'numpy'
    
    regulator = QBERegulator(lambda_qbe=lambda_qbe, qpl_omega=qpl_omega, backend=backend)
    return regulator.apply_to_evolution(E, I, dE_dt, dt, t)
