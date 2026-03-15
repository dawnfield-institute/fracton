"""
Quantum Balance Equation (QBE) Regulator

Enforces the fundamental E<->I coupling constraint:
dI/dt + dE/dt = lambda*QPL(t)

This constraint:
- Prevents runaway growth
- Enforces E<->I equivalence
- Creates natural stability
- Generates quantum mechanics

QPL(t) = Quantum Potential Layer oscillating at 0.020 Hz

Torch-only backend.
"""

import math
import torch
from typing import Union, Tuple, Optional


class QBERegulator:
    """
    Quantum Balance Equation regulator.

    Enforces: dI/dt + dE/dt = lambda*QPL(t)

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
                 backend: str = 'torch'):
        """
        Initialize QBE regulator.

        Args:
            lambda_qbe: QBE coupling constant (dimensionless, default: 1.0)
            qpl_omega: QPL oscillation frequency (Hz, default: 0.020 from experiments)
            backend: Kept for API compatibility (always uses torch)
        """
        self.lambda_qbe = lambda_qbe
        self.qpl_omega = qpl_omega
        self.backend = 'torch'
        self.time = 0.0

    def compute_qpl(self, t: Optional[float] = None) -> torch.Tensor:
        """
        Compute Quantum Potential Layer at time t.

        QPL(t) = cos(omega*t) where omega = 0.020 Hz (universal frequency)

        Args:
            t: Time (uses internal time if None)

        Returns:
            QPL value at time t
        """
        if t is None:
            t = self.time

        return torch.cos(torch.tensor(self.qpl_omega * t))

    def enforce_qbe_constraint(self,
                              dE_dt: torch.Tensor,
                              dI_dt: torch.Tensor,
                              t: Optional[float] = None) -> Tuple[torch.Tensor, ...]:
        """
        Enforce QBE constraint on field derivatives.

        Computes QBE-constrained dI/dt from dE/dt:
        dI_dt_qbe = lambda*QPL(t) - dE_dt

        This ensures: dI/dt + dE/dt = lambda*QPL(t)

        Args:
            dE_dt: Energy time derivative
            dI_dt: Information time derivative (will be replaced)
            t: Current time

        Returns:
            Tuple of (dE_dt, dI_dt_qbe) with constraint enforced
        """
        # Compute QPL at current time
        qpl_t = self.compute_qpl(t)

        if not isinstance(qpl_t, torch.Tensor):
            qpl_t = torch.tensor(qpl_t, device=dE_dt.device)
        dI_dt_qbe = self.lambda_qbe * qpl_t - dE_dt

        return dE_dt, dI_dt_qbe

    def apply_to_evolution(self,
                          E: torch.Tensor,
                          I: torch.Tensor,
                          dE_dt: torch.Tensor,
                          dt: float,
                          t: Optional[float] = None) -> Tuple[torch.Tensor, ...]:
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
            'period': 2 * math.pi / self.qpl_omega if self.qpl_omega > 0 else float('inf')
        }


def enforce_qbe(E: torch.Tensor,
               I: torch.Tensor,
               dE_dt: torch.Tensor,
               dt: float,
               t: float = 0.0,
               lambda_qbe: float = 1.0,
               qpl_omega: float = 0.020,
               backend: str = None) -> Tuple[torch.Tensor, ...]:
    """
    Convenience function for applying QBE constraint.

    Args:
        E, I: Current fields
        dE_dt: Energy time derivative
        dt: Time step
        t: Current time
        lambda_qbe: QBE coupling
        qpl_omega: QPL frequency
        backend: Kept for API compatibility (always uses torch)

    Returns:
        Tuple of (E_new, I_new) with QBE applied

    Example:
        >>> E_new, I_new = enforce_qbe(E, I, dE_dt, dt=0.0001, t=time)
    """
    regulator = QBERegulator(lambda_qbe=lambda_qbe, qpl_omega=qpl_omega)
    return regulator.apply_to_evolution(E, I, dE_dt, dt, t)
