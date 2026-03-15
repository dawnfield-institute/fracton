"""
SEC Field Evolution — continuous field dynamics via energy functional minimization.

Implements the SEC (Symbolic Entropy Collapse) energy functional:
    E(A|P,T) = α||A-P||² + β||∇A||² + γ∫T·|A|²

Evolution via gradient descent with optional Langevin thermal noise:
    ∂A/∂t = -∇E(A|P,T) + thermal_noise

This is FIELD-LEVEL SEC (continuous field evolution toward entropy collapse),
distinct from PAC-TREE SEC (storage/sec_operators.py — discrete merge/branch/gradient
on PAC nodes).

Promoted from reality-engine/conservation/sec_operator.py.
All ops use torch tensors, support batch dimensions, and are GPU-accelerable.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from fracton.backend import get_device


class SECFieldEvolver:
    """SEC operator: energy functional minimization with thermodynamic coupling.

    Evolution via gradient descent on energy functional:
        ∂A/∂t = -α(A-P) + β∇²A - γT·A + thermal_noise

    Generates heat from:
    - Information erasure (Landauer principle)
    - Collapse events (rapid entropy reduction)
    - Spatial smoothing (dissipation)

    Supports batched fields: pass (B, H, W) tensors to evolve B independent
    fields in parallel via a single batched computation.

    Args:
        alpha: Potential-Actual coupling strength (0.05-0.2 typical).
        beta: Spatial smoothing strength / MED (0.01-0.1 typical).
        gamma: Thermodynamic coupling strength (0.001-0.01 typical).
        device: Computation device. If None, uses fracton.backend.get_device().
    """

    # 5-point 2D Laplacian stencil kernel (registered as buffer, not parameter)
    _kernel: torch.Tensor

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.05,
        gamma: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device or get_device()

        # Build 5-point 2D Laplacian stencil: shape (1, 1, 3, 3)
        kernel = torch.zeros(1, 1, 3, 3, dtype=torch.float64, device=self.device)
        kernel[0, 0, 0, 1] = 1.0   # up
        kernel[0, 0, 2, 1] = 1.0   # down
        kernel[0, 0, 1, 0] = 1.0   # left
        kernel[0, 0, 1, 2] = 1.0   # right
        kernel[0, 0, 1, 1] = -4.0  # center
        self._kernel = kernel

        # Thermodynamic tracking
        self.total_heat_generated: float = 0.0
        self.total_entropy_reduced: float = 0.0
        self.collapse_event_count: int = 0
        self.collapse_events: list = []

    # ------------------------------------------------------------------
    # Energy functional
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_energy(
        self,
        A: torch.Tensor,
        P: torch.Tensor,
        T: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute energy functional components.

        E(A|P,T) = α||A-P||² + β||∇A||² + γ∫T·|A|²

        Args:
            A: Actual field (..., H, W).
            P: Potential field, same shape as A.
            T: Temperature field, same shape as A (optional).

        Returns:
            Dict with 'total', 'coupling', 'smoothness', 'thermal' energies.
        """
        # Coupling: α||A-P||²
        E_coupling = self.alpha * (A - P).pow(2).sum()

        # Smoothness: β||∇A||² via finite differences on last two dims
        grad_h = A[..., 1:, :] - A[..., :-1, :]
        grad_w = A[..., :, 1:] - A[..., :, :-1]
        E_smoothness = self.beta * (grad_h.pow(2).sum() + grad_w.pow(2).sum())

        # Thermal: γ∫T·|A|²
        E_thermal: torch.Tensor | float = 0.0
        if T is not None:
            E_thermal = self.gamma * (T * A.pow(2)).sum()

        E_total = E_coupling + E_smoothness + E_thermal

        return {
            "total": E_total.item() if isinstance(E_total, torch.Tensor) else float(E_total),
            "coupling": E_coupling.item(),
            "smoothness": E_smoothness.item(),
            "thermal": E_thermal.item() if isinstance(E_thermal, torch.Tensor) else float(E_thermal),
        }

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evolve(
        self,
        A: torch.Tensor,
        P: torch.Tensor,
        T: torch.Tensor,
        steps: int = 1,
        dt: float = 0.001,
        add_thermal_noise: bool = True,
    ) -> Tuple[torch.Tensor, float]:
        """Evolve Actual field via SEC dynamics.

        ∂A/∂t = -α(A-P) + β∇²A - γT·A + thermal_noise

        Supports batched fields: (B, H, W) evolves B fields in parallel.

        Args:
            A: Current Actual field (..., H, W).
            P: Potential field (target), same shape.
            T: Temperature field, same shape.
            steps: Number of evolution steps.
            dt: Time step size.
            add_thermal_noise: Whether to add Langevin thermal fluctuations.

        Returns:
            (A_new, total_heat_generated) tuple.
        """
        # Ensure kernel dtype/device matches field
        kernel = self._kernel.to(dtype=A.dtype, device=A.device)

        total_heat = 0.0

        for _ in range(steps):
            entropy_before = self._compute_field_entropy(A)

            # 1. Potential-Actual coupling: pulls A toward P
            dA_coupling = -self.alpha * (A - P)

            # 2. Spatial smoothing (MED): β∇²A via conv2d
            laplacian = self._laplacian_2d(A, kernel)
            dA_smooth = self.beta * laplacian

            # 3. Thermal coupling: higher T reduces effective coupling
            dA_thermal = -self.gamma * T * A

            # Total deterministic evolution
            dA_dt = dA_coupling + dA_smooth + dA_thermal

            # Forward Euler step
            A_new = A + dt * dA_dt

            # Langevin thermal noise
            if add_thermal_noise:
                noise_amplitude = torch.sqrt(2 * T.abs() * dt)
                A_new = A_new + noise_amplitude * torch.randn_like(A)

            # Heat generation
            heat = self._compute_heat(A, A_new, dt)
            total_heat += heat
            self.total_heat_generated += heat

            # Entropy tracking
            entropy_after = self._compute_field_entropy(A_new)
            entropy_reduced = entropy_before - entropy_after
            self.total_entropy_reduced += entropy_reduced

            # Collapse event detection
            if entropy_reduced > 0.1:
                self.collapse_event_count += 1
                self.collapse_events.append({
                    "time": self.collapse_event_count,
                    "entropy_reduced": entropy_reduced,
                    "heat_generated": heat,
                })

            A = A_new

        return A, total_heat

    # ------------------------------------------------------------------
    # Laplacian (batched, GPU-accelerable)
    # ------------------------------------------------------------------

    @staticmethod
    def _laplacian_2d(field: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """2D Laplacian via F.conv2d with 5-point stencil.

        Handles arbitrary leading batch dimensions by flattening to (N, 1, H, W).

        Args:
            field: (..., H, W) tensor.
            kernel: (1, 1, 3, 3) stencil kernel on correct device/dtype.

        Returns:
            Laplacian, same shape as field.
        """
        orig_shape = field.shape
        spatial = field.shape[-2:]

        # Flatten leading dims into batch
        flat = field.reshape(-1, 1, *spatial)
        padded = F.pad(flat, (1, 1, 1, 1), mode="replicate")
        result = F.conv2d(padded, kernel)
        return result.reshape(orig_shape)

    # ------------------------------------------------------------------
    # Collapse detection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def detect_collapse_regions(
        self, A: torch.Tensor, threshold: float = 0.1
    ) -> torch.Tensor:
        """Detect regions where field is collapsing (high local gradient).

        Args:
            A: Actual field (..., H, W).
            threshold: Gradient magnitude threshold.

        Returns:
            Binary mask of collapse regions, same shape as A.
        """
        grad_h = F.pad(A[..., 1:, :] - A[..., :-1, :], (0, 0, 0, 1))
        grad_w = F.pad(A[..., :, 1:] - A[..., :, :-1], (0, 1, 0, 0))
        grad_mag = torch.sqrt(grad_h.pow(2) + grad_w.pow(2))
        return (grad_mag > threshold).float()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> Dict:
        """Get current SEC operator state."""
        return {
            "total_heat_generated": self.total_heat_generated,
            "total_entropy_reduced": self.total_entropy_reduced,
            "collapse_event_count": self.collapse_event_count,
            "parameters": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            },
        }

    def reset(self) -> None:
        """Reset thermodynamic tracking state."""
        self.total_heat_generated = 0.0
        self.total_entropy_reduced = 0.0
        self.collapse_event_count = 0
        self.collapse_events = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_field_entropy(field: torch.Tensor, bins: int = 100) -> float:
        """Shannon entropy of field distribution: H = -Σ p(x) log p(x)."""
        if torch.any(~torch.isfinite(field)):
            return float(bins)

        flat = field.flatten()
        field_min = flat.min().item()
        field_max = flat.max().item()

        if abs(field_max - field_min) < 1e-10:
            return 0.0

        hist = torch.histc(flat, bins=bins, min=field_min, max=field_max)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        entropy = -(prob * torch.log(prob)).sum()
        return entropy.item()

    @staticmethod
    def _compute_heat(
        A_before: torch.Tensor, A_after: torch.Tensor, dt: float
    ) -> float:
        """Compute heat from field evolution (kinetic + Landauer)."""
        n_cells = A_before.numel()

        # Kinetic: ½|dA/dt|² per cell
        dA = (A_after - A_before) / dt
        kinetic = 0.5 * dA.pow(2).mean().item()

        # Landauer: E = kT ln(2) per bit erased
        e_before = SECFieldEvolver._compute_field_entropy(A_before)
        e_after = SECFieldEvolver._compute_field_entropy(A_after)
        entropy_reduced = max(0.0, e_before - e_after)
        landauer = (entropy_reduced * math.log(2)) / n_cells

        # Scale to reasonable magnitude
        return (kinetic + landauer) * n_cells * 0.01

    def __repr__(self) -> str:
        return (
            f"SECFieldEvolver(device={self.device})\n"
            f"  Parameters: α={self.alpha:.4f}, β={self.beta:.4f}, γ={self.gamma:.4f}\n"
            f"  Heat generated: {self.total_heat_generated:.6f}\n"
            f"  Entropy reduced: {self.total_entropy_reduced:.6f}\n"
            f"  Collapse events: {self.collapse_event_count}"
        )
