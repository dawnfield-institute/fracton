"""
Projection operators for symmetric/antisymmetric tensor decomposition
and 3D differential operators (gradient, divergence, curl, Laplacian).

The key insight from DFT: Maxwell uses antisymmetric projection (curl),
gravity uses symmetric projection (divergence). Both project from the
same Mobius pre-field.

Promoted from dawn-field-theory/gravity_from_maxwell_pac/core/projections.py.
All ops use torch tensors and support optional leading batch dimensions.
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def symmetric_part(tensor: torch.Tensor) -> torch.Tensor:
    """Extract symmetric part: S_ij = (T_ij + T_ji) / 2.

    For a 3x3 tensor, gives 6 independent components.
    Relates to: stress-energy tensor, metric perturbation.

    Args:
        tensor: (..., N, N) tensor. Supports batch dimensions.

    Returns:
        Symmetric part, same shape.
    """
    return (tensor + tensor.transpose(-2, -1)) / 2


@torch.no_grad()
def antisymmetric_part(tensor: torch.Tensor) -> torch.Tensor:
    """Extract antisymmetric part: A_ij = (T_ij - T_ji) / 2.

    For a 3x3 tensor, gives 3 independent components.
    Relates to: electromagnetic field tensor F_uv.

    Args:
        tensor: (..., N, N) tensor. Supports batch dimensions.

    Returns:
        Antisymmetric part, same shape.
    """
    return (tensor - tensor.transpose(-2, -1)) / 2


@torch.no_grad()
def decompose_tensor(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose tensor into symmetric + antisymmetric parts.

    Args:
        tensor: (..., N, N) tensor.

    Returns:
        (symmetric, antisymmetric) tuple.
    """
    s = symmetric_part(tensor)
    a = antisymmetric_part(tensor)
    return s, a


# ---------------------------------------------------------------------------
# Differential operators — batch-first, GPU-accelerable
# ---------------------------------------------------------------------------

def _gradient_axis(field: torch.Tensor, dx: float, axis: int) -> torch.Tensor:
    """Central-difference gradient along one axis.

    Uses torch.roll for periodic boundary-like behaviour at edges,
    matching numpy.gradient's second-order central differences in the interior.
    """
    # Central difference: (f[i+1] - f[i-1]) / (2*dx)
    fwd = torch.roll(field, -1, dims=axis)
    bwd = torch.roll(field, 1, dims=axis)
    return (fwd - bwd) / (2 * dx)


@torch.no_grad()
def gradient_3d(
    field: torch.Tensor, dx: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 3D gradient of a scalar field.

    Args:
        field: (..., Z, Y, X) tensor. Last 3 dims are spatial.
        dx: Grid spacing (uniform).

    Returns:
        (grad_x, grad_y, grad_z) tuple, each same shape as field.
    """
    grad_x = _gradient_axis(field, dx, axis=-1)
    grad_y = _gradient_axis(field, dx, axis=-2)
    grad_z = _gradient_axis(field, dx, axis=-3)
    return grad_x, grad_y, grad_z


@torch.no_grad()
def divergence_3d(
    Fx: torch.Tensor, Fy: torch.Tensor, Fz: torch.Tensor, dx: float
) -> torch.Tensor:
    """Compute 3D divergence: div(F) = dFx/dx + dFy/dy + dFz/dz.

    This is the GRAVITY operator — sources create scalar potential.

    Args:
        Fx, Fy, Fz: (..., Z, Y, X) vector field components.
        dx: Grid spacing.

    Returns:
        Scalar divergence field, same shape.
    """
    dFx_dx = _gradient_axis(Fx, dx, axis=-1)
    dFy_dy = _gradient_axis(Fy, dx, axis=-2)
    dFz_dz = _gradient_axis(Fz, dx, axis=-3)
    return dFx_dx + dFy_dy + dFz_dz


@torch.no_grad()
def curl_3d(
    Fx: torch.Tensor, Fy: torch.Tensor, Fz: torch.Tensor, dx: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 3D curl of a vector field.

    (curl F)_x = dFz/dy - dFy/dz
    (curl F)_y = dFx/dz - dFz/dx
    (curl F)_z = dFy/dx - dFx/dy

    This is the MAXWELL operator — circulation creates field.

    Args:
        Fx, Fy, Fz: (..., Z, Y, X) vector field components.
        dx: Grid spacing.

    Returns:
        (curl_x, curl_y, curl_z) tuple.
    """
    dFz_dy = _gradient_axis(Fz, dx, axis=-2)
    dFy_dz = _gradient_axis(Fy, dx, axis=-3)
    dFx_dz = _gradient_axis(Fx, dx, axis=-3)
    dFz_dx = _gradient_axis(Fz, dx, axis=-1)
    dFy_dx = _gradient_axis(Fy, dx, axis=-1)
    dFx_dy = _gradient_axis(Fx, dx, axis=-2)

    curl_x = dFz_dy - dFy_dz
    curl_y = dFx_dz - dFz_dx
    curl_z = dFy_dx - dFx_dy
    return curl_x, curl_y, curl_z


@torch.no_grad()
def laplacian_3d(field: torch.Tensor, dx: float) -> torch.Tensor:
    """Compute 3D Laplacian using a 7-point stencil via F.conv3d.

    Appears in both EM (wave equation) and gravity (Poisson equation).

    Args:
        field: (..., Z, Y, X) tensor. Supports batch dims.
        dx: Grid spacing.

    Returns:
        Laplacian, same shape as field.
    """
    # Build 3x3x3 kernel with 7-point stencil
    kernel = torch.zeros(1, 1, 3, 3, 3, dtype=field.dtype, device=field.device)
    kernel[0, 0, 1, 1, 0] = 1.0  # x-
    kernel[0, 0, 1, 1, 2] = 1.0  # x+
    kernel[0, 0, 1, 0, 1] = 1.0  # y-
    kernel[0, 0, 1, 2, 1] = 1.0  # y+
    kernel[0, 0, 0, 1, 1] = 1.0  # z-
    kernel[0, 0, 2, 1, 1] = 1.0  # z+
    kernel[0, 0, 1, 1, 1] = -6.0  # center

    # Reshape for conv3d: need (N, C, D, H, W)
    orig_shape = field.shape
    spatial = field.shape[-3:]

    # Flatten all leading dims into batch
    flat = field.reshape(-1, 1, *spatial)
    # Pad with replication for boundary
    padded = F.pad(flat, (1, 1, 1, 1, 1, 1), mode="replicate")
    result = F.conv3d(padded, kernel) / (dx * dx)
    return result.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Pre-field projections (Maxwell/Gravity from same 4D field)
# ---------------------------------------------------------------------------

@torch.no_grad()
def project_antisymmetric(
    prefield: torch.Tensor, hidden_axis: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project pre-field antisymmetrically (Maxwell).

    The hidden dimension becomes the curl structure.
    Extracts PHASE information.

    Args:
        prefield: (Nz, Ny, Nx) or (B, Nz, Ny, Nx) tensor.
        hidden_axis: Axis to integrate out (default 0, or -3 for batched).

    Returns:
        (Fx, Fy, Fz) tuple of 2D/3D fields.
    """
    if hidden_axis != 0 and hidden_axis != -3:
        raise NotImplementedError("Only axis=0 / -3 supported")

    axis = -3 if prefield.ndim >= 4 else 0
    Nz = prefield.shape[axis]

    # Oscillatory phase weights
    phase = torch.exp(
        2j * math.pi * torch.arange(Nz, device=prefield.device, dtype=torch.float64) / Nz
    ).to(torch.complex128)

    # Reshape for broadcasting
    shape = [1] * prefield.ndim
    shape[axis] = Nz
    phase = phase.reshape(shape)

    # Weight and average over hidden axis
    prefield_c = prefield.to(torch.complex128)
    weighted = prefield_c * phase
    projected = weighted.mean(dim=axis)

    Fx = projected.real.to(prefield.dtype)
    Fy = projected.imag.to(prefield.dtype)
    Fz = (projected.abs() - projected.abs().mean()).to(prefield.dtype)
    return Fx, Fy, Fz


@torch.no_grad()
def project_symmetric(
    prefield: torch.Tensor, hidden_axis: int = 0
) -> torch.Tensor:
    """Project pre-field symmetrically (Gravity).

    Extracts AMPLITUDE information. Returns scalar potential.

    Args:
        prefield: (Nz, Ny, Nx) or (B, Nz, Ny, Nx) tensor.
        hidden_axis: Axis to integrate out.

    Returns:
        Scalar potential field.
    """
    axis = -3 if prefield.ndim >= 4 else 0
    return prefield.abs().mean(dim=axis)


@torch.no_grad()
def depth_2_projection(field_4d: torch.Tensor):
    """MED depth-2 projection.

    The same 4D pre-field produces BOTH EM and gravity.

    Args:
        field_4d: 4D pre-field tensor.

    Returns:
        ((em_x, em_y, em_z), grav_potential) tuple.
    """
    em = project_antisymmetric(field_4d)
    grav = project_symmetric(field_4d)
    return em, grav
