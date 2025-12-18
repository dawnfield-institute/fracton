"""
Spherical Field Encoding

Encodes tokens/patterns into spherical field representations.
Uses φ-modulated angular positioning for natural information geometry.
"""

import torch
import math
from typing import Union, Optional, Tuple

from ..physics.constants import PHI, XI, TAU


def spherical_encode(token_id: Union[int, torch.Tensor],
                     vocab_size: int = 50257,
                     dim: int = 64,
                     device: str = 'cpu') -> torch.Tensor:
    """
    Encode a token ID into a spherical field representation.
    
    Uses golden angle spacing for optimal distribution on sphere.
    Each dimension represents a different frequency component.
    
    Args:
        token_id: Token ID(s) to encode
        vocab_size: Size of vocabulary
        dim: Output field dimension
        device: Torch device
        
    Returns:
        Field tensor of shape (dim,) or (batch, dim)
    """
    # Handle batched input
    if isinstance(token_id, torch.Tensor):
        if token_id.dim() == 0:
            token_id = token_id.item()
        else:
            return torch.stack([
                spherical_encode(t.item(), vocab_size, dim, device)
                for t in token_id
            ])
    
    # Golden angle for optimal spherical distribution
    golden_angle = TAU / (PHI ** 2)  # ~2.4 radians
    
    # Base angle from token position
    base_theta = (token_id / vocab_size) * TAU
    
    # Create field with multiple frequency components
    field = torch.zeros(dim, device=device)
    
    for i in range(dim):
        # Frequency increases with dimension
        freq = 1 + i * XI  # ξ-modulated frequency scaling
        
        # Phase offset using golden angle
        phase = i * golden_angle
        
        # Spherical harmonic-like encoding
        theta = base_theta * freq + phase
        phi = (token_id * golden_angle * (i + 1)) % TAU
        
        # Combine spherical coordinates
        field[i] = (
            math.cos(theta) * math.sin(phi) +
            math.sin(theta) * math.cos(phi) * XI
        )
    
    # Normalize to unit sphere
    norm = torch.norm(field)
    if norm > 1e-10:
        field = field / norm
    
    return field


def spherical_encode_batch(token_ids: torch.Tensor,
                           vocab_size: int = 50257,
                           dim: int = 64) -> torch.Tensor:
    """
    Batch-optimized spherical encoding on GPU.
    
    Args:
        token_ids: Tensor of token IDs, shape (batch,) or (batch, seq)
        vocab_size: Size of vocabulary
        dim: Output field dimension
        
    Returns:
        Field tensor of shape (batch, dim) or (batch, seq, dim)
    """
    device = token_ids.device
    original_shape = token_ids.shape
    token_ids = token_ids.flatten().float()
    batch_size = token_ids.shape[0]
    
    # Golden angle
    golden_angle = TAU / (PHI ** 2)
    
    # Base angles: (batch,)
    base_theta = (token_ids / vocab_size) * TAU
    
    # Dimension indices: (dim,)
    dims = torch.arange(dim, device=device, dtype=torch.float32)
    
    # Frequencies: (dim,)
    freqs = 1 + dims * XI
    
    # Phases: (dim,)
    phases = dims * golden_angle
    
    # Compute theta: (batch, dim)
    theta = base_theta.unsqueeze(1) * freqs.unsqueeze(0) + phases.unsqueeze(0)
    
    # Compute phi: (batch, dim)
    phi = (token_ids.unsqueeze(1) * golden_angle * (dims + 1).unsqueeze(0)) % TAU
    
    # Spherical encoding: (batch, dim)
    field = (
        torch.cos(theta) * torch.sin(phi) +
        torch.sin(theta) * torch.cos(phi) * XI
    )
    
    # Normalize
    norms = torch.norm(field, dim=1, keepdim=True)
    field = field / (norms + 1e-10)
    
    # Reshape if needed
    if len(original_shape) > 1:
        field = field.view(*original_shape, dim)
    
    return field


def decode_spherical(field: torch.Tensor,
                     vocab_size: int = 50257,
                     reference_fields: Optional[torch.Tensor] = None) -> Tuple[int, float]:
    """
    Decode a spherical field back to token ID.
    
    If reference_fields provided, finds best match.
    Otherwise uses inverse of encoding (approximate).
    
    Args:
        field: Field tensor to decode
        vocab_size: Size of vocabulary
        reference_fields: Pre-computed fields for all tokens
        
    Returns:
        Tuple of (predicted_token_id, confidence)
    """
    if reference_fields is not None:
        # Find most similar reference field
        similarities = torch.matmul(reference_fields, field)
        best_idx = similarities.argmax().item()
        confidence = similarities[best_idx].item()
        return best_idx, confidence
    
    # Approximate inverse (less accurate)
    # Use phase angle of first component
    theta_0 = math.atan2(field[1].item(), field[0].item())
    if theta_0 < 0:
        theta_0 += TAU
    
    # Approximate token from angle
    token_id = int((theta_0 / TAU) * vocab_size) % vocab_size
    
    # Confidence based on field magnitude
    confidence = torch.norm(field).item()
    
    return token_id, confidence


def create_reference_fields(vocab_size: int = 50257,
                           dim: int = 64,
                           device: str = 'cpu') -> torch.Tensor:
    """
    Pre-compute reference fields for all tokens in vocabulary.
    
    Used for efficient decoding via similarity search.
    
    Args:
        vocab_size: Size of vocabulary
        dim: Field dimension
        device: Torch device
        
    Returns:
        Reference fields tensor of shape (vocab_size, dim)
    """
    token_ids = torch.arange(vocab_size, device=device)
    return spherical_encode_batch(token_ids, vocab_size, dim)


def field_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute geodesic distance between two field points.
    
    For normalized fields, this is the angle between them.
    
    Args:
        a, b: Field tensors
        
    Returns:
        Distance (0 = identical, π = opposite)
    """
    # Cosine similarity
    cos_sim = torch.dot(a.flatten(), b.flatten())
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    
    # Arc cosine gives angle
    return math.acos(cos_sim.item())


def interpolate_fields(a: torch.Tensor,
                       b: torch.Tensor,
                       t: float) -> torch.Tensor:
    """
    Spherical interpolation (SLERP) between two fields.
    
    Args:
        a, b: Field tensors
        t: Interpolation parameter (0 = a, 1 = b)
        
    Returns:
        Interpolated field
    """
    # Compute angle
    cos_theta = torch.dot(a.flatten(), b.flatten())
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta.item())
    
    if abs(theta) < 1e-6:
        return a
    
    # SLERP formula
    sin_theta = math.sin(theta)
    weight_a = math.sin((1 - t) * theta) / sin_theta
    weight_b = math.sin(t * theta) / sin_theta
    
    result = weight_a * a + weight_b * b
    
    # Normalize
    norm = torch.norm(result)
    if norm > 1e-10:
        result = result / norm
    
    return result
