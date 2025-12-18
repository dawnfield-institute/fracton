"""
Field Resonance Computation

Computes resonance patterns between fields.
Resonance is the fundamental mechanism for pattern matching
and information retrieval in Dawn Field Theory.
"""

import torch
from typing import List, Tuple, Optional

from ..physics.constants import PHI, XI, PHI_XI


def compute_resonance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute resonance between two fields.
    
    Resonance is a generalized similarity measure that accounts
    for phase alignment, not just magnitude overlap.
    
    Args:
        a, b: Field tensors (same shape)
        
    Returns:
        Resonance score in [-1, 1]
    """
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    
    # Cosine similarity (basic resonance)
    dot = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    
    return (dot / (norm_a * norm_b)).item()


def compute_resonance_batch(query: torch.Tensor,
                            candidates: torch.Tensor) -> torch.Tensor:
    """
    Compute resonance between query and multiple candidates.
    
    Optimized for GPU batch processing.
    
    Args:
        query: Query field, shape (dim,)
        candidates: Candidate fields, shape (n, dim)
        
    Returns:
        Resonance scores, shape (n,)
    """
    # Normalize
    query_norm = query / (torch.norm(query) + 1e-10)
    cand_norms = candidates / (torch.norm(candidates, dim=1, keepdim=True) + 1e-10)
    
    # Batch dot product
    return torch.matmul(cand_norms, query_norm)


def find_resonant(query: torch.Tensor,
                  candidates: torch.Tensor,
                  top_k: int = 5,
                  threshold: float = 0.5) -> List[Tuple[int, float]]:
    """
    Find most resonant candidates for a query.
    
    Args:
        query: Query field
        candidates: Candidate fields
        top_k: Number of results
        threshold: Minimum resonance score
        
    Returns:
        List of (index, score) tuples
    """
    scores = compute_resonance_batch(query, candidates)
    
    # Filter by threshold
    mask = scores >= threshold
    valid_indices = torch.where(mask)[0]
    valid_scores = scores[mask]
    
    if len(valid_indices) == 0:
        return []
    
    # Get top k
    k = min(top_k, len(valid_scores))
    top_scores, top_local = torch.topk(valid_scores, k)
    top_indices = valid_indices[top_local]
    
    return [(idx.item(), score.item()) for idx, score in zip(top_indices, top_scores)]


def compute_resonance_matrix(fields: torch.Tensor) -> torch.Tensor:
    """
    Compute full resonance matrix between all pairs.
    
    Args:
        fields: Field tensors, shape (n, dim)
        
    Returns:
        Resonance matrix, shape (n, n)
    """
    # Normalize all fields
    norms = torch.norm(fields, dim=1, keepdim=True) + 1e-10
    normalized = fields / norms
    
    # Compute all pairwise dot products
    return torch.matmul(normalized, normalized.T)


def resonance_energy(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute resonance energy between two fields.
    
    Energy is proportional to squared resonance, representing
    the interaction strength.
    
    Args:
        a, b: Field tensors
        
    Returns:
        Resonance energy (always non-negative)
    """
    r = compute_resonance(a, b)
    return r * r


def harmonic_resonance(a: torch.Tensor,
                       b: torch.Tensor,
                       harmonics: int = 3) -> float:
    """
    Compute multi-harmonic resonance.
    
    Considers resonance at multiple frequency scales,
    weighted by golden ratio powers.
    
    Args:
        a, b: Field tensors
        harmonics: Number of harmonic levels
        
    Returns:
        Weighted harmonic resonance
    """
    total = 0.0
    weight_sum = 0.0
    
    current_a = a
    current_b = b
    
    for h in range(harmonics):
        # Weight by inverse golden power
        weight = 1.0 / (PHI ** h)
        
        # Compute resonance at this scale
        r = compute_resonance(current_a, current_b)
        total += weight * r
        weight_sum += weight
        
        # Downsample for next harmonic (if possible)
        if current_a.shape[0] >= 2:
            current_a = (current_a[::2] + current_a[1::2]) / 2
            current_b = (current_b[::2] + current_b[1::2]) / 2
        else:
            break
    
    return total / weight_sum if weight_sum > 0 else 0.0


def phase_coherence(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute phase coherence between two fields.
    
    Measures how well the phases (signs) of corresponding
    components align.
    
    Args:
        a, b: Field tensors
        
    Returns:
        Phase coherence in [0, 1]
    """
    # Sign agreement
    signs_a = torch.sign(a)
    signs_b = torch.sign(b)
    
    agreement = (signs_a == signs_b).float().mean()
    return agreement.item()


def resonance_gradient(query: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of resonance with respect to query.
    
    Used for optimizing field configurations toward
    higher resonance.
    
    Args:
        query: Query field (will compute gradient w.r.t this)
        target: Target field
        
    Returns:
        Gradient tensor
    """
    # For cosine similarity, gradient is:
    # d(cos)/dq = (t - cos * q) / ||q||
    
    cos_sim = compute_resonance(query, target)
    query_norm = torch.norm(query)
    
    if query_norm < 1e-10:
        return torch.zeros_like(query)
    
    target_normalized = target / (torch.norm(target) + 1e-10)
    query_normalized = query / query_norm
    
    gradient = (target_normalized - cos_sim * query_normalized) / query_norm
    
    return gradient


class ResonanceMesh:
    """
    Maintains a mesh of resonance relationships.
    
    Efficiently tracks pairwise resonances and supports
    incremental updates.
    """
    
    def __init__(self, initial_capacity: int = 1000):
        self._capacity = initial_capacity
        self._count = 0
        self._fields: Optional[torch.Tensor] = None
        self._matrix: Optional[torch.Tensor] = None
        self._ids: List[int] = []
    
    def add(self, field: torch.Tensor, field_id: int) -> None:
        """Add a field to the mesh."""
        device = field.device
        
        if self._fields is None:
            self._fields = torch.zeros(
                self._capacity, field.shape[0], device=device
            )
            self._matrix = torch.zeros(
                self._capacity, self._capacity, device=device
            )
        
        # Expand if needed
        if self._count >= self._capacity:
            self._expand()
        
        # Add field
        self._fields[self._count] = field
        self._ids.append(field_id)
        
        # Update resonance matrix (new row and column)
        if self._count > 0:
            new_resonances = compute_resonance_batch(
                field, self._fields[:self._count]
            )
            self._matrix[self._count, :self._count] = new_resonances
            self._matrix[:self._count, self._count] = new_resonances
        
        self._matrix[self._count, self._count] = 1.0  # Self-resonance
        self._count += 1
    
    def _expand(self) -> None:
        """Double capacity."""
        new_capacity = self._capacity * 2
        
        new_fields = torch.zeros(
            new_capacity, self._fields.shape[1], device=self._fields.device
        )
        new_fields[:self._count] = self._fields[:self._count]
        self._fields = new_fields
        
        new_matrix = torch.zeros(
            new_capacity, new_capacity, device=self._matrix.device
        )
        new_matrix[:self._count, :self._count] = self._matrix[:self._count, :self._count]
        self._matrix = new_matrix
        
        self._capacity = new_capacity
    
    def find_resonant(self, query: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
        """Find most resonant fields in mesh."""
        if self._count == 0:
            return []
        
        scores = compute_resonance_batch(query, self._fields[:self._count])
        
        k = min(top_k, self._count)
        top_scores, top_indices = torch.topk(scores, k)
        
        return [
            (self._ids[idx.item()], score.item())
            for idx, score in zip(top_indices, top_scores)
        ]
    
    def get_neighbors(self, field_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Get most resonant neighbors of a field."""
        try:
            idx = self._ids.index(field_id)
        except ValueError:
            return []
        
        scores = self._matrix[idx, :self._count]
        
        # Exclude self
        scores[idx] = -1
        
        k = min(top_k, self._count - 1)
        top_scores, top_indices = torch.topk(scores, k)
        
        return [
            (self._ids[i.item()], s.item())
            for i, s in zip(top_indices, top_scores)
            if s > 0
        ]
    
    @property
    def size(self) -> int:
        return self._count
