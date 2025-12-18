"""
PAC Conservation Enforcement

Implements Potential-Actualization Conservation (PAC) validation and enforcement.
Core invariant: f(parent) = Σf(children)

This module provides the authoritative implementation of conservation laws
used throughout Fracton and GAIA.
"""

import torch
import numpy as np
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass

from .constants import XI, PHI_XI, LAMBDA_STAR


@dataclass
class ConservationResult:
    """Result of a conservation check."""
    valid: bool
    residual: float
    corrected: bool = False
    correction_magnitude: float = 0.0


class PACValidator:
    """
    Validates and enforces PAC conservation across tree structures.
    
    Key principle: Information is neither created nor destroyed,
    only transformed. f(parent) = Σf(children) must always hold.
    """
    
    def __init__(self, tolerance: float = 1e-10, auto_correct: bool = True):
        """
        Initialize validator.
        
        Args:
            tolerance: Maximum allowed conservation residual
            auto_correct: If True, automatically correct violations
        """
        self.tolerance = tolerance
        self.auto_correct = auto_correct
        self._violation_count = 0
        self._correction_count = 0
    
    def validate(self, parent: torch.Tensor, 
                 children: List[torch.Tensor]) -> ConservationResult:
        """
        Validate PAC conservation for a parent-children relationship.
        
        Args:
            parent: Parent tensor (any shape)
            children: List of child tensors (same shape as parent)
            
        Returns:
            ConservationResult with validation status
        """
        if not children:
            # No children = leaf node, trivially conserved
            return ConservationResult(valid=True, residual=0.0)
        
        # Sum children
        children_sum = torch.zeros_like(parent)
        for child in children:
            children_sum = children_sum + child
        
        # Compute residual
        residual = torch.abs(parent - children_sum).max().item()
        valid = residual < self.tolerance
        
        if not valid:
            self._violation_count += 1
            
            if self.auto_correct:
                # Normalize children to sum to parent
                scale = parent.sum() / (children_sum.sum() + 1e-10)
                for child in children:
                    child.mul_(scale)
                self._correction_count += 1
                return ConservationResult(
                    valid=True, 
                    residual=residual,
                    corrected=True,
                    correction_magnitude=abs(1 - scale.item())
                )
        
        return ConservationResult(valid=valid, residual=residual)
    
    def validate_tree(self, root_value: float, 
                      leaf_values: List[float]) -> ConservationResult:
        """
        Validate conservation for entire tree (root vs all leaves).
        
        Args:
            root_value: Total value at root
            leaf_values: Values at all leaf nodes
            
        Returns:
            ConservationResult
        """
        leaves_sum = sum(leaf_values)
        residual = abs(root_value - leaves_sum)
        valid = residual < self.tolerance
        
        if not valid:
            self._violation_count += 1
        
        return ConservationResult(valid=valid, residual=residual)
    
    @property
    def stats(self) -> dict:
        """Get validation statistics."""
        return {
            "violations": self._violation_count,
            "corrections": self._correction_count
        }


def validate_pac(parent: Union[torch.Tensor, float],
                 children: List[Union[torch.Tensor, float]],
                 tolerance: float = 1e-10) -> Tuple[bool, float]:
    """
    Quick validation of PAC conservation.
    
    Args:
        parent: Parent value or tensor
        children: List of child values or tensors
        tolerance: Maximum allowed residual
        
    Returns:
        Tuple of (is_valid, residual)
    """
    if not children:
        return True, 0.0
    
    if isinstance(parent, torch.Tensor):
        children_sum = sum(children)
        residual = torch.abs(parent - children_sum).max().item()
    else:
        children_sum = sum(children)
        residual = abs(parent - children_sum)
    
    return residual < tolerance, residual


def compute_residual(parent: torch.Tensor, 
                     children: List[torch.Tensor]) -> float:
    """
    Compute conservation residual.
    
    Args:
        parent: Parent tensor
        children: List of child tensors
        
    Returns:
        Maximum absolute residual
    """
    if not children:
        return 0.0
    
    children_sum = sum(children)
    return torch.abs(parent - children_sum).max().item()


def enforce_conservation(parent: torch.Tensor,
                         children: List[torch.Tensor],
                         method: str = "scale") -> List[torch.Tensor]:
    """
    Enforce conservation by modifying children.
    
    Args:
        parent: Parent tensor (not modified)
        children: List of child tensors (will be modified in-place)
        method: "scale" (uniform scaling) or "proportional" (maintain ratios)
        
    Returns:
        Modified children list
    """
    if not children:
        return children
    
    children_sum = sum(children)
    parent_sum = parent.sum()
    children_total = children_sum.sum()
    
    if children_total.abs() < 1e-10:
        # All children near zero, distribute parent equally
        share = parent / len(children)
        for child in children:
            child.copy_(share)
    elif method == "scale":
        # Uniform scaling
        scale = parent_sum / children_total
        for child in children:
            child.mul_(scale)
    else:  # proportional
        # Scale each child proportionally to maintain ratios
        for child in children:
            child_ratio = child.sum() / children_total
            child.mul_(parent_sum / child.sum() * child_ratio)
    
    return children
