"""
PACNode - Delta-Only Storage Primitive

The fundamental building block of PAC-Lazy architecture.
Each node stores only its delta from parent, not absolute values.
Reconstruction requires traversing to root.

Key Invariants:
1. Delta-only storage: node.value = parent.value + node.delta
2. PAC conservation: parent.value = Σ(children.delta) + parent.base
3. Lazy evaluation: only materialize when SEC threshold exceeded
"""

import torch
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import time

from ..physics.constants import XI, PHI_XI, LAMBDA_STAR
from ..physics.phase_transitions import PhaseState, detect_phase


@dataclass
class PACNode:
    """
    PAC-Lazy storage node with delta-only representation.
    
    NEVER stores absolute values. Reconstruction requires parent chain.
    This is the core innovation of PAC-Lazy architecture.
    
    Attributes:
        id: Unique node identifier
        delta: Change from parent (the ONLY stored value)
        potential: Computational budget remaining
        parent_id: ID of parent node (-1 for root)
        children_ids: IDs of child nodes
        label: Optional semantic label
        created_at: Timestamp
        phase: Current phase state
    """
    id: int
    delta: torch.Tensor
    potential: float = 1.0
    parent_id: int = -1
    children_ids: List[int] = field(default_factory=list)
    label: str = ""
    created_at: float = field(default_factory=time.time)
    phase: PhaseState = PhaseState.STABLE
    
    # Lazy evaluation cache (only populated when expanded)
    _materialized: Optional[torch.Tensor] = field(default=None, repr=False)
    _materialized_valid: bool = field(default=False, repr=False)
    
    @property
    def is_root(self) -> bool:
        """Check if this is a root node."""
        return self.parent_id == -1
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children_ids) == 0
    
    @property
    def depth(self) -> int:
        """
        Approximate depth (actual depth requires tree traversal).
        Estimated from potential decay.
        """
        if self.potential >= 1.0:
            return 0
        import math
        return max(0, int(-math.log(self.potential + 1e-10) / math.log(LAMBDA_STAR)))
    
    def should_expand(self) -> bool:
        """Check if this node should expand (trigger lazy evaluation)."""
        return self.potential >= PHI_XI
    
    def should_collapse(self) -> bool:
        """Check if this node should collapse to parent."""
        return self.potential <= XI
    
    def update_phase(self) -> PhaseState:
        """Update and return current phase based on potential."""
        self.phase = detect_phase(self.potential)
        return self.phase
    
    def add_child(self, child_id: int) -> None:
        """Add a child node reference."""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def remove_child(self, child_id: int) -> None:
        """Remove a child node reference."""
        if child_id in self.children_ids:
            self.children_ids.remove(child_id)
    
    def invalidate_cache(self) -> None:
        """Invalidate materialized cache (call when parent changes)."""
        self._materialized_valid = False
        self._materialized = None
    
    def decay_potential(self, factor: float = LAMBDA_STAR) -> float:
        """
        Apply potential decay.
        
        Args:
            factor: Decay multiplier (default: λ* = 1-ξ)
            
        Returns:
            New potential value
        """
        self.potential *= factor
        self.update_phase()
        return self.potential
    
    def transfer_potential(self, amount: float, to_node: 'PACNode') -> float:
        """
        Transfer potential to another node (conserved).
        
        Args:
            amount: Amount to transfer
            to_node: Destination node
            
        Returns:
            Actual amount transferred
        """
        actual = min(amount, self.potential)
        self.potential -= actual
        to_node.potential += actual
        self.update_phase()
        to_node.update_phase()
        return actual
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for storage)."""
        return {
            "id": self.id,
            "delta": self.delta.tolist() if isinstance(self.delta, torch.Tensor) else self.delta,
            "potential": self.potential,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids.copy(),
            "label": self.label,
            "created_at": self.created_at,
            "phase": self.phase.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: str = 'cpu') -> 'PACNode':
        """Deserialize from dictionary."""
        delta = data["delta"]
        if isinstance(delta, list):
            delta = torch.tensor(delta, device=device)
        
        return cls(
            id=data["id"],
            delta=delta,
            potential=data["potential"],
            parent_id=data["parent_id"],
            children_ids=data.get("children_ids", []),
            label=data.get("label", ""),
            created_at=data.get("created_at", time.time()),
            phase=PhaseState(data.get("phase", "stable"))
        )
    
    def __repr__(self) -> str:
        return (f"PACNode(id={self.id}, shape={tuple(self.delta.shape)}, "
                f"potential={self.potential:.4f}, phase={self.phase.value}, "
                f"children={len(self.children_ids)})")


class PACNodeFactory:
    """
    Factory for creating PACNodes with consistent configuration.
    """
    
    def __init__(self, device: str = 'cpu', dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self._next_id = 0
    
    def create(self, 
               delta: torch.Tensor,
               parent_id: int = -1,
               potential: float = 1.0,
               label: str = "") -> PACNode:
        """
        Create a new PACNode.
        
        Args:
            delta: Delta tensor (moved to factory device)
            parent_id: Parent node ID (-1 for root)
            potential: Initial potential
            label: Optional semantic label
            
        Returns:
            New PACNode instance
        """
        node_id = self._next_id
        self._next_id += 1
        
        # Ensure tensor is on correct device
        if delta.device.type != self.device:
            delta = delta.to(self.device)
        if delta.dtype != self.dtype:
            delta = delta.to(self.dtype)
        
        return PACNode(
            id=node_id,
            delta=delta,
            potential=potential,
            parent_id=parent_id,
            label=label
        )
    
    def create_root(self, value: torch.Tensor, label: str = "root") -> PACNode:
        """
        Create a root node.
        
        For root nodes, delta == value (no parent to diff against).
        """
        return self.create(delta=value, parent_id=-1, potential=1.0, label=label)
    
    def create_child(self,
                     parent: PACNode,
                     delta: torch.Tensor,
                     label: str = "") -> PACNode:
        """
        Create a child node with proper parent linkage.
        
        Potential is inherited from parent with decay.
        """
        child = self.create(
            delta=delta,
            parent_id=parent.id,
            potential=parent.potential * LAMBDA_STAR,
            label=label
        )
        parent.add_child(child.id)
        return child
    
    def reset_ids(self) -> None:
        """Reset ID counter (use with caution)."""
        self._next_id = 0
