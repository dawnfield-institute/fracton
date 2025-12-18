"""
Phase Transition Detection (SEC - Selective Entropy Collapse)

Implements the phase transition logic for PAC-Lazy evaluation.
Patterns expand when potential exceeds φ×ξ threshold,
collapse when potential drops below ξ threshold.
"""

import torch
from enum import Enum
from typing import Union, Tuple
from dataclasses import dataclass

from .constants import (
    PHI, XI, PHI_XI,
    SEC_EXPAND_THRESHOLD, 
    SEC_COLLAPSE_THRESHOLD
)


class PhaseState(Enum):
    """Possible phase states for a node."""
    COLLAPSED = "collapsed"  # Below ξ threshold, merged to parent
    STABLE = "stable"        # Between thresholds, lazy storage
    EXPANDED = "expanded"    # Above φ×ξ threshold, materialized


@dataclass
class PhaseTransition:
    """Record of a phase transition event."""
    from_state: PhaseState
    to_state: PhaseState
    potential_before: float
    potential_after: float
    node_id: int = -1


def detect_phase(potential: Union[float, torch.Tensor]) -> PhaseState:
    """
    Determine the current phase based on potential value.
    
    The SEC (Selective Entropy Collapse) thresholds are:
    - Expand: potential ≥ φ×ξ (0.1)
    - Collapse: potential ≤ ξ (0.0618)
    - Stable: ξ < potential < φ×ξ
    
    Args:
        potential: Scalar potential value or tensor
        
    Returns:
        PhaseState enum value
    """
    if isinstance(potential, torch.Tensor):
        potential = potential.item() if potential.numel() == 1 else potential.mean().item()
    
    if potential >= SEC_EXPAND_THRESHOLD:
        return PhaseState.EXPANDED
    elif potential <= SEC_COLLAPSE_THRESHOLD:
        return PhaseState.COLLAPSED
    else:
        return PhaseState.STABLE


def should_expand(potential: Union[float, torch.Tensor]) -> bool:
    """
    Check if a node should expand (trigger lazy evaluation).
    
    Expansion occurs when potential exceeds φ×ξ threshold,
    indicating enough computational budget to justify materialization.
    
    Args:
        potential: Current potential value
        
    Returns:
        True if node should expand
    """
    if isinstance(potential, torch.Tensor):
        potential = potential.item() if potential.numel() == 1 else potential.mean().item()
    
    return potential >= SEC_EXPAND_THRESHOLD


def should_collapse(potential: Union[float, torch.Tensor]) -> bool:
    """
    Check if a node should collapse (merge to parent).
    
    Collapse occurs when potential drops below ξ threshold,
    indicating insufficient budget to maintain separate existence.
    
    Args:
        potential: Current potential value
        
    Returns:
        True if node should collapse
    """
    if isinstance(potential, torch.Tensor):
        potential = potential.item() if potential.numel() == 1 else potential.mean().item()
    
    return potential <= SEC_COLLAPSE_THRESHOLD


def compute_phase_energy(state: PhaseState) -> float:
    """
    Compute the characteristic energy of a phase state.
    
    Used for transitions and conservation checks.
    
    Args:
        state: Current phase state
        
    Returns:
        Characteristic energy value
    """
    if state == PhaseState.COLLAPSED:
        return XI / 2  # Minimum energy
    elif state == PhaseState.STABLE:
        return (XI + PHI_XI) / 2  # Middle energy
    else:  # EXPANDED
        return PHI_XI * PHI  # Maximum energy


def transition_cost(from_state: PhaseState, to_state: PhaseState) -> float:
    """
    Compute the energy cost of a phase transition.
    
    Transitions are not free - they consume potential budget.
    
    Args:
        from_state: Current state
        to_state: Target state
        
    Returns:
        Energy cost (always non-negative)
    """
    from_energy = compute_phase_energy(from_state)
    to_energy = compute_phase_energy(to_state)
    
    # Expansion costs more than collapse
    if to_state == PhaseState.EXPANDED:
        return abs(to_energy - from_energy) * PHI
    elif to_state == PhaseState.COLLAPSED:
        return abs(to_energy - from_energy) * XI
    else:
        return abs(to_energy - from_energy)


class PhaseManager:
    """
    Manages phase transitions for a collection of nodes.
    
    Tracks transition history and ensures conservation during transitions.
    """
    
    def __init__(self):
        self._history: list = []
        self._current_states: dict = {}  # node_id -> PhaseState
    
    def update(self, node_id: int, new_potential: float) -> PhaseTransition | None:
        """
        Update node potential and check for phase transition.
        
        Args:
            node_id: Unique node identifier
            new_potential: Updated potential value
            
        Returns:
            PhaseTransition if transition occurred, None otherwise
        """
        new_state = detect_phase(new_potential)
        old_state = self._current_states.get(node_id, PhaseState.STABLE)
        
        if new_state != old_state:
            transition = PhaseTransition(
                from_state=old_state,
                to_state=new_state,
                potential_before=self._get_last_potential(node_id),
                potential_after=new_potential,
                node_id=node_id
            )
            self._history.append(transition)
            self._current_states[node_id] = new_state
            return transition
        
        self._current_states[node_id] = new_state
        return None
    
    def _get_last_potential(self, node_id: int) -> float:
        """Get last known potential for a node."""
        # Search history in reverse
        for t in reversed(self._history):
            if t.node_id == node_id:
                return t.potential_after
        return 0.0
    
    def get_state(self, node_id: int) -> PhaseState:
        """Get current phase state for a node."""
        return self._current_states.get(node_id, PhaseState.STABLE)
    
    @property
    def transition_count(self) -> int:
        """Total number of transitions recorded."""
        return len(self._history)
    
    @property
    def expansion_count(self) -> int:
        """Number of expansion transitions."""
        return sum(1 for t in self._history if t.to_state == PhaseState.EXPANDED)
    
    @property
    def collapse_count(self) -> int:
        """Number of collapse transitions."""
        return sum(1 for t in self._history if t.to_state == PhaseState.COLLAPSED)
