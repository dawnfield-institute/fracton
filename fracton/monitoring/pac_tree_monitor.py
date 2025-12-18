"""
PAC Tree Monitor - Core monitoring types and tree health analysis.

Provides TreeMetrics, PatternProfile, and PACTreeMonitor for observing
pattern formation and detecting overfitting vs generalization.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Any
import numpy as np


class PatternType(Enum):
    """Classification of pattern behavior."""
    ABSTRACT = "abstract"        # High activation diversity, generalizable
    SPECIFIC = "specific"        # Low diversity, potentially memorized
    TRANSITIONAL = "transitional"  # In transition between states


@dataclass
class TreeMetrics:
    """Structural metrics for PAC tree health."""
    depth: int                    # Maximum tree depth
    breadth: int                  # Number of leaf nodes
    branching_factor: float       # Average children per non-leaf
    compression_ratio: float      # Unique patterns / total tokens seen
    reuse_ratio: float           # Pattern reuse frequency
    byref_candidates: int        # Branches similar enough for byref optimization
    total_patterns: int = 0      # Total number of patterns in tree
    abstract_count: int = 0      # Count of abstract patterns
    specific_count: int = 0      # Count of specific patterns
    transitional_count: int = 0  # Count of transitional patterns


@dataclass
class PatternProfile:
    """Behavioral profile for a single pattern."""
    pattern_id: str
    activation_count: int         # How often activated
    activation_contexts: Set[str] = field(default_factory=set)  # Unique contexts
    activation_diversity: float = 0.0   # len(contexts) / count
    child_count: int = 0          # Number of child patterns
    depth: int = 0                # Position in tree
    pattern_type: PatternType = PatternType.TRANSITIONAL
    entropy_signature: float = 0.0  # Pattern's entropy contribution
    embedding_norm: float = 0.0   # L2 norm of pattern embedding
    last_activation_step: int = 0  # Training step of last activation


# Default classification thresholds
DEFAULT_THRESHOLDS = {
    'abstract_diversity': 0.3,    # Min diversity for ABSTRACT
    'abstract_children': 10,      # Min children for ABSTRACT
    'specific_diversity': 0.1,    # Max diversity for SPECIFIC
    'specific_children': 3,       # Max children for SPECIFIC
    'specific_min_activations': 100  # Min activations to classify as SPECIFIC
}


def classify_pattern(
    profile: PatternProfile, 
    thresholds: Optional[Dict[str, float]] = None
) -> PatternType:
    """
    Classify a pattern based on its activation behavior.
    
    Classification rules:
    - ABSTRACT: High diversity (>0.3), many children (>10), any depth
    - SPECIFIC: Low diversity (<0.1), few children (<3), high activation
    - TRANSITIONAL: Between thresholds, or recently changed classification
    
    Args:
        profile: The pattern profile to classify
        thresholds: Optional custom thresholds (uses defaults if not provided)
        
    Returns:
        PatternType classification
    """
    t = thresholds or DEFAULT_THRESHOLDS
    
    # Check for ABSTRACT pattern
    if (profile.activation_diversity > t.get('abstract_diversity', 0.3) 
        and profile.child_count > t.get('abstract_children', 10)):
        return PatternType.ABSTRACT
    
    # Check for SPECIFIC pattern (memorization)
    if (profile.activation_diversity < t.get('specific_diversity', 0.1)
        and profile.child_count < t.get('specific_children', 3)
        and profile.activation_count > t.get('specific_min_activations', 100)):
        return PatternType.SPECIFIC
    
    return PatternType.TRANSITIONAL


class PACTreeMonitor:
    """
    Monitor for PAC tree health and pattern formation.
    
    Attaches to a PAC tree (or GAIA model's vocabulary) and tracks
    pattern activation, diversity, and generalization metrics.
    
    Usage:
        monitor = PACTreeMonitor(model.pac_tree)
        # or
        monitor = PACTreeMonitor(model.vocabulary.pac_tree)
        
        # During training
        monitor.record_activation(pattern_id, context_id)
        
        # Check health
        metrics = monitor.get_tree_metrics()
        profiles = monitor.get_all_profiles()
    """
    
    def __init__(
        self, 
        pac_tree: Any = None,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the monitor.
        
        Args:
            pac_tree: The PAC tree to monitor (can be set later)
            thresholds: Custom classification thresholds
        """
        self.pac_tree = pac_tree
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        
        # Pattern tracking
        self._profiles: Dict[str, PatternProfile] = {}
        self._tokens_seen: int = 0
        self._current_step: int = 0
        
        # History for trend analysis
        self._metrics_history: List[TreeMetrics] = []
        self._max_history: int = 1000
    
    def attach(self, pac_tree: Any) -> None:
        """Attach to a PAC tree after initialization."""
        self.pac_tree = pac_tree
    
    def record_activation(
        self, 
        pattern_id: str, 
        context_id: str,
        embedding: Optional[np.ndarray] = None
    ) -> None:
        """
        Record a pattern activation.
        
        Args:
            pattern_id: Unique identifier for the pattern
            context_id: Identifier for the activation context
            embedding: Optional pattern embedding for norm calculation
        """
        if pattern_id not in self._profiles:
            self._profiles[pattern_id] = PatternProfile(
                pattern_id=pattern_id,
                activation_count=0,
                activation_contexts=set()
            )
        
        profile = self._profiles[pattern_id]
        profile.activation_count += 1
        profile.activation_contexts.add(context_id)
        profile.activation_diversity = (
            len(profile.activation_contexts) / profile.activation_count
        )
        profile.last_activation_step = self._current_step
        
        if embedding is not None:
            profile.embedding_norm = float(np.linalg.norm(embedding))
        
        self._tokens_seen += 1
    
    def step(self) -> None:
        """Advance the training step counter."""
        self._current_step += 1
    
    def update_tree_structure(self) -> None:
        """
        Update structural information from the PAC tree.
        
        Should be called periodically to sync with actual tree structure.
        """
        if self.pac_tree is None:
            return
        
        # Get tree structure info
        # This depends on the PAC tree implementation
        if hasattr(self.pac_tree, 'get_pattern_children'):
            for pattern_id, profile in self._profiles.items():
                children = self.pac_tree.get_pattern_children(pattern_id)
                profile.child_count = len(children) if children else 0
                
        if hasattr(self.pac_tree, 'get_pattern_depth'):
            for pattern_id, profile in self._profiles.items():
                profile.depth = self.pac_tree.get_pattern_depth(pattern_id)
    
    def classify_all_patterns(self) -> None:
        """Classify all tracked patterns based on current behavior."""
        for profile in self._profiles.values():
            profile.pattern_type = classify_pattern(profile, self.thresholds)
    
    def get_tree_metrics(self) -> TreeMetrics:
        """
        Compute current tree health metrics.
        
        Returns:
            TreeMetrics with current tree state
        """
        self.classify_all_patterns()
        
        # Count by type
        abstract_count = sum(
            1 for p in self._profiles.values() 
            if p.pattern_type == PatternType.ABSTRACT
        )
        specific_count = sum(
            1 for p in self._profiles.values() 
            if p.pattern_type == PatternType.SPECIFIC
        )
        transitional_count = sum(
            1 for p in self._profiles.values() 
            if p.pattern_type == PatternType.TRANSITIONAL
        )
        
        total_patterns = len(self._profiles)
        
        # Compute structural metrics
        if self.pac_tree is not None and hasattr(self.pac_tree, 'max_depth'):
            depth = self.pac_tree.max_depth
        else:
            depth = max((p.depth for p in self._profiles.values()), default=0)
        
        # Breadth = patterns with no children (leaves)
        breadth = sum(
            1 for p in self._profiles.values() 
            if p.child_count == 0
        )
        
        # Branching factor
        non_leaves = [p for p in self._profiles.values() if p.child_count > 0]
        if non_leaves:
            branching_factor = sum(p.child_count for p in non_leaves) / len(non_leaves)
        else:
            branching_factor = 0.0
        
        # Compression ratio
        compression_ratio = total_patterns / max(self._tokens_seen, 1)
        
        # Reuse ratio (avg activations per pattern)
        if total_patterns > 0:
            reuse_ratio = self._tokens_seen / total_patterns
        else:
            reuse_ratio = 0.0
        
        # Byref candidates (placeholder - computed by generalization_monitor)
        byref_candidates = 0
        
        metrics = TreeMetrics(
            depth=depth,
            breadth=breadth,
            branching_factor=branching_factor,
            compression_ratio=compression_ratio,
            reuse_ratio=reuse_ratio,
            byref_candidates=byref_candidates,
            total_patterns=total_patterns,
            abstract_count=abstract_count,
            specific_count=specific_count,
            transitional_count=transitional_count
        )
        
        # Store in history
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._max_history:
            self._metrics_history.pop(0)
        
        return metrics
    
    def get_profile(self, pattern_id: str) -> Optional[PatternProfile]:
        """Get profile for a specific pattern."""
        return self._profiles.get(pattern_id)
    
    def get_all_profiles(self) -> Dict[str, PatternProfile]:
        """Get all pattern profiles."""
        return self._profiles.copy()
    
    def get_patterns_by_type(self, pattern_type: PatternType) -> List[PatternProfile]:
        """Get all patterns of a specific type."""
        self.classify_all_patterns()
        return [p for p in self._profiles.values() if p.pattern_type == pattern_type]
    
    def get_high_risk_patterns(self, min_activations: int = 50) -> List[str]:
        """
        Get patterns at high risk of memorization.
        
        These are SPECIFIC patterns with high activation but low diversity.
        """
        self.classify_all_patterns()
        return [
            p.pattern_id for p in self._profiles.values()
            if p.pattern_type == PatternType.SPECIFIC
            and p.activation_count >= min_activations
        ]
    
    def get_metrics_trend(self, window: int = 10) -> Optional[Dict[str, float]]:
        """
        Get trend of key metrics over recent history.
        
        Returns:
            Dict with trends (positive = increasing, negative = decreasing)
        """
        if len(self._metrics_history) < window:
            return None
        
        recent = self._metrics_history[-window:]
        
        def trend(values):
            if len(values) < 2:
                return 0.0
            return (values[-1] - values[0]) / max(abs(values[0]), 1e-6)
        
        return {
            'compression_trend': trend([m.compression_ratio for m in recent]),
            'specific_trend': trend([m.specific_count for m in recent]),
            'abstract_trend': trend([m.abstract_count for m in recent]),
            'reuse_trend': trend([m.reuse_ratio for m in recent])
        }
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self._profiles.clear()
        self._tokens_seen = 0
        self._current_step = 0
        self._metrics_history.clear()
