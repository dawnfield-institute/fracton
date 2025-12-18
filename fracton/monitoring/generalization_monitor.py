"""
Generalization Monitor - Zone detection and byref optimization.

Identifies connected regions of abstract patterns (generalization zones)
and detects branches that could benefit from byref optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import deque
import numpy as np

from .pac_tree_monitor import (
    PatternType, 
    PatternProfile, 
    PACTreeMonitor,
    classify_pattern
)


@dataclass
class GeneralizationZone:
    """A connected region of abstract patterns."""
    zone_id: str
    root_pattern: str
    abstract_patterns: Set[str] = field(default_factory=set)
    boundary_patterns: Set[str] = field(default_factory=set)  # TRANSITIONAL at edges
    specific_leaves: Set[str] = field(default_factory=set)    # SPECIFIC at leaves
    zone_entropy: float = 0.0
    zone_depth_range: Tuple[int, int] = (0, 0)
    
    @property
    def size(self) -> int:
        return len(self.abstract_patterns)
    
    @property
    def health_score(self) -> float:
        """
        Score from 0-1 indicating zone health.
        
        Healthy zones have:
        - Many abstract patterns
        - Clear boundary (transitional layer)
        - Specific patterns only at leaves
        """
        if not self.abstract_patterns:
            return 0.0
        
        # Penalize if too few abstract patterns
        size_score = min(len(self.abstract_patterns) / 10, 1.0)
        
        # Reward having a boundary layer
        boundary_ratio = len(self.boundary_patterns) / max(len(self.abstract_patterns), 1)
        boundary_score = min(boundary_ratio * 2, 1.0)
        
        # Depth range is good (not too flat)
        depth_range = self.zone_depth_range[1] - self.zone_depth_range[0]
        depth_score = min(depth_range / 5, 1.0)
        
        return (size_score + boundary_score + depth_score) / 3


@dataclass
class BranchSimilarity:
    """Detected similarity between PAC tree branches."""
    branch_a: str
    branch_b: str
    similarity_score: float         # 0.0 to 1.0
    shared_structure_depth: int     # How deep the similarity extends
    estimated_memory_savings: int   # Bytes saved by byref
    
    def __lt__(self, other: 'BranchSimilarity') -> bool:
        return self.estimated_memory_savings < other.estimated_memory_savings


def detect_generalization_zones(
    monitor: PACTreeMonitor,
    min_zone_size: int = 5
) -> List[GeneralizationZone]:
    """
    Find connected regions of abstract patterns.
    
    A healthy tree has:
    - Large generalization zones near the root
    - TRANSITIONAL patterns at zone boundaries
    - SPECIFIC patterns only at leaves (fine-grained distinctions)
    
    Warning signs:
    - SPECIFIC patterns at shallow depths (memorization)
    - Small, disconnected generalization zones
    - No TRANSITIONAL boundary layer
    
    Args:
        monitor: PACTreeMonitor with pattern data
        min_zone_size: Minimum number of abstract patterns for a zone
        
    Returns:
        List of detected GeneralizationZone objects
    """
    zones = []
    visited: Set[str] = set()
    profiles = monitor.get_all_profiles()
    
    # Need tree structure for traversal
    pac_tree = monitor.pac_tree
    
    # Get all abstract patterns as starting points
    abstract_patterns = [
        p for p in profiles.values() 
        if p.pattern_type == PatternType.ABSTRACT
    ]
    
    for start_pattern in abstract_patterns:
        if start_pattern.pattern_id in visited:
            continue
        
        # BFS to find connected abstract region
        zone_patterns: Set[str] = set()
        boundary: Set[str] = set()
        leaves: Set[str] = set()
        depths: List[int] = []
        
        queue = deque([start_pattern.pattern_id])
        
        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)
            
            current = profiles.get(current_id)
            if current is None:
                continue
            
            if current.pattern_type == PatternType.ABSTRACT:
                zone_patterns.add(current_id)
                depths.append(current.depth)
                
                # Get children if we have tree access
                if pac_tree and hasattr(pac_tree, 'get_pattern_children'):
                    children = pac_tree.get_pattern_children(current_id)
                    if children:
                        for child_id in children:
                            if child_id not in visited:
                                queue.append(child_id)
                                
            elif current.pattern_type == PatternType.TRANSITIONAL:
                boundary.add(current_id)
                
            elif current.pattern_type == PatternType.SPECIFIC:
                leaves.add(current_id)
        
        if len(zone_patterns) >= min_zone_size:
            zones.append(GeneralizationZone(
                zone_id=f"zone_{len(zones)}",
                root_pattern=start_pattern.pattern_id,
                abstract_patterns=zone_patterns,
                boundary_patterns=boundary,
                specific_leaves=leaves,
                zone_entropy=_compute_zone_entropy(zone_patterns, profiles),
                zone_depth_range=(
                    min(depths) if depths else 0,
                    max(depths) if depths else 0
                )
            ))
    
    return zones


def _compute_zone_entropy(
    pattern_ids: Set[str], 
    profiles: Dict[str, PatternProfile]
) -> float:
    """Compute aggregate entropy for a zone."""
    entropies = [
        profiles[pid].entropy_signature 
        for pid in pattern_ids 
        if pid in profiles
    ]
    if not entropies:
        return 0.0
    return float(np.mean(entropies))


def detect_byref_candidates(
    monitor: PACTreeMonitor,
    similarity_threshold: float = 0.85,
    embedding_distance_threshold: float = 0.15
) -> List[BranchSimilarity]:
    """
    Find branches that could benefit from byref optimization.
    
    Uses pattern similarity based on:
    - Structural similarity (same depth, similar children)
    - Activation similarity (similar contexts activate them)
    - Embedding similarity (if available)
    
    Args:
        monitor: PACTreeMonitor with pattern data
        similarity_threshold: Minimum similarity for consideration
        embedding_distance_threshold: Max normalized distance for embeddings
        
    Returns:
        List of BranchSimilarity candidates, sorted by memory savings
    """
    candidates = []
    profiles = list(monitor.get_all_profiles().values())
    
    # Only compare patterns at same depth with similar structure
    depth_groups: Dict[int, List[PatternProfile]] = {}
    for p in profiles:
        if p.depth not in depth_groups:
            depth_groups[p.depth] = []
        depth_groups[p.depth].append(p)
    
    for depth, group in depth_groups.items():
        for i, p1 in enumerate(group):
            for p2 in group[i+1:]:
                similarity = _compute_branch_similarity(p1, p2)
                
                if similarity >= similarity_threshold:
                    # Estimate memory savings (rough: embedding size * depth)
                    est_savings = 256 * 4 * max(p1.depth, p2.depth)  # 256 dim * 4 bytes
                    
                    candidates.append(BranchSimilarity(
                        branch_a=p1.pattern_id,
                        branch_b=p2.pattern_id,
                        similarity_score=similarity,
                        shared_structure_depth=depth,
                        estimated_memory_savings=est_savings
                    ))
    
    return sorted(candidates, reverse=True)


def _compute_branch_similarity(p1: PatternProfile, p2: PatternProfile) -> float:
    """
    Compute similarity between two pattern profiles.
    
    Considers:
    - Child count similarity
    - Activation count similarity  
    - Context overlap (Jaccard similarity)
    - Type match bonus
    """
    # Child count similarity
    max_children = max(p1.child_count, p2.child_count, 1)
    child_sim = 1.0 - abs(p1.child_count - p2.child_count) / max_children
    
    # Activation count similarity (log scale)
    log_a1 = np.log1p(p1.activation_count)
    log_a2 = np.log1p(p2.activation_count)
    max_log = max(log_a1, log_a2, 1)
    activation_sim = 1.0 - abs(log_a1 - log_a2) / max_log
    
    # Context overlap (Jaccard)
    if p1.activation_contexts and p2.activation_contexts:
        intersection = len(p1.activation_contexts & p2.activation_contexts)
        union = len(p1.activation_contexts | p2.activation_contexts)
        context_sim = intersection / union if union > 0 else 0
    else:
        context_sim = 0.0
    
    # Type match bonus
    type_bonus = 0.1 if p1.pattern_type == p2.pattern_type else 0.0
    
    # Weighted combination
    similarity = (
        0.3 * child_sim +
        0.2 * activation_sim +
        0.4 * context_sim +
        type_bonus
    )
    
    return min(similarity, 1.0)


class LanguageGeneralizationMonitor:
    """
    High-level monitor for language learning generalization.
    
    Combines PACTreeMonitor with zone detection and byref optimization
    to provide a complete view of how language is being learned.
    
    Usage:
        monitor = LanguageGeneralizationMonitor(model.pac_tree)
        
        # During training
        for batch in dataloader:
            loss = model.train_step(batch)
            monitor.update(batch)
            
            if step % 100 == 0:
                report = monitor.get_report()
                print(report.summary())
    """
    
    def __init__(
        self,
        pac_tree: Any = None,
        check_interval: int = 100,
        byref_threshold: float = 0.85,
        min_zone_size: int = 5
    ):
        self.tree_monitor = PACTreeMonitor(pac_tree)
        self.check_interval = check_interval
        self.byref_threshold = byref_threshold
        self.min_zone_size = min_zone_size
        
        self._step_count = 0
        self._zones: List[GeneralizationZone] = []
        self._byref_candidates: List[BranchSimilarity] = []
        
    def attach(self, pac_tree: Any) -> None:
        """Attach to a PAC tree."""
        self.tree_monitor.attach(pac_tree)
    
    def record_activation(
        self, 
        pattern_id: str, 
        context_id: str,
        embedding: Optional[np.ndarray] = None
    ) -> None:
        """Record a pattern activation."""
        self.tree_monitor.record_activation(pattern_id, context_id, embedding)
    
    def step(self) -> None:
        """Advance training step and run periodic analysis."""
        self._step_count += 1
        self.tree_monitor.step()
        
        if self._step_count % self.check_interval == 0:
            self._run_analysis()
    
    def _run_analysis(self) -> None:
        """Run full analysis of tree state."""
        self.tree_monitor.update_tree_structure()
        self._zones = detect_generalization_zones(
            self.tree_monitor, 
            self.min_zone_size
        )
        self._byref_candidates = detect_byref_candidates(
            self.tree_monitor,
            self.byref_threshold
        )
    
    @property
    def zones(self) -> List[GeneralizationZone]:
        """Get current generalization zones."""
        return self._zones
    
    @property
    def byref_candidates(self) -> List[BranchSimilarity]:
        """Get current byref optimization candidates."""
        return self._byref_candidates
    
    def get_tree_metrics(self) -> 'TreeMetrics':
        """Get current tree metrics."""
        from .pac_tree_monitor import TreeMetrics
        metrics = self.tree_monitor.get_tree_metrics()
        # Update byref count from our analysis
        metrics.byref_candidates = len(self._byref_candidates)
        return metrics
    
    def get_high_risk_patterns(self) -> List[str]:
        """Get patterns at risk of memorization."""
        return self.tree_monitor.get_high_risk_patterns()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of tree health.
        
        Returns:
            Dict with health indicators
        """
        metrics = self.get_tree_metrics()
        
        # Zone health
        if self._zones:
            avg_zone_health = np.mean([z.health_score for z in self._zones])
            largest_zone = max(self._zones, key=lambda z: z.size)
        else:
            avg_zone_health = 0.0
            largest_zone = None
        
        # Risk indicators
        specific_ratio = metrics.specific_count / max(metrics.total_patterns, 1)
        is_overfitting = specific_ratio > 0.5
        
        return {
            'step': self._step_count,
            'total_patterns': metrics.total_patterns,
            'abstract_count': metrics.abstract_count,
            'specific_count': metrics.specific_count,
            'specific_ratio': specific_ratio,
            'compression_ratio': metrics.compression_ratio,
            'num_zones': len(self._zones),
            'avg_zone_health': avg_zone_health,
            'largest_zone_size': largest_zone.size if largest_zone else 0,
            'byref_candidates': len(self._byref_candidates),
            'is_overfitting': is_overfitting,
            'high_risk_patterns': len(self.get_high_risk_patterns())
        }
