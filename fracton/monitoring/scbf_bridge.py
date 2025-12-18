"""
SCBF Bridge - Integration with Standard Consciousness Benchmark Framework.

Provides metrics and analysis that connect PAC tree monitoring with
SCBF consciousness benchmarks and phase transition detection.
"""

from dataclasses import dataclass
from typing import Any, Optional, Dict, List
import numpy as np

from .pac_tree_monitor import PatternType, PACTreeMonitor


# Physics constants from Dawn Field Theory
PHI_XI = 0.915965594177  # Golden ratio of criticality


@dataclass
class SCBFMetrics:
    """SCBF metrics relevant to PAC tree health."""
    entropy_collapse_risk: float    # How close to collapse (0-1)
    phase_alignment: float          # Alignment with PHI_XI
    criticality: float              # Self-organized criticality measure
    field_coherence: float          # Cross-pattern coherence
    
    @property
    def health_score(self) -> float:
        """Overall health from 0-1 (higher is better)."""
        # Low collapse risk is good
        # High phase alignment is good
        # Criticality near 1.0 is good
        # High coherence is good
        return (
            (1.0 - self.entropy_collapse_risk) * 0.3 +
            self.phase_alignment * 0.3 +
            self.criticality * 0.2 +
            self.field_coherence * 0.2
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'entropy_collapse_risk': self.entropy_collapse_risk,
            'phase_alignment': self.phase_alignment,
            'criticality': self.criticality,
            'field_coherence': self.field_coherence,
            'health_score': self.health_score
        }


class EntropyCollapseDetector:
    """
    Detects when a PAC tree is approaching entropy collapse.
    
    Entropy collapse occurs when too many patterns become SPECIFIC,
    indicating the tree is memorizing rather than generalizing.
    """
    
    def __init__(
        self,
        warning_threshold: float = 0.3,
        critical_threshold: float = 0.5,
        window_size: int = 100
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.window_size = window_size
        self._history: List[float] = []
    
    def update(self, specific_ratio: float) -> None:
        """Update with current specific pattern ratio."""
        self._history.append(specific_ratio)
        if len(self._history) > self.window_size:
            self._history.pop(0)
    
    def get_risk(self) -> float:
        """
        Get current entropy collapse risk (0-1).
        
        Uses quadratic scaling to penalize high specific ratios.
        """
        if not self._history:
            return 0.0
        
        current = self._history[-1]
        return min(current ** 2 / (self.critical_threshold ** 2), 1.0)
    
    def get_trend(self) -> float:
        """
        Get trend direction (-1 to 1).
        
        Positive = getting worse (more specific patterns)
        Negative = improving (more abstract patterns)
        """
        if len(self._history) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(self._history))
        y = np.array(self._history)
        
        # Normalize
        x_norm = x - x.mean()
        slope = np.sum(x_norm * y) / max(np.sum(x_norm ** 2), 1e-6)
        
        # Scale to -1 to 1
        return np.clip(slope * 100, -1.0, 1.0)
    
    def is_warning(self) -> bool:
        """Check if at warning level."""
        return len(self._history) > 0 and self._history[-1] >= self.warning_threshold
    
    def is_critical(self) -> bool:
        """Check if at critical level."""
        return len(self._history) > 0 and self._history[-1] >= self.critical_threshold


class PhaseAlignmentTracker:
    """
    Tracks phase alignment with the critical PHI_XI constant.
    
    Good learning maintains patterns near the critical point,
    allowing for both stability and adaptability.
    """
    
    def __init__(self, phi_xi: float = PHI_XI):
        self.phi_xi = phi_xi
        self._measurements: List[float] = []
    
    def measure_alignment(self, tree_monitor: PACTreeMonitor) -> float:
        """
        Measure phase alignment for current tree state.
        
        Computes a criticality measure based on pattern distribution
        and compares to PHI_XI.
        
        Args:
            tree_monitor: PACTreeMonitor with pattern data
            
        Returns:
            Phase alignment score (0-1, higher = better aligned)
        """
        metrics = tree_monitor.get_tree_metrics()
        
        # Compute criticality from pattern distribution
        if metrics.total_patterns == 0:
            return 0.0
        
        # Ratio of transitional patterns (the "edge of chaos")
        transitional_ratio = metrics.transitional_count / metrics.total_patterns
        
        # Abstract/Specific balance
        if metrics.abstract_count + metrics.specific_count > 0:
            abstract_ratio = metrics.abstract_count / (
                metrics.abstract_count + metrics.specific_count
            )
        else:
            abstract_ratio = 0.5
        
        # Combined criticality measure
        # Ideal: transitional_ratio near PHI_XI, abstract_ratio balanced
        criticality = (transitional_ratio + abstract_ratio) / 2
        
        # Distance from PHI_XI
        distance = abs(criticality - self.phi_xi)
        alignment = 1.0 - min(distance / self.phi_xi, 1.0)
        
        self._measurements.append(alignment)
        return alignment
    
    def get_stability(self, window: int = 10) -> float:
        """
        Get alignment stability over recent measurements.
        
        Returns:
            Stability score (0-1, higher = more stable)
        """
        if len(self._measurements) < window:
            return 0.5  # Unknown
        
        recent = self._measurements[-window:]
        variance = np.var(recent)
        
        # Low variance = high stability
        return 1.0 / (1.0 + variance * 100)


class SCBFBridge:
    """
    Bridge between SCBF metrics and PAC tree monitoring.
    
    Combines entropy collapse detection, phase alignment tracking,
    and coherence measurement into unified SCBF metrics.
    
    Usage:
        bridge = SCBFBridge()
        monitor = PACTreeMonitor(pac_tree)
        
        # During training
        metrics = bridge.get_tree_health(monitor)
        if metrics.entropy_collapse_risk > 0.3:
            print("Warning: Risk of memorization")
    """
    
    def __init__(
        self,
        entropy_detector: Optional[EntropyCollapseDetector] = None,
        phase_tracker: Optional[PhaseAlignmentTracker] = None
    ):
        self.entropy_detector = entropy_detector or EntropyCollapseDetector()
        self.phase_tracker = phase_tracker or PhaseAlignmentTracker()
    
    def get_tree_health(self, tree_monitor: PACTreeMonitor) -> SCBFMetrics:
        """
        Compute comprehensive SCBF metrics for current tree state.
        
        Args:
            tree_monitor: PACTreeMonitor with pattern data
            
        Returns:
            SCBFMetrics with all health indicators
        """
        metrics = tree_monitor.get_tree_metrics()
        
        # Entropy collapse risk
        if metrics.total_patterns > 0:
            specific_ratio = metrics.specific_count / metrics.total_patterns
        else:
            specific_ratio = 0.0
        
        self.entropy_detector.update(specific_ratio)
        collapse_risk = self.entropy_detector.get_risk()
        
        # Phase alignment
        phase_alignment = self.phase_tracker.measure_alignment(tree_monitor)
        
        # Criticality (distance from PHI_XI)
        tree_criticality = self._compute_criticality(metrics)
        criticality = 1.0 - abs(tree_criticality - PHI_XI)
        
        # Field coherence
        coherence = self._compute_coherence(tree_monitor)
        
        return SCBFMetrics(
            entropy_collapse_risk=collapse_risk,
            phase_alignment=phase_alignment,
            criticality=criticality,
            field_coherence=coherence
        )
    
    def _compute_criticality(self, metrics: 'TreeMetrics') -> float:
        """
        Compute tree criticality measure.
        
        Criticality is maximized when the tree is at the "edge of chaos" -
        balanced between order (specific) and disorder (abstract).
        """
        if metrics.total_patterns == 0:
            return 0.5
        
        # Branching factor contributes to criticality
        bf_contribution = min(metrics.branching_factor / 10, 1.0)
        
        # Compression ratio contributes
        comp_contribution = min(metrics.compression_ratio * 10, 1.0)
        
        # Type balance
        abstract = metrics.abstract_count
        specific = metrics.specific_count
        transitional = metrics.transitional_count
        
        total = abstract + specific + transitional
        if total == 0:
            type_balance = 0.5
        else:
            # Transitional patterns are the critical state
            type_balance = transitional / total
        
        return (bf_contribution + comp_contribution + type_balance) / 3
    
    def _compute_coherence(self, tree_monitor: PACTreeMonitor) -> float:
        """
        Compute field coherence across patterns.
        
        Coherence measures how well patterns work together -
        high coherence means good information flow.
        """
        profiles = tree_monitor.get_all_profiles()
        if not profiles:
            return 0.5
        
        # Coherence indicators:
        # 1. High reuse (patterns activated multiple times)
        avg_activation = np.mean([
            p.activation_count for p in profiles.values()
        ])
        reuse_coherence = min(avg_activation / 10, 1.0)
        
        # 2. Diverse contexts (patterns used in different situations)
        avg_diversity = np.mean([
            p.activation_diversity for p in profiles.values()
        ])
        diversity_coherence = avg_diversity
        
        # 3. Connected structure (patterns have children)
        patterns_with_children = sum(
            1 for p in profiles.values() if p.child_count > 0
        )
        structure_coherence = patterns_with_children / max(len(profiles), 1)
        
        return (reuse_coherence + diversity_coherence + structure_coherence) / 3
    
    def get_recommendations(self, metrics: SCBFMetrics) -> List[str]:
        """
        Get actionable recommendations based on current metrics.
        
        Args:
            metrics: Current SCBF metrics
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if metrics.entropy_collapse_risk > 0.5:
            recommendations.append(
                "CRITICAL: High memorization risk. Apply noise injection or dropout."
            )
        elif metrics.entropy_collapse_risk > 0.3:
            recommendations.append(
                "WARNING: Moderate memorization risk. Consider token masking."
            )
        
        if metrics.phase_alignment < 0.5:
            recommendations.append(
                "Phase misalignment detected. Tree may be stuck in local minimum."
            )
        
        if metrics.criticality < 0.5:
            recommendations.append(
                "Low criticality. System may need perturbation to reach edge of chaos."
            )
        
        if metrics.field_coherence < 0.3:
            recommendations.append(
                "Low coherence. Patterns not well connected. Consider byref optimization."
            )
        
        if not recommendations:
            recommendations.append("Tree health is good. Continue training.")
        
        return recommendations
