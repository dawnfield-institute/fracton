"""
Fracton Monitoring Module

PAC Tree monitoring for detecting overfitting vs generalization during training.
Provides real-time observation of pattern formation and tree health.

Usage:
    from fracton.monitoring import (
        PACTreeMonitor,
        PatternType,
        TreeMetrics,
        PatternProfile,
        GeneralizationZone,
        detect_generalization_zones,
        GeneralizationNurturingTrainer,
        visualize_tree_health
    )
"""

from .pac_tree_monitor import (
    PatternType,
    TreeMetrics,
    PatternProfile,
    PACTreeMonitor,
    classify_pattern
)

from .generalization_monitor import (
    GeneralizationZone,
    detect_generalization_zones,
    LanguageGeneralizationMonitor
)

from .interventions import (
    InterventionType,
    Intervention,
    GeneralizationNurturingTrainer
)

from .scbf_bridge import (
    SCBFMetrics,
    SCBFBridge
)

from .visualization import (
    visualize_tree_health,
    visualize_pattern_distribution
)

__all__ = [
    # Core types
    "PatternType",
    "TreeMetrics", 
    "PatternProfile",
    "PACTreeMonitor",
    "classify_pattern",
    # Generalization
    "GeneralizationZone",
    "detect_generalization_zones",
    "LanguageGeneralizationMonitor",
    # Interventions
    "InterventionType",
    "Intervention",
    "GeneralizationNurturingTrainer",
    # SCBF
    "SCBFMetrics",
    "SCBFBridge",
    # Visualization
    "visualize_tree_health",
    "visualize_pattern_distribution"
]
