"""
Visualization - ASCII and text-based tree health visualization.

Provides human-readable output for monitoring PAC tree health
during training without requiring graphical dependencies.
"""

from typing import List, Dict, Any, Optional
from .pac_tree_monitor import TreeMetrics, PatternType
from .generalization_monitor import GeneralizationZone
from .scbf_bridge import SCBFMetrics


def visualize_tree_health(
    tree_metrics: TreeMetrics,
    zones: Optional[List[GeneralizationZone]] = None,
    scbf_metrics: Optional[SCBFMetrics] = None
) -> str:
    """
    Generate ASCII visualization of tree health.
    
    Args:
        tree_metrics: Current TreeMetrics
        zones: Optional list of GeneralizationZones
        scbf_metrics: Optional SCBF metrics
        
    Returns:
        Multi-line string visualization
    """
    lines = []
    
    # Header
    lines.append("╔══════════════════════════════════════════╗")
    lines.append("║         PAC Tree Health Report           ║")
    lines.append("╠══════════════════════════════════════════╣")
    
    # Structure metrics
    lines.append(f"║ Depth: {tree_metrics.depth:>4}  │  Breadth: {tree_metrics.breadth:>6}     ║")
    lines.append(f"║ Branching Factor: {tree_metrics.branching_factor:>6.2f}              ║")
    lines.append(f"║ Compression: {tree_metrics.compression_ratio:>8.4f}                 ║")
    lines.append(f"║ Reuse Ratio: {tree_metrics.reuse_ratio:>8.2f}                 ║")
    lines.append("╠──────────────────────────────────────────╣")
    
    # Pattern distribution
    total = tree_metrics.total_patterns or 1
    abstract_pct = tree_metrics.abstract_count / total * 100
    specific_pct = tree_metrics.specific_count / total * 100
    trans_pct = tree_metrics.transitional_count / total * 100
    
    lines.append("║ Pattern Distribution:                    ║")
    lines.append(f"║   Abstract:     {tree_metrics.abstract_count:>5} ({abstract_pct:>5.1f}%)       ║")
    lines.append(f"║   Specific:     {tree_metrics.specific_count:>5} ({specific_pct:>5.1f}%)       ║")
    lines.append(f"║   Transitional: {tree_metrics.transitional_count:>5} ({trans_pct:>5.1f}%)       ║")
    lines.append(f"║   Total:        {tree_metrics.total_patterns:>5}                 ║")
    
    # Pattern bar visualization
    bar_width = 30
    abs_bar = int(abstract_pct / 100 * bar_width)
    spec_bar = int(specific_pct / 100 * bar_width)
    trans_bar = bar_width - abs_bar - spec_bar
    
    bar = "█" * abs_bar + "▓" * trans_bar + "░" * spec_bar
    lines.append(f"║   [{bar}] ║")
    lines.append("║   █=Abstract ▓=Trans ░=Specific         ║")
    
    # Byref candidates
    if tree_metrics.byref_candidates > 0:
        lines.append("╠──────────────────────────────────────────╣")
        lines.append(f"║ Byref Candidates: {tree_metrics.byref_candidates:>4}                   ║")
    
    # Generalization zones
    if zones:
        lines.append("╠──────────────────────────────────────────╣")
        lines.append("║ Generalization Zones:                    ║")
        for zone in zones[:5]:  # Top 5 zones
            health = _health_indicator(zone.health_score)
            lines.append(f"║   {health} {zone.zone_id}: {zone.size:>3} patterns, "
                        f"depth {zone.zone_depth_range[0]}-{zone.zone_depth_range[1]} ║")
        if len(zones) > 5:
            lines.append(f"║   ... and {len(zones) - 5} more zones              ║")
    
    # SCBF metrics
    if scbf_metrics:
        lines.append("╠══════════════════════════════════════════╣")
        lines.append("║ SCBF Metrics:                            ║")
        lines.append(f"║   Collapse Risk:  {_risk_bar(scbf_metrics.entropy_collapse_risk)} ║")
        lines.append(f"║   Phase Align:    {_health_bar(scbf_metrics.phase_alignment)} ║")
        lines.append(f"║   Criticality:    {_health_bar(scbf_metrics.criticality)} ║")
        lines.append(f"║   Coherence:      {_health_bar(scbf_metrics.field_coherence)} ║")
        lines.append(f"║   Overall:        {_health_bar(scbf_metrics.health_score)} ║")
    
    # Footer
    lines.append("╚══════════════════════════════════════════╝")
    
    return "\n".join(lines)


def _health_indicator(score: float) -> str:
    """Get emoji indicator for health score."""
    if score >= 0.7:
        return "🟢"
    elif score >= 0.4:
        return "🟡"
    else:
        return "🔴"


def _health_bar(value: float, width: int = 20) -> str:
    """Generate a health bar (higher = better)."""
    filled = int(value * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    indicator = _health_indicator(value)
    return f"{indicator} {bar} {value:.2f}"


def _risk_bar(value: float, width: int = 20) -> str:
    """Generate a risk bar (lower = better)."""
    filled = int(value * width)
    empty = width - filled
    bar = "▓" * filled + "░" * empty
    # Invert for indicator (high risk = bad)
    indicator = _health_indicator(1.0 - value)
    return f"{indicator} {bar} {value:.2f}"


def visualize_pattern_distribution(
    tree_metrics: TreeMetrics,
    history: Optional[List[TreeMetrics]] = None,
    width: int = 60
) -> str:
    """
    Visualize pattern type distribution over time.
    
    Args:
        tree_metrics: Current metrics
        history: Optional list of historical metrics
        width: Character width for the visualization
        
    Returns:
        Multi-line string visualization
    """
    lines = []
    lines.append("Pattern Distribution Over Time")
    lines.append("─" * width)
    
    if history and len(history) > 1:
        # Show sparkline for each type
        lines.append(_sparkline("Abstract", [m.abstract_count for m in history], width))
        lines.append(_sparkline("Specific", [m.specific_count for m in history], width))
        lines.append(_sparkline("Trans", [m.transitional_count for m in history], width))
    else:
        # Just show current state
        total = tree_metrics.total_patterns or 1
        lines.append(f"Abstract:     {'█' * int(tree_metrics.abstract_count / total * 40)}")
        lines.append(f"Specific:     {'▓' * int(tree_metrics.specific_count / total * 40)}")
        lines.append(f"Transitional: {'░' * int(tree_metrics.transitional_count / total * 40)}")
    
    lines.append("─" * width)
    return "\n".join(lines)


def _sparkline(label: str, values: List[int], width: int) -> str:
    """Generate a sparkline for a series of values."""
    if not values:
        return f"{label:>10}: [no data]"
    
    # Normalize to available width
    chart_width = width - 12  # Leave room for label
    
    if len(values) > chart_width:
        # Downsample
        step = len(values) / chart_width
        values = [values[int(i * step)] for i in range(chart_width)]
    
    max_val = max(values) or 1
    
    # Sparkline characters (increasing height)
    chars = " ▁▂▃▄▅▆▇█"
    
    sparkline = ""
    for v in values:
        idx = int(v / max_val * (len(chars) - 1))
        sparkline += chars[idx]
    
    return f"{label:>10}: {sparkline}"


def format_health_summary(summary: Dict[str, Any]) -> str:
    """
    Format a health summary dict as readable text.
    
    Args:
        summary: Dict from LanguageGeneralizationMonitor.get_health_summary()
        
    Returns:
        Formatted string
    """
    lines = []
    
    # Status indicator
    if summary.get('is_overfitting', False):
        status = "⚠️  OVERFITTING DETECTED"
    elif summary.get('specific_ratio', 0) > 0.3:
        status = "🟡 CAUTION: High specific ratio"
    else:
        status = "🟢 HEALTHY"
    
    lines.append(f"Status: {status}")
    lines.append(f"Step: {summary.get('step', 'N/A')}")
    lines.append("")
    
    lines.append(f"Patterns: {summary.get('total_patterns', 0)}")
    lines.append(f"  Abstract: {summary.get('abstract_count', 0)}")
    lines.append(f"  Specific: {summary.get('specific_count', 0)} "
                f"({summary.get('specific_ratio', 0):.1%})")
    lines.append("")
    
    lines.append(f"Compression: {summary.get('compression_ratio', 0):.4f}")
    lines.append(f"Zones: {summary.get('num_zones', 0)} "
                f"(avg health: {summary.get('avg_zone_health', 0):.2f})")
    lines.append(f"Largest zone: {summary.get('largest_zone_size', 0)} patterns")
    lines.append("")
    
    if summary.get('byref_candidates', 0) > 0:
        lines.append(f"💡 {summary['byref_candidates']} byref optimization opportunities")
    
    if summary.get('high_risk_patterns', 0) > 0:
        lines.append(f"⚠️  {summary['high_risk_patterns']} high-risk patterns")
    
    return "\n".join(lines)


def create_training_dashboard(
    step: int,
    loss: float,
    tree_metrics: TreeMetrics,
    scbf_metrics: Optional[SCBFMetrics] = None,
    interventions: Optional[List[Any]] = None
) -> str:
    """
    Create a compact training dashboard for console output.
    
    Args:
        step: Current training step
        loss: Current loss value
        tree_metrics: Current tree metrics
        scbf_metrics: Optional SCBF metrics
        interventions: Optional list of active interventions
        
    Returns:
        Single-line or compact multi-line dashboard
    """
    # Compact single-line format
    total = tree_metrics.total_patterns or 1
    spec_pct = tree_metrics.specific_count / total * 100
    
    health = "🟢" if spec_pct < 30 else "🟡" if spec_pct < 50 else "🔴"
    
    line = (
        f"[{step:>6}] loss={loss:.4f} "
        f"│ {health} patterns={total} spec={spec_pct:.1f}% "
        f"│ compress={tree_metrics.compression_ratio:.4f}"
    )
    
    if scbf_metrics:
        line += f" │ health={scbf_metrics.health_score:.2f}"
    
    if interventions:
        active = [i.type.value[:4] for i in interventions[:2]]
        line += f" │ ⚡{','.join(active)}"
    
    return line
