"""
GeometricConfidence - Topology-based confidence scoring.

Confidence derived from graph geometry, NOT model self-assessment.
The structure of the knowledge graph reveals what we can trust.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class GeometricConfidence:
    """
    Confidence derived from graph topology.

    The geometry of the knowledge graph reveals confidence:
    - Dense, symmetric regions = well-explored territory
    - Sparse, asymmetric regions = extrapolation/uncertainty
    - Orphaned nodes = potential hallucination
    - Missing expected children = knowledge gaps

    This is fundamentally different from asking a model "how sure are you?"
    Instead, we look at the STRUCTURE of knowledge.
    """

    # === Density Metrics ===
    local_density: float = 0.0  # 0-1: How tightly clustered is this region?
    # High density = many nearby concepts (well-explored)
    # Low density = isolated concept (uncertain)

    branch_symmetry: float = 0.0  # 0-1: Is the tree balanced?
    # High symmetry = well-developed concept tree
    # Low symmetry = asymmetric development (missing branches)

    # === Distance Metrics ===
    traversal_distance: float = 0.0  # Distance from well-supported nodes
    # Low distance = close to documented concepts
    # High distance = extrapolating far from knowledge base

    documentation_depth: int = 0  # Number of supporting documents
    # High count = well-documented
    # Low count = poorly documented

    # === Anomaly Detection ===
    orphan_score: float = 0.0  # 0-1: Is this concept orphaned?
    # High score = concept without clear lineage (suspicious)

    confluence_bottleneck: bool = False  # Disproportionate parent contribution?
    # True = one parent dominates (confluence_weight > 0.8)
    # Indicates potential mis-modeling of relationships

    missing_expected_children: List[str] = field(default_factory=list)
    # Detected gaps based on sibling patterns
    # If siblings have children X, Y, Z but this node doesn't, flag it

    # === Computed Scores ===
    retrieval_confidence: float = 0.0  # 0-1: Overall confidence
    hallucination_risk: float = 1.0  # 0-1: 1 - retrieval_confidence

    def compute(self) -> float:
        """
        Compute overall confidence from geometry.

        High confidence indicators:
        - Dense neighborhood (local_density > 0.7)
        - Symmetric branches (branch_symmetry > 0.7)
        - Good documentation (docs > 5)
        - Close to well-supported nodes (distance < 3)

        Low confidence indicators:
        - Sparse region (local_density < 0.3)
        - Asymmetric branches (branch_symmetry < 0.3)
        - Poor documentation (docs < 2)
        - Far from support (distance > 5)
        - Orphaned (orphan_score > 0.7)
        - Bottlenecked (confluence_bottleneck)

        Returns:
            Overall confidence score (0-1)
        """
        # Base score from positive indicators
        base = (
            self.local_density * 0.3
            + self.branch_symmetry * 0.2
            + (1 / (1 + self.traversal_distance)) * 0.3
            + min(self.documentation_depth / 10, 1.0) * 0.2
        )

        # Penalties for anomalies
        penalties = 0.0

        # Orphan penalty (heavy)
        penalties += self.orphan_score * 0.3

        # Bottleneck penalty
        if self.confluence_bottleneck:
            penalties += 0.1

        # Missing children penalty
        if self.missing_expected_children:
            penalties += 0.05 * min(len(self.missing_expected_children), 5)

        # Final score
        score = max(0.0, min(1.0, base - penalties))

        # Update fields
        self.retrieval_confidence = score
        self.hallucination_risk = 1.0 - score

        return score

    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of confidence."""
        if self.retrieval_confidence >= 0.8:
            return "High confidence - well-trodden territory"
        elif self.retrieval_confidence >= 0.5:
            return "Moderate confidence - reasonable extrapolation"
        elif self.retrieval_confidence >= 0.3:
            return "Low confidence - weak support"
        elif self.retrieval_confidence >= 0.1:
            return "Very low confidence - potential hallucination"
        else:
            return "No confidence - likely fabrication"

    @property
    def action_recommendation(self) -> str:
        """Recommended action based on confidence."""
        if self.retrieval_confidence >= 0.8:
            return "Trust retrieval"
        elif self.retrieval_confidence >= 0.5:
            return "Use with context"
        elif self.retrieval_confidence >= 0.3:
            return "Flag uncertainty"
        elif self.retrieval_confidence >= 0.1:
            return "Investigate/verify"
        else:
            return "Reject"

    @property
    def is_trustworthy(self) -> bool:
        """Can we trust this knowledge?"""
        return self.retrieval_confidence >= 0.7

    @property
    def is_suspicious(self) -> bool:
        """Should this be flagged as suspicious?"""
        return self.hallucination_risk >= 0.7

    @property
    def has_anomalies(self) -> bool:
        """Are there anomalies detected?"""
        return (
            self.orphan_score > 0.5
            or self.confluence_bottleneck
            or len(self.missing_expected_children) > 0
        )

    def get_anomaly_report(self) -> List[str]:
        """Get list of detected anomalies."""
        anomalies = []

        if self.orphan_score > 0.7:
            anomalies.append(f"High orphan score ({self.orphan_score:.2f}) - concept lacks clear lineage")

        if self.confluence_bottleneck:
            anomalies.append("Confluence bottleneck - one parent dominates (>80%)")

        if self.missing_expected_children:
            anomalies.append(
                f"Missing {len(self.missing_expected_children)} expected children: "
                f"{', '.join(self.missing_expected_children[:3])}"
                + ("..." if len(self.missing_expected_children) > 3 else "")
            )

        if self.local_density < 0.2:
            anomalies.append(f"Very sparse region (density={self.local_density:.2f})")

        if self.branch_symmetry < 0.3:
            anomalies.append(f"Asymmetric tree (symmetry={self.branch_symmetry:.2f})")

        if self.documentation_depth == 0:
            anomalies.append("No supporting documentation")

        if self.traversal_distance > 7:
            anomalies.append(
                f"Far from supported nodes (distance={self.traversal_distance:.1f})"
            )

        return anomalies

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "local_density": self.local_density,
            "branch_symmetry": self.branch_symmetry,
            "traversal_distance": self.traversal_distance,
            "documentation_depth": self.documentation_depth,
            "orphan_score": self.orphan_score,
            "confluence_bottleneck": self.confluence_bottleneck,
            "missing_expected_children": self.missing_expected_children.copy(),
            "retrieval_confidence": self.retrieval_confidence,
            "hallucination_risk": self.hallucination_risk,
            "interpretation": self.interpretation,
            "action_recommendation": self.action_recommendation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeometricConfidence":
        """Create from dictionary."""
        return cls(
            local_density=data.get("local_density", 0.0),
            branch_symmetry=data.get("branch_symmetry", 0.0),
            traversal_distance=data.get("traversal_distance", 0.0),
            documentation_depth=data.get("documentation_depth", 0),
            orphan_score=data.get("orphan_score", 0.0),
            confluence_bottleneck=data.get("confluence_bottleneck", False),
            missing_expected_children=data.get("missing_expected_children", []),
            retrieval_confidence=data.get("retrieval_confidence", 0.0),
            hallucination_risk=data.get("hallucination_risk", 1.0),
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GeometricConfidence("
            f"confidence={self.retrieval_confidence:.2f}, "
            f"risk={self.hallucination_risk:.2f}, "
            f"interpretation='{self.interpretation}')"
        )


def compute_local_density(
    node_id: str, graph: "KronosGraph", radius: int = 2
) -> float:
    """
    Compute local density around a node.

    Density = (actual_neighbors / expected_neighbors) for given radius.

    Args:
        node_id: Node to measure around
        graph: Knowledge graph
        radius: How many hops to consider

    Returns:
        Density score 0-1 (1 = very dense, 0 = isolated)
    """
    # Get all nodes within radius
    neighbors = graph.get_neighbors_within_radius(node_id, radius)

    # Expected neighbors for a balanced tree at this depth
    node = graph.get_node(node_id)
    if not node:
        return 0.0

    # Rough estimate: balanced tree has ~branching_factor^radius neighbors
    branching_factor = 3  # Assume average branching of 3
    expected = branching_factor**radius

    # Actual density
    actual = len(neighbors)
    density = min(1.0, actual / expected)

    return density


def compute_branch_symmetry(node_id: str, graph: "KronosGraph") -> float:
    """
    Compute branch symmetry for a node.

    Symmetry = how balanced is the tree structure?

    High symmetry:
    - All children have similar numbers of children
    - All parents contribute equally to confluence

    Low symmetry:
    - Some children have many descendants, others none
    - One parent dominates confluence pattern

    Args:
        node_id: Node to measure
        graph: Knowledge graph

    Returns:
        Symmetry score 0-1 (1 = perfectly balanced, 0 = very asymmetric)
    """
    node = graph.get_node(node_id)
    if not node:
        return 0.0

    symmetry_scores = []

    # Check child symmetry
    if node.child_actualizations:
        child_descendant_counts = []
        for child_id in node.child_actualizations:
            descendants = graph.get_descendants(child_id, max_depth=2)
            child_descendant_counts.append(len(descendants))

        if child_descendant_counts:
            # Coefficient of variation (low = symmetric)
            import statistics

            if len(child_descendant_counts) > 1:
                mean = statistics.mean(child_descendant_counts)
                if mean > 0:
                    stdev = statistics.stdev(child_descendant_counts)
                    cv = stdev / mean
                    child_symmetry = max(0.0, 1.0 - cv)  # Lower CV = higher symmetry
                else:
                    child_symmetry = 1.0
            else:
                child_symmetry = 1.0

            symmetry_scores.append(child_symmetry)

    # Check confluence symmetry
    if node.confluence_pattern:
        weights = list(node.confluence_pattern.values())
        if len(weights) > 1:
            # Check if one parent dominates (>0.8)
            max_weight = max(weights)
            if max_weight > 0.8:
                confluence_symmetry = 0.0  # Bottleneck
            else:
                # Measure uniformity
                ideal_weight = 1.0 / len(weights)
                deviations = [abs(w - ideal_weight) for w in weights]
                avg_deviation = sum(deviations) / len(deviations)
                confluence_symmetry = max(0.0, 1.0 - avg_deviation * 2)
        else:
            confluence_symmetry = 1.0  # Single parent is "symmetric"

        symmetry_scores.append(confluence_symmetry)

    # Average symmetry scores
    if symmetry_scores:
        return sum(symmetry_scores) / len(symmetry_scores)
    else:
        return 0.5  # Neutral for leaf nodes


def compute_traversal_distance(
    node_id: str, graph: "KronosGraph", well_documented_threshold: int = 5
) -> float:
    """
    Compute distance to nearest well-documented node.

    Uses BFS to find closest node with documentation_depth >= threshold.

    Args:
        node_id: Starting node
        graph: Knowledge graph
        well_documented_threshold: Min docs to be considered well-documented

    Returns:
        Distance (number of hops) to nearest well-documented node
    """
    from collections import deque

    visited = set()
    queue = deque([(node_id, 0)])  # (node_id, distance)

    while queue:
        current_id, distance = queue.popleft()

        if current_id in visited:
            continue
        visited.add(current_id)

        current_node = graph.get_node(current_id)
        if not current_node:
            continue

        # Check if this node is well-documented
        if current_node.documentation_depth >= well_documented_threshold:
            return float(distance)

        # Add neighbors to queue
        # Parents (up the tree)
        for parent_id in current_node.parent_potentials:
            if parent_id not in visited:
                queue.append((parent_id, distance + 1))

        # Children (down the tree)
        for child_id in current_node.child_actualizations:
            if child_id not in visited:
                queue.append((child_id, distance + 1))

        # Siblings (same level)
        for sibling_id in current_node.sibling_nodes:
            if sibling_id not in visited:
                queue.append((sibling_id, distance + 1))

    # No well-documented nodes found
    return float("inf")
