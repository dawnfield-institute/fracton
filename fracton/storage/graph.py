"""
KronosGraph - Conceptual genealogy tree container.

Manages nodes, edges, and tree operations for the knowledge graph.
Implements PAC conservation, lineage tracking, and confidence computation.
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from fracton.storage.confidence import (
    GeometricConfidence,
    compute_branch_symmetry,
    compute_local_density,
    compute_traversal_distance,
)
from fracton.storage.edge import KronosEdge, RelationType
from fracton.storage.node import KronosNode

logger = logging.getLogger(__name__)


class KronosGraph:
    """
    Conceptual genealogy tree with lineage-aware operations.

    Maintains the knowledge graph as a tree where identity emerges
    from confluence patterns, not labels.
    """

    def __init__(self):
        """Initialize empty graph."""
        self.nodes: Dict[str, KronosNode] = {}
        self.edges: List[KronosEdge] = []
        self.roots: List[str] = []  # Top-level potentials (no parents)

        # Index for fast edge lookup
        self._edge_index: Dict[Tuple[str, str, RelationType], KronosEdge] = {}

        logger.info("Initialized empty KronosGraph")

    # === Node Management ===

    def add_node(self, node: KronosNode) -> bool:
        """
        Add node to graph and establish lineage relationships.

        Args:
            node: Node to add

        Returns:
            True if added successfully, False if already exists
        """
        if node.id in self.nodes:
            logger.warning(f"Node {node.id} already exists")
            return False

        # Add node
        self.nodes[node.id] = node

        # Update roots list
        if node.is_root:
            self.roots.append(node.id)

        # Establish parent relationships
        for parent_id in node.parent_potentials:
            if parent_id in self.nodes:
                parent = self.nodes[parent_id]
                parent.add_child(node.id)

                # Create parent-child edges
                self.add_edge(
                    KronosEdge(
                        source_id=parent_id,
                        target_id=node.id,
                        relationship_type=RelationType.PARENT_OF,
                        strength=node.confluence_pattern.get(parent_id, 1.0),
                    )
                )

        # Establish sibling relationships
        self._update_siblings(node.id)

        logger.info(f"Added node: {node.id} (depth={node.actualization_depth})")
        return True

    def get_node(self, node_id: str) -> Optional[KronosNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        """
        Remove node and rebalance tree.

        Args:
            node_id: Node to remove

        Returns:
            True if removed, False if not found
        """
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]

        # Remove from parents' child lists
        for parent_id in node.parent_potentials:
            if parent_id in self.nodes:
                parent = self.nodes[parent_id]
                if node_id in parent.child_actualizations:
                    parent.child_actualizations.remove(node_id)

        # Remove from siblings' sibling lists
        for sibling_id in node.sibling_nodes:
            if sibling_id in self.nodes:
                sibling = self.nodes[sibling_id]
                if node_id in sibling.sibling_nodes:
                    sibling.sibling_nodes.remove(node_id)

        # Remove edges
        self.edges = [
            e
            for e in self.edges
            if e.source_id != node_id and e.target_id != node_id
        ]
        self._rebuild_edge_index()

        # Remove from roots if applicable
        if node_id in self.roots:
            self.roots.remove(node_id)

        # Remove node
        del self.nodes[node_id]

        logger.info(f"Removed node: {node_id}")
        return True

    # === Edge Management ===

    def add_edge(self, edge: KronosEdge) -> bool:
        """
        Add edge to graph.

        Args:
            edge: Edge to add

        Returns:
            True if added, False if already exists
        """
        key = (edge.source_id, edge.target_id, edge.relationship_type)

        if key in self._edge_index:
            logger.debug(f"Edge already exists: {edge}")
            return False

        self.edges.append(edge)
        self._edge_index[key] = edge

        return True

    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relationship_type: Optional[RelationType] = None,
    ) -> List[KronosEdge]:
        """
        Get edges matching criteria.

        Args:
            source_id: Filter by source
            target_id: Filter by target
            relationship_type: Filter by type

        Returns:
            List of matching edges
        """
        results = self.edges

        if source_id:
            results = [e for e in results if e.source_id == source_id]
        if target_id:
            results = [e for e in results if e.target_id == target_id]
        if relationship_type:
            results = [e for e in results if e.relationship_type == relationship_type]

        return results

    # === Tree Traversal ===

    def get_ancestors(self, node_id: str, max_depth: int = -1) -> List[KronosNode]:
        """
        Get all parent potentials up the tree.

        Returns concepts this crystallized FROM (more general).

        Args:
            node_id: Starting node
            max_depth: Maximum depth (-1 = unlimited)

        Returns:
            List of ancestor nodes
        """
        if node_id not in self.nodes:
            return []

        ancestors = []
        visited = set()
        queue = deque([(node_id, 0)])

        while queue:
            current_id, depth = queue.popleft()

            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id != node_id:  # Don't include self
                current_node = self.nodes.get(current_id)
                if current_node:
                    ancestors.append(current_node)

            # Stop if max_depth reached
            if max_depth >= 0 and depth >= max_depth:
                continue

            # Add parents to queue
            current_node = self.nodes.get(current_id)
            if current_node:
                for parent_id in current_node.parent_potentials:
                    if parent_id not in visited:
                        queue.append((parent_id, depth + 1))

        return ancestors

    def get_descendants(self, node_id: str, max_depth: int = 2) -> List[KronosNode]:
        """
        Get all child actualizations down the tree.

        Returns concepts that crystallized FROM this (more specific).

        Args:
            node_id: Starting node
            max_depth: Maximum depth (default=2)

        Returns:
            List of descendant nodes
        """
        if node_id not in self.nodes:
            return []

        descendants = []
        visited = set()
        queue = deque([(node_id, 0)])

        while queue:
            current_id, depth = queue.popleft()

            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id != node_id:  # Don't include self
                current_node = self.nodes.get(current_id)
                if current_node:
                    descendants.append(current_node)

            # Stop if max_depth reached
            if depth >= max_depth:
                continue

            # Add children to queue
            current_node = self.nodes.get(current_id)
            if current_node:
                for child_id in current_node.child_actualizations:
                    if child_id not in visited:
                        queue.append((child_id, depth + 1))

        return descendants

    def get_siblings(self, node_id: str) -> List[KronosNode]:
        """
        Get alternative actualizations from same parent potentials.

        Returns concepts that are "cousins" in the tree.

        Args:
            node_id: Starting node

        Returns:
            List of sibling nodes
        """
        if node_id not in self.nodes:
            return []

        node = self.nodes[node_id]
        siblings = []

        for sibling_id in node.sibling_nodes:
            sibling = self.nodes.get(sibling_id)
            if sibling:
                siblings.append(sibling)

        return siblings

    def get_derivation_path(self, node_id: str) -> List[KronosNode]:
        """
        Get full path from root potential to this node.

        Example: [physics] → [quantum_mechanics] → [quantum_foundations]
                 → [EPR_paradox] → [bells_theorem]

        Args:
            node_id: Target node

        Returns:
            List of nodes from root to target
        """
        if node_id not in self.nodes:
            return []

        node = self.nodes[node_id]
        path = []

        # Build path from derivation_path IDs
        for path_node_id in node.derivation_path:
            path_node = self.nodes.get(path_node_id)
            if path_node:
                path.append(path_node)

        # Add target node
        path.append(node)

        return path

    def get_neighbors_within_radius(
        self, node_id: str, radius: int
    ) -> List[KronosNode]:
        """
        Get all nodes within radius hops.

        Args:
            node_id: Center node
            radius: Maximum distance

        Returns:
            List of nodes within radius
        """
        if node_id not in self.nodes:
            return []

        neighbors = []
        visited = set()
        queue = deque([(node_id, 0)])

        while queue:
            current_id, distance = queue.popleft()

            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id != node_id:  # Don't include self
                current_node = self.nodes.get(current_id)
                if current_node:
                    neighbors.append(current_node)

            if distance >= radius:
                continue

            # Add all connected nodes
            current_node = self.nodes.get(current_id)
            if current_node:
                # Parents
                for parent_id in current_node.parent_potentials:
                    if parent_id not in visited:
                        queue.append((parent_id, distance + 1))

                # Children
                for child_id in current_node.child_actualizations:
                    if child_id not in visited:
                        queue.append((child_id, distance + 1))

                # Siblings
                for sibling_id in current_node.sibling_nodes:
                    if sibling_id not in visited:
                        queue.append((sibling_id, distance + 1))

        return neighbors

    # === PAC Conservation ===

    def verify_conservation(self, node_id: str, tolerance: float = 0.01) -> bool:
        """
        Verify PAC conservation: Parent ≈ Σ weighted_children

        Checks both:
        - Conceptual: parent_meaning = Σ (child_meaning * confluence_weight)
        - Storage: parent_embedding = parent.Δ + Σ child.Δ

        Args:
            node_id: Node to verify
            tolerance: Acceptable error (default=1%)

        Returns:
            True if conservation holds within tolerance
        """
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]

        # Root nodes trivially satisfy conservation
        if node.is_root:
            return True

        # Get primary parent
        if not node.primary_parent:
            return False

        parent = self.nodes.get(node.primary_parent)
        if not parent:
            return False

        # Check embedding conservation (if embeddings present)
        if (
            node.delta_embedding is not None
            and parent.delta_embedding is not None
        ):
            # Reconstruct should equal parent
            reconstructed = self._reconstruct_embedding(node)
            if parent.delta_embedding is not None:
                parent_full = (
                    parent.delta_embedding
                    if parent.is_root
                    else self._reconstruct_embedding(parent)
                )

                if parent_full is not None and reconstructed is not None:
                    error = np.linalg.norm(reconstructed - parent_full) / np.linalg.norm(
                        parent_full
                    )
                    if error > tolerance:
                        logger.warning(
                            f"PAC conservation violated for {node_id}: "
                            f"error={error:.4f} > tolerance={tolerance}"
                        )
                        return False

        return True

    def _reconstruct_embedding(self, node: KronosNode) -> Optional[np.ndarray]:
        """
        Reconstruct full embedding by summing deltas to root.

        Args:
            node: Node to reconstruct

        Returns:
            Full embedding or None if reconstruction fails
        """
        if node.is_root:
            return node.delta_embedding

        if not node.primary_parent:
            return node.delta_embedding

        parent = self.nodes.get(node.primary_parent)
        if not parent:
            return node.delta_embedding

        # Recursive reconstruction
        parent_embedding = self._reconstruct_embedding(parent)
        if parent_embedding is None or node.delta_embedding is None:
            return None

        return parent_embedding + node.delta_embedding

    def recompute_confluence(self, node_id: str) -> Dict[str, float]:
        """
        Recompute confluence pattern based on current children.

        Used when tree structure changes.

        Args:
            node_id: Node to recompute

        Returns:
            New confluence pattern
        """
        # TODO: Implement using embedding similarity or other heuristic
        # For now, return existing pattern
        node = self.nodes.get(node_id)
        if node:
            return node.confluence_pattern.copy()
        return {}

    # === Confidence Computation ===

    def compute_geometric_confidence(self, node_id: str) -> GeometricConfidence:
        """
        Compute confidence from local graph topology.

        Analyzes:
        - Density of neighborhood
        - Symmetry of branches
        - Distance to well-documented nodes
        - Anomaly detection (orphans, bottlenecks, gaps)

        Args:
            node_id: Node to analyze

        Returns:
            GeometricConfidence object
        """
        if node_id not in self.nodes:
            return GeometricConfidence()  # Zero confidence

        node = self.nodes[node_id]

        # Compute metrics
        local_density = compute_local_density(node_id, self, radius=2)
        branch_symmetry = compute_branch_symmetry(node_id, self)
        traversal_distance = compute_traversal_distance(
            node_id, self, well_documented_threshold=5
        )
        documentation_depth = node.documentation_depth

        # Anomaly detection
        orphan_score = self._compute_orphan_score(node)
        confluence_bottleneck = self._has_confluence_bottleneck(node)
        missing_children = self.find_knowledge_gaps(node_id)

        # Create confidence object
        confidence = GeometricConfidence(
            local_density=local_density,
            branch_symmetry=branch_symmetry,
            traversal_distance=traversal_distance,
            documentation_depth=documentation_depth,
            orphan_score=orphan_score,
            confluence_bottleneck=confluence_bottleneck,
            missing_expected_children=missing_children,
        )

        # Compute final score
        confidence.compute()

        return confidence

    def _compute_orphan_score(self, node: KronosNode) -> float:
        """
        Compute orphan score (0-1).

        High score = node lacks clear lineage.

        Args:
            node: Node to analyze

        Returns:
            Orphan score
        """
        score = 0.0

        # No parents (except roots)
        if not node.is_root and not node.parent_potentials:
            score += 0.5

        # Weak confluence pattern
        if node.confluence_pattern:
            max_weight = max(node.confluence_pattern.values())
            if max_weight < 0.3:  # All parents weak
                score += 0.3

        # Not in anyone else's child list
        referenced_count = sum(
            1
            for other_node in self.nodes.values()
            if node.id in other_node.child_actualizations
        )
        if referenced_count == 0 and not node.is_root:
            score += 0.2

        return min(1.0, score)

    def _has_confluence_bottleneck(self, node: KronosNode) -> bool:
        """
        Check if one parent dominates confluence (>80%).

        Args:
            node: Node to check

        Returns:
            True if bottleneck detected
        """
        if not node.confluence_pattern or len(node.confluence_pattern) <= 1:
            return False

        max_weight = max(node.confluence_pattern.values())
        return max_weight > 0.8

    def find_knowledge_gaps(self, node_id: str) -> List[str]:
        """
        Detect missing expected children.

        Based on:
        - Sibling patterns (what children do they have?)
        - Confluence symmetry (unbalanced parent contributions?)

        Args:
            node_id: Node to analyze

        Returns:
            List of expected but missing child concept names
        """
        if node_id not in self.nodes:
            return []

        node = self.nodes[node_id]
        gaps = []

        # Get siblings
        siblings = self.get_siblings(node_id)
        if not siblings:
            return gaps

        # Collect children that siblings have
        sibling_children: Set[str] = set()
        for sibling in siblings:
            sibling_children.update(sibling.child_actualizations)

        # Get this node's children
        node_children = set(node.child_actualizations)

        # Find gaps (concepts siblings have but this node doesn't)
        potential_gaps = sibling_children - node_children

        # Only report if multiple siblings have this child
        for gap_id in potential_gaps:
            # Count how many siblings have this child
            count = sum(
                1 for sibling in siblings if gap_id in sibling.child_actualizations
            )

            # If >50% of siblings have it, it's a gap
            if count > len(siblings) / 2:
                gap_node = self.nodes.get(gap_id)
                if gap_node:
                    gaps.append(gap_node.name)

        return gaps[:5]  # Limit to top 5 gaps

    # === Helpers ===

    def _update_siblings(self, node_id: str) -> None:
        """Update sibling relationships for a node."""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # Find nodes with same parents
        for other_id, other_node in self.nodes.items():
            if other_id == node_id:
                continue

            # Check for shared parents
            shared_parents = set(node.parent_potentials) & set(
                other_node.parent_potentials
            )

            if shared_parents:
                # Add as siblings
                node.add_sibling(other_id)
                other_node.add_sibling(node_id)

    def _rebuild_edge_index(self) -> None:
        """Rebuild edge index from edges list."""
        self._edge_index = {}
        for edge in self.edges:
            key = (edge.source_id, edge.target_id, edge.relationship_type)
            self._edge_index[key] = edge

    # === Graph Statistics ===

    def get_stats(self) -> dict:
        """Get graph statistics."""
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "root_count": len(self.roots),
            "max_depth": max(
                (node.actualization_depth for node in self.nodes.values()), default=0
            ),
            "avg_children": (
                sum(len(node.child_actualizations) for node in self.nodes.values())
                / len(self.nodes)
                if self.nodes
                else 0
            ),
            "avg_documentation": (
                sum(node.documentation_depth for node in self.nodes.values())
                / len(self.nodes)
                if self.nodes
                else 0
            ),
        }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"KronosGraph(nodes={stats['node_count']}, "
            f"edges={stats['edge_count']}, "
            f"roots={stats['root_count']}, "
            f"max_depth={stats['max_depth']})"
        )
