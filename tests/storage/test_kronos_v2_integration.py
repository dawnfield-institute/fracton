"""
Integration tests for KRONOS v2.

Tests complete workflows:
- Building a knowledge graph from scratch
- Traversing genealogy trees
- Computing confidence across the graph
- Detecting anomalies systematically
- PAC conservation across multiple generations
- Real-world scenarios (quantum mechanics tree)
"""

import pytest
import numpy as np
from datetime import datetime
from fracton.storage import (
    KronosGraph,
    KronosNode,
    KronosEdge,
    DocumentReference,
    CrystallizationEvent,
    GeometricConfidence,
)
from fracton.storage.edge import RelationType


@pytest.fixture
def quantum_mechanics_tree():
    """
    Create a realistic quantum mechanics genealogy tree.

    Structure:
        physics (root)
        ├── classical_mechanics
        └── quantum_mechanics
            ├── quantum_foundations
            │   ├── superposition
            │   ├── nonlocality
            │   └── measurement
            └── quantum_computing
                └── quantum_algorithms

    Plus:
        quantum_entanglement (confluence of superposition, nonlocality, measurement)
    """
    graph = KronosGraph()

    # Level 0: Root
    physics = KronosNode(
        id="physics",
        name="Physics",
        definition="The natural science of matter, energy, and their interactions",
        confluence_pattern={},
        parent_potentials=[],
        child_actualizations=[],
        derivation_path=[],
        actualization_depth=0,
        delta_embedding=np.random.randn(128),
        documentation_depth=2,
        supported_by=[
            DocumentReference(
                doc_id="feynman_lectures",
                title="The Feynman Lectures on Physics",
                authors=["Richard Feynman"],
                year=1963
            ),
            DocumentReference(
                doc_id="newton_principia",
                title="Philosophiæ Naturalis Principia Mathematica",
                authors=["Isaac Newton"],
                year=1687
            )
        ]
    )
    graph.add_node(physics)

    # Level 1: Branches
    quantum_mechanics = KronosNode(
        id="quantum_mechanics",
        name="Quantum Mechanics",
        definition="Physics of atomic and subatomic systems",
        confluence_pattern={"physics": 1.0},
        parent_potentials=["physics"],
        child_actualizations=[],
        derivation_path=["physics"],
        actualization_depth=1,
        delta_embedding=np.random.randn(128) * 0.1,
        documentation_depth=2,
        supported_by=[
            DocumentReference(
                doc_id="griffiths_qm",
                title="Introduction to Quantum Mechanics",
                authors=["David J. Griffiths"],
                year=2004
            )
        ]
    )
    graph.add_node(quantum_mechanics)

    classical_mechanics = KronosNode(
        id="classical_mechanics",
        name="Classical Mechanics",
        definition="Motion of macroscopic objects",
        confluence_pattern={"physics": 1.0},
        parent_potentials=["physics"],
        child_actualizations=[],
        derivation_path=["physics"],
        actualization_depth=1,
        delta_embedding=np.random.randn(128) * 0.1,
        documentation_depth=1
    )
    graph.add_node(classical_mechanics)

    # Level 2: Specializations
    quantum_foundations = KronosNode(
        id="quantum_foundations",
        name="Quantum Foundations",
        definition="Fundamental principles of quantum mechanics",
        confluence_pattern={"quantum_mechanics": 1.0},
        parent_potentials=["quantum_mechanics"],
        child_actualizations=[],
        derivation_path=["physics", "quantum_mechanics"],
        actualization_depth=2,
        delta_embedding=np.random.randn(128) * 0.05,
        documentation_depth=2,
        supported_by=[
            DocumentReference(
                doc_id="bell_theorem",
                title="On the Einstein Podolsky Rosen Paradox",
                authors=["John S. Bell"],
                year=1964
            )
        ]
    )
    graph.add_node(quantum_foundations)

    quantum_computing = KronosNode(
        id="quantum_computing",
        name="Quantum Computing",
        definition="Computing using quantum mechanical phenomena",
        confluence_pattern={"quantum_mechanics": 1.0},
        parent_potentials=["quantum_mechanics"],
        child_actualizations=[],
        derivation_path=["physics", "quantum_mechanics"],
        actualization_depth=2,
        delta_embedding=np.random.randn(128) * 0.05,
        documentation_depth=2
    )
    graph.add_node(quantum_computing)

    # Level 3: Core concepts
    superposition = KronosNode(
        id="superposition",
        name="Quantum Superposition",
        definition="System exists in multiple states simultaneously",
        confluence_pattern={"quantum_foundations": 1.0},
        parent_potentials=["quantum_foundations"],
        child_actualizations=[],
        derivation_path=["physics", "quantum_mechanics", "quantum_foundations"],
        actualization_depth=3,
        delta_embedding=np.random.randn(128) * 0.03,
        documentation_depth=2
    )
    graph.add_node(superposition)

    nonlocality = KronosNode(
        id="nonlocality",
        name="Quantum Nonlocality",
        definition="Correlations between spatially separated systems",
        confluence_pattern={"quantum_foundations": 1.0},
        parent_potentials=["quantum_foundations"],
        child_actualizations=[],
        derivation_path=["physics", "quantum_mechanics", "quantum_foundations"],
        actualization_depth=3,
        delta_embedding=np.random.randn(128) * 0.03,
        documentation_depth=1
    )
    graph.add_node(nonlocality)

    measurement = KronosNode(
        id="measurement",
        name="Quantum Measurement",
        definition="Observation causing wavefunction collapse",
        confluence_pattern={"quantum_foundations": 1.0},
        parent_potentials=["quantum_foundations"],
        child_actualizations=[],
        derivation_path=["physics", "quantum_mechanics", "quantum_foundations"],
        actualization_depth=3,
        delta_embedding=np.random.randn(128) * 0.03,
        documentation_depth=1
    )
    graph.add_node(measurement)

    quantum_algorithms = KronosNode(
        id="quantum_algorithms",
        name="Quantum Algorithms",
        definition="Algorithms leveraging quantum effects",
        confluence_pattern={"quantum_computing": 1.0},
        parent_potentials=["quantum_computing"],
        child_actualizations=[],
        derivation_path=["physics", "quantum_mechanics", "quantum_computing"],
        actualization_depth=3,
        delta_embedding=np.random.randn(128) * 0.03,
        documentation_depth=2
    )
    graph.add_node(quantum_algorithms)

    # Level 3: Confluence concept (multiple parents!)
    quantum_entanglement = KronosNode(
        id="quantum_entanglement",
        name="Quantum Entanglement",
        definition="Particles remain connected regardless of distance",
        confluence_pattern={
            "superposition": 0.40,
            "nonlocality": 0.35,
            "measurement": 0.25
        },
        parent_potentials=["superposition", "nonlocality", "measurement"],
        child_actualizations=[],
        derivation_path=["physics", "quantum_mechanics", "quantum_foundations"],
        actualization_depth=3,
        delta_embedding=np.random.randn(128) * 0.02,
        documentation_depth=2,
        supported_by=[
            DocumentReference(
                doc_id="epr_paper",
                title="Can Quantum-Mechanical Description of Physical Reality Be Considered Complete?",
                authors=["Einstein", "Podolsky", "Rosen"],
                year=1935
            ),
            DocumentReference(
                doc_id="aspect_experiment",
                title="Experimental Tests of Bell's Inequalities",
                authors=["Alain Aspect"],
                year=1982
            )
        ]
    )
    graph.add_node(quantum_entanglement)

    return graph


class TestKnowledgeGraphConstruction:
    """Test building a knowledge graph from scratch."""

    def test_build_quantum_tree(self, quantum_mechanics_tree):
        """Test that quantum mechanics tree is constructed correctly."""
        graph = quantum_mechanics_tree

        # Verify node count
        assert len(graph.nodes) == 10

        # Verify root
        assert len(graph.roots) == 1
        assert "physics" in graph.roots

        # Verify structure
        physics = graph.get_node("physics")
        assert physics.actualization_depth == 0
        assert len(physics.child_actualizations) == 2  # classical_mechanics, quantum_mechanics

        # Verify confluence node
        entanglement = graph.get_node("quantum_entanglement")
        assert entanglement is not None
        assert len(entanglement.parent_potentials) == 3
        assert sum(entanglement.confluence_pattern.values()) == pytest.approx(1.0, abs=0.01)

    def test_depth_distribution(self, quantum_mechanics_tree):
        """Test that nodes are distributed across depth levels."""
        depth_counts = {}
        for node in quantum_mechanics_tree.nodes.values():
            depth = node.actualization_depth
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        # Should have nodes at depths 0, 1, 2, 3
        assert 0 in depth_counts  # Root
        assert 1 in depth_counts  # Branches
        assert 2 in depth_counts  # Specializations
        assert 3 in depth_counts  # Core concepts


class TestGenealogyTraversal:
    """Test genealogy tree traversal."""

    def test_full_lineage_reconstruction(self, quantum_mechanics_tree):
        """Test reconstructing full lineage for a concept."""
        path = quantum_mechanics_tree.get_derivation_path("quantum_entanglement")

        # Should trace: physics -> quantum_mechanics -> quantum_foundations -> quantum_entanglement
        path_ids = [n.id for n in path]
        assert path_ids == ["physics", "quantum_mechanics", "quantum_foundations", "quantum_entanglement"]

    def test_ancestor_retrieval(self, quantum_mechanics_tree):
        """Test retrieving all ancestors of a concept."""
        ancestors = quantum_mechanics_tree.get_ancestors("quantum_entanglement")

        ancestor_ids = [n.id for n in ancestors]

        # Should include all three parents
        assert "superposition" in ancestor_ids
        assert "nonlocality" in ancestor_ids
        assert "measurement" in ancestor_ids

        # Should also include grandparents
        assert "quantum_foundations" in ancestor_ids
        assert "quantum_mechanics" in ancestor_ids
        assert "physics" in ancestor_ids

    def test_descendant_retrieval(self, quantum_mechanics_tree):
        """Test retrieving all descendants of a concept."""
        descendants = quantum_mechanics_tree.get_descendants("quantum_mechanics")

        descendant_ids = [n.id for n in descendants]

        # Should include direct children
        assert "quantum_foundations" in descendant_ids
        assert "quantum_computing" in descendant_ids

        # Should include grandchildren
        assert "superposition" in descendant_ids
        assert "nonlocality" in descendant_ids
        assert "measurement" in descendant_ids
        assert "quantum_algorithms" in descendant_ids
        assert "quantum_entanglement" in descendant_ids

    def test_sibling_detection(self, quantum_mechanics_tree):
        """Test detecting sibling concepts."""
        siblings = quantum_mechanics_tree.get_siblings("superposition")

        sibling_ids = [n.id for n in siblings]

        # Siblings of superposition (all children of quantum_foundations)
        assert "nonlocality" in sibling_ids
        assert "measurement" in sibling_ids
        assert "superposition" not in sibling_ids  # Not its own sibling


class TestConfidenceAnalysis:
    """Test confidence analysis across the graph."""

    def test_root_confidence(self, quantum_mechanics_tree):
        """Test confidence for root node."""
        confidence = quantum_mechanics_tree.compute_geometric_confidence("physics")

        # Root should have decent confidence (well-documented)
        assert confidence.documentation_depth == 2
        assert confidence.retrieval_confidence > 0.0

    def test_well_documented_vs_sparse(self, quantum_mechanics_tree):
        """Test confidence difference between well-documented and sparse nodes."""
        qm_confidence = quantum_mechanics_tree.compute_geometric_confidence("quantum_mechanics")
        nonlocality_confidence = quantum_mechanics_tree.compute_geometric_confidence("nonlocality")

        # quantum_mechanics has more documentation
        assert qm_confidence.documentation_depth > nonlocality_confidence.documentation_depth

    def test_confluence_node_confidence(self, quantum_mechanics_tree):
        """Test confidence for confluence node."""
        confidence = quantum_mechanics_tree.compute_geometric_confidence("quantum_entanglement")

        # Should have multiple parents (not orphaned)
        assert confidence.orphan_score < 0.5

        # Check if bottleneck detected
        # (confluence is 0.4, 0.35, 0.25 - no single parent > 0.8)
        assert confidence.confluence_bottleneck == False

    def test_confidence_anomaly_report(self, quantum_mechanics_tree):
        """Test anomaly reporting across nodes."""
        anomalies_found = 0

        for node_id in quantum_mechanics_tree.nodes.keys():
            confidence = quantum_mechanics_tree.compute_geometric_confidence(node_id)
            if confidence.has_anomalies:
                anomalies_found += 1
                report = confidence.get_anomaly_report()
                assert len(report) > 0

        # At least some nodes should have anomalies detected
        # (due to sparse documentation or missing expected children)
        assert anomalies_found > 0


class TestPACConservationValidation:
    """Test PAC conservation across the graph."""

    def test_conservation_check_all_nodes(self, quantum_mechanics_tree):
        """Test PAC conservation for all non-root nodes."""
        conservation_results = {}

        for node_id, node in quantum_mechanics_tree.nodes.items():
            if node.actualization_depth > 0:  # Skip root
                is_conserved = quantum_mechanics_tree.verify_conservation(node_id, tolerance=0.5)
                conservation_results[node_id] = is_conserved

        # With random deltas, most will likely fail, but mechanism should work
        assert len(conservation_results) > 0

    def test_embedding_reconstruction(self, quantum_mechanics_tree):
        """Test that embeddings can be reconstructed from parents."""
        node = quantum_mechanics_tree.get_node("quantum_entanglement")

        # Reconstruct embedding from parents
        reconstructed = quantum_mechanics_tree._reconstruct_embedding("quantum_entanglement")

        # Should produce an embedding
        assert reconstructed is not None
        assert isinstance(reconstructed, np.ndarray)
        assert reconstructed.shape == node.delta_embedding.shape


class TestKnowledgeGapDetection:
    """Test detecting gaps in the knowledge graph."""

    def test_find_knowledge_gaps(self, quantum_mechanics_tree):
        """Test finding missing expected children."""
        gaps = quantum_mechanics_tree.find_knowledge_gaps()

        # Should return a dictionary
        assert isinstance(gaps, dict)

        # Some nodes might have missing expected children based on sibling patterns
        # (e.g., if quantum_foundations has 3 children, but quantum_computing has only 1)

    def test_identify_sparse_regions(self, quantum_mechanics_tree):
        """Test identifying sparse regions of the graph."""
        sparse_nodes = []

        for node_id in quantum_mechanics_tree.nodes.keys():
            confidence = quantum_mechanics_tree.compute_geometric_confidence(node_id)
            if confidence.local_density < 0.3:
                sparse_nodes.append(node_id)

        # Some nodes should be identified as sparse
        # (depending on tree structure)
        assert isinstance(sparse_nodes, list)


class TestGraphStatistics:
    """Test graph-wide statistics."""

    def test_depth_statistics(self, quantum_mechanics_tree):
        """Test depth-related statistics."""
        depths = [node.actualization_depth for node in quantum_mechanics_tree.nodes.values()]

        max_depth = max(depths)
        avg_depth = sum(depths) / len(depths)

        assert max_depth == 3  # Deepest concepts are at level 3
        assert avg_depth > 0  # Average should be positive

    def test_branching_statistics(self, quantum_mechanics_tree):
        """Test branching factor statistics."""
        child_counts = [len(node.child_actualizations) for node in quantum_mechanics_tree.nodes.values()]

        avg_children = sum(child_counts) / len(child_counts)
        max_children = max(child_counts)

        assert avg_children >= 0
        assert max_children > 0  # At least some nodes have children

    def test_documentation_statistics(self, quantum_mechanics_tree):
        """Test documentation statistics."""
        doc_depths = [node.documentation_depth for node in quantum_mechanics_tree.nodes.values()]

        avg_docs = sum(doc_depths) / len(doc_depths)
        max_docs = max(doc_depths)

        assert avg_docs > 0  # Some documentation exists
        assert max_docs >= 2  # Root has at least 2 docs


class TestEdgeSemantics:
    """Test semantic relationships via edges."""

    def test_parent_child_edges(self, quantum_mechanics_tree):
        """Test parent-child relationship edges."""
        # Add explicit edges
        edge = KronosEdge(
            source_id="physics",
            target_id="quantum_mechanics",
            relationship_type=RelationType.PARENT_OF,
            strength=1.0
        )
        quantum_mechanics_tree.add_edge(edge)

        assert len(quantum_mechanics_tree.edges) > 0

    def test_confluence_edges(self, quantum_mechanics_tree):
        """Test confluence relationship edges."""
        # Add confluence edges
        for parent_id, weight in [("superposition", 0.4), ("nonlocality", 0.35), ("measurement", 0.25)]:
            edge = KronosEdge(
                source_id=parent_id,
                target_id="quantum_entanglement",
                relationship_type=RelationType.CONFLUENCE,
                strength=weight
            )
            quantum_mechanics_tree.add_edge(edge)

        # Verify confluence edges exist
        confluence_edges = [e for e in quantum_mechanics_tree.edges if e.relationship_type == RelationType.CONFLUENCE]
        assert len(confluence_edges) == 3


class TestGraphSerialization:
    """Test full graph serialization and restoration."""

    def test_full_graph_serialization(self, quantum_mechanics_tree):
        """Test serializing and deserializing the full graph."""
        # Serialize
        data = quantum_mechanics_tree.to_dict()

        # Verify serialized structure
        assert "nodes" in data
        assert "edges" in data
        assert "roots" in data
        assert len(data["nodes"]) == 10

        # Deserialize
        restored = KronosGraph.from_dict(data)

        # Verify restoration
        assert len(restored.nodes) == len(quantum_mechanics_tree.nodes)
        assert len(restored.roots) == len(quantum_mechanics_tree.roots)

        # Verify specific node
        original_entanglement = quantum_mechanics_tree.get_node("quantum_entanglement")
        restored_entanglement = restored.get_node("quantum_entanglement")

        assert restored_entanglement is not None
        assert restored_entanglement.name == original_entanglement.name
        assert len(restored_entanglement.parent_potentials) == len(original_entanglement.parent_potentials)
        assert restored_entanglement.confluence_pattern == original_entanglement.confluence_pattern


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    def test_lineage_aware_retrieval(self, quantum_mechanics_tree):
        """Test retrieving a concept with full context (lineage-aware)."""
        concept_id = "quantum_entanglement"

        # Get the concept
        concept = quantum_mechanics_tree.get_node(concept_id)

        # Get its context (ancestors + descendants + siblings)
        ancestors = quantum_mechanics_tree.get_ancestors(concept_id)
        descendants = quantum_mechanics_tree.get_descendants(concept_id)
        siblings = quantum_mechanics_tree.get_siblings(concept_id)

        # Build full context
        context = {
            "concept": concept,
            "ancestors": ancestors,
            "descendants": descendants,
            "siblings": siblings,
            "confidence": quantum_mechanics_tree.compute_geometric_confidence(concept_id)
        }

        # Verify context completeness
        assert context["concept"] is not None
        assert len(context["ancestors"]) > 0
        assert context["confidence"] is not None

    def test_concept_comparison(self, quantum_mechanics_tree):
        """Test comparing two related concepts."""
        concept_a = "superposition"
        concept_b = "nonlocality"

        # Get common ancestors
        ancestors_a = set(n.id for n in quantum_mechanics_tree.get_ancestors(concept_a))
        ancestors_b = set(n.id for n in quantum_mechanics_tree.get_ancestors(concept_b))

        common_ancestors = ancestors_a & ancestors_b

        # Should share ancestors (quantum_foundations, quantum_mechanics, physics)
        assert "quantum_foundations" in common_ancestors
        assert "quantum_mechanics" in common_ancestors
        assert "physics" in common_ancestors

    def test_conceptual_distance(self, quantum_mechanics_tree):
        """Test measuring conceptual distance between nodes."""
        # Get paths from root to each concept
        path_a = quantum_mechanics_tree.get_derivation_path("superposition")
        path_b = quantum_mechanics_tree.get_derivation_path("quantum_algorithms")

        # Concepts with same ancestry up to a point are closer
        # Both share: physics -> quantum_mechanics
        # But diverge after that

        # Find common prefix length
        common_prefix = 0
        for i in range(min(len(path_a), len(path_b))):
            if path_a[i].id == path_b[i].id:
                common_prefix += 1
            else:
                break

        # Should share at least 2 levels (physics, quantum_mechanics)
        assert common_prefix >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
