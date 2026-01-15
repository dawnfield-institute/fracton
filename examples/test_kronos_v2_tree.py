"""
KRONOS v2 - Quantum Mechanics Genealogy Tree Test

Creates a conceptual genealogy tree of quantum mechanics concepts
to validate the KRONOS v2 architecture.

Tree Structure:
    physics (root)
    ├─ classical_mechanics
    │  └─ newtonian_mechanics
    └─ quantum_mechanics
       ├─ quantum_foundations
       │  ├─ wave_particle_duality
       │  ├─ uncertainty_principle
       │  └─ superposition
       │     └─ quantum_entanglement (40% superposition + 35% nonlocality + 25% measurement)
       ├─ quantum_field_theory
       └─ quantum_computing
          └─ quantum_algorithms
"""

import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add fracton to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fracton.storage.node import KronosNode, DocumentReference, CrystallizationEvent
from fracton.storage.edge import KronosEdge, RelationType
from fracton.storage.graph import KronosGraph


def create_quantum_tree() -> KronosGraph:
    """Create quantum mechanics genealogy tree."""
    print("Creating quantum mechanics genealogy tree...")
    graph = KronosGraph()

    # === Level 0: Root ===
    print("\n=== Level 0: Root Potential ===")

    physics = KronosNode(
        id="physics",
        name="Physics",
        definition="The natural science that studies matter, energy, and their interactions",
        confluence_pattern={},  # Root has no parents
        parent_potentials=[],
        derivation_path=[],
        actualization_depth=0,
        supported_by=[
            DocumentReference(
                doc_id="physics_textbook",
                title="Fundamentals of Physics",
                authors=["Halliday, D.", "Resnick, R.", "Walker, J."],
                year=2013,
                uri="https://example.com/physics",
                excerpt="Physics is the fundamental science..."
            ),
            DocumentReference(
                doc_id="feynman_lectures",
                title="The Feynman Lectures on Physics",
                authors=["Feynman, R.", "Leighton, R.", "Sands, M."],
                year=1963,
                uri="https://example.com/feynman",
                excerpt="The principle of science is to understand nature..."
            ),
        ],
        delta_embedding=np.random.randn(384) * 0.1,  # Root embedding (normalized)
    )
    graph.add_node(physics)
    print(f"[OK] Added: {physics.name} (depth={physics.actualization_depth})")

    # === Level 1: Major Branches ===
    print("\n=== Level 1: Major Branches ===")

    classical_mechanics = KronosNode(
        id="classical_mechanics",
        name="Classical Mechanics",
        definition="Branch of physics dealing with motion of macroscopic objects",
        confluence_pattern={"physics": 1.0},
        parent_potentials=["physics"],
        derivation_path=["physics"],
        actualization_depth=1,
        supported_by=[
            DocumentReference(
                doc_id="classical_mech_goldstein",
                title="Classical Mechanics",
                authors=["Goldstein, H."],
                year=1980,
                uri="https://example.com/classical",
                excerpt="Classical mechanics is the study of motion..."
            )
        ],
        delta_embedding=np.random.randn(384) * 0.05,  # Small delta from physics
    )
    graph.add_node(classical_mechanics)
    print(f"[OK] Added: {classical_mechanics.name} (depth={classical_mechanics.actualization_depth})")

    quantum_mechanics = KronosNode(
        id="quantum_mechanics",
        name="Quantum Mechanics",
        definition="Branch of physics dealing with atomic and subatomic systems",
        confluence_pattern={"physics": 1.0},
        parent_potentials=["physics"],
        derivation_path=["physics"],
        actualization_depth=1,
        supported_by=[
            DocumentReference(
                doc_id="qm_dirac",
                title="The Principles of Quantum Mechanics",
                authors=["Dirac, P.A.M."],
                year=1930,
                uri="https://example.com/qm_dirac",
                excerpt="Quantum mechanics provides a description of atomic processes..."
            ),
            DocumentReference(
                doc_id="qm_griffiths",
                title="Introduction to Quantum Mechanics",
                authors=["Griffiths, D.J."],
                year=2004,
                uri="https://example.com/qm_griffiths",
                excerpt="Quantum mechanics is the physics of the very small..."
            ),
        ],
        delta_embedding=np.random.randn(384) * 0.05,
    )
    graph.add_node(quantum_mechanics)
    print(f"[OK] Added: {quantum_mechanics.name} (depth={quantum_mechanics.actualization_depth})")

    # === Level 2: Specializations ===
    print("\n=== Level 2: Specializations ===")

    newtonian_mechanics = KronosNode(
        id="newtonian_mechanics",
        name="Newtonian Mechanics",
        definition="Classical mechanics based on Newton's three laws",
        confluence_pattern={"classical_mechanics": 1.0},
        parent_potentials=["classical_mechanics"],
        derivation_path=["physics", "classical_mechanics"],
        actualization_depth=2,
        supported_by=[
            DocumentReference(
                doc_id="newton_principia",
                title="Philosophiæ Naturalis Principia Mathematica",
                authors=["Newton, I."],
                year=1687,
                uri="https://example.com/principia",
                excerpt="The laws of motion form the foundation..."
            )
        ],
        delta_embedding=np.random.randn(384) * 0.03,
    )
    graph.add_node(newtonian_mechanics)
    print(f"[OK] Added: {newtonian_mechanics.name} (depth={newtonian_mechanics.actualization_depth})")

    quantum_foundations = KronosNode(
        id="quantum_foundations",
        name="Quantum Foundations",
        definition="Fundamental principles and interpretations of quantum mechanics",
        confluence_pattern={"quantum_mechanics": 1.0},
        parent_potentials=["quantum_mechanics"],
        derivation_path=["physics", "quantum_mechanics"],
        actualization_depth=2,
        supported_by=[
            DocumentReference(
                doc_id="qm_foundations_ballentine",
                title="Quantum Mechanics: A Modern Development",
                authors=["Ballentine, L.E."],
                year=1998,
                uri="https://example.com/foundations",
                excerpt="The foundations of quantum mechanics remain debated..."
            ),
            DocumentReference(
                doc_id="bell_theorem",
                title="On the Einstein Podolsky Rosen Paradox",
                authors=["Bell, J.S."],
                year=1964,
                uri="https://example.com/bell",
                excerpt="Bell's theorem shows that no local hidden variable theory..."
            ),
        ],
        delta_embedding=np.random.randn(384) * 0.04,
    )
    graph.add_node(quantum_foundations)
    print(f"[OK] Added: {quantum_foundations.name} (depth={quantum_foundations.actualization_depth})")

    quantum_field_theory = KronosNode(
        id="quantum_field_theory",
        name="Quantum Field Theory",
        definition="Quantum mechanical description of fields",
        confluence_pattern={"quantum_mechanics": 1.0},
        parent_potentials=["quantum_mechanics"],
        derivation_path=["physics", "quantum_mechanics"],
        actualization_depth=2,
        supported_by=[
            DocumentReference(
                doc_id="qft_peskin",
                title="An Introduction to Quantum Field Theory",
                authors=["Peskin, M.E.", "Schroeder, D.V."],
                year=1995,
                uri="https://example.com/qft",
                excerpt="QFT combines quantum mechanics with special relativity..."
            )
        ],
        delta_embedding=np.random.randn(384) * 0.04,
    )
    graph.add_node(quantum_field_theory)
    print(f"[OK] Added: {quantum_field_theory.name} (depth={quantum_field_theory.actualization_depth})")

    quantum_computing = KronosNode(
        id="quantum_computing",
        name="Quantum Computing",
        definition="Computing using quantum mechanical phenomena",
        confluence_pattern={"quantum_mechanics": 1.0},
        parent_potentials=["quantum_mechanics"],
        derivation_path=["physics", "quantum_mechanics"],
        actualization_depth=2,
        supported_by=[
            DocumentReference(
                doc_id="qc_nielsen",
                title="Quantum Computation and Quantum Information",
                authors=["Nielsen, M.A.", "Chuang, I.L."],
                year=2000,
                uri="https://example.com/qc",
                excerpt="Quantum computers harness superposition and entanglement..."
            ),
            DocumentReference(
                doc_id="qc_preskill",
                title="Quantum Computing in the NISQ era",
                authors=["Preskill, J."],
                year=2018,
                uri="https://example.com/nisq",
                excerpt="We are in the Noisy Intermediate-Scale Quantum era..."
            ),
        ],
        delta_embedding=np.random.randn(384) * 0.04,
    )
    graph.add_node(quantum_computing)
    print(f"[OK] Added: {quantum_computing.name} (depth={quantum_computing.actualization_depth})")

    # === Level 3: Core Concepts ===
    print("\n=== Level 3: Core Quantum Concepts ===")

    wave_particle_duality = KronosNode(
        id="wave_particle_duality",
        name="Wave-Particle Duality",
        definition="Matter and light exhibit properties of both waves and particles",
        confluence_pattern={"quantum_foundations": 1.0},
        parent_potentials=["quantum_foundations"],
        derivation_path=["physics", "quantum_mechanics", "quantum_foundations"],
        actualization_depth=3,
        supported_by=[
            DocumentReference(
                doc_id="debroglie",
                title="Recherches sur la théorie des quanta",
                authors=["de Broglie, L."],
                year=1924,
                uri="https://example.com/debroglie",
                excerpt="Matter waves possess wavelength λ = h/p..."
            )
        ],
        delta_embedding=np.random.randn(384) * 0.02,
    )
    graph.add_node(wave_particle_duality)
    print(f"[OK] Added: {wave_particle_duality.name} (depth={wave_particle_duality.actualization_depth})")

    uncertainty_principle = KronosNode(
        id="uncertainty_principle",
        name="Uncertainty Principle",
        definition="Fundamental limit on precision of simultaneous measurements",
        confluence_pattern={"quantum_foundations": 1.0},
        parent_potentials=["quantum_foundations"],
        derivation_path=["physics", "quantum_mechanics", "quantum_foundations"],
        actualization_depth=3,
        supported_by=[
            DocumentReference(
                doc_id="heisenberg",
                title="Über den anschaulichen Inhalt der quantentheoretischen Kinematik und Mechanik",
                authors=["Heisenberg, W."],
                year=1927,
                uri="https://example.com/heisenberg",
                excerpt="The more precisely position is determined, the less precisely momentum..."
            )
        ],
        delta_embedding=np.random.randn(384) * 0.02,
    )
    graph.add_node(uncertainty_principle)
    print(f"[OK] Added: {uncertainty_principle.name} (depth={uncertainty_principle.actualization_depth})")

    superposition = KronosNode(
        id="superposition",
        name="Quantum Superposition",
        definition="Quantum system exists in multiple states simultaneously until measured",
        confluence_pattern={"quantum_foundations": 1.0},
        parent_potentials=["quantum_foundations"],
        derivation_path=["physics", "quantum_mechanics", "quantum_foundations"],
        actualization_depth=3,
        supported_by=[
            DocumentReference(
                doc_id="schrodinger_cat",
                title="Die gegenwärtige Situation in der Quantenmechanik",
                authors=["Schrödinger, E."],
                year=1935,
                uri="https://example.com/cat",
                excerpt="The cat is both alive and dead until observed..."
            ),
            DocumentReference(
                doc_id="superposition_review",
                title="Quantum Superposition: An Overview",
                authors=["Zurek, W.H."],
                year=2003,
                uri="https://example.com/superposition",
                excerpt="Superposition is the cornerstone of quantum mechanics..."
            ),
        ],
        delta_embedding=np.random.randn(384) * 0.02,
    )
    graph.add_node(superposition)
    print(f"[OK] Added: {superposition.name} (depth={superposition.actualization_depth})")

    # Add nonlocality and measurement as additional parents for entanglement
    nonlocality = KronosNode(
        id="nonlocality",
        name="Quantum Nonlocality",
        definition="Correlations between spatially separated quantum systems",
        confluence_pattern={"quantum_foundations": 1.0},
        parent_potentials=["quantum_foundations"],
        derivation_path=["physics", "quantum_mechanics", "quantum_foundations"],
        actualization_depth=3,
        supported_by=[
            DocumentReference(
                doc_id="epr_paper",
                title="Can Quantum-Mechanical Description of Physical Reality Be Considered Complete?",
                authors=["Einstein, A.", "Podolsky, B.", "Rosen, N."],
                year=1935,
                uri="https://example.com/epr",
                excerpt="EPR paradox questions completeness of quantum mechanics..."
            )
        ],
        delta_embedding=np.random.randn(384) * 0.02,
    )
    graph.add_node(nonlocality)
    print(f"[OK] Added: {nonlocality.name} (depth={nonlocality.actualization_depth})")

    measurement = KronosNode(
        id="measurement",
        name="Quantum Measurement",
        definition="Process of observing a quantum system causing wavefunction collapse",
        confluence_pattern={"quantum_foundations": 1.0},
        parent_potentials=["quantum_foundations"],
        derivation_path=["physics", "quantum_mechanics", "quantum_foundations"],
        actualization_depth=3,
        supported_by=[
            DocumentReference(
                doc_id="von_neumann_measurement",
                title="Mathematical Foundations of Quantum Mechanics",
                authors=["von Neumann, J."],
                year=1932,
                uri="https://example.com/measurement",
                excerpt="Measurement causes the quantum state to collapse..."
            )
        ],
        delta_embedding=np.random.randn(384) * 0.02,
    )
    graph.add_node(measurement)
    print(f"[OK] Added: {measurement.name} (depth={measurement.actualization_depth})")

    quantum_algorithms = KronosNode(
        id="quantum_algorithms",
        name="Quantum Algorithms",
        definition="Algorithms leveraging quantum mechanical effects for computation",
        confluence_pattern={"quantum_computing": 1.0},
        parent_potentials=["quantum_computing"],
        derivation_path=["physics", "quantum_mechanics", "quantum_computing"],
        actualization_depth=3,
        supported_by=[
            DocumentReference(
                doc_id="shor_algorithm",
                title="Polynomial-Time Algorithms for Prime Factorization",
                authors=["Shor, P.W."],
                year=1994,
                uri="https://example.com/shor",
                excerpt="Shor's algorithm factors integers in polynomial time..."
            ),
            DocumentReference(
                doc_id="grover_algorithm",
                title="A Fast Quantum Mechanical Algorithm for Database Search",
                authors=["Grover, L.K."],
                year=1996,
                uri="https://example.com/grover",
                excerpt="Grover's algorithm searches unstructured databases..."
            ),
        ],
        delta_embedding=np.random.randn(384) * 0.02,
    )
    graph.add_node(quantum_algorithms)
    print(f"[OK] Added: {quantum_algorithms.name} (depth={quantum_algorithms.actualization_depth})")

    # === Level 4: Confluence Concept (Multiple Parents!) ===
    print("\n=== Level 4: Confluence Concept ===")

    quantum_entanglement = KronosNode(
        id="quantum_entanglement",
        name="Quantum Entanglement",
        definition="Quantum phenomenon where particles remain connected regardless of distance",
        confluence_pattern={
            "superposition": 0.40,  # 40% from superposition
            "nonlocality": 0.35,    # 35% from nonlocality
            "measurement": 0.25      # 25% from measurement
        },
        parent_potentials=["superposition", "nonlocality", "measurement"],
        derivation_path=["physics", "quantum_mechanics", "quantum_foundations"],
        actualization_depth=4,
        supported_by=[
            DocumentReference(
                doc_id="entanglement_horodecki",
                title="Quantum Entanglement",
                authors=["Horodecki, R.", "Horodecki, P.", "Horodecki, M.", "Horodecki, K."],
                year=2009,
                uri="https://example.com/entanglement",
                excerpt="Entanglement is the essence of quantum mechanics..."
            ),
            DocumentReference(
                doc_id="aspect_experiments",
                title="Experimental Test of Bell's Inequalities",
                authors=["Aspect, A.", "Grangier, P.", "Roger, G."],
                year=1982,
                uri="https://example.com/aspect",
                excerpt="Experimental violation of Bell's inequalities confirms..."
            ),
        ],
        delta_embedding=np.random.randn(384) * 0.015,
    )
    graph.add_node(quantum_entanglement)
    print(f"[OK] Added: {quantum_entanglement.name} (depth={quantum_entanglement.actualization_depth})")
    print(f"  Confluence: {quantum_entanglement.confluence_pattern}")

    # Record crystallization event
    quantum_entanglement.record_crystallization(
        CrystallizationEvent(
            timestamp=datetime(1935, 5, 15),
            document=DocumentReference(
                doc_id="epr_paper",
                title="Can Quantum-Mechanical Description of Physical Reality Be Considered Complete?",
                authors=["Einstein, A.", "Podolsky, B.", "Rosen, N."],
                year=1935,
                uri="https://example.com/epr",
            ),
            context="EPR paradox first described entanglement phenomenon",
            confidence=0.95,
        )
    )

    print(f"\n[PASS] Created tree with {len(graph.nodes)} nodes")
    return graph


def test_tree_traversal(graph: KronosGraph):
    """Test tree traversal operations."""
    print("\n" + "="*80)
    print("TESTING TREE TRAVERSAL")
    print("="*80)

    # Test 1: Get ancestors
    print("\n--- Test 1: Ancestors of quantum_entanglement ---")
    ancestors = graph.get_ancestors("quantum_entanglement")
    print(f"Ancestors ({len(ancestors)}):")
    for i, ancestor in enumerate(ancestors, 1):
        print(f"  {i}. {ancestor.name} (depth={ancestor.actualization_depth})")

    # Test 2: Get descendants
    print("\n--- Test 2: Descendants of quantum_mechanics ---")
    descendants = graph.get_descendants("quantum_mechanics", max_depth=3)
    print(f"Descendants ({len(descendants)}):")
    for i, desc in enumerate(descendants, 1):
        print(f"  {i}. {desc.name} (depth={desc.actualization_depth})")

    # Test 3: Get siblings
    print("\n--- Test 3: Siblings of superposition ---")
    siblings = graph.get_siblings("superposition")
    print(f"Siblings ({len(siblings)}):")
    for i, sibling in enumerate(siblings, 1):
        print(f"  {i}. {sibling.name}")

    # Test 4: Derivation path
    print("\n--- Test 4: Full derivation path to quantum_entanglement ---")
    path = graph.get_derivation_path("quantum_entanglement")
    print("Path:")
    for i, node in enumerate(path):
        indent = "  " * i
        arrow = "-> " if i > 0 else ""
        print(f"{indent}{arrow}{node.name} (depth={node.actualization_depth})")

    # Test 5: Neighbors within radius
    print("\n--- Test 5: Neighbors within radius=2 of quantum_foundations ---")
    neighbors = graph.get_neighbors_within_radius("quantum_foundations", radius=2)
    print(f"Neighbors ({len(neighbors)}):")
    for i, neighbor in enumerate(neighbors, 1):
        print(f"  {i}. {neighbor.name} (depth={neighbor.actualization_depth})")


def test_pac_conservation(graph: KronosGraph):
    """Test PAC conservation."""
    print("\n" + "="*80)
    print("TESTING PAC CONSERVATION")
    print("="*80)

    test_nodes = [
        "quantum_mechanics",
        "quantum_foundations",
        "superposition",
        "quantum_entanglement",
    ]

    for node_id in test_nodes:
        node = graph.get_node(node_id)
        if not node:
            continue

        print(f"\n--- {node.name} ---")
        print(f"  Parents: {node.parent_potentials}")
        print(f"  Confluence: {node.confluence_pattern}")

        # Check conservation
        is_conserved = graph.verify_conservation(node_id, tolerance=0.01)
        status = "[PASS] CONSERVED" if is_conserved else "[FAIL] VIOLATED"
        print(f"  PAC Conservation: {status}")

        # Show embedding info
        if node.delta_embedding is not None:
            print(f"  Delta embedding: shape={node.delta_embedding.shape}, "
                  f"norm={np.linalg.norm(node.delta_embedding):.4f}")


def test_geometric_confidence(graph: KronosGraph):
    """Test geometric confidence computation."""
    print("\n" + "="*80)
    print("TESTING GEOMETRIC CONFIDENCE")
    print("="*80)

    test_nodes = [
        ("physics", "Root - should have high confidence (well-documented)"),
        ("quantum_mechanics", "Level 1 - should have high confidence (many children, well-documented)"),
        ("quantum_entanglement", "Level 4 - confluence node with multiple parents"),
        ("quantum_algorithms", "Level 3 - moderate confidence"),
    ]

    for node_id, description in test_nodes:
        print(f"\n--- {node_id}: {description} ---")
        node = graph.get_node(node_id)
        if not node:
            print("  Node not found!")
            continue

        confidence = graph.compute_geometric_confidence(node_id)

        print(f"  Local Density: {confidence.local_density:.3f}")
        print(f"  Branch Symmetry: {confidence.branch_symmetry:.3f}")
        print(f"  Traversal Distance: {confidence.traversal_distance:.1f}")
        print(f"  Documentation Depth: {confidence.documentation_depth}")
        print(f"  Orphan Score: {confidence.orphan_score:.3f}")
        print(f"  Confluence Bottleneck: {confidence.confluence_bottleneck}")
        print(f"  Missing Children: {len(confidence.missing_expected_children)}")
        print(f"\n  > Confidence: {confidence.retrieval_confidence:.3f}")
        print(f"  > Hallucination Risk: {confidence.hallucination_risk:.3f}")
        print(f"  > Interpretation: {confidence.interpretation}")
        print(f"  > Recommendation: {confidence.action_recommendation}")

        if confidence.has_anomalies:
            print(f"\n  [!]  Anomalies detected:")
            for anomaly in confidence.get_anomaly_report():
                print(f"      • {anomaly}")


def test_graph_stats(graph: KronosGraph):
    """Show graph statistics."""
    print("\n" + "="*80)
    print("GRAPH STATISTICS")
    print("="*80)

    stats = graph.get_stats()
    print(f"\nNodes: {stats['node_count']}")
    print(f"Edges: {stats['edge_count']}")
    print(f"Roots: {stats['root_count']}")
    print(f"Max Depth: {stats['max_depth']}")
    print(f"Avg Children per Node: {stats['avg_children']:.2f}")
    print(f"Avg Documentation per Node: {stats['avg_documentation']:.2f}")

    print(f"\n{graph}")


def main():
    """Run all tests."""
    print("="*80)
    print("KRONOS v2: QUANTUM MECHANICS GENEALOGY TREE TEST")
    print("="*80)

    # Create tree
    graph = create_quantum_tree()

    # Run tests
    test_tree_traversal(graph)
    test_pac_conservation(graph)
    test_geometric_confidence(graph)
    test_graph_stats(graph)

    print("\n" + "="*80)
    print("[PASS] ALL TESTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
