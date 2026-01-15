"""
Comprehensive KRONOS Graph Builder and Validator

Creates a rich quantum mechanics knowledge graph with:
- Multiple depth levels (0-4)
- Confluence nodes with weighted parents
- Well-documented vs sparse regions
- Citations and crystallization events
- Historical timeline

Then validates:
- Tree structure correctness
- Confidence scores vary appropriately
- Response personality matches topology
"""

import sys
import io
from pathlib import Path
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add fracton to path
fracton_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(fracton_path))

from fracton.storage import (
    KronosGraph,
    KronosNode,
    DocumentReference,
    CrystallizationEvent,
    KronosResponseGenerator,
)


def create_comprehensive_graph():
    """
    Build a comprehensive quantum mechanics graph.

    Structure:
    - Level 0: Physics (root, well-documented)
    - Level 1: Quantum Mechanics (well-documented)
    - Level 2: Foundations (superposition, nonlocality, measurement)
    - Level 3: Phenomena (entanglement - confluence node)
    - Level 4: Applications (cryptography, computing, teleportation)
    - Level 4: Sparse areas (some poorly documented leaf nodes)
    """
    graph = KronosGraph()

    # ========================================================================
    # LEVEL 0: ROOT (Well-documented, authoritative)
    # ========================================================================

    physics = KronosNode(
        id="physics",
        name="Physics",
        definition="The natural science that studies matter, energy, and their fundamental interactions.",
        actualization_depth=0,
        derivation_path=["physics"],
        parent_potentials=[],
        child_actualizations=["classical_mechanics", "quantum_mechanics", "relativity"],
        supported_by=[
            DocumentReference(
                doc_id="newton_1687",
                title="Philosophiæ Naturalis Principia Mathematica",
                authors=["Newton, Isaac"],
                year=1687,
                uri="https://example.com/principia",
                excerpt="The laws of motion and universal gravitation"
            ),
            DocumentReference(
                doc_id="einstein_1905",
                title="On the Electrodynamics of Moving Bodies",
                authors=["Einstein, Albert"],
                year=1905,
                uri="https://example.com/relativity",
                excerpt="Special relativity and E=mc²"
            ),
        ],
        first_crystallization=datetime(1687, 7, 5),
    )
    graph.add_node(physics)

    # ========================================================================
    # LEVEL 1: QUANTUM MECHANICS (Well-documented)
    # ========================================================================

    qm = KronosNode(
        id="quantum_mechanics",
        name="Quantum Mechanics",
        definition="A fundamental theory in physics describing the physical properties of nature at atomic and subatomic scales.",
        actualization_depth=1,
        derivation_path=["physics", "quantum_mechanics"],
        parent_potentials=["physics"],
        child_actualizations=["superposition", "nonlocality", "wave_function", "measurement"],
        supported_by=[
            DocumentReference(
                doc_id="planck_1900",
                title="On the Theory of the Energy Distribution Law of the Normal Spectrum",
                authors=["Planck, Max"],
                year=1900,
                uri="https://example.com/planck",
                excerpt="Introduced the quantum of action"
            ),
            DocumentReference(
                doc_id="heisenberg_1925",
                title="Quantum-Theoretical Re-interpretation of Kinematic and Mechanical Relations",
                authors=["Heisenberg, Werner"],
                year=1925,
                uri="https://example.com/heisenberg",
                excerpt="Matrix mechanics formulation"
            ),
            DocumentReference(
                doc_id="schrodinger_1926",
                title="An Undulatory Theory of the Mechanics of Atoms and Molecules",
                authors=["Schrödinger, Erwin"],
                year=1926,
                uri="https://example.com/schrodinger",
                excerpt="Wave mechanics and the Schrödinger equation"
            ),
            DocumentReference(
                doc_id="dirac_1930",
                title="The Principles of Quantum Mechanics",
                authors=["Dirac, Paul"],
                year=1930,
                uri="https://example.com/dirac",
                excerpt="Unified formulation of quantum mechanics"
            ),
        ],
        first_crystallization=datetime(1900, 12, 14),
    )
    graph.add_node(qm)

    # ========================================================================
    # LEVEL 2: QUANTUM FOUNDATIONS (Well-documented)
    # ========================================================================

    superposition = KronosNode(
        id="superposition",
        name="Quantum Superposition",
        definition="The principle that a quantum system can exist in multiple states simultaneously until measured, represented as a linear combination of basis states.",
        actualization_depth=2,
        derivation_path=["physics", "quantum_mechanics", "superposition"],
        parent_potentials=["quantum_mechanics"],
        child_actualizations=["quantum_entanglement", "quantum_computing"],
        supported_by=[
            DocumentReference(
                doc_id="schrodinger_1926_wave",
                title="An Undulatory Theory of the Mechanics of Atoms and Molecules",
                authors=["Schrödinger, Erwin"],
                year=1926,
                uri="https://example.com/wave",
                excerpt="Wave function as superposition of eigenstates"
            ),
            DocumentReference(
                doc_id="schrodinger_1935_cat",
                title="The Present Situation in Quantum Mechanics",
                authors=["Schrödinger, Erwin"],
                year=1935,
                uri="https://example.com/cat",
                excerpt="Schrödinger's cat thought experiment"
            ),
        ],
        first_crystallization=datetime(1926, 3, 1),
    )
    graph.add_node(superposition)

    nonlocality = KronosNode(
        id="nonlocality",
        name="Quantum Nonlocality",
        definition="The phenomenon where quantum correlations between particles persist regardless of spatial separation, violating classical locality constraints.",
        actualization_depth=2,
        derivation_path=["physics", "quantum_mechanics", "nonlocality"],
        parent_potentials=["quantum_mechanics"],
        child_actualizations=["quantum_entanglement"],
        supported_by=[
            DocumentReference(
                doc_id="epr_1935",
                title="Can Quantum-Mechanical Description of Physical Reality Be Considered Complete?",
                authors=["Einstein, Albert", "Podolsky, Boris", "Rosen, Nathan"],
                year=1935,
                uri="https://example.com/epr",
                excerpt="Spooky action at a distance"
            ),
            DocumentReference(
                doc_id="bell_1964",
                title="On the Einstein Podolsky Rosen Paradox",
                authors=["Bell, John S."],
                year=1964,
                uri="https://example.com/bell",
                excerpt="Bell's theorem and inequality"
            ),
        ],
        first_crystallization=datetime(1935, 5, 15),
    )
    graph.add_node(nonlocality)

    measurement = KronosNode(
        id="measurement",
        name="Quantum Measurement",
        definition="The process by which quantum superposition collapses to a definite outcome, fundamentally linking observation to physical reality.",
        actualization_depth=2,
        derivation_path=["physics", "quantum_mechanics", "measurement"],
        parent_potentials=["quantum_mechanics"],
        child_actualizations=["quantum_entanglement", "decoherence"],
        supported_by=[
            DocumentReference(
                doc_id="von_neumann_1932",
                title="Mathematical Foundations of Quantum Mechanics",
                authors=["von Neumann, John"],
                year=1932,
                uri="https://example.com/vonneumann",
                excerpt="Measurement problem and wave function collapse"
            ),
            DocumentReference(
                doc_id="wigner_1961",
                title="Remarks on the Mind-Body Question",
                authors=["Wigner, Eugene"],
                year=1961,
                uri="https://example.com/wigner",
                excerpt="The role of consciousness in measurement"
            ),
        ],
        first_crystallization=datetime(1932, 1, 1),
    )
    graph.add_node(measurement)

    # ========================================================================
    # LEVEL 3: QUANTUM ENTANGLEMENT (Confluence node - well-documented)
    # ========================================================================

    entanglement = KronosNode(
        id="quantum_entanglement",
        name="Quantum Entanglement",
        definition="A phenomenon where quantum states of multiple particles become correlated such that measuring one instantaneously affects the others, regardless of spatial separation.",
        actualization_depth=3,
        derivation_path=["physics", "quantum_mechanics", "quantum_entanglement"],
        parent_potentials=["superposition", "nonlocality", "measurement"],
        child_actualizations=["quantum_cryptography", "quantum_computing", "quantum_teleportation"],
        confluence_pattern={
            "superposition": 0.40,  # Primary component
            "nonlocality": 0.35,    # Secondary component
            "measurement": 0.25,    # Tertiary component
        },
        supported_by=[
            DocumentReference(
                doc_id="epr_1935_entanglement",
                title="Can Quantum-Mechanical Description of Physical Reality Be Considered Complete?",
                authors=["Einstein, Albert", "Podolsky, Boris", "Rosen, Nathan"],
                year=1935,
                uri="https://example.com/epr",
                excerpt="First description of entangled states"
            ),
            DocumentReference(
                doc_id="bohm_1951",
                title="Quantum Theory",
                authors=["Bohm, David"],
                year=1951,
                uri="https://example.com/bohm",
                excerpt="EPR paradox with spin-1/2 particles"
            ),
            DocumentReference(
                doc_id="aspect_1982",
                title="Experimental Test of Bell's Inequalities Using Time-Varying Analyzers",
                authors=["Aspect, Alain", "Dalibard, Jean", "Roger, Gérard"],
                year=1982,
                uri="https://example.com/aspect",
                excerpt="Experimental violation of Bell inequalities"
            ),
            DocumentReference(
                doc_id="gisin_2014",
                title="Quantum Chance: Nonlocality, Teleportation and Other Quantum Marvels",
                authors=["Gisin, Nicolas"],
                year=2014,
                uri="https://example.com/gisin",
                excerpt="Modern perspective on entanglement"
            ),
        ],
        first_crystallization=datetime(1935, 5, 15),
        crystallization_events=[
            CrystallizationEvent(
                timestamp=datetime(1935, 5, 15),
                document=DocumentReference(
                    doc_id="epr_1935_event",
                    title="EPR Paper",
                    authors=["Einstein, Albert", "Podolsky, Boris", "Rosen, Nathan"],
                    year=1935,
                ),
                context="First conceptual description of entangled states as EPR paradox",
                confidence=0.90,
            ),
            CrystallizationEvent(
                timestamp=datetime(1964, 11, 4),
                document=DocumentReference(
                    doc_id="bell_1964_event",
                    title="Bell's Theorem",
                    authors=["Bell, John S."],
                    year=1964,
                ),
                context="Theoretical framework for testing entanglement via inequalities",
                confidence=0.95,
            ),
            CrystallizationEvent(
                timestamp=datetime(1982, 12, 10),
                document=DocumentReference(
                    doc_id="aspect_1982_event",
                    title="Aspect Experiments",
                    authors=["Aspect, Alain"],
                    year=1982,
                ),
                context="Experimental confirmation of entanglement violating Bell inequalities",
                confidence=0.98,
            ),
        ],
    )
    graph.add_node(entanglement)

    # ========================================================================
    # LEVEL 4: APPLICATIONS (Well-documented)
    # ========================================================================

    quantum_crypto = KronosNode(
        id="quantum_cryptography",
        name="Quantum Cryptography",
        definition="Cryptographic protocols that exploit quantum mechanical properties, particularly entanglement and superposition, to achieve provably secure communication.",
        actualization_depth=4,
        derivation_path=["physics", "quantum_mechanics", "quantum_entanglement", "quantum_cryptography"],
        parent_potentials=["quantum_entanglement"],
        child_actualizations=["qkd", "quantum_money"],
        supported_by=[
            DocumentReference(
                doc_id="bennett_brassard_1984",
                title="Quantum Cryptography: Public Key Distribution and Coin Tossing",
                authors=["Bennett, Charles H.", "Brassard, Gilles"],
                year=1984,
                uri="https://example.com/bb84",
                excerpt="BB84 protocol for quantum key distribution"
            ),
            DocumentReference(
                doc_id="ekert_1991",
                title="Quantum Cryptography Based on Bell's Theorem",
                authors=["Ekert, Artur K."],
                year=1991,
                uri="https://example.com/ekert",
                excerpt="Entanglement-based quantum key distribution"
            ),
        ],
        first_crystallization=datetime(1984, 1, 1),
    )
    graph.add_node(quantum_crypto)

    quantum_comp = KronosNode(
        id="quantum_computing",
        name="Quantum Computing",
        definition="Computing paradigm that exploits quantum mechanical phenomena like superposition and entanglement to perform computations that are intractable for classical computers.",
        actualization_depth=4,
        derivation_path=["physics", "quantum_mechanics", "quantum_entanglement", "quantum_computing"],
        parent_potentials=["quantum_entanglement", "superposition"],
        child_actualizations=["quantum_algorithms", "quantum_error_correction"],
        confluence_pattern={
            "quantum_entanglement": 0.5,
            "superposition": 0.5,
        },
        supported_by=[
            DocumentReference(
                doc_id="feynman_1982",
                title="Simulating Physics with Computers",
                authors=["Feynman, Richard P."],
                year=1982,
                uri="https://example.com/feynman",
                excerpt="Proposal for quantum simulation"
            ),
            DocumentReference(
                doc_id="deutsch_1985",
                title="Quantum Theory, the Church-Turing Principle and the Universal Quantum Computer",
                authors=["Deutsch, David"],
                year=1985,
                uri="https://example.com/deutsch",
                excerpt="Universal quantum computer concept"
            ),
            DocumentReference(
                doc_id="shor_1994",
                title="Algorithms for Quantum Computation: Discrete Logarithms and Factoring",
                authors=["Shor, Peter W."],
                year=1994,
                uri="https://example.com/shor",
                excerpt="Shor's algorithm for factoring"
            ),
        ],
        first_crystallization=datetime(1982, 5, 1),
    )
    graph.add_node(quantum_comp)

    quantum_teleport = KronosNode(
        id="quantum_teleportation",
        name="Quantum Teleportation",
        definition="A protocol for transferring quantum states from one location to another using entanglement and classical communication, without physically transmitting the particle itself.",
        actualization_depth=4,
        derivation_path=["physics", "quantum_mechanics", "quantum_entanglement", "quantum_teleportation"],
        parent_potentials=["quantum_entanglement"],
        child_actualizations=[],
        supported_by=[
            DocumentReference(
                doc_id="bennett_1993",
                title="Teleporting an Unknown Quantum State via Dual Classical and Einstein-Podolsky-Rosen Channels",
                authors=["Bennett, Charles H.", "Brassard, Gilles", "Crépeau, Claude", "Jozsa, Richard", "Peres, Asher", "Wootters, William K."],
                year=1993,
                uri="https://example.com/teleportation",
                excerpt="Theoretical protocol for quantum state teleportation"
            ),
            DocumentReference(
                doc_id="bouwmeester_1997",
                title="Experimental Quantum Teleportation",
                authors=["Bouwmeester, Dik", "Pan, Jian-Wei", "Mattle, Klaus", "Eibl, Manfred", "Weinfurter, Harald", "Zeilinger, Anton"],
                year=1997,
                uri="https://example.com/bouwmeester",
                excerpt="First experimental demonstration"
            ),
        ],
        first_crystallization=datetime(1993, 3, 29),
    )
    graph.add_node(quantum_teleport)

    # ========================================================================
    # LEVEL 4: SPARSE/EMERGING CONCEPTS (Poorly documented)
    # ========================================================================

    # This should have LOW confidence - sparse region, no docs
    quantum_biology = KronosNode(
        id="quantum_biology",
        name="Quantum Biology",
        definition="An emerging field exploring whether quantum mechanical phenomena play functional roles in biological processes like photosynthesis and bird navigation.",
        actualization_depth=4,
        derivation_path=["physics", "quantum_mechanics", "quantum_entanglement", "quantum_biology"],
        parent_potentials=["quantum_entanglement"],
        child_actualizations=[],
        supported_by=[],  # NO DOCUMENTATION - should flag low confidence
        first_crystallization=datetime(2007, 1, 1),
    )
    graph.add_node(quantum_biology)

    # Another sparse concept - speculative
    quantum_consciousness = KronosNode(
        id="quantum_consciousness",
        name="Quantum Consciousness",
        definition="A speculative hypothesis that quantum mechanics may be necessary to explain consciousness and cognitive processes.",
        actualization_depth=4,
        derivation_path=["physics", "quantum_mechanics", "measurement", "quantum_consciousness"],
        parent_potentials=["measurement"],
        child_actualizations=[],
        supported_by=[
            DocumentReference(
                doc_id="penrose_1989",
                title="The Emperor's New Mind",
                authors=["Penrose, Roger"],
                year=1989,
                uri="https://example.com/penrose",
                excerpt="Speculative connection between quantum mechanics and consciousness"
            ),
        ],
        first_crystallization=datetime(1989, 1, 1),
    )
    graph.add_node(quantum_consciousness)

    return graph


def validate_graph_structure(graph: KronosGraph):
    """Validate the graph structure before testing responses."""
    print("=" * 80)
    print("GRAPH STRUCTURE VALIDATION")
    print("=" * 80)

    print(f"\nTotal nodes: {len(graph.nodes)}")
    print(f"\nNodes by depth:")

    by_depth = {}
    for node_id, node in graph.nodes.items():
        depth = node.actualization_depth
        if depth not in by_depth:
            by_depth[depth] = []
        by_depth[depth].append((node_id, node.name))

    for depth in sorted(by_depth.keys()):
        print(f"\n  Depth {depth}:")
        for node_id, name in by_depth[depth]:
            node = graph.nodes[node_id]
            doc_count = len(node.supported_by)
            conf = graph.compute_geometric_confidence(node_id)
            confluence_str = ""
            if len(node.parent_potentials) > 1:
                confluence_str = " [CONFLUENCE]"

            print(f"    - {name} ({node_id})")
            print(f"      Docs: {doc_count}, Confidence: {conf.retrieval_confidence:.2f}{confluence_str}")

    # Test tree traversal
    print("\n" + "=" * 80)
    print("TREE TRAVERSAL TESTS")
    print("=" * 80)

    # Test 1: Ancestors of entanglement
    print("\n1. Ancestors of quantum_entanglement:")
    ancestors = graph.get_ancestors("quantum_entanglement")
    print(f"   Found {len(ancestors)} ancestors:")
    for anc in ancestors:
        # Check if anc is node or ID
        if isinstance(anc, str):
            print(f"   - {graph.nodes[anc].name} ({anc})")
        else:
            print(f"   - {anc.name} ({anc.id})")

    # Test 2: Descendants of quantum mechanics
    print("\n2. Descendants of quantum_mechanics:")
    descendants = graph.get_descendants("quantum_mechanics")
    print(f"   Found {len(descendants)} descendants:")
    for desc in descendants:
        if isinstance(desc, str):
            print(f"   - {graph.nodes[desc].name} ({desc})")
        else:
            print(f"   - {desc.name} ({desc.id})")

    # Test 3: Siblings of superposition
    print("\n3. Siblings of superposition:")
    siblings = graph.get_siblings("superposition")
    print(f"   Found {len(siblings)} siblings:")
    for sib in siblings:
        if isinstance(sib, str):
            print(f"   - {graph.nodes[sib].name} ({sib})")
        else:
            print(f"   - {sib.name} ({sib.id})")

    # Test 4: Confidence comparison
    print("\n" + "=" * 80)
    print("CONFIDENCE COMPARISON")
    print("=" * 80)

    test_nodes = [
        ("physics", "Root concept, well-documented"),
        ("quantum_entanglement", "Confluence node, well-documented"),
        ("quantum_cryptography", "Application, well-documented"),
        ("quantum_biology", "Sparse, NO documentation"),
        ("quantum_consciousness", "Sparse, minimal documentation"),
    ]

    print("\nExpected: High confidence for well-documented, low for sparse")
    print()

    for node_id, description in test_nodes:
        conf = graph.compute_geometric_confidence(node_id)
        node = graph.nodes[node_id]

        print(f"{node.name} ({description}):")
        print(f"  Confidence: {conf.retrieval_confidence:.2f}")
        print(f"  Local density: {conf.local_density:.2f}")
        print(f"  Docs: {len(node.supported_by)}")
        print(f"  Anomalies: {conf.has_anomalies}")
        if conf.has_anomalies:
            print(f"  Issues: {conf.get_anomaly_report()[:2]}")  # First 2 issues
        print()


def test_response_personalities(graph: KronosGraph):
    """Test that different topology produces different response personalities."""
    print("=" * 80)
    print("RESPONSE PERSONALITY TESTS")
    print("=" * 80)

    generator = KronosResponseGenerator(graph)

    test_cases = [
        {
            "concept_id": "physics",
            "query": "what is physics",
            "expected_tone": "Authoritative (root, well-documented)",
            "expected_confidence": "High (>0.7)",
        },
        {
            "concept_id": "quantum_entanglement",
            "query": "what is quantum entanglement",
            "expected_tone": "Integrative/Comprehensive (confluence, well-documented)",
            "expected_confidence": "High (>0.6)",
        },
        {
            "concept_id": "quantum_cryptography",
            "query": "what is quantum cryptography",
            "expected_tone": "Comprehensive (well-documented application)",
            "expected_confidence": "Medium-High (>0.5)",
        },
        {
            "concept_id": "quantum_biology",
            "query": "what is quantum biology",
            "expected_tone": "Tentative (sparse, no docs)",
            "expected_confidence": "Low (<0.3)",
        },
        {
            "concept_id": "quantum_consciousness",
            "query": "what is quantum consciousness",
            "expected_tone": "Tentative (sparse, speculative)",
            "expected_confidence": "Low (<0.4)",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'-' * 80}")
        print(f"TEST {i}: {test['concept_id']}")
        print(f"{'-' * 80}")
        print(f"Query: {test['query']}")
        print(f"Expected Tone: {test['expected_tone']}")
        print(f"Expected Confidence: {test['expected_confidence']}")
        print()

        response = generator.generate_response(
            query=test['query'],
            concept_id=test['concept_id']
        )

        # Extract key metrics
        confidence = response['confidence']['score']
        level = response['confidence']['level']
        strategy = response['metadata']['strategy']
        tone = response['metadata']['tone']

        print(f"ACTUAL RESULTS:")
        print(f"  Confidence: {confidence:.2f} ({level})")
        print(f"  Strategy: {strategy}")
        print(f"  Tone: {tone}")
        print(f"  Citations: {len(response['citations'])}")
        print(f"  Lineage depth: {len(response['lineage']['path'])}")

        # Show response preview
        content = response['content']
        preview_lines = content.split('\n')[:8]  # First 8 lines
        print(f"\nRESPONSE PREVIEW:")
        for line in preview_lines:
            print(f"  {line}")
        if len(content.split('\n')) > 8:
            print(f"  ...")

        # Validation
        print(f"\nVALIDATION:")

        # Check confidence matches expectation
        if "High" in test['expected_confidence']:
            threshold = float(test['expected_confidence'].split('>')[1].rstrip(')'))
            if confidence >= threshold:
                print(f"  ✓ Confidence {confidence:.2f} >= {threshold}")
            else:
                print(f"  ✗ Confidence {confidence:.2f} < {threshold} (expected high)")
        elif "Low" in test['expected_confidence']:
            threshold = float(test['expected_confidence'].split('<')[1].rstrip(')'))
            if confidence < threshold:
                print(f"  ✓ Confidence {confidence:.2f} < {threshold}")
            else:
                print(f"  ✗ Confidence {confidence:.2f} >= {threshold} (expected low)")

        # Check tone/strategy appropriateness
        if "Tentative" in test['expected_tone']:
            if confidence < 0.4:
                print(f"  ✓ Low confidence produces appropriate tentative tone")
            else:
                print(f"  ? Confidence unexpectedly high for sparse region")

        if "confluence" in test['expected_tone'].lower():
            if strategy == "confluence_synthesis":
                print(f"  ✓ Confluence node uses synthesis strategy")
            else:
                print(f"  ? Confluence node not using synthesis strategy")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("KRONOS COMPREHENSIVE GRAPH VALIDATION")
    print("=" * 80)

    # Build graph
    print("\nBuilding comprehensive quantum mechanics knowledge graph...")
    graph = create_comprehensive_graph()
    print(f"✓ Created graph with {len(graph.nodes)} nodes")

    # Validate structure
    validate_graph_structure(graph)

    # Test response personalities
    test_response_personalities(graph)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nReady for end-to-end CLI avatar testing!")
    print("Run: python test_kronos_integration.py")
