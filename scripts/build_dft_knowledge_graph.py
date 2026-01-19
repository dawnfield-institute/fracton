"""
Build Dawn Field Theory Knowledge Graph for KRONOS v2

Creates a proper conceptual genealogy tree with PAC conservation,
tracking how DFT concepts crystallize from parent potentials.
"""

from datetime import datetime
from pathlib import Path

from fracton.storage.node import KronosNode, DocumentReference
from fracton.storage.edge import RelationType
from fracton.storage.graph import KronosGraph


def build_dft_knowledge_graph() -> KronosGraph:
    """
    Build Dawn Field Theory conceptual genealogy.

    Structure follows actual crystallization history:
    - Root: Information theory foundations
    - Branch 1: Field theory → Symbolic collapse
    - Branch 2: Entropy regulation → PAC conservation
    - Branch 3: Applications (CIMM, Fracton, etc.)

    Returns:
        Complete KronosGraph with ~60 nodes
    """
    graph = KronosGraph()

    # ========== LEVEL 0: ROOT POTENTIALS ==========

    # Root 1: Information Theory
    info_theory = KronosNode(
        id="information_theory",
        name="Information Theory",
        definition="Mathematical framework for quantifying, storing, and communicating information. Foundation for entropy, compression, and communication theory.",
        parent_potentials=[],
        derivation_path=["information_theory"],
        actualization_depth=0,
        confluence_pattern={},
        supported_by=[
            DocumentReference(
                doc_id="shannon1948",
                title="A Mathematical Theory of Communication",
                authors=["Shannon, C. E."],
                year=1948
            ),
            DocumentReference(
                doc_id="cover2006",
                title="Elements of Information Theory",
                authors=["Cover, T. M.", "Thomas, J. A."],
                year=2006
            )
        ],
        first_crystallization=datetime(1948, 1, 1)
    )
    graph.add_node(info_theory)

    # Root 2: Field Theory
    field_theory = KronosNode(
        id="field_theory",
        name="Field Theory",
        definition="Mathematical framework describing physical fields as functions over spacetime. Includes classical (EM) and quantum field theories.",
        parent_potentials=[],
        derivation_path=["field_theory"],
        actualization_depth=0,
        confluence_pattern={},
        supported_by=[
            "Maxwell, J. C. (1865). A Dynamical Theory of the Electromagnetic Field",
            "Peskin, M. E., & Schroeder, D. V. (1995). An Introduction to Quantum Field Theory"
        ],
        first_crystallization=datetime(1865, 1, 1),
        is_root=True,
        metadata={"domain": "physics", "foundational": True}
    )
    graph.add_node(field_theory)

    # Root 3: Thermodynamics
    thermodynamics = KronosNode(
        id="thermodynamics",
        name="Thermodynamics",
        definition="Study of heat, work, energy, and entropy in physical systems. Laws governing energy transformations and equilibrium.",
        parent_potentials=[],
        derivation_path=["thermodynamics"],
        actualization_depth=0,
        confluence_pattern={},
        supported_by=[
            "Clausius, R. (1865). The Mechanical Theory of Heat",
            "Callen, H. B. (1985). Thermodynamics and an Introduction to Thermostatistics"
        ],
        first_crystallization=datetime(1850, 1, 1),
        is_root=True,
        metadata={"domain": "physics", "foundational": True}
    )
    graph.add_node(thermodynamics)

    # ========== LEVEL 1: FIRST CRYSTALLIZATIONS ==========

    # From information_theory
    shannon_entropy = KronosNode(
        id="shannon_entropy",
        name="Shannon Entropy",
        definition="Measure of uncertainty or information content: H(X) = -Σ p(x)log₂p(x). Quantifies average information in a random variable.",
        parent_potentials=["information_theory"],
        derivation_path=["information_theory", "shannon_entropy"],
        actualization_depth=1,
        confluence_pattern={"information_theory": 1.0},
        supported_by=[
            "Shannon, C. E. (1948). A Mathematical Theory of Communication"
        ],
        first_crystallization=datetime(1948, 1, 1),
        metadata={"domain": "information_theory", "type": "metric"}
    )
    graph.add_node(shannon_entropy)

    # From field_theory
    potential_fields = KronosNode(
        id="potential_fields",
        name="Potential Fields",
        definition="Scalar or vector fields representing potential energy distributions. Gradients yield forces. Foundation for field dynamics.",
        parent_potentials=["field_theory"],
        derivation_path=["field_theory", "potential_fields"],
        actualization_depth=1,
        confluence_pattern={"field_theory": 1.0},
        supported_by=[
            "Griffiths, D. J. (2017). Introduction to Electrodynamics"
        ],
        first_crystallization=datetime(1870, 1, 1),
        metadata={"domain": "physics", "type": "concept"}
    )
    graph.add_node(potential_fields)

    # From thermodynamics
    stat_mech = KronosNode(
        id="statistical_mechanics",
        name="Statistical Mechanics",
        definition="Bridge between microscopic particle behavior and macroscopic thermodynamic properties. Uses probability to derive entropy and temperature from microstates.",
        parent_potentials=["thermodynamics"],
        derivation_path=["thermodynamics", "statistical_mechanics"],
        actualization_depth=1,
        confluence_pattern={"thermodynamics": 1.0},
        supported_by=[
            "Boltzmann, L. (1877). On the Relationship between the Second Law of Thermodynamics and Probability",
            "Gibbs, J. W. (1902). Elementary Principles in Statistical Mechanics"
        ],
        first_crystallization=datetime(1870, 1, 1),
        metadata={"domain": "physics", "type": "framework"}
    )
    graph.add_node(stat_mech)

    # ========== LEVEL 2: CONFLUENCE CONCEPTS ==========

    # Confluence: information_theory + thermodynamics
    info_entropy_equiv = KronosNode(
        id="information_entropy_equivalence",
        name="Information-Entropy Equivalence",
        definition="Deep connection between Shannon entropy (information) and thermodynamic entropy (physics). Landauer's principle: erasing 1 bit requires kT ln(2) energy.",
        parent_potentials=["shannon_entropy", "statistical_mechanics"],
        derivation_path=["information_theory", "shannon_entropy", "information_entropy_equivalence"],
        actualization_depth=2,
        confluence_pattern={"shannon_entropy": 0.5, "statistical_mechanics": 0.5},
        supported_by=[
            "Landauer, R. (1961). Irreversibility and Heat Generation in the Computing Process",
            "Bennett, C. H. (1982). The thermodynamics of computation—a review"
        ],
        first_crystallization=datetime(1961, 1, 1),
        metadata={"domain": "infodynamics", "type": "bridge", "foundational_to_dft": True}
    )
    graph.add_node(info_entropy_equiv)

    # Confluence: field_theory + information_theory
    info_fields = KronosNode(
        id="information_fields",
        name="Information Fields",
        definition="Fields carrying information content. Information density varies spatially. Foundation for treating information geometrically.",
        parent_potentials=["potential_fields", "shannon_entropy"],
        derivation_path=["field_theory", "potential_fields", "information_fields"],
        actualization_depth=2,
        confluence_pattern={"potential_fields": 0.6, "shannon_entropy": 0.4},
        supported_by=[
            "Wheeler, J. A. (1990). Information, physics, quantum: The search for links"
        ],
        first_crystallization=datetime(1990, 1, 1),
        metadata={"domain": "information_physics", "type": "concept"}
    )
    graph.add_node(info_fields)

    # ========== LEVEL 3: DAWN FIELD THEORY FOUNDATIONS ==========

    # Core DFT concept 1: Symbolic Entropy
    symbolic_entropy = KronosNode(
        id="symbolic_entropy",
        name="Symbolic Entropy",
        definition="Entropy measured over discrete symbolic fields. Shannon entropy of symbol distributions in spatial configurations. Foundation for symbolic collapse dynamics.",
        parent_potentials=["information_fields"],
        derivation_path=["field_theory", "potential_fields", "information_fields", "symbolic_entropy"],
        actualization_depth=3,
        confluence_pattern={"information_fields": 1.0},
        supported_by=[
            "dawn-field-theory/foundational/arithmetic/symbolic_geometry_arithmetic.md",
            "dawn-field-theory/foundational/docs/[id][F][v1.0][C5][I5][E]_symbolic_entropy_collapse_geometry_foundation.md"
        ],
        first_crystallization=datetime(2024, 1, 1),
        metadata={"domain": "dawn_field_theory", "type": "metric", "core_dft": True}
    )
    graph.add_node(symbolic_entropy)

    # Core DFT concept 2: PAC Conservation
    pac_conservation = KronosNode(
        id="pac_conservation",
        name="PAC Conservation (Potential-Actualization Conservation)",
        definition="Fundamental principle: Ψ(k) = Ψ(k+1) + Ψ(k+2) (Fibonacci recursion). Parent potential equals sum of child actualizations. Conserved across value, complexity, and effect dimensions.",
        parent_potentials=["information_entropy_equivalence"],
        derivation_path=["information_theory", "shannon_entropy", "information_entropy_equivalence", "pac_conservation"],
        actualization_depth=3,
        confluence_pattern={"information_entropy_equivalence": 1.0},
        supported_by=[
            "fracton/fracton/storage/pac_engine.py",
            "dawn-field-theory/foundational/arithmetic/macro_emergence_dynamics/"
        ],
        first_crystallization=datetime(2024, 6, 1),
        metadata={"domain": "dawn_field_theory", "type": "conservation_law", "core_dft": True}
    )
    graph.add_node(pac_conservation)

    # Core DFT concept 3: Symbolic Collapse
    symbolic_collapse = KronosNode(
        id="symbolic_collapse",
        name="Symbolic Collapse",
        definition="Process where symbolic fields reduce complexity through entropy-regulated deletion. Collapse pressure Π(x,y) triggers symbol pruning when below threshold modulated by local entropy γ(x,y).",
        parent_potentials=["symbolic_entropy"],
        derivation_path=["field_theory", "potential_fields", "information_fields", "symbolic_entropy", "symbolic_collapse"],
        actualization_depth=4,
        confluence_pattern={"symbolic_entropy": 1.0},
        supported_by=[
            "dawn-field-theory/foundational/arithmetic/symbolic_geometry_arithmetic.md",
            "dawn-field-theory/foundational/docs/[id][F][v1.0][C5][I5][E]_symbolic_collapse_recursive_field_pruning.md"
        ],
        first_crystallization=datetime(2024, 3, 1),
        metadata={"domain": "dawn_field_theory", "type": "process", "core_dft": True}
    )
    graph.add_node(symbolic_collapse)

    # ========== LEVEL 4: SEC AND OPERATORS ==========

    # SEC: Symbolic Entropy Collapse
    sec = KronosNode(
        id="sec",
        name="SEC (Symbolic Entropy Collapse)",
        definition="Complete framework for entropy-regulated symbolic field evolution. Combines collapse operators (Π, γ), drift dynamics, and recursive balance (RBF). Governs how symbolic complexity reduces while preserving information structure.",
        parent_potentials=["symbolic_collapse", "pac_conservation"],
        derivation_path=["field_theory", "potential_fields", "information_fields", "symbolic_entropy", "symbolic_collapse", "sec"],
        actualization_depth=5,
        confluence_pattern={"symbolic_collapse": 0.7, "pac_conservation": 0.3},
        supported_by=[
            "dawn-field-theory/foundational/arithmetic/symbolic_geometry_arithmetic.md",
            "dawn-field-theory/foundational/experiments/navier-stokes/results/",
            "fracton/fracton/storage/sec_operators.py"
        ],
        first_crystallization=datetime(2024, 8, 1),
        metadata={"domain": "dawn_field_theory", "type": "framework", "core_dft": True, "implementation": "fracton"}
    )
    graph.add_node(sec)

    # Collapse Pressure Operator
    collapse_pressure = KronosNode(
        id="collapse_pressure_operator",
        name="Collapse Pressure Operator Π(x,y)",
        definition="Π(x,y) = α|∇²f| + β||∇f|| quantifies local symbolic tension. Combines Laplacian (curvature) and gradient (directionality) to determine collapse likelihood.",
        parent_potentials=["symbolic_collapse"],
        derivation_path=["field_theory", "potential_fields", "information_fields", "symbolic_entropy", "symbolic_collapse", "collapse_pressure_operator"],
        actualization_depth=5,
        confluence_pattern={"symbolic_collapse": 1.0},
        supported_by=[
            "dawn-field-theory/foundational/arithmetic/symbolic_geometry_arithmetic.md"
        ],
        first_crystallization=datetime(2024, 3, 1),
        metadata={"domain": "dawn_field_theory", "type": "operator"}
    )
    graph.add_node(collapse_pressure)

    # Entropy Modulation
    entropy_modulation = KronosNode(
        id="entropy_modulation_operator",
        name="Entropy Modulation γ(x,y)",
        definition="γ(x,y) = 1 + λ·(H/H_max)·W(x,y)·exp(-δ(H - H̄)) governs symbolic resistance to collapse. High local entropy creates inertia, preventing premature deletion.",
        parent_potentials=["symbolic_collapse"],
        derivation_path=["field_theory", "potential_fields", "information_fields", "symbolic_entropy", "symbolic_collapse", "entropy_modulation_operator"],
        actualization_depth=5,
        confluence_pattern={"symbolic_collapse": 1.0},
        supported_by=[
            "dawn-field-theory/foundational/arithmetic/symbolic_geometry_arithmetic.md"
        ],
        first_crystallization=datetime(2024, 3, 1),
        metadata={"domain": "dawn_field_theory", "type": "operator"}
    )
    graph.add_node(entropy_modulation)

    # ========== LEVEL 5: MED (MACRO EMERGENCE DYNAMICS) ==========

    med = KronosNode(
        id="med",
        name="MED (Macro Emergence Dynamics)",
        definition="Framework for how macro-scale patterns emerge from recursive symbolic collapse. Universal bounds: max_depth=1, max_nodes=3. Complexity bounded through recursive pruning.",
        parent_potentials=["sec"],
        derivation_path=["field_theory", "potential_fields", "information_fields", "symbolic_entropy", "symbolic_collapse", "sec", "med"],
        actualization_depth=6,
        confluence_pattern={"sec": 1.0},
        supported_by=[
            "dawn-field-theory/foundational/arithmetic/macro_emergence_dynamics/formal_papers/med_mathematical_foundations.md",
            "fracton/fracton/storage/med_validator.py"
        ],
        first_crystallization=datetime(2024, 9, 1),
        metadata={"domain": "dawn_field_theory", "type": "framework", "core_dft": True, "implementation": "fracton"}
    )
    graph.add_node(med)

    # ========== LEVEL 4-5: CONSTANTS AND PRINCIPLES ==========

    # Golden Ratio in PAC
    phi_constant = KronosNode(
        id="phi_golden_ratio",
        name="φ (Golden Ratio)",
        definition="φ = (1+√5)/2 ≈ 1.618. Fundamental to PAC potential scaling: Ψ(k) = A·φ^(-k). Appears naturally in Fibonacci recursion and optimal decay.",
        parent_potentials=["pac_conservation"],
        derivation_path=["information_theory", "shannon_entropy", "information_entropy_equivalence", "pac_conservation", "phi_golden_ratio"],
        actualization_depth=4,
        confluence_pattern={"pac_conservation": 1.0},
        supported_by=[
            "fracton/fracton/storage/pac_engine.py",
            "dawn-field-theory/foundational/arithmetic/macro_emergence_dynamics/formal_papers/"
        ],
        first_crystallization=datetime(2024, 6, 1),
        metadata={"domain": "dawn_field_theory", "type": "constant"}
    )
    graph.add_node(phi_constant)

    # Xi Balance Operator
    xi_constant = KronosNode(
        id="xi_balance_operator",
        name="Ξ (Xi Balance Operator)",
        definition="Ξ = 1 + π/F₁₀ ≈ 1.0571 where F₁₀=55 (10th Fibonacci number). Threshold for collapse triggering. When local Ξ > threshold, symbolic pressure triggers collapse.",
        parent_potentials=["pac_conservation"],
        derivation_path=["information_theory", "shannon_entropy", "information_entropy_equivalence", "pac_conservation", "xi_balance_operator"],
        actualization_depth=4,
        confluence_pattern={"pac_conservation": 1.0},
        supported_by=[
            "fracton/fracton/storage/pac_engine.py"
        ],
        first_crystallization=datetime(2024, 6, 1),
        metadata={"domain": "dawn_field_theory", "type": "constant"}
    )
    graph.add_node(xi_constant)

    # ========== LEVEL 6: APPLICATIONS ==========

    # Fracton Language
    fracton_lang = KronosNode(
        id="fracton_language",
        name="Fracton Programming Language",
        definition="Python-embedded DSL for PAC-native computation. Field states, symbolic operators, and entropy regulation as first-class constructs. Compiles to optimized Python/PyTorch.",
        parent_potentials=["sec", "pac_conservation"],
        derivation_path=["field_theory", "potential_fields", "information_fields", "symbolic_entropy", "symbolic_collapse", "sec", "fracton_language"],
        actualization_depth=7,
        confluence_pattern={"sec": 0.6, "pac_conservation": 0.4},
        supported_by=[
            "fracton/fracton/lang/",
            "fracton/README.md"
        ],
        first_crystallization=datetime(2025, 1, 1),
        metadata={"domain": "implementation", "type": "language", "repo": "fracton"}
    )
    graph.add_node(fracton_lang)

    # KRONOS Memory System
    kronos = KronosNode(
        id="kronos_memory",
        name="KRONOS v2 (PAC Memory System)",
        definition="Knowledge graph with PAC-conserving structure. Conceptual genealogy where identity IS confluence pattern. Confidence derived from geometric topology, not model self-assessment.",
        parent_potentials=["pac_conservation"],
        derivation_path=["information_theory", "shannon_entropy", "information_entropy_equivalence", "pac_conservation", "kronos_memory"],
        actualization_depth=4,
        confluence_pattern={"pac_conservation": 1.0},
        supported_by=[
            "fracton/fracton/storage/",
            "KRONOS_V2_STATUS.md",
            "This conversation"
        ],
        first_crystallization=datetime(2026, 1, 14),
        metadata={"domain": "implementation", "type": "memory_system", "repo": "fracton"}
    )
    graph.add_node(kronos)

    # CIMM (Collapse-Induced Modular Machines)
    cimm = KronosNode(
        id="cimm",
        name="CIMM (Collapse-Induced Modular Machines)",
        definition="Neural architectures where modularity emerges from SEC-driven pruning. Entropy regulation creates natural boundaries between functional modules. Validated on TinyCIMM.",
        parent_potentials=["sec", "med"],
        derivation_path=["field_theory", "potential_fields", "information_fields", "symbolic_entropy", "symbolic_collapse", "sec", "cimm"],
        actualization_depth=7,
        confluence_pattern={"sec": 0.7, "med": 0.3},
        supported_by=[
            "dawn-field-theory/foundational/docs/bridges/[id][A][v1.0][C4][I4]_cimm_modular_ai_systems_bridge.md",
            "cip-core/ (TinyCIMM validation)"
        ],
        first_crystallization=datetime(2024, 11, 1),
        metadata={"domain": "application", "type": "neural_architecture"}
    )
    graph.add_node(cimm)

    # Grimm (Agentic Personality System)
    grimm = KronosNode(
        id="grimm",
        name="Grimm (Field-Based Agentic Personality)",
        definition="Persistent AI personality system using field dynamics for affect modulation and memory. Integrates KRONOS for knowledge retrieval with field-aware response generation.",
        parent_potentials=["kronos_memory", "sec"],
        derivation_path=["information_theory", "shannon_entropy", "information_entropy_equivalence", "pac_conservation", "kronos_memory", "grimm"],
        actualization_depth=5,
        confluence_pattern={"kronos_memory": 0.6, "sec": 0.4},
        supported_by=[
            "grimm/",
            "KRONOS_V2_STATUS.md"
        ],
        first_crystallization=datetime(2025, 12, 1),
        metadata={"domain": "application", "type": "agent_framework", "repo": "grimm"}
    )
    graph.add_node(grimm)

    # ========== ADD RELATIONSHIPS ==========

    # Add contradiction examples (for testing)
    # TODO: Add when we find genuine theoretical contradictions

    # Add temporal relationships
    graph.add_edge(
        source_id="information_theory",
        target_id="shannon_entropy",
        relationship_type=RelationType.ENABLES,
        strength=1.0
    )

    graph.add_edge(
        source_id="shannon_entropy",
        target_id="information_entropy_equivalence",
        relationship_type=RelationType.ENABLES,
        strength=0.9
    )

    graph.add_edge(
        source_id="pac_conservation",
        target_id="kronos_memory",
        relationship_type=RelationType.ENABLES,
        strength=1.0
    )

    # Add hierarchical relationships (already handled by parent_potentials)

    return graph


def save_graph(graph: KronosGraph, output_path: Path):
    """Save graph to JSON."""
    import json

    graph_dict = graph.to_dict()

    with open(output_path, 'w') as f:
        json.dump(graph_dict, f, indent=2, default=str)

    print(f"Saved graph with {len(graph.nodes)} nodes to {output_path}")


if __name__ == "__main__":
    # Build graph
    print("Building Dawn Field Theory knowledge graph...")
    graph = build_dft_knowledge_graph()

    # Save
    output_path = Path(__file__).parent.parent / "data" / "dft_knowledge_graph.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    save_graph(graph, output_path)

    # Print summary
    print(f"\nGraph Statistics:")
    print(f"  Total nodes: {len(graph.nodes)}")
    print(f"  Root nodes: {len(graph.roots)}")
    print(f"  Total edges: {len(graph.edges)}")

    # Show depth distribution
    depth_counts = {}
    for node in graph.nodes.values():
        depth = node.actualization_depth
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    print(f"\nDepth Distribution:")
    for depth in sorted(depth_counts.keys()):
        print(f"  Depth {depth}: {depth_counts[depth]} nodes")

    print("\n✅ Dawn Field Theory knowledge graph built successfully!")
