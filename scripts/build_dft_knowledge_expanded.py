"""
Build Expanded Dawn Field Theory Knowledge Graph for KRONOS v2

~60 nodes covering DFT foundations, core concepts, operators, and applications.
"""

from datetime import datetime
from pathlib import Path
import json

from fracton.storage.node import KronosNode
from fracton.storage.edge import RelationType, KronosEdge
from fracton.storage.graph import KronosGraph


def build_dft_knowledge_graph() -> KronosGraph:
    """Build comprehensive DFT conceptual genealogy."""
    graph = KronosGraph()

    # (id, name, definition, parents, confluence, depth)
    nodes = [
        # ========== LEVEL 0: ROOT POTENTIALS (5 roots) ==========
        ("information_theory", "Information Theory",
         "Mathematical framework for quantifying information. Foundation for entropy and communication theory.",
         [], {}, 0),

        ("field_theory", "Field Theory",
         "Framework describing physical fields as functions over spacetime.",
         [], {}, 0),

        ("thermodynamics", "Thermodynamics",
         "Study of heat, work, energy, and entropy in physical systems.",
         [], {}, 0),

        ("mathematics", "Mathematics",
         "Study of numbers, structures, space, and change. Foundation for formal reasoning.",
         [], {}, 0),

        ("quantum_mechanics", "Quantum Mechanics",
         "Physical theory describing nature at atomic and subatomic scales.",
         [], {}, 0),

        # ========== LEVEL 1: FIRST CRYSTALLIZATIONS ==========
        ("shannon_entropy", "Shannon Entropy",
         "H(X) = -Σ p(x)log p(x). Quantifies average information in a random variable.",
         ["information_theory"], {"information_theory": 1.0}, 1),

        ("statistical_mechanics", "Statistical Mechanics",
         "Bridges microscopic particle behavior and macroscopic thermodynamics.",
         ["thermodynamics"], {"thermodynamics": 1.0}, 1),

        ("potential_fields", "Potential Fields",
         "Scalar/vector fields representing potential energy. Gradients yield forces.",
         ["field_theory"], {"field_theory": 1.0}, 1),

        ("fibonacci_sequence", "Fibonacci Sequence",
         "Recursive sequence: F(n) = F(n-1) + F(n-2). Appears in nature and optimal growth.",
         ["mathematics"], {"mathematics": 1.0}, 1),

        ("golden_ratio_math", "Golden Ratio (Mathematical)",
         "φ = (1+√5)/2 ≈ 1.618. Limit of Fibonacci ratios. Optimal proportion.",
         ["fibonacci_sequence"], {"fibonacci_sequence": 1.0}, 2),

        ("wave_function", "Wave Function",
         "ψ: Mathematical description of quantum state. Contains all information about system.",
         ["quantum_mechanics"], {"quantum_mechanics": 1.0}, 1),

        ("measurement_problem", "Quantum Measurement Problem",
         "How/when wave function collapse occurs during observation.",
         ["wave_function"], {"wave_function": 1.0}, 2),

        ("kolmogorov_complexity", "Kolmogorov Complexity",
         "Shortest program that produces a string. Fundamental measure of information content.",
         ["information_theory"], {"information_theory": 1.0}, 1),

        # ========== LEVEL 2: CONFLUENCE & BRIDGES ==========
        ("information_entropy_equivalence", "Information-Entropy Equivalence",
         "Deep connection between Shannon entropy and thermodynamic entropy. Landauer's principle.",
         ["shannon_entropy", "statistical_mechanics"],
         {"shannon_entropy": 0.5, "statistical_mechanics": 0.5}, 2),

        ("information_fields", "Information Fields",
         "Fields carrying information content with spatial density variation.",
         ["potential_fields", "shannon_entropy"],
         {"potential_fields": 0.6, "shannon_entropy": 0.4}, 2),

        ("algorithmic_information", "Algorithmic Information Theory",
         "Bridge between computation and information. Kolmogorov complexity foundations.",
         ["kolmogorov_complexity", "shannon_entropy"],
         {"kolmogorov_complexity": 0.6, "shannon_entropy": 0.4}, 2),

        # ========== LEVEL 3: DFT FOUNDATIONS ==========
        ("symbolic_entropy", "Symbolic Entropy",
         "Entropy measured over discrete symbolic fields. Foundation for symbolic collapse.",
         ["information_fields"], {"information_fields": 1.0}, 3),

        ("pac_conservation", "PAC Conservation",
         "Fundamental principle: Ψ(k) = Ψ(k+1) + Ψ(k+2). Parent potential equals sum of children.",
         ["information_entropy_equivalence", "golden_ratio_math"],
         {"information_entropy_equivalence": 0.7, "golden_ratio_math": 0.3}, 3),

        ("infodynamics", "Infodynamics",
         "Study of information flow and transformation in dynamic systems.",
         ["information_fields", "algorithmic_information"],
         {"information_fields": 0.6, "algorithmic_information": 0.4}, 3),

        # ========== LEVEL 4: CORE DFT CONCEPTS ==========
        ("symbolic_collapse", "Symbolic Collapse",
         "Entropy-regulated reduction of symbolic field complexity. Π(x,y) triggers pruning.",
         ["symbolic_entropy"], {"symbolic_entropy": 1.0}, 4),

        ("phi_constant", "φ (Golden Ratio Constant)",
         "φ ≈ 1.618. Fundamental to PAC scaling: Ψ(k) = A·φ^(-k).",
         ["pac_conservation"], {"pac_conservation": 1.0}, 4),

        ("xi_constant", "Ξ (Xi Balance Operator)",
         "Ξ ≈ 1.0571. Threshold for collapse triggering. Ξ = 1 + π/F₁₀.",
         ["pac_conservation"], {"pac_conservation": 1.0}, 4),

        ("lambda_star", "λ* (Optimal Decay Constant)",
         "λ* ≈ 0.9816. Optimal temporal decay rate for PAC systems.",
         ["pac_conservation"], {"pac_conservation": 1.0}, 4),

        ("duty_cycle", "Duty Cycle (PAC Equilibrium)",
         "φ/(φ+1) ≈ 0.618 (61.8%). Natural equilibrium ratio in PAC systems.",
         ["phi_constant"], {"phi_constant": 1.0}, 5),

        # ========== LEVEL 5: OPERATORS & DYNAMICS ==========
        ("collapse_pressure", "Collapse Pressure Operator Π",
         "Π(x,y) = α|∇²f| + β||∇f||. Quantifies local symbolic tension.",
         ["symbolic_collapse"], {"symbolic_collapse": 1.0}, 5),

        ("entropy_modulation", "Entropy Modulation Operator γ",
         "γ(x,y) governs symbolic resistance to collapse based on local entropy.",
         ["symbolic_collapse"], {"symbolic_collapse": 1.0}, 5),

        ("symbolic_drift", "Symbolic Drift",
         "Probabilistic transfer of symbols to adjacent lower-gradient sites.",
         ["symbolic_collapse"], {"symbolic_collapse": 1.0}, 5),

        ("rbf_regulation", "RBF (Recursive Balance Field)",
         "Modulates collapse by deviation from entropy equilibrium.",
         ["symbolic_collapse"], {"symbolic_collapse": 1.0}, 5),

        # ========== LEVEL 5-6: SEC FRAMEWORK ==========
        ("sec", "SEC (Symbolic Entropy Collapse)",
         "Complete framework: collapse operators (Π, γ), drift, recursive balance.",
         ["symbolic_collapse", "pac_conservation"],
         {"symbolic_collapse": 0.7, "pac_conservation": 0.3}, 5),

        ("sec_navier_stokes", "SEC-Navier-Stokes Equivalence",
         "SEC dynamics equivalent to Navier-Stokes under certain boundary conditions.",
         ["sec"], {"sec": 1.0}, 6),

        ("sec_validation_quantum", "SEC Quantum Alignment",
         "SEC symbolic collapse aligns with quantum measurement collapse phenomenology.",
         ["sec", "measurement_problem"],
         {"sec": 0.7, "measurement_problem": 0.3}, 6),

        # ========== LEVEL 6-7: MED & BOUNDS ==========
        ("med", "MED (Macro Emergence Dynamics)",
         "Framework for macro patterns from recursive collapse. Bounds: depth≤1, nodes≤3.",
         ["sec"], {"sec": 1.0}, 6),

        ("med_depth_bound", "MED Depth Bound (max_depth=1)",
         "Universal bound: recursive symbolic structures collapse beyond depth 1.",
         ["med"], {"med": 1.0}, 7),

        ("med_node_bound", "MED Node Bound (max_nodes=3)",
         "Universal bound: macro patterns emerge with ≤3 nodes at each level.",
         ["med"], {"med": 1.0}, 7),

        ("complexity_bound", "Bounded Complexity Theorem",
         "Total complexity remains bounded through recursive pruning despite local amplification.",
         ["med"], {"med": 1.0}, 7),

        # ========== LEVEL 6: ADVANCED CONCEPTS ==========
        ("bifractal_time", "Bifractal Time",
         "Dual-timescale emergence from recursive collapse dynamics.",
         ["sec"], {"sec": 1.0}, 6),

        ("pi_harmonics", "π Harmonics",
         "Harmonic oscillations based on π appearing in collapse geometry.",
         ["sec"], {"sec": 1.0}, 6),

        ("feigenbaum_constant", "Feigenbaum Constants in DFT",
         "Period-doubling route to chaos appearing in recursive symbolic systems.",
         ["sec"], {"sec": 1.0}, 6),

        ("hodge_mapping", "Hodge Mapping (Symbolic)",
         "Symbolic field-theoretic approach to algebraic geometry.",
         ["symbolic_entropy"], {"symbolic_entropy": 1.0}, 4),

        ("landauer_field", "Landauer Erasure in Field Dynamics",
         "Energy cost of information erasure mapped to field dissipation.",
         ["information_entropy_equivalence", "sec"],
         {"information_entropy_equivalence": 0.5, "sec": 0.5}, 6),

        # ========== LEVEL 4-7: APPLICATIONS & IMPLEMENTATIONS ==========
        ("kronos_memory", "KRONOS v2",
         "PAC-conserving knowledge graph. Identity IS confluence pattern.",
         ["pac_conservation"], {"pac_conservation": 1.0}, 4),

        ("fracton_language", "Fracton Language",
         "Python DSL for PAC-native computation. Field states as first-class constructs.",
         ["sec", "pac_conservation"],
         {"sec": 0.6, "pac_conservation": 0.4}, 7),

        ("cimm", "CIMM (Collapse-Induced Modular Machines)",
         "Neural architectures where modularity emerges from SEC pruning.",
         ["sec", "med"], {"sec": 0.7, "med": 0.3}, 7),

        ("grimm", "Grimm (Field-Based Agent)",
         "Persistent AI personality using field dynamics and KRONOS memory.",
         ["kronos_memory", "sec"],
         {"kronos_memory": 0.6, "sec": 0.4}, 5),

        ("tinycimm", "TinyCIMM",
         "Experimental validation of CIMM on tiny neural architectures.",
         ["cimm"], {"cimm": 1.0}, 8),

        ("gaia_framework", "GAIA (General Agentic Integration Architecture)",
         "Multi-agent orchestration using field dynamics.",
         ["grimm"], {"grimm": 1.0}, 6),

        # ========== EXPERIMENTAL VALIDATIONS ==========
        ("quantum_eraser_alignment", "Quantum Eraser Alignment",
         "DFT interpretation of delayed-choice quantum eraser experiments.",
         ["sec_validation_quantum"], {"sec_validation_quantum": 1.0}, 7),

        ("weak_measurement_alignment", "Weak Measurement Alignment",
         "DFT symbolic collapse matches weak measurement phenomenology.",
         ["sec_validation_quantum"], {"sec_validation_quantum": 1.0}, 7),

        ("evolution_symbolic_collapse", "Evolution as Symbolic Collapse",
         "Biological evolution modeled as entropy-regulated symbolic pruning.",
         ["sec"], {"sec": 1.0}, 6),

        ("dna_repair_entropy", "DNA Repair via Entropy Minimization",
         "Experimental: Using entropy landscapes to detect and repair mutations.",
         ["evolution_symbolic_collapse"], {"evolution_symbolic_collapse": 1.0}, 7),

        # ========== THEORETICAL BRIDGES ==========
        ("gradient_descent_bridge", "Gradient Descent as SEC",
         "Neural network training interpreted as symbolic entropy collapse.",
         ["sec", "cimm"], {"sec": 0.6, "cimm": 0.4}, 7),

        ("continual_learning_bridge", "Continual Learning via RBF",
         "Catastrophic forgetting prevented by recursive balance fields.",
         ["rbf_regulation", "cimm"],
         {"rbf_regulation": 0.5, "cimm": 0.5}, 7),

        ("superfluid_crystallization", "Superfluid Informational Crystallization",
         "Physical superfluids as limiting case of zero-entropy information fields.",
         ["information_fields", "thermodynamics"],
         {"information_fields": 0.6, "thermodynamics": 0.4}, 3),

        ("herniation_hypothesis", "Reality as Herniation Precipitate",
         "Philosophical: Physical reality as crystallized defects in information substrate.",
         ["superfluid_crystallization"], {"superfluid_crystallization": 1.0}, 4),

        # ========== TECHNICAL IMPLEMENTATIONS ==========
        ("mobius_tensor", "Möbius Tensor",
         "Tensor structure encoding topological twists in symbolic fields.",
         ["fracton_language"], {"fracton_language": 1.0}, 8),

        ("gpu_acceleration", "GPU-Accelerated Memory Fields",
         "CUDA/PyTorch optimization for large-scale PAC computations.",
         ["fracton_language"], {"fracton_language": 1.0}, 8),

        ("fdo_format", "FDO (Fractal Document Object)",
         "Hierarchical document format preserving PAC structure.",
         ["kronos_memory"], {"kronos_memory": 1.0}, 5),

        ("scbf_monitoring", "SCBF (Symbolic Collapse Boundary Flags)",
         "Real-time monitoring system for detecting collapse anomalies.",
         ["sec"], {"sec": 1.0}, 6),

        # ========== ADDITIONAL MATHEMATICAL FOUNDATIONS ==========
        ("recursive_sequences", "Recursive Sequences",
         "Sequences defined by recurrence relations. Foundation for PAC dynamics.",
         ["mathematics"], {"mathematics": 1.0}, 1),

        ("lucas_numbers", "Lucas Numbers",
         "L(n) = L(n-1) + L(n-2), L(0)=2, L(1)=1. Related to Fibonacci, same ratio φ.",
         ["fibonacci_sequence"], {"fibonacci_sequence": 1.0}, 2),

        ("continued_fractions", "Continued Fractions",
         "Representation of numbers as nested fractions. φ has simplest continued fraction: [1;1,1,1,...]",
         ["golden_ratio_math"], {"golden_ratio_math": 1.0}, 3),

        ("metallic_ratios", "Metallic Ratios",
         "Family including golden (φ), silver (δ_S), bronze ratios. All appear in recursive systems.",
         ["golden_ratio_math"], {"golden_ratio_math": 1.0}, 3),

        ("pisot_numbers", "Pisot Numbers",
         "Algebraic integers >1 where all conjugates <1. φ is smallest. Related to quasicrystals.",
         ["golden_ratio_math"], {"golden_ratio_math": 1.0}, 3),

        # ========== INFORMATION THEORY EXTENSIONS ==========
        ("mutual_information", "Mutual Information",
         "I(X;Y) measures shared information between variables. Foundation for correlation detection.",
         ["shannon_entropy"], {"shannon_entropy": 1.0}, 2),

        ("transfer_entropy", "Transfer Entropy",
         "Directional information flow between processes. Used in causality detection.",
         ["mutual_information"], {"mutual_information": 1.0}, 3),

        ("fisher_information", "Fisher Information",
         "Measures information a random variable carries about a parameter.",
         ["information_theory"], {"information_theory": 1.0}, 1),

        ("information_geometry", "Information Geometry",
         "Geometric structure on probability distributions. Connects information theory to differential geometry.",
         ["fisher_information", "shannon_entropy"],
         {"fisher_information": 0.5, "shannon_entropy": 0.5}, 2),

        # ========== QUANTUM MECHANICS EXTENSIONS ==========
        ("quantum_entanglement", "Quantum Entanglement",
         "Non-local correlations between quantum systems. Cannot be explained by classical physics.",
         ["wave_function"], {"wave_function": 1.0}, 2),

        ("decoherence", "Quantum Decoherence",
         "Loss of quantum coherence through environmental interaction. Explains classical emergence.",
         ["measurement_problem"], {"measurement_problem": 1.0}, 3),

        ("von_neumann_entropy", "Von Neumann Entropy",
         "S(ρ) = -Tr(ρ log ρ). Quantum generalization of Shannon entropy.",
         ["shannon_entropy", "wave_function"],
         {"shannon_entropy": 0.5, "wave_function": 0.5}, 2),

        ("quantum_darwinism", "Quantum Darwinism",
         "Redundant information about system spreads to environment. Explains objective reality emergence.",
         ["decoherence"], {"decoherence": 1.0}, 4),

        ("quantum_darwinism_alignment", "Quantum Darwinism DFT Alignment",
         "DFT symbolic redundancy matches quantum Darwinism information spreading.",
         ["quantum_darwinism", "sec"],
         {"quantum_darwinism": 0.6, "sec": 0.4}, 5),

        # ========== THERMODYNAMICS EXTENSIONS ==========
        ("maximum_entropy", "Maximum Entropy Principle",
         "Subject to constraints, distribution with maximum entropy is most likely.",
         ["statistical_mechanics"], {"statistical_mechanics": 1.0}, 2),

        ("fluctuation_theorem", "Fluctuation Theorem",
         "Quantifies probability of entropy-decreasing fluctuations in small systems.",
         ["statistical_mechanics"], {"statistical_mechanics": 1.0}, 2),

        ("jarzynski_equality", "Jarzynski Equality",
         "Relates non-equilibrium work to equilibrium free energy difference.",
         ["fluctuation_theorem"], {"fluctuation_theorem": 1.0}, 3),

        ("crooks_theorem", "Crooks Fluctuation Theorem",
         "Detailed fluctuation theorem relating forward/reverse process probabilities.",
         ["fluctuation_theorem"], {"fluctuation_theorem": 1.0}, 3),

        # ========== DFT EXPERIMENTAL VALIDATIONS ==========
        ("delayed_choice_alignment", "Wheeler Delayed Choice Alignment",
         "DFT symbolic collapse matches delayed-choice experiment phenomenology.",
         ["sec_validation_quantum"], {"sec_validation_quantum": 1.0}, 7),

        ("flux_qubit_alignment", "Flux Qubit Alignment",
         "DFT predictions match flux qubit decoherence patterns.",
         ["sec_validation_quantum"], {"sec_validation_quantum": 1.0}, 7),

        ("mipt_alignment", "MIPT (Measurement-Induced Phase Transition) Alignment",
         "DFT collapse dynamics match measurement-induced phase transitions.",
         ["sec_validation_quantum"], {"sec_validation_quantum": 1.0}, 7),

        ("neural_collapse_validation", "Neural Collapse Validation",
         "TinyCIMM exhibits neural collapse phenomenon predicted by SEC.",
         ["tinycimm"], {"tinycimm": 1.0}, 9),

        # ========== FIELD THEORY EXTENSIONS ==========
        ("gauge_theory", "Gauge Theory",
         "Field theories with local symmetry. Foundation of Standard Model.",
         ["field_theory"], {"field_theory": 1.0}, 1),

        ("yang_mills", "Yang-Mills Theory",
         "Non-abelian gauge theory. Describes strong and weak forces.",
         ["gauge_theory"], {"gauge_theory": 1.0}, 2),

        ("higgs_mechanism", "Higgs Mechanism",
         "Spontaneous symmetry breaking gives mass to particles.",
         ["gauge_theory"], {"gauge_theory": 1.0}, 2),

        ("effective_field_theory", "Effective Field Theory",
         "Low-energy approximation of fundamental theory. Scale-dependent description.",
         ["field_theory"], {"field_theory": 1.0}, 1),

        # ========== COMPLEXITY & EMERGENCE ==========
        ("emergence", "Emergence",
         "Higher-level patterns arising from lower-level interactions. Cannot be reduced to components.",
         ["statistical_mechanics"], {"statistical_mechanics": 1.0}, 2),

        ("self_organization", "Self-Organization",
         "Spontaneous pattern formation in open systems far from equilibrium.",
         ["emergence"], {"emergence": 1.0}, 3),

        ("criticality", "Criticality & Phase Transitions",
         "Systems at critical points exhibit scale-free behavior. Power-law distributions.",
         ["emergence"], {"emergence": 1.0}, 3),

        ("renormalization_group", "Renormalization Group",
         "How physical systems change with scale. Explains universality in phase transitions.",
         ["criticality"], {"criticality": 1.0}, 4),

        ("soc", "Self-Organized Criticality",
         "Systems naturally evolve toward critical state. Explains 1/f noise, fractals.",
         ["criticality"], {"criticality": 1.0}, 4),

        # ========== DFT-PHYSICS BRIDGES ==========
        ("sec_renormalization_bridge", "SEC as Renormalization Flow",
         "Symbolic collapse interpreted as information-theoretic renormalization.",
         ["sec", "renormalization_group"],
         {"sec": 0.6, "renormalization_group": 0.4}, 7),

        ("pac_gauge_symmetry", "PAC as Gauge Symmetry",
         "PAC conservation as emergent gauge symmetry in information space.",
         ["pac_conservation", "gauge_theory"],
         {"pac_conservation": 0.7, "gauge_theory": 0.3}, 4),

        # ========== COMPUTATIONAL COMPLEXITY ==========
        ("computational_complexity", "Computational Complexity",
         "Resources required to solve problems. P, NP, PSPACE hierarchy.",
         ["algorithmic_information"], {"algorithmic_information": 1.0}, 3),

        ("p_vs_np", "P vs NP Problem",
         "Is every problem whose solution can be verified quickly also solvable quickly?",
         ["computational_complexity"], {"computational_complexity": 1.0}, 4),

        ("circuit_complexity", "Circuit Complexity",
         "Minimum circuit size to compute function. Related to Kolmogorov complexity.",
         ["computational_complexity", "kolmogorov_complexity"],
         {"computational_complexity": 0.6, "kolmogorov_complexity": 0.4}, 4),

        # ========== BIOLOGICAL APPLICATIONS ==========
        ("protein_folding", "Protein Folding",
         "How amino acid sequence determines 3D structure. Entropy-driven process.",
         ["thermodynamics"], {"thermodynamics": 1.0}, 1),

        ("protein_folding_sec", "Protein Folding via SEC",
         "Protein folding as entropy-guided symbolic collapse in amino acid configuration space.",
         ["protein_folding", "sec"],
         {"protein_folding": 0.5, "sec": 0.5}, 6),

        ("morphogenesis", "Morphogenesis",
         "Biological form development. Pattern formation from genetic/chemical fields.",
         ["self_organization"], {"self_organization": 1.0}, 4),

        ("morphogenesis_sec", "Morphogenesis as Field Collapse",
         "Developmental biology interpreted through SEC lens. Form from entropy regulation.",
         ["morphogenesis", "sec"],
         {"morphogenesis": 0.5, "sec": 0.5}, 6),

        # ========== COSMOLOGY & GRAVITY ==========
        ("general_relativity", "General Relativity",
         "Gravity as spacetime curvature. Geometric theory of gravitation.",
         ["field_theory"], {"field_theory": 1.0}, 1),

        ("black_holes", "Black Holes",
         "Regions where spacetime curvature prevents escape. Information paradox.",
         ["general_relativity"], {"general_relativity": 1.0}, 2),

        ("hawking_radiation", "Hawking Radiation",
         "Black holes emit thermal radiation. Connects GR, QM, and thermodynamics.",
         ["black_holes", "quantum_mechanics"],
         {"black_holes": 0.6, "quantum_mechanics": 0.4}, 3),

        ("holographic_principle", "Holographic Principle",
         "Information content of region bounded by surface area, not volume.",
         ["black_holes"], {"black_holes": 1.0}, 3),

        ("ads_cft", "AdS/CFT Correspondence",
         "Duality between gravity in AdS space and CFT on boundary. Holography realized.",
         ["holographic_principle"], {"holographic_principle": 1.0}, 4),

        ("gravity_as_entropic_force", "Gravity as Entropic Force",
         "Verlinde's proposal: gravity emerges from information/entropy considerations.",
         ["holographic_principle", "thermodynamics"],
         {"holographic_principle": 0.6, "thermodynamics": 0.4}, 4),

        ("gravity_dft_bridge", "Gravity-DFT Bridge",
         "Entropic gravity interpretation through DFT information field dynamics.",
         ["gravity_as_entropic_force", "infodynamics"],
         {"gravity_as_entropic_force": 0.6, "infodynamics": 0.4}, 5),

        # ========== FRACTAL & CHAOTIC SYSTEMS ==========
        ("fractals", "Fractals",
         "Self-similar structures across scales. Fractional dimension.",
         ["mathematics"], {"mathematics": 1.0}, 1),

        ("mandelbrot_set", "Mandelbrot Set",
         "Iconic fractal from complex dynamics. Boundary exhibits infinite detail.",
         ["fractals"], {"fractals": 1.0}, 2),

        ("chaos_theory", "Chaos Theory",
         "Deterministic systems with sensitive dependence on initial conditions.",
         ["mathematics"], {"mathematics": 1.0}, 1),

        ("lyapunov_exponents", "Lyapunov Exponents",
         "Quantify rate of separation of infinitesimally close trajectories.",
         ["chaos_theory"], {"chaos_theory": 1.0}, 2),

        ("strange_attractors", "Strange Attractors",
         "Fractal structures in phase space of chaotic systems.",
         ["chaos_theory", "fractals"],
         {"chaos_theory": 0.6, "fractals": 0.4}, 2),

        ("bifurcation_theory", "Bifurcation Theory",
         "Study of qualitative changes in system behavior as parameters vary.",
         ["chaos_theory"], {"chaos_theory": 1.0}, 2),

        ("feigenbaum_universality", "Feigenbaum Universality",
         "Universal constants (δ ≈ 4.669) in period-doubling route to chaos.",
         ["bifurcation_theory"], {"bifurcation_theory": 1.0}, 3),
    ]

    # Build nodes
    for node_id, name, definition, parents, confluence, depth in nodes:
        if depth == 0:
            derivation_path = [node_id]
        elif len(parents) == 1:
            derivation_path = [parents[0], node_id]
        else:
            # Multiple parents - use primary
            primary = max(confluence.items(), key=lambda x: x[1])[0] if confluence else parents[0]
            derivation_path = [primary, node_id]

        node = KronosNode(
            id=node_id,
            name=name,
            definition=definition,
            parent_potentials=parents,
            derivation_path=derivation_path,
            actualization_depth=depth,
            confluence_pattern=confluence,
            first_crystallization=datetime(2024, 1, 1)
        )
        graph.add_node(node)

    # Add important relationships
    edges = [
        # Temporal enablement
        ("information_theory", "shannon_entropy", RelationType.ENABLES),
        ("shannon_entropy", "information_entropy_equivalence", RelationType.ENABLES),
        ("pac_conservation", "kronos_memory", RelationType.ENABLES),
        ("sec", "cimm", RelationType.ENABLES),
        ("sec", "med", RelationType.ENABLES),

        # Extensions
        ("symbolic_collapse", "sec", RelationType.EXTENDS),
        ("sec", "med", RelationType.EXTENDS),
        ("cimm", "tinycimm", RelationType.EXTENDS),

        # Supports (experimental validation)
        ("sec_validation_quantum", "sec", RelationType.SUPPORTS),
        ("quantum_eraser_alignment", "sec_validation_quantum", RelationType.SUPPORTS),
        ("weak_measurement_alignment", "sec_validation_quantum", RelationType.SUPPORTS),
    ]

    for source, target, rel_type in edges:
        edge = KronosEdge(
            source_id=source,
            target_id=target,
            relationship_type=rel_type,
            strength=1.0
        )
        graph.add_edge(edge)

    return graph


def save_graph(graph: KronosGraph, output_path: Path):
    """Save graph to JSON."""
    graph_dict = {
        "nodes": {},
        "edges": [],
        "roots": graph.roots
    }

    for node_id, node in graph.nodes.items():
        graph_dict["nodes"][node_id] = {
            "id": node.id,
            "name": node.name,
            "definition": node.definition,
            "parent_potentials": node.parent_potentials,
            "child_actualizations": node.child_actualizations,
            "sibling_nodes": node.sibling_nodes,
            "derivation_path": node.derivation_path,
            "actualization_depth": node.actualization_depth,
            "confluence_pattern": node.confluence_pattern,
            "first_crystallization": node.first_crystallization.isoformat() if node.first_crystallization else None,
        }

    for edge in graph.edges:
        graph_dict["edges"].append({
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "relationship_type": edge.relationship_type.value,
            "strength": edge.strength,
        })

    with open(output_path, 'w') as f:
        json.dump(graph_dict, f, indent=2)

    print(f"[OK] Saved graph with {len(graph.nodes)} nodes to {output_path}")


if __name__ == "__main__":
    print("Building expanded Dawn Field Theory knowledge graph...")
    graph = build_dft_knowledge_graph()

    # Save
    output_path = Path(__file__).parent.parent / "data" / "dft_knowledge_graph.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    save_graph(graph, output_path)

    # Statistics
    print(f"\n[STATS] Graph Statistics:")
    print(f"  Total nodes: {len(graph.nodes)}")
    print(f"  Root nodes: {len(graph.roots)}")
    print(f"  Total edges: {len(graph.edges)}")

    # Depth distribution
    depth_counts = {}
    for node in graph.nodes.values():
        depth = node.actualization_depth
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    print(f"\n[DEPTH] Depth Distribution:")
    for depth in sorted(depth_counts.keys()):
        print(f"  Depth {depth}: {depth_counts[depth]} nodes")

    # Show categories
    print(f"\n[CATEGORIES]")
    roots = [n for n in graph.nodes.values() if n.is_root]
    leaves = [n for n in graph.nodes.values() if n.is_leaf]
    confluence = [n for n in graph.nodes.values() if len(n.parent_potentials) > 1]

    print(f"  Roots: {len(roots)}")
    print(f"  Leaves (applications): {len(leaves)}")
    print(f"  Confluence nodes: {len(confluence)}")

    print("\n[DONE] Expanded DFT knowledge graph ready!")
