"""
Build Dawn Field Theory Knowledge Graph for KRONOS v2 (Simplified)

Creates a proper conceptual genealogy tree with PAC conservation.
Uses simple string references for documents (can enhance later).
"""

from datetime import datetime
from pathlib import Path
import json

from fracton.storage.node import KronosNode
from fracton.storage.edge import RelationType, KronosEdge
from fracton.storage.graph import KronosGraph


def build_dft_knowledge_graph() -> KronosGraph:
    """
    Build Dawn Field Theory conceptual genealogy.

    ~30 core concepts arranged in proper PAC hierarchy.
    """
    graph = KronosGraph()

    # ========== LEVEL 0: ROOT POTENTIALS ==========

    # Root nodes have empty parent_potentials and confluence_pattern
    nodes = [
        # Roots
        ("information_theory", "Information Theory",
         "Mathematical framework for quantifying information. Foundation for entropy and communication theory.",
         [], {}, 0),

        ("field_theory", "Field Theory",
         "Framework describing physical fields as functions over spacetime.",
         [], {}, 0),

        ("thermodynamics", "Thermodynamics",
         "Study of heat, work, energy, and entropy in physical systems.",
         [], {}, 0),

        # Level 1 - Direct children of roots
        ("shannon_entropy", "Shannon Entropy",
         "H(X) = -Σ p(x)log p(x). Quantifies average information in a random variable.",
         ["information_theory"], {"information_theory": 1.0}, 1),

        ("statistical_mechanics", "Statistical Mechanics",
         "Bridges microscopic particle behavior and macroscopic thermodynamics.",
         ["thermodynamics"], {"thermodynamics": 1.0}, 1),

        ("potential_fields", "Potential Fields",
         "Scalar/vector fields representing potential energy. Gradients yield forces.",
         ["field_theory"], {"field_theory": 1.0}, 1),

        # Level 2 - Confluence concepts
        ("information_entropy_equivalence", "Information-Entropy Equivalence",
         "Deep connection between Shannon entropy and thermodynamic entropy. Landauer's principle.",
         ["shannon_entropy", "statistical_mechanics"],
         {"shannon_entropy": 0.5, "statistical_mechanics": 0.5}, 2),

        ("information_fields", "Information Fields",
         "Fields carrying information content with spatial density variation.",
         ["potential_fields", "shannon_entropy"],
         {"potential_fields": 0.6, "shannon_entropy": 0.4}, 2),

        # Level 3 - DFT Foundations
        ("symbolic_entropy", "Symbolic Entropy",
         "Entropy measured over discrete symbolic fields. Foundation for symbolic collapse.",
         ["information_fields"], {"information_fields": 1.0}, 3),

        ("pac_conservation", "PAC Conservation",
         "Fundamental principle: Ψ(k) = Ψ(k+1) + Ψ(k+2). Parent potential equals sum of children.",
         ["information_entropy_equivalence"], {"information_entropy_equivalence": 1.0}, 3),

        # Level 4 - Core DFT
        ("symbolic_collapse", "Symbolic Collapse",
         "Entropy-regulated reduction of symbolic field complexity. Π(x,y) triggers pruning.",
         ["symbolic_entropy"], {"symbolic_entropy": 1.0}, 4),

        ("phi_golden_ratio", "φ (Golden Ratio)",
         "φ ≈ 1.618. Fundamental to PAC scaling: Ψ(k) = A·φ^(-k).",
         ["pac_conservation"], {"pac_conservation": 1.0}, 4),

        ("xi_balance_operator", "Ξ (Xi Balance Operator)",
         "Ξ ≈ 1.0571. Threshold for collapse triggering.",
         ["pac_conservation"], {"pac_conservation": 1.0}, 4),

        # Level 5 - Operators & SEC
        ("collapse_pressure_operator", "Collapse Pressure Π(x,y)",
         "Π = α|∇²f| + β||∇f||. Quantifies local symbolic tension.",
         ["symbolic_collapse"], {"symbolic_collapse": 1.0}, 5),

        ("entropy_modulation_operator", "Entropy Modulation γ(x,y)",
         "γ governs symbolic resistance to collapse based on local entropy.",
         ["symbolic_collapse"], {"symbolic_collapse": 1.0}, 5),

        ("sec", "SEC (Symbolic Entropy Collapse)",
         "Complete framework: collapse operators (Π, γ), drift, recursive balance.",
         ["symbolic_collapse", "pac_conservation"],
         {"symbolic_collapse": 0.7, "pac_conservation": 0.3}, 5),

        # Level 6 - MED & Applications
        ("med", "MED (Macro Emergence Dynamics)",
         "Framework for macro patterns from recursive collapse. Bounds: depth≤1, nodes≤3.",
         ["sec"], {"sec": 1.0}, 6),

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

        # Additional important concepts
        ("rbf_regulation", "RBF (Recursive Balance Field)",
         "Modulates collapse by deviation from entropy equilibrium.",
         ["symbolic_collapse"], {"symbolic_collapse": 1.0}, 5),

        ("bifractal_time", "Bifractal Time",
         "Dual-timescale emergence from recursive collapse dynamics.",
         ["sec"], {"sec": 1.0}, 6),

        ("hodge_mapping", "Hodge Mapping",
         "Symbolic field-theoretic approach to algebraic geometry.",
         ["symbolic_entropy"], {"symbolic_entropy": 1.0}, 4),

        ("navier_stokes_sec", "Navier-Stokes SEC Equivalence",
         "SEC dynamics equivalent to Navier-Stokes under certain conditions.",
         ["sec"], {"sec": 1.0}, 6),

        ("quantum_collapse_alignment", "Quantum Measurement Alignment",
         "DFT symbolic collapse aligns with quantum measurement phenomena.",
         ["symbolic_collapse"], {"symbolic_collapse": 1.0}, 5),
    ]

    # Build nodes
    for node_id, name, definition, parents, confluence, depth in nodes:
        # Build derivation path
        if depth == 0:
            derivation_path = [node_id]
        elif len(parents) == 1:
            # Single parent - extend their path
            parent_node = None
            for n_id, n_name, _, n_parents, _, n_depth in nodes:
                if n_id == parents[0]:
                    if n_depth == 0:
                        parent_path = [n_id]
                    else:
                        parent_path = []  # Will be filled when we process parent
                    break
            derivation_path = (parent_path if parent_path else [parents[0]]) + [node_id]
        else:
            # Multiple parents - use primary parent's path
            primary = max(confluence.items(), key=lambda x: x[1])[0]
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

    # Add edges for important relationships
    edges = [
        # Temporal enablement
        ("information_theory", "shannon_entropy", RelationType.ENABLES),
        ("shannon_entropy", "information_entropy_equivalence", RelationType.ENABLES),
        ("pac_conservation", "kronos_memory", RelationType.ENABLES),
        ("sec", "cimm", RelationType.ENABLES),

        # Extensions
        ("symbolic_collapse", "sec", RelationType.EXTENDS),
        ("sec", "med", RelationType.EXTENDS),
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
    # Manual serialization
    graph_dict = {
        "nodes": {},
        "edges": [],
        "roots": graph.roots
    }

    # Serialize nodes
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

    # Serialize edges
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
    print("Building Dawn Field Theory knowledge graph...")
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

    # Show some example lineages
    print(f"\n[LINEAGE] Example Lineages:")
    examples = ["grimm", "cimm", "sec", "kronos_memory"]
    for ex_id in examples:
        if ex_id in graph.nodes:
            node = graph.nodes[ex_id]
            path = " > ".join(node.derivation_path)
            print(f"  {node.name}: {path}")

    print("\n[DONE] Dawn Field Theory knowledge graph ready for KRONOS!")
