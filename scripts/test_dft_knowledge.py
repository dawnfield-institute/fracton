"""
Test querying the DFT knowledge graph with KRONOS
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from fracton.storage.graph import KronosGraph
from fracton.storage.node import KronosNode


def load_graph(graph_path: Path) -> KronosGraph:
    """Load graph from JSON."""
    with open(graph_path, 'r') as f:
        data = json.load(f)

    graph = KronosGraph()

    # Load nodes
    for node_data in data["nodes"].values():
        node = KronosNode(
            id=node_data["id"],
            name=node_data["name"],
            definition=node_data["definition"],
            parent_potentials=node_data["parent_potentials"],
            derivation_path=node_data["derivation_path"],
            actualization_depth=node_data["actualization_depth"],
            confluence_pattern=node_data["confluence_pattern"],
            first_crystallization=datetime.fromisoformat(node_data["first_crystallization"]) if node_data["first_crystallization"] else None
        )
        # Restore child actualizations (populated during graph.add_node)
        # but we need to set them manually for loaded graphs
        node.child_actualizations = node_data["child_actualizations"]
        graph.nodes[node.id] = node

    # Restore roots
    graph.roots = data["roots"]

    return graph


def query_concept(graph: KronosGraph, concept_id: str):
    """Query a concept and show what KRONOS knows."""
    print(f"\n{'='*70}")
    print(f"QUERY: {concept_id}")
    print(f"{'='*70}")

    node = graph.get_node(concept_id)
    if not node:
        print(f"  [NOT FOUND] No such concept: {concept_id}")
        return

    # Basic info
    print(f"\n[NAME] {node.name}")
    print(f"[DEFINITION] {node.definition}")
    print(f"\n[LINEAGE]")
    print(f"  Depth: {node.actualization_depth}")
    print(f"  Path: {' > '.join(node.derivation_path)}")

    # Parents (what this crystallized from)
    if node.parent_potentials:
        print(f"\n[CRYSTALLIZED FROM]")
        for parent_id in node.parent_potentials:
            weight = node.confluence_pattern.get(parent_id, 0.0)
            parent = graph.get_node(parent_id)
            if parent:
                print(f"  - {parent.name} (weight: {weight:.2f})")

    # Children (what crystallized from this)
    if node.child_actualizations:
        print(f"\n[ENABLED]")
        for child_id in node.child_actualizations:
            child = graph.get_node(child_id)
            if child:
                print(f"  - {child.name}")

    # Siblings (alternative actualizations)
    if node.sibling_nodes:
        print(f"\n[SIBLINGS]")
        for sib_id in node.sibling_nodes:
            sib = graph.get_node(sib_id)
            if sib:
                print(f"  - {sib.name}")

    # Simple topology metrics
    print(f"\n[TOPOLOGY]")
    print(f"  Parents: {len(node.parent_potentials)}")
    print(f"  Children: {len(node.child_actualizations)}")
    print(f"  Siblings: {len(node.sibling_nodes)}")
    print(f"  Is Root: {node.is_root}")
    print(f"  Is Leaf: {node.is_leaf}")


if __name__ == "__main__":
    # Load graph
    graph_path = Path(__file__).parent.parent / "data" / "dft_knowledge_graph.json"
    print(f"Loading graph from {graph_path}...")
    graph = load_graph(graph_path)
    print(f"Loaded {len(graph.nodes)} nodes")

    # Test queries
    test_concepts = [
        "pac_conservation",
        "sec",
        "symbolic_collapse",
        "kronos_memory",
        "grimm",
        "cimm",
        "information_entropy_equivalence",
    ]

    for concept in test_concepts:
        query_concept(graph, concept)

    print(f"\n{'='*70}")
    print("[DONE] Knowledge graph test complete!")
    print(f"{'='*70}\n")
