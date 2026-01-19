"""Quick test of expanded knowledge graph"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from test_dft_knowledge import load_graph, query_concept

graph_path = Path(__file__).parent.parent / "data" / "dft_knowledge_graph.json"
graph = load_graph(graph_path)

print(f"Testing {len(graph.nodes)}-node knowledge graph\n")

# Test new concepts from expansion
test_concepts = [
    "quantum_darwinism",
    "gravity_dft_bridge",
    "feigenbaum_universality",
    "holographic_principle",
    "protein_folding_sec"
]

for concept_id in test_concepts:
    if concept_id in graph.nodes:
        query_concept(graph, concept_id)
    else:
        print(f"\n[NOT FOUND] {concept_id}")
