"""
KRONOS Knowledge Ingestion Pipeline

Add new concepts to the DFT knowledge graph while maintaining PAC structure.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from fracton.storage.node import KronosNode
from fracton.storage.edge import RelationType, KronosEdge
from fracton.storage.graph import KronosGraph


class ConceptIngestion:
    """Pipeline for adding concepts to KRONOS graph."""

    def __init__(self, graph_path: Path):
        """Initialize with existing graph."""
        self.graph_path = graph_path
        self.graph = self.load_graph()

    def load_graph(self) -> KronosGraph:
        """Load existing graph from JSON."""
        with open(self.graph_path, 'r') as f:
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
            node.child_actualizations = node_data["child_actualizations"]
            graph.nodes[node.id] = node

        graph.roots = data["roots"]

        # Load edges
        for edge_data in data["edges"]:
            edge = KronosEdge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                relationship_type=RelationType(edge_data["relationship_type"]),
                strength=edge_data["strength"]
            )
            graph.edges.append(edge)

        return graph

    def save_graph(self):
        """Save graph back to JSON."""
        graph_dict = {
            "nodes": {},
            "edges": [],
            "roots": self.graph.roots
        }

        for node_id, node in self.graph.nodes.items():
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

        for edge in self.graph.edges:
            graph_dict["edges"].append({
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relationship_type": edge.relationship_type.value,
                "strength": edge.strength,
            })

        with open(self.graph_path, 'w') as f:
            json.dump(graph_dict, f, indent=2)

        print(f"\n[SAVED] Graph updated with {len(self.graph.nodes)} nodes")

    def suggest_parents(self, concept_name: str, definition: str, limit: int = 5) -> List[Tuple[str, float]]:
        """
        Suggest potential parent concepts based on semantic similarity.

        Uses simple keyword matching for now.
        Future: Use embeddings for better matching.
        """
        concept_words = set(concept_name.lower().split()) | set(definition.lower().split())
        scores = []

        for node_id, node in self.graph.nodes.items():
            # Skip leaves (applications) - they shouldn't be parents
            if node.is_leaf:
                continue

            # Keyword overlap score
            node_words = set(node.name.lower().split()) | set(node.definition.lower().split())
            overlap = len(concept_words & node_words)

            # Boost score for foundational concepts
            if node.actualization_depth < 3:
                overlap *= 1.5

            if overlap > 0:
                scores.append((node_id, overlap, node.name, node.actualization_depth))

        # Sort by score, then by depth (prefer less deep)
        scores.sort(key=lambda x: (x[1], -x[2]), reverse=True)

        return [(node_id, score) for node_id, score, _, _ in scores[:limit]]

    def interactive_add_concept(self):
        """Interactive CLI for adding a concept."""
        print("\n" + "="*70)
        print("KRONOS Concept Ingestion")
        print("="*70)
        print(f"Current graph: {len(self.graph.nodes)} nodes\n")

        # Get basic info
        concept_id = input("Concept ID (snake_case): ").strip()
        if not concept_id:
            print("[CANCELLED]")
            return

        if concept_id in self.graph.nodes:
            print(f"[ERROR] Concept '{concept_id}' already exists!")
            return

        name = input("Concept Name: ").strip()
        if not name:
            print("[CANCELLED]")
            return

        definition = input("Definition: ").strip()
        if not definition:
            print("[CANCELLED]")
            return

        # Suggest parents
        print("\n[SUGGESTING PARENTS]")
        suggestions = self.suggest_parents(name, definition, limit=10)

        if not suggestions:
            print("No parent suggestions found. This will be a root concept.")
            print("\nCurrent roots:")
            for root_id in self.graph.roots:
                root = self.graph.nodes[root_id]
                print(f"  - {root.name}")

            is_root = input("\nMake this a root concept? (y/n): ").lower() == 'y'
            if not is_root:
                print("[CANCELLED]")
                return

            parents = []
            confluence = {}
        else:
            print("\nTop parent suggestions:")
            for i, (node_id, score) in enumerate(suggestions, 1):
                node = self.graph.nodes[node_id]
                print(f"  {i}. {node.name} (score: {score:.1f}, depth: {node.actualization_depth})")

            # Select parents
            print("\nSelect parent(s) (comma-separated numbers, or 'none' for root):")
            selection = input("> ").strip()

            if selection.lower() == 'none':
                parents = []
                confluence = {}
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in selection.split(',')]
                    parents = [suggestions[i][0] for i in indices if 0 <= i < len(suggestions)]

                    if not parents:
                        print("[ERROR] No valid parents selected")
                        return

                    # Get confluence weights
                    print(f"\nAssign confluence weights for {len(parents)} parent(s) (must sum to 1.0):")
                    confluence = {}
                    remaining = 1.0

                    for i, parent_id in enumerate(parents[:-1]):
                        parent = self.graph.nodes[parent_id]
                        weight = float(input(f"  {parent.name}: "))
                        confluence[parent_id] = weight
                        remaining -= weight

                    # Last parent gets remaining weight
                    confluence[parents[-1]] = remaining
                    print(f"  {self.graph.nodes[parents[-1]].name}: {remaining:.2f} (auto)")

                    # Validate
                    total = sum(confluence.values())
                    if not (0.95 <= total <= 1.05):
                        print(f"[ERROR] Weights sum to {total:.2f}, must be ~1.0")
                        return

                except (ValueError, IndexError) as e:
                    print(f"[ERROR] Invalid selection: {e}")
                    return

        # Calculate depth
        if not parents:
            depth = 0
        else:
            max_parent_depth = max(self.graph.nodes[p].actualization_depth for p in parents)
            depth = max_parent_depth + 1

        # Build derivation path
        if not parents:
            derivation_path = [concept_id]
        elif len(parents) == 1:
            parent_path = self.graph.nodes[parents[0]].derivation_path
            derivation_path = parent_path + [concept_id]
        else:
            # Multiple parents - use primary (highest confluence)
            primary = max(confluence.items(), key=lambda x: x[1])[0]
            parent_path = self.graph.nodes[primary].derivation_path
            derivation_path = parent_path + [concept_id]

        # Create node
        node = KronosNode(
            id=concept_id,
            name=name,
            definition=definition,
            parent_potentials=parents,
            derivation_path=derivation_path,
            actualization_depth=depth,
            confluence_pattern=confluence,
            first_crystallization=datetime.now()
        )

        # Preview
        print("\n" + "="*70)
        print("PREVIEW")
        print("="*70)
        print(f"ID: {node.id}")
        print(f"Name: {node.name}")
        print(f"Definition: {node.definition}")
        print(f"Depth: {node.actualization_depth}")
        print(f"Path: {' > '.join(node.derivation_path)}")
        if node.parent_potentials:
            print(f"\nParents:")
            for parent_id, weight in confluence.items():
                parent = self.graph.nodes[parent_id]
                print(f"  - {parent.name} ({weight:.2f})")
        print("="*70)

        # Confirm
        confirm = input("\nAdd this concept? (y/n): ").lower()
        if confirm != 'y':
            print("[CANCELLED]")
            return

        # Add to graph
        self.graph.add_node(node)

        # Save
        self.save_graph()

        print(f"\n[SUCCESS] Added '{name}' to knowledge graph!")
        print(f"Total nodes: {len(self.graph.nodes)}")

    def batch_add_concepts(self, concepts_file: Path):
        """
        Add multiple concepts from JSON file.

        Format:
        [
            {
                "id": "concept_id",
                "name": "Concept Name",
                "definition": "Definition here",
                "parents": ["parent1", "parent2"],
                "confluence": {"parent1": 0.6, "parent2": 0.4}
            },
            ...
        ]
        """
        with open(concepts_file, 'r') as f:
            concepts = json.load(f)

        print(f"\n[BATCH INGESTION] Loading {len(concepts)} concepts...")

        added = 0
        for concept_data in concepts:
            try:
                concept_id = concept_data["id"]

                if concept_id in self.graph.nodes:
                    print(f"[SKIP] {concept_id} already exists")
                    continue

                parents = concept_data.get("parents", [])
                confluence = concept_data.get("confluence", {})

                # Calculate depth
                if not parents:
                    depth = 0
                    derivation_path = [concept_id]
                else:
                    max_parent_depth = max(self.graph.nodes[p].actualization_depth for p in parents)
                    depth = max_parent_depth + 1

                    if len(parents) == 1:
                        parent_path = self.graph.nodes[parents[0]].derivation_path
                        derivation_path = parent_path + [concept_id]
                    else:
                        primary = max(confluence.items(), key=lambda x: x[1])[0]
                        parent_path = self.graph.nodes[primary].derivation_path
                        derivation_path = parent_path + [concept_id]

                node = KronosNode(
                    id=concept_id,
                    name=concept_data["name"],
                    definition=concept_data["definition"],
                    parent_potentials=parents,
                    derivation_path=derivation_path,
                    actualization_depth=depth,
                    confluence_pattern=confluence,
                    first_crystallization=datetime.now()
                )

                self.graph.add_node(node)
                added += 1
                print(f"[ADDED] {concept_data['name']}")

            except Exception as e:
                print(f"[ERROR] Failed to add {concept_data.get('id', 'unknown')}: {e}")

        self.save_graph()
        print(f"\n[BATCH COMPLETE] Added {added} new concepts")


def main():
    """Main CLI entry point."""
    graph_path = Path(__file__).parent.parent / "data" / "dft_knowledge_graph.json"

    if not graph_path.exists():
        print(f"[ERROR] Graph not found: {graph_path}")
        return

    ingestion = ConceptIngestion(graph_path)

    # Check for batch file argument
    if len(sys.argv) > 1:
        batch_file = Path(sys.argv[1])
        if batch_file.exists():
            ingestion.batch_add_concepts(batch_file)
        else:
            print(f"[ERROR] Batch file not found: {batch_file}")
    else:
        # Interactive mode
        ingestion.interactive_add_concept()


if __name__ == "__main__":
    main()
