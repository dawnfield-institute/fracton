"""
Build Concept Relationships

Analyze extracted concepts and build parent-child relationships to create
a true genealogical tree structure.

Strategy:
1. Identify fundamental concepts (trunk) - PAC, SEC, Infodynamics, Field Theory
2. Build parent-child relationships via:
   - Explicit references ("builds on", "derived from", "based on")
   - Co-occurrence patterns (concepts mentioned together)
   - Hierarchical structure (H2 → H3 in markdown)
   - Definition analysis (concepts defined in terms of others)
3. Calculate depth (distance from trunk)
4. Visualize the tree structure

Author: Claude (Dawn Field Institute)
Date: 2026-01-19
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


class ConceptRelationshipBuilder:
    """Build parent-child relationships for concepts."""

    def __init__(self, graph_path: Path):
        self.graph_path = graph_path
        self.concepts = {}  # id -> concept data
        self.relationships = defaultdict(list)  # parent_id -> [child_ids]
        self.reverse_relationships = defaultdict(list)  # child_id -> [parent_ids]

        # Fundamental concepts will be discovered dynamically
        # (concepts that are referenced by many others but reference few themselves)
        self.trunk_concepts = set()  # Will be populated during analysis

        # Relationship keywords
        self.parent_indicators = {
            'builds on', 'based on', 'derived from', 'extends',
            'implements', 'uses', 'applies', 'leverages',
            'depends on', 'requires', 'assumes', 'builds upon',
            'is a type of', 'is a kind of', 'inherits from',
            'specializes', 'generalizes'
        }

        # Load graph
        with open(graph_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.concepts = data['nodes']

    def build_relationships(self):
        """Main pipeline to build relationships."""
        print("="*70)
        print("Building Concept Relationships")
        print("="*70)

        # Step 1: Find explicit relationships
        print("\nStep 1: Finding explicit relationships...")
        explicit_rels = self._find_explicit_relationships()
        print(f"  Found {len(explicit_rels)} explicit parent-child links")

        # Step 2: Co-occurrence analysis
        print("\nStep 2: Analyzing co-occurrence patterns...")
        cooccurrence_rels = self._find_cooccurrence_relationships()
        print(f"  Found {len(cooccurrence_rels)} co-occurrence links")

        # Step 3: Term containment (concepts defined using other concepts)
        print("\nStep 3: Analyzing term containment...")
        containment_rels = self._find_containment_relationships()
        print(f"  Found {len(containment_rels)} containment links")

        # Step 4: Merge all relationships
        print("\nStep 4: Merging relationships...")
        all_rels = self._merge_relationships([
            explicit_rels,
            cooccurrence_rels,
            containment_rels
        ])
        print(f"  Total unique relationships: {len(all_rels)}")

        # Step 5: Identify trunk concepts (AFTER building relationships)
        print("\nStep 5: Identifying trunk concepts...")
        trunk_nodes = self._identify_trunk_nodes(all_rels)
        print(f"  Found {len(trunk_nodes)} trunk concepts")

        # Step 6: Build final graph structure
        print("\nStep 6: Building tree structure...")
        self._build_tree_structure(all_rels, trunk_nodes)

        # Step 7: Calculate depths
        print("\nStep 7: Calculating depths from trunk...")
        depths = self._calculate_depths(trunk_nodes)

        # Step 8: Save updated graph
        print("\nStep 8: Saving updated graph...")
        self._save_updated_graph(depths)

        print("\n" + "="*70)
        print("Relationship Building Complete")
        print("="*70)

        return all_rels, depths

    def _identify_trunk_nodes(self, relationships: Dict[str, Set[str]]) -> Set[str]:
        """
        Identify fundamental trunk concepts dynamically.

        Trunk concepts are those that:
        1. Are referenced by many other concepts (high in-degree)
        2. Reference few other concepts themselves (low out-degree)
        3. Have high "fundamentalness" score = in_degree / (out_degree + 1)
        """
        # Calculate in-degree (how many concepts reference this one)
        in_degree = defaultdict(int)
        for parent_id, children in relationships.items():
            for child_id in children:
                in_degree[parent_id] += 1  # parent is referenced by child

        # Calculate out-degree (how many concepts this one references)
        out_degree = defaultdict(int)
        for parent_id, children in relationships.items():
            out_degree[parent_id] = len(children)

        # Calculate fundamentalness score
        fundamentalness = {}
        for concept_id in self.concepts:
            in_deg = in_degree.get(concept_id, 0)
            out_deg = out_degree.get(concept_id, 0)

            # High score = referenced a lot, references little
            if in_deg > 0:
                fundamentalness[concept_id] = in_deg / (out_deg + 1)
            else:
                fundamentalness[concept_id] = 0

        # Sort by fundamentalness
        sorted_concepts = sorted(
            fundamentalness.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Top 10% or concepts with score > 2.0 are trunk
        threshold = max(2.0, sorted_concepts[len(sorted_concepts) // 10][1] if sorted_concepts else 0)
        trunk_nodes = set()

        print(f"\n  Using fundamentalness threshold: {threshold:.2f}")
        print(f"\n  Top fundamental concepts:")

        for concept_id, score in sorted_concepts[:20]:
            if score >= threshold:
                trunk_nodes.add(concept_id)
                in_deg = in_degree.get(concept_id, 0)
                out_deg = out_degree.get(concept_id, 0)
                print(f"    {score:.2f} - {self.concepts[concept_id]['name']}")
                print(f"           (referenced by {in_deg}, references {out_deg})")

        # If no trunk found, pick roots (concepts with no parents)
        if not trunk_nodes:
            print("\n  No high-fundamentalness concepts found, using orphan roots")
            trunk_nodes = {cid for cid in self.concepts if concept_id not in relationships}

        return trunk_nodes

    def _find_explicit_relationships(self) -> List[Tuple[str, str, str]]:
        """Find explicit parent-child relationships from text."""
        relationships = []

        for concept_id, concept in self.concepts.items():
            definition = concept.get('definition', '').lower()

            # Look for relationship keywords
            for indicator in self.parent_indicators:
                if indicator in definition:
                    # Find concepts mentioned after the indicator
                    pattern = f"{indicator}\\s+([A-Z][a-zA-Z\\s]+)"
                    matches = re.findall(pattern, concept['definition'])

                    for match in matches:
                        # Try to find matching concept
                        parent_id = self._find_concept_by_name(match.strip())
                        if parent_id:
                            relationships.append((
                                parent_id,  # parent
                                concept_id,  # child
                                indicator    # relationship type
                            ))

        return relationships

    def _find_cooccurrence_relationships(self) -> List[Tuple[str, str, str]]:
        """Find relationships based on co-occurrence in definitions."""
        relationships = []

        # Build co-occurrence matrix
        cooccurrence = defaultdict(lambda: defaultdict(int))

        for concept_id, concept in self.concepts.items():
            definition = concept.get('definition', '').lower()

            # Find all other concepts mentioned in this definition
            for other_id, other_concept in self.concepts.items():
                if other_id == concept_id:
                    continue

                other_name = other_concept['name'].lower()
                if other_name in definition:
                    cooccurrence[concept_id][other_id] += 1

        # Convert high co-occurrence to relationships
        # If concept A mentions concept B frequently, B might be parent of A
        for concept_id, mentions in cooccurrence.items():
            for mentioned_id, count in mentions.items():
                if count >= 2:  # Mentioned at least twice
                    relationships.append((
                        mentioned_id,  # parent (the mentioned concept)
                        concept_id,    # child (the concept doing the mentioning)
                        'cooccurrence'
                    ))

        return relationships

    def _find_containment_relationships(self) -> List[Tuple[str, str, str]]:
        """Find relationships where concept names contain other concept names."""
        relationships = []

        for concept_id, concept in self.concepts.items():
            name = concept['name'].lower()

            # Check if this concept's name contains other concept names
            for other_id, other_concept in self.concepts.items():
                if other_id == concept_id:
                    continue

                other_name = other_concept['name'].lower()

                # If "PAC Conservation" contains "PAC", PAC is parent
                if other_name in name and len(other_name) >= 3:
                    relationships.append((
                        other_id,      # parent (the contained term)
                        concept_id,    # child (the larger term)
                        'containment'
                    ))

        return relationships

    def _merge_relationships(self, relationship_lists: List[List[Tuple]]) -> Dict[str, Set[str]]:
        """Merge relationship lists, removing duplicates."""
        merged = defaultdict(set)

        for rel_list in relationship_lists:
            for parent_id, child_id, rel_type in rel_list:
                merged[parent_id].add(child_id)

        return merged

    def _build_tree_structure(self, relationships: Dict[str, Set[str]], trunk_nodes: Set[str]):
        """Build forward and reverse relationship maps."""
        self.relationships = relationships

        # Build reverse map
        for parent_id, children in relationships.items():
            for child_id in children:
                self.reverse_relationships[child_id].append(parent_id)

        # Ensure trunk nodes are roots (no parents)
        for trunk_id in trunk_nodes:
            if trunk_id in self.reverse_relationships:
                del self.reverse_relationships[trunk_id]

    def _calculate_depths(self, trunk_nodes: Set[str]) -> Dict[str, int]:
        """Calculate depth of each concept from trunk (BFS)."""
        depths = {}

        # Trunk nodes at depth 0
        queue = [(trunk_id, 0) for trunk_id in trunk_nodes]
        for trunk_id in trunk_nodes:
            depths[trunk_id] = 0

        # BFS to calculate depths
        visited = set(trunk_nodes)

        while queue:
            concept_id, depth = queue.pop(0)

            # Process children
            for child_id in self.relationships.get(concept_id, []):
                if child_id not in visited:
                    visited.add(child_id)
                    depths[child_id] = depth + 1
                    queue.append((child_id, depth + 1))

        # Concepts with no depth (orphans) get max depth
        max_depth = max(depths.values()) if depths else 0
        for concept_id in self.concepts:
            if concept_id not in depths:
                depths[concept_id] = max_depth + 1

        return depths

    def _find_concept_by_name(self, name: str) -> str | None:
        """Find concept ID by name (fuzzy matching)."""
        name_lower = name.lower().strip()

        # Exact match
        for concept_id, concept in self.concepts.items():
            if concept['name'].lower() == name_lower:
                return concept_id

        # Contains match
        for concept_id, concept in self.concepts.items():
            if name_lower in concept['name'].lower():
                return concept_id

        return None

    def _save_updated_graph(self, depths: Dict[str, int]):
        """Save graph with updated relationships and depths."""
        # Update nodes with parent/child relationships and depths
        for concept_id, concept in self.concepts.items():
            # Update parent potentials
            parents = self.reverse_relationships.get(concept_id, [])
            concept['parent_potentials'] = parents

            # Update child actualizations
            children = list(self.relationships.get(concept_id, []))
            concept['child_actualizations'] = children

            # Update depth
            concept['actualization_depth'] = depths.get(concept_id, 999)

            # Update derivation path (simplified - just parent names)
            if parents:
                parent_names = [self.concepts[p]['name'] for p in parents if p in self.concepts]
                concept['derivation_path'] = parent_names + [concept['name']]
            else:
                concept['derivation_path'] = [concept['name']]

        # Update roots (concepts with no parents)
        roots = [cid for cid in self.concepts if not self.reverse_relationships.get(cid)]

        # Save
        data = {
            'nodes': self.concepts,
            'roots': roots,
            'metadata': {
                'source': 'Dawn Field Institute Repository',
                'generated_by': 'build_concept_relationships.py',
                'total_concepts': len(self.concepts),
                'total_relationships': sum(len(children) for children in self.relationships.values()),
                'max_depth': max(depths.values()) if depths else 0
            }
        }

        output_path = self.graph_path.parent / "dfi_repository_knowledge_with_relationships.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  Saved to {output_path}")

    def print_tree_statistics(self, depths: Dict[str, int]):
        """Print tree statistics."""
        print("\n" + "="*70)
        print("Tree Statistics")
        print("="*70)

        # Depth distribution
        depth_counts = defaultdict(int)
        for depth in depths.values():
            depth_counts[depth] += 1

        print("\nDepth Distribution:")
        for depth in sorted(depth_counts.keys()):
            if depth < 10:  # Only show first 10 levels
                print(f"  Depth {depth}: {depth_counts[depth]} concepts")

        # Trunk concepts
        trunk = [cid for cid, d in depths.items() if d == 0]
        print(f"\nTrunk Concepts ({len(trunk)}):")
        for cid in trunk[:20]:  # Show first 20
            print(f"  - {self.concepts[cid]['name']}")

        # Most connected concepts
        print(f"\nMost Connected Concepts (by children):")
        by_children = sorted(
            [(cid, len(self.relationships.get(cid, []))) for cid in self.concepts],
            key=lambda x: x[1],
            reverse=True
        )
        for cid, count in by_children[:10]:
            if count > 0:
                print(f"  {self.concepts[cid]['name']}: {count} children")

        # Sample branches
        print(f"\nSample Branches (trunk → leaves):")
        for trunk_id in list(trunk)[:3]:
            self._print_branch(trunk_id, max_depth=3)

    def _print_branch(self, concept_id: str, indent: int = 0, max_depth: int = 3):
        """Recursively print a branch."""
        if indent > max_depth:
            return

        concept = self.concepts[concept_id]
        print("  " + "  " * indent + f"└─ {concept['name']}")

        children = self.relationships.get(concept_id, [])
        for child_id in list(children)[:3]:  # Max 3 children per level
            self._print_branch(child_id, indent + 1, max_depth)


if __name__ == "__main__":
    # Load repository graph
    repo_root = Path(__file__).parent.parent.parent
    graph_path = repo_root / "fracton" / "data" / "dfi_repository_knowledge.json"

    if not graph_path.exists():
        print(f"ERROR: Graph not found at {graph_path}")
        sys.exit(1)

    # Build relationships
    builder = ConceptRelationshipBuilder(graph_path)
    relationships, depths = builder.build_relationships()

    # Print statistics
    builder.print_tree_statistics(depths)

    print("\n✅ Done!")
    print("\nNext steps:")
    print("  1. Review tree structure and verify trunk concepts")
    print("  2. Visualize the tree (3D plot like recursive_tree.py)")
    print("  3. Query the graph with updated relationships")
    print("  4. Test confidence scores with proper depth/lineage")
