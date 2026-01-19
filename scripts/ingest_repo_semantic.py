"""
Semantic Repository Ingestion

Uses LLM-based semantic analysis to:
1. Extract concepts from documentation
2. Understand what each concept means
3. Identify parent-child relationships
4. Generalize specific implementations to principles
5. Build true genealogical tree structure

This is the "interpretation layer" that transforms syntactic extraction
into semantic understanding.

Author: Claude (Dawn Field Institute)
Date: 2026-01-19
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


@dataclass
class ConceptExtraction:
    """Result of semantic concept extraction."""
    name: str
    definition: str
    source_file: str
    is_fundamental: bool  # Near trunk?
    parent_concepts: List[str]  # Concepts this builds on
    generalizes_to: str | None  # More abstract concept
    keywords: Set[str]


def analyze_concept_semantically(concept_name: str, definition: str, context: str) -> Dict:
    """
    Use LLM to semantically analyze a concept.

    This would call an LLM with a prompt like:
    "Given this concept and definition, identify:
     1. Is this a fundamental principle or specific implementation?
     2. What concepts does this build on? (parents)
     3. What is the more general principle? (generalization)
     4. Key domain keywords

     Concept: {concept_name}
     Definition: {definition}
     Context: {context}
    "

    Returns structured semantic analysis.
    """
    # TODO: Replace with actual LLM call
    # For now, use heuristics

    # Detect fundamental concepts
    fundamental_keywords = {
        'pac', 'sec', 'infodynamics', 'field', 'theory',
        'principle', 'law', 'fundamental', 'axiom', 'postulate',
        'conservation', 'entropy', 'information', 'energy'
    }

    name_lower = concept_name.lower()
    def_lower = definition.lower()

    is_fundamental = any(kw in name_lower for kw in fundamental_keywords)

    # Extract parent concepts (concepts mentioned in definition)
    parent_concepts = []
    for kw in ['builds on', 'based on', 'derived from', 'extends', 'uses']:
        if kw in def_lower:
            # Try to extract what comes after
            pattern = f"{kw}\\s+([A-Z][a-zA-Z\\s]+)"
            matches = re.findall(pattern, definition)
            parent_concepts.extend(m.strip() for m in matches)

    # Generalization heuristic
    generalizes_to = None
    if 'implementation' in def_lower or 'specific' in def_lower:
        # Try to find general principle mentioned
        for kw in fundamental_keywords:
            if kw in def_lower and kw not in name_lower:
                generalizes_to = kw.title()
                break

    # Extract keywords
    words = re.findall(r'\b[a-z]{4,}\b', def_lower)
    stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'will'}
    keywords = {w for w in words if w not in stop_words}

    return {
        'is_fundamental': is_fundamental,
        'parent_concepts': parent_concepts,
        'generalizes_to': generalizes_to,
        'keywords': keywords
    }


def prompt_for_relationships(concepts: List[Dict]) -> List[Tuple[str, str, str]]:
    """
    Use LLM to identify relationships between concepts in batch.

    Prompt would be:
    "Given these concepts, identify parent-child relationships.
     Return as: (parent_name, child_name, relationship_type)

     Concepts:
     - PAC Conservation: f(Parent) = Σ f(Children)
     - Fibonacci: Recursive number sequence
     - Golden Ratio: φ = 1.618...
     ...
    "

    This is MORE ACCURATE than heuristics because LLM understands:
    - "Fibonacci emerges from PAC" (parent: PAC, child: Fibonacci)
    - "Golden ratio appears in Fibonacci" (parent: Fibonacci, child: Golden ratio)
    """
    # TODO: Replace with actual LLM call
    # For now, return empty
    return []


class SemanticRepoIngestion:
    """Semantic repository ingestion with LLM interpretation."""

    def __init__(self, repo_root: Path, output_path: Path):
        self.repo_root = repo_root
        self.output_path = output_path
        self.concepts = []  # List[ConceptExtraction]
        self.relationships = []  # List[(parent, child, type)]

    def ingest_repository(self):
        """
        Main semantic ingestion flow.

        1. Extract concepts (syntactic - same as before)
        2. Semantic analysis per concept (LLM)
        3. Batch relationship identification (LLM)
        4. Build genealogical tree
        5. Calculate depths and confluence
        6. Save structured graph
        """
        print("="*70)
        print("Semantic Repository Ingestion")
        print("="*70)

        print(f"\nRepository: {self.repo_root}")
        print(f"Output: {self.output_path}")

        # Step 1: Find documentation
        print("\nStep 1: Scanning repository...")
        doc_files = self._find_documentation()
        print(f"  Found {len(doc_files)} documentation files")

        # Step 2: Extract concepts (syntactic)
        print("\nStep 2: Extracting concepts...")
        for doc_file in doc_files:
            self._extract_from_file(doc_file)
        print(f"  Extracted {len(self.concepts)} concepts")

        # Step 3: Semantic analysis
        print("\nStep 3: Semantic analysis (LLM)...")
        self._semantic_analysis()
        print(f"  Analyzed {len(self.concepts)} concepts")

        # Step 4: Relationship identification
        print("\nStep 4: Identifying relationships (LLM)...")
        self._identify_relationships()
        print(f"  Identified {len(self.relationships)} relationships")

        # Step 5: Build tree
        print("\nStep 5: Building genealogical tree...")
        tree = self._build_tree()
        print(f"  Tree depth: {tree['max_depth']}")
        print(f"  Trunk concepts: {len(tree['trunk'])}")

        # Step 6: Save
        print("\nStep 6: Saving semantic graph...")
        self._save_graph(tree)
        print(f"  Saved to {self.output_path}")

        print("\n" + "="*70)
        print("Semantic Ingestion Complete!")
        print("="*70)

        # Analysis
        self._print_analysis(tree)

    def _find_documentation(self) -> List[Path]:
        """Find documentation files (same as before)."""
        doc_files = []

        scan_dirs = [
            self.repo_root / "grimm",
            self.repo_root / "fracton",
        ]

        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue

            for md_file in scan_dir.rglob("*.md"):
                if any(skip in str(md_file) for skip in ['node_modules', '.git', '_archive', 'venv', 'test_']):
                    continue
                doc_files.append(md_file)

        return sorted(doc_files)

    def _extract_from_file(self, file_path: Path):
        """Extract concepts from file (syntactic extraction)."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"  ⚠ Could not read {file_path}: {e}")
            return

        # Extract H2 headers with definitions
        header_pattern = r'^##\s+(.+)$'
        lines = content.split('\n')

        # Skip generic headers
        skip_headers = {
            'Overview', 'Introduction', 'Background', 'Conclusion',
            'Summary', 'Usage', 'Installation', 'Examples',
            'Contributing', 'License', 'References', 'See Also',
            'Change Log', 'Changelog', 'TODO', 'Notes',
            'Next Steps', 'Impact', 'Status', 'Related',
            'Validation', 'Testing', 'Test Results', 'Documentation',
            'Performance', 'Known Limitations', 'Security', 'Future Work',
            'Motivation', 'Metrics', 'Results', 'Analysis'
        }

        i = 0
        while i < len(lines):
            match = re.match(header_pattern, lines[i])
            if match:
                concept_name = match.group(1).strip()

                if concept_name in skip_headers:
                    i += 1
                    continue

                # Get definition
                definition_lines = []
                j = i + 1
                while j < len(lines) and len(definition_lines) < 5:
                    line = lines[j].strip()
                    if line and not line.startswith('#') and not line.startswith('```'):
                        definition_lines.append(line)
                        if len(' '.join(definition_lines)) > 200:
                            break
                    elif line.startswith('#'):
                        break
                    j += 1

                if definition_lines:
                    definition = ' '.join(definition_lines)

                    # Store for semantic analysis
                    extraction = ConceptExtraction(
                        name=concept_name,
                        definition=definition,
                        source_file=str(file_path.relative_to(self.repo_root)),
                        is_fundamental=False,  # Will be determined by LLM
                        parent_concepts=[],
                        generalizes_to=None,
                        keywords=set()
                    )

                    self.concepts.append(extraction)

            i += 1

    def _semantic_analysis(self):
        """Perform semantic analysis on each concept."""
        for concept in self.concepts:
            # Get context (surrounding concepts from same file)
            context = self._get_context(concept)

            # Analyze semantically
            analysis = analyze_concept_semantically(
                concept.name,
                concept.definition,
                context
            )

            # Update concept with analysis
            concept.is_fundamental = analysis['is_fundamental']
            concept.parent_concepts = analysis['parent_concepts']
            concept.generalizes_to = analysis['generalizes_to']
            concept.keywords = analysis['keywords']

    def _get_context(self, concept: ConceptExtraction) -> str:
        """Get context for concept (other concepts from same file)."""
        same_file = [c for c in self.concepts if c.source_file == concept.source_file]
        return " | ".join(f"{c.name}: {c.definition[:50]}" for c in same_file[:5])

    def _identify_relationships(self):
        """Identify relationships between concepts using batch LLM analysis."""
        # Prepare concept summaries for LLM
        concept_summaries = [
            {
                'name': c.name,
                'definition': c.definition[:100],
                'is_fundamental': c.is_fundamental
            }
            for c in self.concepts
        ]

        # Get relationships from LLM
        llm_relationships = prompt_for_relationships(concept_summaries)

        # Combine with extracted parent concepts
        for concept in self.concepts:
            for parent_name in concept.parent_concepts:
                # Find matching concept
                parent = self._find_concept_by_name(parent_name)
                if parent:
                    self.relationships.append((parent.name, concept.name, 'builds_on'))

            # Add generalization relationship
            if concept.generalizes_to:
                self.relationships.append((concept.generalizes_to, concept.name, 'generalizes'))

        # Add LLM-identified relationships
        self.relationships.extend(llm_relationships)

    def _find_concept_by_name(self, name: str) -> ConceptExtraction | None:
        """Find concept by name (fuzzy matching)."""
        name_lower = name.lower().strip()

        for concept in self.concepts:
            if concept.name.lower() == name_lower:
                return concept
            if name_lower in concept.name.lower():
                return concept

        return None

    def _build_tree(self) -> Dict:
        """Build genealogical tree structure."""
        # Build adjacency lists
        children_map = defaultdict(list)
        parent_map = defaultdict(list)

        for parent_name, child_name, rel_type in self.relationships:
            children_map[parent_name].append(child_name)
            parent_map[child_name].append(parent_name)

        # Identify trunk (fundamental concepts with no parents)
        trunk = [c.name for c in self.concepts if c.is_fundamental and not parent_map.get(c.name)]

        # Calculate depths (BFS from trunk)
        depths = {}
        queue = [(name, 0) for name in trunk]
        for name in trunk:
            depths[name] = 0

        visited = set(trunk)
        while queue:
            name, depth = queue.pop(0)
            for child in children_map.get(name, []):
                if child not in visited:
                    visited.add(child)
                    depths[child] = depth + 1
                    queue.append((child, depth + 1))

        # Orphans get max depth + 1
        max_depth = max(depths.values()) if depths else 0
        for concept in self.concepts:
            if concept.name not in depths:
                depths[concept.name] = max_depth + 1

        return {
            'trunk': trunk,
            'depths': depths,
            'max_depth': max_depth,
            'children_map': dict(children_map),
            'parent_map': dict(parent_map)
        }

    def _save_graph(self, tree: Dict):
        """Save semantic graph with relationships."""
        nodes = {}

        for concept in self.concepts:
            node_id = self._name_to_id(concept.name)

            parents = tree['parent_map'].get(concept.name, [])
            children = tree['children_map'].get(concept.name, [])
            depth = tree['depths'].get(concept.name, 999)

            nodes[node_id] = {
                'id': node_id,
                'name': concept.name,
                'definition': concept.definition,
                'source_file': concept.source_file,
                'is_fundamental': concept.is_fundamental,
                'parent_potentials': [self._name_to_id(p) for p in parents],
                'child_actualizations': [self._name_to_id(c) for c in children],
                'actualization_depth': depth,
                'derivation_path': parents + [concept.name],
                'keywords': list(concept.keywords),
                'generalizes_to': concept.generalizes_to
            }

        data = {
            'nodes': nodes,
            'roots': [self._name_to_id(name) for name in tree['trunk']],
            'metadata': {
                'source': 'Dawn Field Institute Repository (Semantic)',
                'generated_by': 'ingest_repo_semantic.py',
                'total_concepts': len(self.concepts),
                'total_relationships': len(self.relationships),
                'trunk_concepts': len(tree['trunk']),
                'max_depth': tree['max_depth']
            }
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _name_to_id(self, name: str) -> str:
        """Convert name to ID."""
        id_str = name.lower()
        id_str = re.sub(r'[^a-z0-9\\s_-]', '', id_str)
        id_str = re.sub(r'\\s+', '_', id_str)
        return id_str

    def _print_analysis(self, tree: Dict):
        """Print analysis."""
        print(f"\n📊 Analysis:")
        print(f"  Total concepts: {len(self.concepts)}")
        print(f"  Trunk concepts: {len(tree['trunk'])}")
        print(f"  Max depth: {tree['max_depth']}")
        print(f"  Total relationships: {len(self.relationships)}")

        print(f"\n🌳 Trunk Concepts:")
        for name in tree['trunk'][:10]:
            print(f"  - {name}")

        print(f"\n📏 Depth Distribution:")
        depth_counts = defaultdict(int)
        for depth in tree['depths'].values():
            depth_counts[depth] += 1
        for depth in sorted(depth_counts.keys())[:10]:
            print(f"  Depth {depth}: {depth_counts[depth]} concepts")


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent.parent
    output_path = repo_root / "fracton" / "data" / "dfi_repository_semantic.json"

    ingestion = SemanticRepoIngestion(repo_root, output_path)
    ingestion.ingest_repository()

    print(f"\n✅ Done! Semantic graph saved to {output_path}")
    print(f"\nNOTE: This uses heuristics. For real semantic analysis, integrate LLM calls.")
    print(f"See functions: analyze_concept_semantically(), prompt_for_relationships()")
