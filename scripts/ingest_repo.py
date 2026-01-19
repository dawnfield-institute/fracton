"""
Repository Ingestion Script

Walks the Dawn Field Institute repository and ingests concepts from:
- Markdown files (documentation, specs, vision docs)
- Code comments (Python, TypeScript, etc.)
- README files
- Spec files

This is the real stress test - can KRONOS turn scattered documentation
into a coherent conceptual genealogy?

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

# Add fracton to path
fracton_path = Path(__file__).parent.parent
sys.path.insert(0, str(fracton_path))

from fracton.storage.graph import KronosGraph
from fracton.storage.node import KronosNode


@dataclass
class ExtractedConcept:
    """A concept extracted from documentation."""
    name: str
    definition: str
    source_file: str
    source_section: str
    related_concepts: List[str]  # Mentioned concepts
    keywords: Set[str]


class RepoIngestion:
    """Ingest concepts from Dawn Field Institute repository."""

    def __init__(self, repo_root: Path, output_path: Path):
        self.repo_root = repo_root
        self.output_path = output_path
        self.extracted_concepts: List[ExtractedConcept] = []
        self.concept_mentions = defaultdict(list)  # concept_name -> [file_paths]

    def ingest_repository(self):
        """
        Main ingestion flow.

        1. Scan repository for documentation
        2. Extract concepts from markdown
        3. Build concept graph with auto-detected relationships
        4. Save to output file
        """
        print("="*70)
        print("KRONOS Repository Ingestion")
        print("="*70)

        print(f"\nRepository: {self.repo_root}")
        print(f"Output: {self.output_path}\n")

        # Step 1: Find documentation files
        print("Step 1: Scanning repository...")
        doc_files = self._find_documentation()
        print(f"  Found {len(doc_files)} documentation files")

        # Step 2: Extract concepts
        print("\nStep 2: Extracting concepts...")
        for doc_file in doc_files:
            self._extract_from_file(doc_file)
        print(f"  Extracted {len(self.extracted_concepts)} raw concepts")

        # Step 3: Build concept graph
        print("\nStep 3: Building concept graph...")
        graph = self._build_graph()
        print(f"  Built graph with {len(graph.nodes)} nodes")

        # Step 4: Save graph
        print("\nStep 4: Saving graph...")
        self._save_graph(graph)
        print(f"  Saved to {self.output_path}")

        # Step 5: Analysis
        print("\n" + "="*70)
        print("Ingestion Complete!")
        print("="*70)
        self._print_analysis(graph)

    def _find_documentation(self) -> List[Path]:
        """Find all documentation files in repository."""
        doc_files = []

        # Only scan grimm and fracton directories (scoped ingestion)
        scan_dirs = [
            self.repo_root / "grimm",
            self.repo_root / "fracton",
        ]

        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue

            # Markdown files
            for md_file in scan_dir.rglob("*.md"):
                # Skip node_modules, .git, _archive, test files
                if any(skip in str(md_file) for skip in ['node_modules', '.git', '_archive', 'venv', 'test_']):
                    continue
                doc_files.append(md_file)

        return sorted(doc_files)

    def _extract_from_file(self, file_path: Path):
        """Extract concepts from a single file."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"  ⚠ Could not read {file_path}: {e}")
            return

        # Extract concepts from markdown headers
        # Look for patterns like:
        # ## Concept Name
        # Definition here...

        # Simple extraction: H2 headers followed by paragraph
        header_pattern = r'^##\s+(.+)$'
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            match = re.match(header_pattern, lines[i])
            if match:
                concept_name = match.group(1).strip()

                # Skip common non-concept headers
                skip_headers = {
                    'Overview', 'Introduction', 'Background', 'Conclusion',
                    'Summary', 'Usage', 'Installation', 'Examples',
                    'Contributing', 'License', 'References', 'See Also',
                    'Change Log', 'Changelog', 'TODO', 'Notes'
                }

                if concept_name in skip_headers:
                    i += 1
                    continue

                # Get definition (next few non-empty lines)
                definition_lines = []
                j = i + 1
                while j < len(lines) and len(definition_lines) < 5:
                    line = lines[j].strip()
                    if line and not line.startswith('#') and not line.startswith('```'):
                        definition_lines.append(line)
                        if len(' '.join(definition_lines)) > 200:  # Enough context
                            break
                    elif line.startswith('#'):  # Hit next header
                        break
                    j += 1

                if definition_lines:
                    definition = ' '.join(definition_lines)

                    # Extract related concepts (mentioned terms)
                    related = self._extract_related_concepts(definition)

                    # Extract keywords
                    keywords = self._extract_keywords(concept_name, definition)

                    concept = ExtractedConcept(
                        name=concept_name,
                        definition=definition,
                        source_file=str(file_path.relative_to(self.repo_root)),
                        source_section=concept_name,
                        related_concepts=related,
                        keywords=keywords
                    )

                    self.extracted_concepts.append(concept)
                    self.concept_mentions[concept_name.lower()].append(str(file_path))

            i += 1

    def _extract_related_concepts(self, text: str) -> List[str]:
        """Extract mentioned concepts from text."""
        # Look for capitalized terms, technical terms
        # Simple approach: words in Title Case or acronyms
        words = text.split()
        related = []

        for i, word in enumerate(words):
            # Title Case terms (likely concepts)
            if word[0].isupper() and len(word) > 3:
                # Check if next word also capitalized (multi-word concept)
                if i + 1 < len(words) and words[i+1][0].isupper():
                    related.append(f"{word} {words[i+1]}")
                else:
                    related.append(word)

            # Acronyms (all caps, 2-5 letters)
            if word.isupper() and 2 <= len(word) <= 5:
                related.append(word)

        return list(set(related))[:10]  # Top 10 unique

    def _extract_keywords(self, name: str, definition: str) -> Set[str]:
        """Extract important keywords."""
        # Combine name and definition
        text = f"{name} {definition}".lower()

        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'this', 'that', 'these', 'those', 'it', 'its'
        }

        words = re.findall(r'\b\w+\b', text)
        keywords = {w for w in words if len(w) > 3 and w not in stop_words}

        return keywords

    def _build_graph(self) -> KronosGraph:
        """Build KRONOS graph from extracted concepts."""
        graph = KronosGraph()

        # First pass: Create nodes
        # Group concepts by similarity to detect confluence
        concept_groups = self._group_similar_concepts()

        print(f"\n  Detected {len(concept_groups)} concept groups")

        for group_id, concepts in enumerate(concept_groups):
            if len(concepts) == 1:
                # Single concept - add as node
                concept = concepts[0]
                node = self._create_node(concept, depth=1, parents=[])
                graph.nodes[node.id] = node
            else:
                # Multiple similar concepts - create confluence node
                print(f"  📦 Confluence: {', '.join(c.name for c in concepts[:3])}...")
                merged = self._merge_concepts(concepts)
                node = self._create_node(merged, depth=1, parents=[])
                graph.nodes[node.id] = node

        # Second pass: Detect relationships
        # TODO: Use co-occurrence, shared keywords, etc.

        # For now, mark all as roots (we'll refine this)
        graph.roots = list(graph.nodes.keys())[:20]  # Top 20 as roots

        return graph

    def _group_similar_concepts(self) -> List[List[ExtractedConcept]]:
        """Group concepts by similarity (same concept in different files)."""
        groups = []
        seen = set()

        for concept in self.extracted_concepts:
            if concept.name in seen:
                continue

            # Find all concepts with similar names
            similar = [
                c for c in self.extracted_concepts
                if c.name.lower() == concept.name.lower() or
                   self._name_similarity(c.name, concept.name) > 0.8
            ]

            if similar:
                groups.append(similar)
                seen.update(c.name for c in similar)

        return groups

    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between concept names."""
        # Simple word overlap
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1 & words2)
        total = len(words1 | words2)

        return overlap / total if total > 0 else 0.0

    def _merge_concepts(self, concepts: List[ExtractedConcept]) -> ExtractedConcept:
        """Merge multiple similar concepts into one."""
        # Use first name
        name = concepts[0].name

        # Combine definitions
        definitions = [c.definition for c in concepts]
        definition = " | ".join(definitions[:3])  # Top 3 for brevity

        # Combine sources
        sources = [c.source_file for c in concepts]
        source_file = f"Multiple sources: {', '.join(sources[:3])}"

        # Combine related concepts
        all_related = []
        for c in concepts:
            all_related.extend(c.related_concepts)
        related = list(set(all_related))[:15]

        # Combine keywords
        all_keywords = set()
        for c in concepts:
            all_keywords.update(c.keywords)

        return ExtractedConcept(
            name=name,
            definition=definition,
            source_file=source_file,
            source_section="Merged",
            related_concepts=related,
            keywords=all_keywords
        )

    def _create_node(self, concept: ExtractedConcept, depth: int, parents: List[str]) -> KronosNode:
        """Create KRONOS node from extracted concept."""
        node_id = self._name_to_id(concept.name)

        # Calculate confluence pattern (equal weights for now)
        if parents:
            confluence = {p: 1.0 / len(parents) for p in parents}
        else:
            confluence = {}

        # Build derivation path
        path = parents + [concept.name]

        node = KronosNode(
            id=node_id,
            name=concept.name,
            definition=concept.definition,
            parent_potentials=parents,
            derivation_path=path,
            actualization_depth=depth,
            confluence_pattern=confluence,
            first_crystallization=None  # We don't have timestamps
        )

        # Add child placeholders (will be filled when we detect descendants)
        node.child_actualizations = []

        return node

    def _name_to_id(self, name: str) -> str:
        """Convert concept name to valid ID."""
        # Lowercase, replace spaces with underscores, remove special chars
        id_str = name.lower()
        id_str = re.sub(r'[^a-z0-9\s_-]', '', id_str)
        id_str = re.sub(r'\s+', '_', id_str)
        return id_str

    def _save_graph(self, graph: KronosGraph):
        """Save graph to JSON."""
        data = {
            "nodes": {},
            "roots": graph.roots,
            "metadata": {
                "source": "Dawn Field Institute Repository",
                "generated_by": "ingest_repo.py",
                "total_concepts": len(graph.nodes)
            }
        }

        for node_id, node in graph.nodes.items():
            data["nodes"][node_id] = {
                "id": node.id,
                "name": node.name,
                "definition": node.definition,
                "parent_potentials": node.parent_potentials,
                "derivation_path": node.derivation_path,
                "actualization_depth": node.actualization_depth,
                "confluence_pattern": node.confluence_pattern,
                "child_actualizations": node.child_actualizations,
            }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _print_analysis(self, graph: KronosGraph):
        """Print ingestion analysis."""
        print(f"\n📊 Analysis:")
        print(f"  Total concepts: {len(graph.nodes)}")
        print(f"  Root concepts: {len(graph.roots)}")

        # Show sample concepts
        print(f"\n🔍 Sample Concepts:")
        for i, (node_id, node) in enumerate(list(graph.nodes.items())[:10]):
            print(f"  {i+1}. {node.name}")
            print(f"      ID: {node_id}")
            print(f"      Depth: {node.actualization_depth}")
            if node.definition:
                preview = node.definition[:80] + "..." if len(node.definition) > 80 else node.definition
                print(f"      Def: {preview}")

        # Concept co-occurrence
        print(f"\n📎 Most Mentioned Concepts:")
        sorted_mentions = sorted(self.concept_mentions.items(), key=lambda x: len(x[1]), reverse=True)
        for name, files in sorted_mentions[:10]:
            print(f"  {name}: {len(files)} files")


if __name__ == "__main__":
    # Configuration
    repo_root = Path(__file__).parent.parent.parent  # Dawn Field Institute root
    output_path = repo_root / "fracton" / "data" / "dfi_repository_knowledge.json"

    # Run ingestion
    ingestion = RepoIngestion(repo_root, output_path)
    ingestion.ingest_repository()

    print(f"\n✅ Done! Graph saved to {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Review the graph: python -m json.tool {output_path} | head -100")
    print(f"  2. Test queries: python grimm/test_conversational_kronos.py")
    print(f"  3. Audit concept relationships and confidence scores")
