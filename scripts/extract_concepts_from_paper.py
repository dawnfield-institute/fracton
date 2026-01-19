"""
Extract concepts from DFT papers/documents for ingestion into KRONOS.

This is a helper tool to identify new concepts from research papers
and prepare them for batch ingestion.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


class ConceptExtractor:
    """Extract concepts from text for KRONOS ingestion."""

    def __init__(self):
        self.concepts = []

    def extract_from_file(self, file_path: Path) -> List[Dict]:
        """
        Extract concepts from a markdown/text file.

        Looks for common patterns:
        - ## Section headers as potential concepts
        - Definitions (lines with "Definition:", "is defined as", etc.)
        - Key terms in **bold** or *italic*
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        concepts = []

        for i, line in enumerate(lines):
            # Section headers as concepts
            if line.startswith('## ') or line.startswith('### '):
                header = line.lstrip('#').strip()

                # Get next few lines as definition
                definition_lines = []
                for j in range(i+1, min(i+4, len(lines))):
                    if lines[j].strip() and not lines[j].startswith('#'):
                        definition_lines.append(lines[j].strip())

                if definition_lines:
                    definition = ' '.join(definition_lines)

                    # Create concept ID
                    concept_id = header.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')

                    concepts.append({
                        "id": concept_id,
                        "name": header,
                        "definition": definition[:200] + ("..." if len(definition) > 200 else ""),
                        "source": str(file_path),
                        "line": i+1
                    })

        return concepts

    def interactive_review(self, concepts: List[Dict]) -> List[Dict]:
        """Review extracted concepts and assign parents."""
        print(f"\n[EXTRACTED {len(concepts)} CONCEPTS]")
        print("Review and assign parent concepts:\n")

        reviewed = []

        for i, concept in enumerate(concepts, 1):
            print(f"\n{'='*70}")
            print(f"Concept {i}/{len(concepts)}")
            print(f"{'='*70}")
            print(f"ID: {concept['id']}")
            print(f"Name: {concept['name']}")
            print(f"Definition: {concept['definition']}")
            print(f"Source: {concept['source']} (line {concept['line']})")

            # Keep or skip
            keep = input("\nKeep this concept? (y/n/q to quit): ").lower()
            if keep == 'q':
                break
            if keep != 'y':
                continue

            # Assign parents
            parents_str = input("Parent concept IDs (comma-separated): ").strip()
            if not parents_str:
                print("[SKIPPED] No parents specified")
                continue

            parents = [p.strip() for p in parents_str.split(',')]

            # Assign confluence weights
            if len(parents) == 1:
                confluence = {parents[0]: 1.0}
            else:
                print(f"\nAssign weights for {len(parents)} parents (must sum to 1.0):")
                confluence = {}
                remaining = 1.0

                for parent in parents[:-1]:
                    weight = float(input(f"  {parent}: "))
                    confluence[parent] = weight
                    remaining -= weight

                confluence[parents[-1]] = remaining
                print(f"  {parents[-1]}: {remaining:.2f} (auto)")

            concept['parents'] = parents
            concept['confluence'] = confluence
            del concept['source']
            del concept['line']

            reviewed.append(concept)
            print(f"[ADDED TO BATCH]")

        return reviewed

    def save_batch(self, concepts: List[Dict], output_path: Path):
        """Save reviewed concepts as batch file."""
        with open(output_path, 'w') as f:
            json.dump(concepts, f, indent=2)

        print(f"\n[SAVED] {len(concepts)} concepts to {output_path}")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python extract_concepts_from_paper.py <paper.md> [output.json]")
        print("\nExtracts concepts from a paper and prepares batch ingestion file.")
        return

    paper_path = Path(sys.argv[1])
    if not paper_path.exists():
        print(f"[ERROR] File not found: {paper_path}")
        return

    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("concepts_batch.json")

    extractor = ConceptExtractor()

    print(f"[EXTRACTING] Concepts from {paper_path}...")
    concepts = extractor.extract_from_file(paper_path)

    if not concepts:
        print("[NO CONCEPTS FOUND]")
        return

    # Interactive review
    reviewed = extractor.interactive_review(concepts)

    if reviewed:
        extractor.save_batch(reviewed, output_path)
        print(f"\n[NEXT STEP] Run: python scripts/ingest_concept.py {output_path}")
    else:
        print("\n[NO CONCEPTS REVIEWED]")


if __name__ == "__main__":
    main()
