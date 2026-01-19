"""
Relationship Extraction Agent

Uses LLM to identify semantic relationships between concepts.

This agent analyzes concept definitions and extracts:
- Parent-child relationships (A builds on B)
- Dependency relationships (A requires B)
- Application relationships (A applies B)
- Generalization relationships (A is a type of B)

Author: Claude (Dawn Field Institute)
Date: 2026-01-19
"""

import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


@dataclass
class ConceptPair:
    """A concept with its definition."""
    id: str
    name: str
    definition: str


@dataclass
class Relationship:
    """An identified relationship between concepts."""
    parent_id: str
    parent_name: str
    child_id: str
    child_name: str
    relationship_type: str  # builds_on, requires, applies, generalizes
    confidence: float  # 0-1


class RelationshipExtractionAgent:
    """
    Agent that uses LLM to extract semantic relationships.

    This is the first and most critical agent for building the tree.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize agent with LLM client.

        Args:
            api_key: Anthropic API key (optional, will use env var if not provided)
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found. Set environment variable or pass to constructor.")

        # Import anthropic client
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def extract_relationships_batch(
        self,
        concepts: List[ConceptPair],
        batch_size: int = 10
    ) -> List[Relationship]:
        """
        Extract relationships from a batch of concepts.

        Uses LLM to understand semantic meaning and identify how concepts relate.

        Args:
            concepts: List of concepts to analyze
            batch_size: How many concepts to analyze at once

        Returns:
            List of identified relationships
        """
        all_relationships = []

        # Process in batches
        for i in range(0, len(concepts), batch_size):
            batch = concepts[i:i+batch_size]
            print(f"\n  Processing batch {i//batch_size + 1}/{(len(concepts)-1)//batch_size + 1}...")

            batch_rels = self._extract_batch(batch)
            all_relationships.extend(batch_rels)

            print(f"    Found {len(batch_rels)} relationships in this batch")

        return all_relationships

    def _extract_batch(self, concepts: List[ConceptPair]) -> List[Relationship]:
        """Extract relationships from one batch."""
        # Build prompt
        prompt = self._build_extraction_prompt(concepts)

        # Call LLM
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.0,  # Deterministic for relationship extraction
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Parse response
            response_text = response.content[0].text
            relationships = self._parse_response(response_text, concepts)

            return relationships

        except Exception as e:
            print(f"    ⚠ LLM call failed: {e}")
            return []

    def _build_extraction_prompt(self, concepts: List[ConceptPair]) -> str:
        """Build prompt for LLM to extract relationships."""
        concept_list = "\n".join([
            f"{i+1}. **{c.name}** (ID: {c.id})\n   {c.definition[:200]}{'...' if len(c.definition) > 200 else ''}"
            for i, c in enumerate(concepts)
        ])

        prompt = f"""You are analyzing concepts from a technical knowledge base to identify semantic relationships.

Given these concepts:

{concept_list}

Identify parent-child relationships where one concept builds on, extends, or specializes another.

Relationship types:
- **builds_on**: Child concept is built on/derived from parent concept
- **requires**: Child concept requires/depends on parent concept
- **applies**: Child concept applies/uses parent concept
- **specializes**: Child concept is a specific instance of parent concept

Return your analysis as JSON array:
```json
[
  {{
    "parent_id": "concept_id_1",
    "child_id": "concept_id_2",
    "relationship_type": "builds_on",
    "confidence": 0.9,
    "reasoning": "Brief explanation"
  }}
]
```

Guidelines:
- Only include relationships you are confident about (confidence > 0.5)
- Focus on direct relationships (not transitive)
- If a concept mentions another by name, that's a strong signal
- Consider domain hierarchy (field theory → quantum mechanics → quantum entanglement)
- Look for phrases like "builds on", "based on", "extends", "uses", "applies"

Return ONLY the JSON array, no other text."""

        return prompt

    def _parse_response(self, response_text: str, concepts: List[ConceptPair]) -> List[Relationship]:
        """Parse LLM response into Relationship objects."""
        relationships = []

        # Extract JSON from response
        try:
            # Find JSON array in response
            start = response_text.find('[')
            end = response_text.rfind(']') + 1

            if start == -1 or end == 0:
                print("    ⚠ No JSON array found in response")
                return []

            json_str = response_text[start:end]
            data = json.loads(json_str)

            # Convert to Relationship objects
            concept_map = {c.id: c for c in concepts}

            for item in data:
                parent_id = item.get('parent_id')
                child_id = item.get('child_id')

                # Validate IDs exist
                if parent_id not in concept_map or child_id not in concept_map:
                    continue

                parent = concept_map[parent_id]
                child = concept_map[child_id]

                rel = Relationship(
                    parent_id=parent_id,
                    parent_name=parent.name,
                    child_id=child_id,
                    child_name=child.name,
                    relationship_type=item.get('relationship_type', 'builds_on'),
                    confidence=float(item.get('confidence', 0.7))
                )

                relationships.append(rel)

        except json.JSONDecodeError as e:
            print(f"    ⚠ Failed to parse JSON: {e}")
            print(f"    Response: {response_text[:200]}")

        return relationships

    def extract_pairwise_relationship(
        self,
        concept1: ConceptPair,
        concept2: ConceptPair
    ) -> Relationship | None:
        """
        Extract relationship between two specific concepts.

        Useful for targeted analysis.
        """
        rels = self._extract_batch([concept1, concept2])
        return rels[0] if rels else None


def main():
    """Test relationship extraction on DFI repository concepts."""
    print("="*70)
    print("Relationship Extraction Agent - Test")
    print("="*70)

    # Load repository graph
    repo_root = Path(__file__).parent.parent.parent.parent
    graph_path = repo_root / "fracton" / "data" / "dfi_repository_knowledge.json"

    if not graph_path.exists():
        print(f"\n✗ ERROR: Graph not found at {graph_path}")
        return

    print(f"\n✓ Loading graph from {graph_path}")

    with open(graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = data['nodes']
    print(f"✓ Loaded {len(nodes)} concepts")

    # Select interesting concepts for testing
    # Focus on technical concepts (not generic headers)
    technical_keywords = [
        'kronos', 'pac', 'sec', 'field', 'cortex', 'lobe',
        'soul', 'avatar', 'confluence', 'actualization'
    ]

    test_concepts = []
    for node_id, node in nodes.items():
        name_lower = node['name'].lower()
        if any(kw in name_lower for kw in technical_keywords):
            concept = ConceptPair(
                id=node_id,
                name=node['name'],
                definition=node.get('definition', '')[:500]  # Limit for API
            )
            test_concepts.append(concept)

    print(f"\n✓ Selected {len(test_concepts)} technical concepts for testing")

    # Check for API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("\n✗ ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("\nTo run this test:")
        print("  1. Get API key from https://console.anthropic.com/")
        print("  2. Set environment variable: export ANTHROPIC_API_KEY='your-key'")
        print("  3. Run again")
        return

    # Initialize agent
    print("\n" + "-"*70)
    print("Initializing Relationship Extraction Agent...")
    print("-"*70)

    try:
        agent = RelationshipExtractionAgent()
        print("✓ Agent initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return

    # Extract relationships
    print("\n" + "-"*70)
    print("Extracting Relationships...")
    print("-"*70)

    relationships = agent.extract_relationships_batch(
        test_concepts[:20],  # Test with first 20 concepts
        batch_size=10
    )

    print(f"\n✓ Extracted {len(relationships)} relationships")

    # Display results
    print("\n" + "="*70)
    print("Extracted Relationships")
    print("="*70)

    if relationships:
        # Sort by confidence
        relationships.sort(key=lambda r: r.confidence, reverse=True)

        print("\nTop relationships (by confidence):\n")
        for i, rel in enumerate(relationships[:15], 1):
            print(f"{i}. {rel.parent_name} → {rel.child_name}")
            print(f"   Type: {rel.relationship_type}")
            print(f"   Confidence: {rel.confidence:.2f}")
            print()

        # Save to file
        output_path = graph_path.parent / "extracted_relationships.json"
        output_data = [
            {
                'parent_id': r.parent_id,
                'parent_name': r.parent_name,
                'child_id': r.child_id,
                'child_name': r.child_name,
                'relationship_type': r.relationship_type,
                'confidence': r.confidence
            }
            for r in relationships
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved relationships to {output_path}")

    else:
        print("\n⚠ No relationships extracted")

    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)


if __name__ == "__main__":
    main()
