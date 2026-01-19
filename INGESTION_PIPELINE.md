# KRONOS Knowledge Ingestion Pipeline

**Purpose:** Add new concepts to the DFT knowledge graph while maintaining PAC structure.

---

## Overview

The ingestion pipeline provides two main workflows:

1. **Interactive Mode** - Add one concept at a time with guided prompts
2. **Batch Mode** - Add multiple concepts from a JSON file

Both modes ensure:
- PAC conservation (confluence weights sum to 1.0)
- Proper lineage tracking
- Automatic depth calculation
- Parent-child relationship management

---

## Quick Start

### Interactive: Add Single Concept

```bash
cd fracton
python scripts/ingest_concept.py
```

You'll be prompted for:
- **Concept ID** - snake_case identifier (e.g., `entropic_time`)
- **Concept Name** - Human-readable (e.g., "Entropic Time")
- **Definition** - Clear, concise description
- **Parents** - Select from suggested parent concepts
- **Confluence weights** - How much from each parent (must sum to 1.0)

### Batch: Add Multiple Concepts

```bash
python scripts/ingest_concept.py data/my_concepts.json
```

---

## Workflows

### Workflow 1: Manual Concept Addition

Use when you have a single new concept to add:

```bash
python scripts/ingest_concept.py
```

**Example Session:**
```
KRONOS Concept Ingestion
Current graph: 112 nodes

Concept ID (snake_case): entropic_time
Concept Name: Entropic Time
Definition: Time as emergent from entropy gradients in information fields

[SUGGESTING PARENTS]
Top parent suggestions:
  1. Bifractal Time (score: 5.2, depth: 6)
  2. Infodynamics (score: 4.8, depth: 3)
  3. Thermodynamics (score: 3.1, depth: 0)

Select parent(s) (comma-separated numbers): 1,2

Assign confluence weights for 2 parent(s):
  Bifractal Time: 0.6
  Infodynamics: 0.4 (auto)

PREVIEW
ID: entropic_time
Name: Entropic Time
Definition: Time as emergent from entropy gradients
Depth: 7
Path: sec > bifractal_time > entropic_time

Parents:
  - Bifractal Time (0.60)
  - Infodynamics (0.40)

Add this concept? (y/n): y

[SUCCESS] Added 'Entropic Time' to knowledge graph!
Total nodes: 113
```

---

### Workflow 2: Extract from Paper

Use when adding concepts from a research paper or document:

**Step 1: Extract concepts from paper**
```bash
python scripts/extract_concepts_from_paper.py paper.md concepts_batch.json
```

This will:
1. Parse markdown/text file for section headers and definitions
2. Present each found concept for review
3. Let you assign parents and confluence weights
4. Save as batch file

**Step 2: Ingest batch**
```bash
python scripts/ingest_concept.py concepts_batch.json
```

---

### Workflow 3: Manual Batch File

Create a JSON file with your concepts:

**File: `my_concepts.json`**
```json
[
  {
    "id": "entropic_time",
    "name": "Entropic Time",
    "definition": "Time as emergent from entropy gradients in information fields.",
    "parents": ["bifractal_time", "infodynamics"],
    "confluence": {"bifractal_time": 0.6, "infodynamics": 0.4}
  },
  {
    "id": "topological_memory",
    "name": "Topological Memory",
    "definition": "Memory storage via topological defects in symbolic fields.",
    "parents": ["kronos_memory"],
    "confluence": {"kronos_memory": 1.0}
  }
]
```

**Ingest:**
```bash
python scripts/ingest_concept.py my_concepts.json
```

---

## Parent Suggestion Algorithm

The interactive mode suggests parents based on:

1. **Keyword overlap** - Shared words between concept and candidate parents
2. **Depth preference** - Favors less deep concepts (more foundational)
3. **Node type** - Excludes leaf nodes (applications)

Current implementation uses simple keyword matching.

**Future enhancements:**
- Semantic embeddings for better matching
- Analyze concept definition structure
- Learn from existing confluence patterns

---

## PAC Conservation Rules

### Confluence Weights Must Sum to 1.0

```python
# Valid
{"parent1": 0.7, "parent2": 0.3}  # Sum = 1.0 ✓

# Invalid
{"parent1": 0.5, "parent2": 0.3}  # Sum = 0.8 ✗
```

Tolerance: ±0.05 (95% - 105%)

### Single Parent = 1.0

```python
{"parent": 1.0}  # Always for single parent
```

### Depth Calculation

```python
if no parents:
    depth = 0  # Root concept
else:
    depth = max(parent.depth for parent in parents) + 1
```

### Derivation Path

- **Single parent:** Extend parent's path
- **Multiple parents:** Use primary parent's path (highest confluence)

---

## Examples

### Example 1: Pure DFT Concept

```json
{
  "id": "symbolic_phase_transition",
  "name": "Symbolic Phase Transition",
  "definition": "Discontinuous changes in symbolic field structure driven by entropy regulation.",
  "parents": ["sec", "criticality"],
  "confluence": {"sec": 0.7, "criticality": 0.3}
}
```

**Lineage:** SEC (70%) + Criticality (30%) → Symbolic Phase Transition

### Example 2: Bridge to Existing Field

```json
{
  "id": "neural_pac_dynamics",
  "name": "Neural PAC Dynamics",
  "definition": "Neural network training dynamics interpreted through PAC conservation.",
  "parents": ["pac_conservation", "gradient_descent_bridge"],
  "confluence": {"pac_conservation": 0.6, "gradient_descent_bridge": 0.4}
}
```

**Lineage:** Bridges PAC theory to neural networks

### Example 3: Experimental Validation

```json
{
  "id": "superconductor_sec_alignment",
  "name": "Superconductor SEC Alignment",
  "definition": "Superconducting phase transitions exhibit SEC collapse signatures.",
  "parents": ["sec_validation_quantum"],
  "confluence": {"sec_validation_quantum": 1.0}
}
```

**Lineage:** Adds to experimental validation tree

---

## Batch File Format

```json
[
  {
    "id": "concept_id",           // Required: snake_case identifier
    "name": "Concept Name",       // Required: Human-readable name
    "definition": "...",          // Required: Clear definition
    "parents": ["parent1", ...],  // Required: List of parent IDs
    "confluence": {               // Required: Weights (sum to 1.0)
      "parent1": 0.6,
      "parent2": 0.4
    }
  }
]
```

**Optional Fields (calculated automatically):**
- `depth` - Computed from parents
- `derivation_path` - Built from primary parent
- `first_crystallization` - Set to current time

---

## Validation

The pipeline automatically validates:

✅ **Unique IDs** - No duplicate concept IDs
✅ **Valid parents** - All parent IDs must exist in graph
✅ **Confluence sum** - Weights must sum to ~1.0
✅ **Proper depth** - Depth = max(parent depths) + 1
✅ **Path consistency** - Derivation path matches parent lineage

---

## Error Handling

### Duplicate ID
```
[ERROR] Concept 'sec' already exists!
```
**Fix:** Use a different ID or update existing concept manually

### Invalid Parent
```
[ERROR] Parent 'unknown_concept' not found in graph
```
**Fix:** Ensure parent concept exists or add it first

### Invalid Confluence
```
[ERROR] Weights sum to 0.8, must be ~1.0
```
**Fix:** Adjust weights to sum to 1.0

---

## Files

### Scripts
- `scripts/ingest_concept.py` - Main ingestion tool
- `scripts/extract_concepts_from_paper.py` - Extract from documents

### Data
- `data/dft_knowledge_graph.json` - Main graph (auto-updated)
- `data/concept_batch_template.json` - Example batch file

---

## Tips & Best Practices

### Choosing Parents

1. **Ask "What did this crystallize from?"**
   - Not "what is related to this"
   - But "what potential did this actualize from"

2. **Prefer fewer parents**
   - Single parent when possible
   - Multiple only when genuinely a confluence

3. **Check existing children**
   - Look at similar concepts
   - Follow existing patterns

### Writing Definitions

1. **Be concise** - 1-2 sentences ideal
2. **State what it IS** - Not just what it does
3. **Include key formulas** - If applicable (e.g., "H = -Σ p log p")
4. **Avoid circular definitions** - Don't reference the concept itself

### Confluence Weights

1. **Reflect actual contribution**
   - 0.7/0.3 = heavily from first parent
   - 0.5/0.5 = equal contribution

2. **Common patterns:**
   - Pure extension: Single parent (1.0)
   - Bridge: ~0.5/0.5
   - Enhancement: ~0.7/0.3 (primary + modifier)

---

## Future Enhancements

### Planned Features

- **Semantic search** - Use embeddings to find similar concepts
- **Citation management** - Add paper references automatically
- **Duplicate detection** - Warn if concept similar to existing
- **Validation strengthening** - Check for cycles, orphans
- **Undo/rollback** - Revert recent additions

### Integration

- **LLM-assisted extraction** - Automatically suggest concepts from papers
- **Grimm integration** - Add concepts through conversational interface
- **Web interface** - Visual graph editor
- **Git integration** - Track graph changes over time

---

## Troubleshooting

### Parent suggestions seem wrong

The current algorithm uses simple keyword matching. Manually select better parents from the full node list.

To see all nodes:
```python
from pathlib import Path
from test_dft_knowledge import load_graph

graph = load_graph(Path("data/dft_knowledge_graph.json"))
for node_id, node in sorted(graph.nodes.items()):
    print(f"{node_id}: {node.name} (depth {node.actualization_depth})")
```

### Graph file corrupted

Restore from git:
```bash
git checkout data/dft_knowledge_graph.json
```

### Batch file format errors

Validate JSON:
```bash
python -m json.tool my_concepts.json
```

---

## Summary

The KRONOS ingestion pipeline makes it easy to grow your knowledge graph while maintaining PAC structure. Start with interactive mode to learn the patterns, then move to batch mode for efficiency.

**Remember:** Every concept's identity IS its confluence pattern. Choose parents and weights carefully.
