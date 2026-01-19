"""
Analyze Ingested Concepts

Deep analysis of the repository ingestion to understand:
1. What concepts were extracted?
2. Which concepts have confluence (multiple sources)?
3. What's the signal-to-noise ratio?
4. Do related concepts appear together?

Author: Claude (Dawn Field Institute)
Date: 2026-01-19
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Load repository graph
repo_root = Path(__file__).parent.parent.parent
graph_path = repo_root / "fracton" / "data" / "dfi_repository_knowledge.json"

print("="*70)
print("Repository Ingestion Analysis")
print("="*70)

with open(graph_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

nodes = data['nodes']
print(f"\nTotal nodes: {len(nodes)}")

# 1. Categorize concepts
print("\n" + "="*70)
print("Concept Categories")
print("="*70)

# Generic section headers (noise)
generic_headers = {
    'overview', 'introduction', 'background', 'conclusion', 'summary',
    'usage', 'installation', 'examples', 'contributing', 'license',
    'references', 'see also', 'changelog', 'todo', 'notes',
    'motivation', 'next steps', 'impact', 'status', 'related',
    'validation', 'testing', 'test results', 'documentation',
    'performance', 'known limitations', 'security', 'future work'
}

# Technical keywords
technical_keywords = {
    'kronos', 'cortex', 'lobe', 'soul', 'avatar', 'grimm', 'fracton',
    'pac', 'sec', 'field', 'confluence', 'actualization', 'resonance',
    'potential', 'coherence', 'uncertainty', 'valence', 'arousal',
    'memory', 'graph', 'node', 'edge', 'topology', 'lineage',
    'chrysalis', 'wave', 'collapse', 'crystallization', 'emergence'
}

generic_concepts = []
technical_concepts = []
ambiguous_concepts = []

for node_id, node in nodes.items():
    name = node['name']
    name_lower = name.lower()

    # Check if generic
    is_generic = any(gh in name_lower for gh in generic_headers)

    # Check if technical
    is_technical = any(tk in name_lower for tk in technical_keywords)

    if is_generic and not is_technical:
        generic_concepts.append((node_id, name))
    elif is_technical:
        technical_concepts.append((node_id, name, node.get('definition', '')[:100]))
    else:
        ambiguous_concepts.append((node_id, name))

print(f"\nGeneric headers (noise): {len(generic_concepts)}")
print(f"Technical concepts (signal): {len(technical_concepts)}")
print(f"Ambiguous concepts: {len(ambiguous_concepts)}")

signal_ratio = len(technical_concepts) / len(nodes) * 100
print(f"\nSignal-to-noise ratio: {signal_ratio:.1f}% technical")

# 2. Technical concepts by category
print("\n" + "="*70)
print("Technical Concepts by Domain")
print("="*70)

domains = {
    'KRONOS': [],
    'Cortex/Lobes': [],
    'Soul/Avatar': [],
    'Field Theory': [],
    'PAC/SEC': [],
    'Memory': [],
    'Other': []
}

for node_id, name, definition in technical_concepts:
    name_lower = name.lower()

    if 'kronos' in name_lower:
        domains['KRONOS'].append((name, definition))
    elif any(kw in name_lower for kw in ['cortex', 'lobe']):
        domains['Cortex/Lobes'].append((name, definition))
    elif any(kw in name_lower for kw in ['soul', 'avatar']):
        domains['Soul/Avatar'].append((name, definition))
    elif any(kw in name_lower for kw in ['field', 'confluence', 'actualization', 'resonance']):
        domains['Field Theory'].append((name, definition))
    elif any(kw in name_lower for kw in ['pac', 'sec']):
        domains['PAC/SEC'].append((name, definition))
    elif 'memory' in name_lower:
        domains['Memory'].append((name, definition))
    else:
        domains['Other'].append((name, definition))

for domain, concepts in domains.items():
    if concepts:
        print(f"\n{domain} ({len(concepts)} concepts):")
        for name, definition in concepts[:5]:  # Top 5
            print(f"  - {name}")
            if definition:
                print(f"    {definition}...")

# 3. Confluence analysis (concepts with rich definitions = multiple sources)
print("\n" + "="*70)
print("Confluence Detection")
print("="*70)

# Check for definitions with " | " separator (indicates merged concepts)
confluence_nodes = []
for node_id, node in nodes.items():
    definition = node.get('definition', '')
    if ' | ' in definition:
        # This is a merged concept from multiple files
        sources = definition.count(' | ') + 1
        confluence_nodes.append((node['name'], sources, definition[:150]))

print(f"\nDetected {len(confluence_nodes)} confluence nodes (concepts from multiple files)")

if confluence_nodes:
    print("\nTop confluence concepts:")
    for name, sources, definition in sorted(confluence_nodes, key=lambda x: x[1], reverse=True)[:10]:
        print(f"\n  {name} ({sources} sources):")
        print(f"    {definition}...")

# 4. Coverage analysis - which files contributed most?
print("\n" + "="*70)
print("Source File Analysis")
print("="*70)

# Extract source files from definitions (if they're in "Multiple sources: ..." format)
file_mentions = defaultdict(int)

for node_id, node in nodes.items():
    source = node.get('definition', '')
    if source.startswith('Multiple sources:'):
        # Extract file names
        files = source.split('Multiple sources:')[1].split(' | ')[0]
        for file_part in files.split(','):
            file_mentions[file_part.strip()] += 1

if file_mentions:
    print("\nMost referenced files:")
    for file, count in sorted(file_mentions.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {file}: {count} concepts")

# 5. Key findings
print("\n" + "="*70)
print("Key Findings")
print("="*70)

print("\n✅ What's Working:")
if len(technical_concepts) > 20:
    print(f"  - Extracted {len(technical_concepts)} technical concepts")
if len(confluence_nodes) > 10:
    print(f"  - Detected {len(confluence_nodes)} confluence nodes (multi-document concepts)")
if signal_ratio > 5:
    print(f"  - Signal-to-noise ratio: {signal_ratio:.1f}%")

print("\n⚠️  Issues:")
if len(generic_concepts) > len(technical_concepts) * 2:
    print(f"  - Too many generic headers: {len(generic_concepts)} noise vs {len(technical_concepts)} signal")
    print("  - Recommendation: Add better header filtering")
if signal_ratio < 20:
    print(f"  - Low signal-to-noise ratio: {signal_ratio:.1f}%")
    print("  - Recommendation: Improve concept extraction patterns")
if len(confluence_nodes) < 10:
    print(f"  - Few confluence nodes: {len(confluence_nodes)}")
    print("  - Recommendation: Better name normalization across files")

print("\n📝 Recommendations:")
print("  1. Filter out generic section headers (Next Steps, Impact, Status, etc.)")
print("  2. Focus on:")
print("     - Class/function definitions")
print("     - Technical term definitions (capitalized multi-word phrases)")
print("     - Concepts with code examples")
print("  3. Add relationship detection:")
print("     - Co-occurrence analysis (concepts mentioned together)")
print("     - Hierarchical structure from markdown depth (H2 -> H3)")
print("     - Cross-references in text")

print("\n" + "="*70)
print("Story Coherence Test")
print("="*70)

# Check if key architectural concepts are connected
key_story = {
    'KRONOS': False,
    'Cortex': False,
    'Lobes': False,
    'Soul': False,
    'Field Theory': False,
    'PAC Conservation': False,
    'Confluence': False
}

for node_id, node in nodes.items():
    name = node['name']
    if 'kronos' in name.lower():
        key_story['KRONOS'] = True
    if 'cortex' in name.lower():
        key_story['Cortex'] = True
    if 'lobe' in name.lower():
        key_story['Lobes'] = True
    if 'soul' in name.lower():
        key_story['Soul'] = True
    if any(kw in name.lower() for kw in ['field', 'actualization']):
        key_story['Field Theory'] = True
    if 'pac' in name.lower():
        key_story['PAC Conservation'] = True
    if 'confluence' in name.lower():
        key_story['Confluence'] = True

print("\nCore architectural concepts found:")
for concept, found in key_story.items():
    status = "✓" if found else "✗"
    print(f"  {status} {concept}")

if all(key_story.values()):
    print("\n✅ STORY COHERENCE: All core concepts extracted!")
    print("   Scattered documentation successfully unified")
else:
    missing = [k for k, v in key_story.items() if not v]
    print(f"\n⚠️  STORY COHERENCE: Missing {len(missing)} core concepts: {', '.join(missing)}")

print("\n" + "="*70)
