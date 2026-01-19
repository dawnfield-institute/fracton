"""
Test the ingestion pipeline with a sample concept.
"""

import sys
import json
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Create test batch file
test_concepts = [
    {
        "id": "entropic_time_flow",
        "name": "Entropic Time Flow",
        "definition": "Time as emergent directionality from entropy gradients in information fields. Clock emerges from irreversibility.",
        "parents": ["bifractal_time", "infodynamics"],
        "confluence": {"bifractal_time": 0.6, "infodynamics": 0.4}
    },
    {
        "id": "topological_memory_defects",
        "name": "Topological Memory Defects",
        "definition": "Memory storage via stable topological defects in symbolic field configurations. Information persists in topology.",
        "parents": ["kronos_memory", "mobius_tensor"],
        "confluence": {"kronos_memory": 0.7, "mobius_tensor": 0.3}
    },
    {
        "id": "consciousness_field_collapse",
        "name": "Consciousness as Field Collapse",
        "definition": "Consciousness emerges from recursive symbolic collapse creating observer-dependent information structures.",
        "parents": ["sec", "measurement_problem"],
        "confluence": {"sec": 0.6, "measurement_problem": 0.4}
    }
]

# Save test batch
batch_path = Path(__file__).parent.parent / "data" / "test_concepts_batch.json"
with open(batch_path, 'w') as f:
    json.dump(test_concepts, f, indent=2)

print(f"Created test batch file: {batch_path}")
print(f"Contains {len(test_concepts)} concepts:")
for c in test_concepts:
    print(f"  - {c['name']}")

print(f"\nTo ingest these concepts, run:")
print(f"  python scripts/ingest_concept.py {batch_path}")
