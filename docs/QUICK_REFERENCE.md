# Fracton Quick Reference

**Version**: 2.0.0

Quick reference guide for Fracton classes and common patterns.

---

## Imports

```python
# Core memory
from fracton.storage import KronosMemory, NodeType

# Foundation engines (advanced)
from fracton.storage.pac_engine import PACEngine, PACNode
from fracton.storage.sec_operators import SECOperators
from fracton.storage.med_validator import MEDValidator
from fracton.storage.distance_validator import DistanceValidator
from fracton.storage.foundation_integration import FoundationIntegration

# Types
from fracton.storage.types import NodeType, RelationType
```

---

## KronosMemory Cheat Sheet

### Initialization

```python
# Basic
memory = KronosMemory(storage_path="./data", namespace="demo")

# With GPU
memory = KronosMemory(
    storage_path="./data",
    namespace="demo",
    device="cuda",
    embedding_model="base"  # larger model
)

# Connect
await memory.connect()
await memory.create_graph("my_graph")
```

### Storing Nodes

```python
# Root node
root_id = await memory.store(
    content="Root concept",
    graph="my_graph",
    node_type=NodeType.CONCEPT
)

# Child with validation
child_id = await memory.store(
    content="Child concept",
    graph="my_graph",
    node_type=NodeType.CONCEPT,
    parent_id=root_id,  # PAC validates Fibonacci recursion
    importance=0.8,
    metadata={"author": "Alice", "tags": ["ml", "ai"]}
)
```

### Querying

```python
# Basic query
results = await memory.query(
    query_text="machine learning",
    graphs=["my_graph"]
)

# Filtered query
results = await memory.query(
    query_text="neural networks",
    graphs=["research", "code"],
    node_types=[NodeType.CONCEPT, NodeType.PAPER],
    limit=20,
    min_similarity=0.6
)

# Process results
for r in results:
    print(f"[{r.score:.3f}] {r.node.content}")
    print(f"  Metadata: {r.node.metadata}")
```

### Health Monitoring

```python
# Get health metrics
health = memory.get_foundation_health()

# Key metrics
print(f"Balance Ξ: {health['balance_operator']['latest']:.4f}")
print(f"Target Ξ: {health['constants']['xi']:.4f}")
print(f"Duty cycle: {health['duty_cycle']['latest']:.3f}")
print(f"c²: {health['c_squared']['latest']:.2f}")

# Check for issues
if health['balance_operator']['latest'] > 1.2:
    print("⚠️  Collapse risk!")
elif health['balance_operator']['latest'] < 0.5:
    print("⚠️  Field decay!")
```

### Statistics

```python
stats = await memory.get_stats()
print(f"Nodes: {stats['total_nodes']}")
print(f"Graphs: {stats['total_graphs']}")
print(f"Collapses: {stats['collapses']}")
```

---

## Foundation Engines

### PAC Engine

```python
pac = PACEngine()

# Compute potential at depth
pot = pac.compute_potential(depth=0)  # 1.0 (root)
pot = pac.compute_potential(depth=1)  # 0.618 (child)
pot = pac.compute_potential(depth=2)  # 0.382 (grandchild)

# Verify Fibonacci recursion
is_valid, residual = pac.verify_fibonacci_recursion(
    parent_potential=1.0,
    child1_potential=0.618,
    child2_potential=0.382
)

# Compute balance operator
from fracton.storage.pac_engine import PACNode
parent = PACNode(potential=1.0, depth=0)
child1 = PACNode(potential=0.618, depth=1)
child2 = PACNode(potential=0.382, depth=1)
xi = pac.compute_balance_operator(parent, [child1, child2])

# Check collapse
status = pac.check_collapse_trigger(xi)  # "STABLE", "COLLAPSE", or "DECAY"
```

### SEC Operators

```python
import torch
sec = SECOperators()

# Merge embeddings
emb1 = torch.randn(384)
emb2 = torch.randn(384)
merged = sec.merge([emb1, emb2])

# Branch embedding
branches = sec.branch(emb1, num_branches=3)

# Compute gradient
grad = sec.gradient(emb1, emb2)

# Duty cycle
duty = sec.compute_duty_cycle(
    attraction_time=0.618,
    repulsion_time=0.382
)  # ~0.618

# SEC score
score = sec.compute_sec_score(
    similarity=0.85,
    entropy=0.3,
    recency=0.9
)
```

### MED Validator

```python
med = MEDValidator()

# Validate structure
is_valid, quality, msg = med.validate_structure(
    depth=1,  # Must be ≤ 1
    num_children=2  # Must be ≤ 3
)

# Compute quality
quality = med.compute_emergence_quality(
    depth_compliance=1.0,
    node_compliance=0.667  # 2/3 children
)
```

### Distance Validator

```python
dist = DistanceValidator()

# Measure c² constant
embeddings = [torch.randn(384) for _ in range(100)]
c_squared = dist.measure_model_constant(embeddings)

# Validate conservation
parent_emb = torch.randn(384)
child_embs = [torch.randn(384), torch.randn(384)]
is_conserved, ratio = dist.validate_conservation(
    parent_emb,
    child_embs,
    tolerance=0.2
)

# Fractal dimension
dimension = dist.compute_fractal_dimension(embeddings)
```

### Foundation Integration

```python
integration = FoundationIntegration()

# Validate node storage (all foundations)
results = integration.validate_storage(
    node=child_node,
    parent=parent_node,
    siblings=[sibling1, sibling2]
)

print(f"Overall valid: {results['overall_valid']}")
print(f"PAC valid: {results['pac_valid']}, residual: {results['pac_residual']:.2e}")
print(f"Balance Ξ: {results['balance_operator']:.4f}")
print(f"Collapse: {results['collapse_status']}")
print(f"MED valid: {results['med_valid']}, quality: {results['med_quality']:.3f}")
print(f"Distance conserved: {results['distance_conserved']}")
print(f"c²: {results['c_squared']:.2f}")
print(f"Duty cycle: {results['duty_cycle']:.3f}")
```

---

## Common Patterns

### Hierarchical Knowledge

```python
# Create hierarchy
root = await memory.store("AI", graph="kb", node_type=NodeType.CONCEPT)
ml = await memory.store("Machine Learning", graph="kb",
                        node_type=NodeType.CONCEPT, parent_id=root)
dl = await memory.store("Deep Learning", graph="kb",
                        node_type=NodeType.CONCEPT, parent_id=ml)

# Query finds all levels
results = await memory.query("learning algorithms", graphs=["kb"])
```

### Versioned Content

```python
# Version 1
v1 = await memory.store("Initial idea", graph="docs", node_type=NodeType.CONCEPT)

# Version 2 (delta stored automatically)
v2 = await memory.store("Initial idea refined", graph="docs",
                        node_type=NodeType.CONCEPT, parent_id=v1)

# Version 3
v3 = await memory.store("Initial idea refined and validated", graph="docs",
                        node_type=NodeType.CONCEPT, parent_id=v2)
```

### Cross-Graph Knowledge

```python
# Research paper
paper_id = await memory.store(
    "Attention Is All You Need",
    graph="research",
    node_type=NodeType.PAPER
)

# Code implementation
code_id = await memory.store(
    "Transformer implementation",
    graph="code",
    node_type=NodeType.COMMIT
)

# Query across both
results = await memory.query(
    "transformer architecture",
    graphs=["research", "code"]
)
```

### Metadata Filtering

```python
# Store with rich metadata
await memory.store(
    "Research findings",
    graph="lab",
    node_type=NodeType.EXPERIMENT,
    metadata={
        "researcher": "Alice",
        "date": "2024-12-29",
        "status": "completed",
        "confidence": 0.95
    }
)

# Query and filter
results = await memory.query("findings", graphs=["lab"])
high_conf = [r for r in results if r.node.metadata.get("confidence", 0) > 0.9]
```

### Importance Weighting

```python
# Critical knowledge
await memory.store(
    "Core algorithm",
    graph="code",
    node_type=NodeType.FUNCTION,
    importance=1.0  # Highest importance
)

# Supporting detail
await memory.store(
    "Helper utility",
    graph="code",
    node_type=NodeType.FUNCTION,
    importance=0.3  # Lower importance
)

# SEC ranking considers importance
results = await memory.query("algorithm", graphs=["code"])
```

---

## Constants Reference

### PAC Constants

```python
PHI = 1.618033988749895  # Golden ratio
XI = 1.0571238898  # Balance operator target (1 + π/F₁₀)
LAMBDA_STAR = 0.97  # Potential decay rate
DUTY_CYCLE = 0.618  # φ/(φ+1), 61.8% attraction
CONSERVATION_TOLERANCE = 1e-10  # Precision threshold
```

### SEC Constants

```python
ATTRACTION_CYCLES = 4
REPULSION_CYCLES = 1
BALANCE_RATIO = 4.0  # 4:1 attraction/repulsion
```

### MED Constants

```python
MAX_DEPTH = 1  # Universal depth bound
MAX_CHILDREN = 3  # Universal sibling count bound
```

---

## Formulas Quick Reference

### PAC

```python
# Potential at depth
Ψ(depth) = amplitude * φ^(-depth)

# Fibonacci recursion
Ψ(k) = Ψ(k+1) + Ψ(k+2)

# Balance operator
Ξ = 1 + (excess_pressure / parent_potential)
where excess_pressure = Σ(children_potential) - parent_potential
```

### SEC

```python
# Duty cycle
duty = φ / (φ + 1) = 0.618

# SEC resonance score
sec_score = similarity * (1 - entropy) * recency
```

### MED

```python
# Quality score
quality = sqrt(depth_compliance * node_compliance)

# Universal bounds
depth ≤ 1
children ≤ 3
```

### E=mc²

```python
# Energy
E = ||embedding||²

# Conservation
E_parent ≈ Σ(E_children) + binding_energy

# Fractal dimension
D = log(N) / log(1/r)
```

---

## Node Types Quick List

```python
# Code/Repository
NodeType.FILE
NodeType.FUNCTION
NodeType.CLASS
NodeType.MODULE
NodeType.COMMIT

# Research/Concepts
NodeType.CONCEPT
NodeType.PAPER
NodeType.EXPERIMENT
NodeType.HYPOTHESIS

# Personal/Context
NodeType.GOAL
NodeType.PREFERENCE
NodeType.FACT

# Social/Content
NodeType.POST
NodeType.THREAD
NodeType.ANNOUNCEMENT

# Services/Monitoring
NodeType.SERVICE
NodeType.STATUS
NodeType.ALERT
```

---

## Health Thresholds

### Balance Operator (Ξ)

- `Ξ > 1.2`: 🔴 Collapse risk
- `1.0 < Ξ ≤ 1.2`: 🟡 Slightly elevated
- `0.5 ≤ Ξ ≤ 1.0`: 🟢 Stable
- `Ξ < 0.5`: 🔴 Field decay

### Duty Cycle

- Target: `0.618` (golden ratio)
- Range: `0.5 - 0.7` acceptable

### MED Quality

- `> 0.9`: 🟢 Excellent
- `0.7 - 0.9`: 🟡 Good
- `< 0.7`: 🔴 Poor structure

---

## Error Handling

```python
try:
    await memory.store(...)
except ValueError as e:
    # Parent not found or graph doesn't exist
    print(f"Storage error: {e}")

try:
    results = await memory.query(...)
except Exception as e:
    # Query error
    print(f"Query error: {e}")

# Validation warnings (logged, not raised)
# - PAC conservation residual > 1e-6
# - Balance operator < 0.5 or > 1.2
# - MED bounds violated
```

---

## Performance Tips

1. **Use GPU for large queries**: `device="cuda"`
2. **Scope queries**: Always specify `graphs` parameter
3. **Filter by type**: Use `node_types` to narrow search
4. **Batch storage**: Store related nodes in sequence
5. **Monitor health**: Check `get_foundation_health()` periodically
6. **Set importance**: Use `importance` parameter for critical nodes

---

## See Also

- [Full API Reference](API_REFERENCE.md)
- [KronosMemory Spec](../.spec/kronos-memory.spec.md)
- [Testing Guide](../tests/TESTING_GUIDE.md)
- [Examples](../examples/)
