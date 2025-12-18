# Fracton Language Specification

## Table of Contents
1. [Language Overview](#language-overview)
2. [Core Concepts](#core-concepts)
3. [Syntax Reference](#syntax-reference)
4. [Built-in Functions](#built-in-functions)
5. [Memory Model](#memory-model)
6. [Tool Expression](#tool-expression)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)
9. [PAC Tree Monitoring](#pac-tree-monitoring)
10. [Kronos Integration](#kronos-integration)

## Language Overview

Fracton is a recursive, entropy-aware computational modeling language designed for infodynamics research. It treats recursion as the fundamental computational primitive and uses entropy dynamics to control execution flow.

### Key Characteristics
- **Recursive-First**: All computation flows through recursive function calls
- **Entropy-Driven**: Execution controlled by entropy thresholds and field dynamics
- **Bifractal**: Forward and reverse traces maintained for all operations
- **Context-Aware**: Execution context includes entropy, depth, and field state
- **Tool-Expressive**: External systems accessed through contextual expression

## Core Concepts

### 1. Recursive Functions

All Fracton functions are potentially recursive and receive two parameters:
- `memory`: Shared memory field containing persistent state
- `context`: Execution context with entropy, depth, and metadata

```python
@fracton.recursive
def my_function(memory, context):
    # Function body
    return result
```

### 2. Entropy Gates

Functions can specify minimum entropy thresholds for execution:

```python
@fracton.entropy_gate(min_threshold=0.5, max_threshold=0.9)
def entropy_sensitive_function(memory, context):
    # Only executes when 0.5 <= context.entropy <= 0.9
    pass
```

### 3. Context Management

The execution context carries metadata through recursive calls:

```python
class Context:
    entropy: float          # Current entropy level (0.0 - 1.0)
    depth: int             # Recursion depth
    trace_id: str          # Unique identifier for trace
    field_state: dict      # Field-specific metadata
    parent_context: Context # Reference to calling context
```

### 4. Memory Fields

Shared memory structures that maintain state across recursive calls:

```python
with fracton.memory_field(capacity=1000) as field:
    # Operations within this field share memory
    result = recursive_function(field, context)
```

## Syntax Reference

### Function Decorators

#### @fracton.recursive
Marks a function as recursively callable within the Fracton runtime.

```python
@fracton.recursive
def process_data(memory, context):
    return memory.transform(context.entropy)
```

#### @fracton.entropy_gate(min_threshold, max_threshold=1.0)
Sets entropy thresholds for function execution.

```python
@fracton.entropy_gate(0.7)  # Only execute if entropy >= 0.7
def high_entropy_operation(memory, context):
    pass
```

#### @fracton.tool_binding(tool_name)
Marks a function as a tool expression interface.

```python
@fracton.tool_binding("github")
def github_operations(memory, context):
    return fracton.express_tool("github", context)
```

### Control Flow

#### fracton.recurse(function, memory, context)
Initiates a recursive call with proper tracing.

```python
@fracton.recursive
def fibonacci(memory, context):
    if context.depth <= 1:
        return 1
    
    a = fracton.recurse(fibonacci, memory, context.deeper(1))
    b = fracton.recurse(fibonacci, memory, context.deeper(2))
    return a + b
```

#### fracton.crystallize(data, patterns=None)
Crystallizes data into stable structures based on entropy patterns.

```python
result = fracton.crystallize(
    processed_data, 
    patterns=context.discovered_patterns
)
```

#### fracton.branch(condition, if_true, if_false, memory, context)
Entropy-aware conditional branching.

```python
result = fracton.branch(
    context.entropy > 0.5,
    high_entropy_path,
    low_entropy_path,
    memory,
    context
)
```

### Memory Operations

#### memory.get(key, default=None)
Retrieves value from shared memory field.

```python
symbols = memory.get("symbols", [])
```

#### memory.set(key, value)
Stores value in shared memory field.

```python
memory.set("processed_symbols", crystallized_symbols)
```

#### memory.transform(entropy_level)
Applies entropy-based transformation to memory contents.

```python
transformed = memory.transform(context.entropy)
```

#### memory.snapshot()
Creates a snapshot of current memory state for rollback.

```python
snapshot = memory.snapshot()
# ... operations ...
memory.restore(snapshot)  # Rollback if needed
```

### Context Operations

#### context.deeper(steps=1)
Creates a new context with increased depth.

```python
child_context = context.deeper(2)  # Increase depth by 2
```

#### context.with_entropy(new_entropy)
Creates a new context with modified entropy.

```python
high_entropy_context = context.with_entropy(0.9)
```

#### context.with_metadata(**kwargs)
Adds metadata to context.

```python
annotated_context = context.with_metadata(
    operation="pattern_analysis",
    timestamp=time.now()
)
```

## Built-in Functions

### Core Primitives

#### fracton.initialize_field(capacity=1000, entropy=0.5)
Creates a new memory field with specified capacity and initial entropy.

#### fracton.merge_fields(*fields)
Merges multiple memory fields into a single field.

#### fracton.analyze_trace(trace)
Analyzes a bifractal trace for patterns and performance metrics.

#### fracton.visualize_trace(trace, format="graph")
Generates visualization of recursive execution trace.

### Entropy Functions

#### fracton.calculate_entropy(data)
Calculates entropy of given data structure.

#### fracton.entropy_gradient(field, direction="forward")
Calculates entropy gradient across memory field.

#### fracton.regulate_entropy(field, target_entropy)
Adjusts field entropy toward target value.

### Pattern Recognition

#### fracton.detect_patterns(memory, min_confidence=0.7)
Identifies patterns in memory contents.

#### fracton.validate_patterns(patterns, test_data)
Validates discovered patterns against test data.

## Memory Model

### Field Structure

Memory fields are hierarchical structures that maintain:
- **Content**: Actual data being processed
- **Metadata**: Information about data relationships and entropy
- **Traces**: Forward and reverse operation histories
- **Snapshots**: Point-in-time states for rollback

### Memory Isolation

Each memory field provides isolation boundaries:
```python
with fracton.memory_field() as field1:
    with fracton.memory_field() as field2:
        # field1 and field2 are isolated
        # Operations in field2 don't affect field1
        pass
```

### Cross-Field Communication

Fields can communicate through controlled interfaces:
```python
field1.send_message(field2, data, entropy_threshold=0.6)
response = field1.receive_message(timeout=1.0)
```

## Tool Expression

### Tool Registry

Tools are registered with the Fracton runtime:
```python
fracton.register_tool("database", DatabaseConnector())
fracton.register_tool("github", GitHubInterface())
```

### Context-Aware Tool Access

Tools are accessed based on current execution context:
```python
@fracton.tool_binding("database")
def data_operations(memory, context):
    if context.entropy > 0.8:
        return fracton.express_tool("database", "high_entropy_query")
    else:
        return fracton.express_tool("database", "stable_query")
```

### Tool Chaining

Tools can be chained through recursive calls:
```python
@fracton.recursive
def data_pipeline(memory, context):
    # Fetch data
    raw_data = fracton.express_tool("database", "fetch")
    
    # Process recursively
    processed = fracton.recurse(process_data, memory, context)
    
    # Store results
    fracton.express_tool("storage", "save", processed)
```

## Error Handling

### Entropy-Based Error Recovery

Fracton provides entropy-aware error handling:
```python
try:
    result = fracton.recurse(risky_operation, memory, context)
except fracton.EntropyError as e:
    # Adjust entropy and retry
    adjusted_context = context.with_entropy(e.suggested_entropy)
    result = fracton.recurse(risky_operation, memory, adjusted_context)
```

### Trace-Based Debugging

Errors include full bifractal traces for debugging:
```python
try:
    result = complex_recursive_operation(memory, context)
except fracton.RecursionError as e:
    # Analyze the trace to understand the failure
    fracton.visualize_trace(e.trace)
    print(f"Failed at depth {e.failure_depth}")
```

### Graceful Degradation

Functions can specify fallback behaviors:
```python
@fracton.recursive
@fracton.fallback(simple_fallback_function)
def complex_operation(memory, context):
    # Complex logic that might fail
    pass
```

## Performance Considerations

### Tail Recursion Optimization

Fracton optimizes tail-recursive calls:
```python
@fracton.recursive
@fracton.tail_recursive
def countdown(memory, context):
    if context.depth <= 0:
        return "done"
    return fracton.recurse(countdown, memory, context.deeper(-1))
```

### Memory Management

- Use `memory.compact()` to reduce memory footprint
- Set appropriate field capacities to avoid memory exhaustion
- Use `memory.snapshot()` sparingly as it consumes additional memory

### Entropy Calculation

- Entropy calculations can be expensive for large data structures
- Cache entropy values when possible using `context.cache_entropy()`
- Use entropy approximation for performance-critical paths

### Trace Management

- Bifractal traces grow with recursion depth
- Use `fracton.prune_trace()` to remove unnecessary trace elements
- Set trace depth limits for production systems

---

## 9. PAC Tree Monitoring

PAC Tree (Pattern-Activated Context) Monitoring enables real-time observation of how patterns form, generalize, and potentially overfit during learning. This section specifies the monitoring subsystem for Fracton-based systems.

### 9.1 Core Monitoring Types

```python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple
import numpy as np

class PatternType(Enum):
    """Classification of pattern behavior."""
    ABSTRACT = "abstract"        # High activation diversity, generalizable
    SPECIFIC = "specific"        # Low diversity, potentially memorized
    TRANSITIONAL = "transitional"  # In transition between states

@dataclass
class TreeMetrics:
    """Structural metrics for PAC tree health."""
    depth: int                    # Maximum tree depth
    breadth: int                  # Number of leaf nodes
    branching_factor: float       # Average children per non-leaf
    compression_ratio: float      # Unique patterns / total tokens seen
    reuse_ratio: float           # Pattern reuse frequency
    byref_candidates: int        # Branches similar enough for byref optimization

@dataclass
class PatternProfile:
    """Behavioral profile for a single pattern."""
    pattern_id: str
    activation_count: int         # How often activated
    activation_contexts: Set[str]  # Unique contexts that activated it
    activation_diversity: float    # len(contexts) / count
    child_count: int              # Number of child patterns
    depth: int                    # Position in tree
    pattern_type: PatternType     # Classified type
    entropy_signature: float      # Pattern's entropy contribution
```

### 9.2 Pattern Classification

```python
def classify_pattern(profile: PatternProfile, thresholds: dict) -> PatternType:
    """
    Classify a pattern based on its activation behavior.
    
    Classification rules:
    - ABSTRACT: High diversity (>0.3), many children (>10), any depth
    - SPECIFIC: Low diversity (<0.1), few children (<3), high activation
    - TRANSITIONAL: Between thresholds, or recently changed classification
    """
    if (profile.activation_diversity > thresholds.get('abstract_diversity', 0.3) 
        and profile.child_count > thresholds.get('abstract_children', 10)):
        return PatternType.ABSTRACT
    
    if (profile.activation_diversity < thresholds.get('specific_diversity', 0.1)
        and profile.child_count < thresholds.get('specific_children', 3)
        and profile.activation_count > thresholds.get('specific_min_activations', 100)):
        return PatternType.SPECIFIC
    
    return PatternType.TRANSITIONAL
```

### 9.3 Generalization Zone Detection

Generalization zones are connected regions of ABSTRACT patterns. Healthy learning produces expanding generalization zones with SPECIFIC patterns at the leaves.

```python
@dataclass
class GeneralizationZone:
    """A connected region of abstract patterns."""
    zone_id: str
    root_pattern: str
    abstract_patterns: Set[str]
    boundary_patterns: Set[str]    # TRANSITIONAL at the edges
    specific_leaves: Set[str]      # SPECIFIC patterns at leaves
    zone_entropy: float            # Aggregate entropy of zone
    zone_depth_range: Tuple[int, int]

def detect_generalization_zones(
    tree: 'PACTree',
    min_zone_size: int = 5
) -> List[GeneralizationZone]:
    """
    Find connected regions of abstract patterns.
    
    A healthy tree has:
    - Large generalization zones near the root
    - TRANSITIONAL patterns at zone boundaries
    - SPECIFIC patterns only at leaves (fine-grained distinctions)
    
    Warning signs:
    - SPECIFIC patterns at shallow depths (memorization)
    - Small, disconnected generalization zones
    - No TRANSITIONAL boundary layer
    """
    zones = []
    visited = set()
    
    for pattern in tree.patterns:
        if pattern.id in visited:
            continue
        if pattern.type != PatternType.ABSTRACT:
            continue
            
        # BFS to find connected abstract region
        zone_patterns = set()
        boundary = set()
        leaves = set()
        queue = [pattern]
        
        while queue:
            current = queue.pop(0)
            if current.id in visited:
                continue
            visited.add(current.id)
            
            if current.type == PatternType.ABSTRACT:
                zone_patterns.add(current.id)
                for child in current.children:
                    queue.append(child)
            elif current.type == PatternType.TRANSITIONAL:
                boundary.add(current.id)
            elif current.type == PatternType.SPECIFIC:
                leaves.add(current.id)
        
        if len(zone_patterns) >= min_zone_size:
            zones.append(GeneralizationZone(
                zone_id=f"zone_{len(zones)}",
                root_pattern=pattern.id,
                abstract_patterns=zone_patterns,
                boundary_patterns=boundary,
                specific_leaves=leaves,
                zone_entropy=tree.compute_zone_entropy(zone_patterns),
                zone_depth_range=(
                    min(tree.get_depth(p) for p in zone_patterns),
                    max(tree.get_depth(p) for p in zone_patterns)
                )
            ))
    
    return zones
```

### 9.4 Byref Optimization Detection

When similar patterns emerge in different branches, they can be connected by reference (byref) for memory efficiency and improved generalization.

```python
@dataclass
class BranchSimilarity:
    """Detected similarity between PAC tree branches."""
    branch_a: str
    branch_b: str
    similarity_score: float         # 0.0 to 1.0
    shared_structure_depth: int     # How deep the similarity extends
    estimated_memory_savings: int   # Bytes saved by byref
    
def detect_byref_candidates(
    tree: 'PACTree',
    similarity_threshold: float = 0.85
) -> List[BranchSimilarity]:
    """
    Find branches that could benefit from byref optimization.
    
    Uses euclidean distance validation pattern:
    - Branches with same structural depth
    - Similar activation patterns
    - Similar child distributions
    
    Returns candidates sorted by estimated memory savings.
    """
    candidates = []
    branches = tree.get_all_branches()
    
    for i, branch_a in enumerate(branches):
        for branch_b in branches[i+1:]:
            similarity = compute_branch_similarity(branch_a, branch_b)
            
            if similarity.score >= similarity_threshold:
                candidates.append(BranchSimilarity(
                    branch_a=branch_a.id,
                    branch_b=branch_b.id,
                    similarity_score=similarity.score,
                    shared_structure_depth=similarity.depth,
                    estimated_memory_savings=estimate_byref_savings(
                        branch_a, branch_b
                    )
                ))
    
    return sorted(candidates, key=lambda x: x.estimated_memory_savings, reverse=True)
```

### 9.5 SCBF Integration

The Standard Consciousness Benchmark Framework provides metrics that integrate with PAC tree monitoring.

```python
@dataclass
class SCBFMetrics:
    """SCBF metrics relevant to PAC tree health."""
    entropy_collapse_risk: float    # How close to collapse (0-1)
    phase_alignment: float          # Alignment with PHI_XI
    criticality: float              # Self-organized criticality measure
    field_coherence: float          # Cross-pattern coherence

class SCBFBridge:
    """Bridge between SCBF metrics and PAC tree monitoring."""
    
    PHI_XI = 0.915965594177  # Golden ratio of criticality
    
    def __init__(self, entropy_detector, phase_tracker):
        self.entropy_detector = entropy_detector
        self.phase_tracker = phase_tracker
    
    def get_tree_health(self, tree: 'PACTree') -> SCBFMetrics:
        """Compute SCBF metrics for current tree state."""
        # Entropy collapse: too many SPECIFIC patterns
        specific_ratio = tree.count_by_type(PatternType.SPECIFIC) / tree.total_patterns
        collapse_risk = specific_ratio ** 2  # Quadratic penalty
        
        # Phase alignment: are transitions happening at right depths?
        phase_alignment = self.phase_tracker.measure_alignment(tree)
        
        # Criticality: distance from PHI_XI
        tree_criticality = tree.compute_criticality()
        criticality = 1.0 - abs(tree_criticality - self.PHI_XI)
        
        # Field coherence: do patterns work together?
        coherence = tree.compute_pattern_coherence()
        
        return SCBFMetrics(
            entropy_collapse_risk=collapse_risk,
            phase_alignment=phase_alignment,
            criticality=criticality,
            field_coherence=coherence
        )
```

### 9.6 Training Interventions

When monitoring detects problematic patterns, interventions can be applied to encourage generalization.

```python
class InterventionType(Enum):
    """Types of training interventions."""
    ADD_NOISE = "add_noise"           # Add Gaussian noise to embeddings
    ADD_DROPOUT = "add_dropout"       # Temporary dropout increase
    TOKEN_MASKING = "token_masking"   # Mask tokens to force context
    POSITION_PERTURB = "position_perturb"  # Shuffle positions slightly
    BRANCH_PRUNE = "branch_prune"     # Remove overly specific branches
    BYREF_MERGE = "byref_merge"       # Connect similar branches

@dataclass
class Intervention:
    """A specific intervention to apply."""
    type: InterventionType
    target_patterns: List[str]
    strength: float  # 0.0 to 1.0
    duration_steps: int

class GeneralizationNurturingTrainer:
    """Trainer that actively nurtures generalization."""
    
    def __init__(self, model, monitor, base_trainer):
        self.model = model
        self.monitor = monitor
        self.base_trainer = base_trainer
        self.intervention_history = []
    
    def training_step(self, batch):
        # Get current tree state
        tree_state = self.monitor.analyze()
        
        # Check for concerning patterns
        if tree_state.entropy_collapse_risk > 0.3:
            self.apply_intervention(Intervention(
                type=InterventionType.ADD_NOISE,
                target_patterns=tree_state.high_risk_patterns,
                strength=0.1,
                duration_steps=100
            ))
        
        if tree_state.byref_candidates:
            top_candidate = tree_state.byref_candidates[0]
            if top_candidate.similarity_score > 0.95:
                self.apply_intervention(Intervention(
                    type=InterventionType.BYREF_MERGE,
                    target_patterns=[top_candidate.branch_a, top_candidate.branch_b],
                    strength=1.0,
                    duration_steps=1  # Immediate
                ))
        
        # Normal training step with active interventions
        return self.base_trainer.step_with_interventions(
            batch, self.active_interventions
        )
```

### 9.7 Monitoring Visualization

For interactive debugging, PAC tree state can be visualized:

```python
def visualize_tree_health(tree_state: TreeMetrics, zones: List[GeneralizationZone]) -> str:
    """Generate ASCII visualization of tree health."""
    lines = []
    lines.append("═══ PAC Tree Health ═══")
    lines.append(f"Depth: {tree_state.depth} | Breadth: {tree_state.breadth}")
    lines.append(f"Branching Factor: {tree_state.branching_factor:.2f}")
    lines.append(f"Compression: {tree_state.compression_ratio:.4f}")
    lines.append(f"Byref Candidates: {tree_state.byref_candidates}")
    lines.append("")
    lines.append("Generalization Zones:")
    for zone in zones:
        health = "🟢" if len(zone.abstract_patterns) > 10 else "🟡" if len(zone.abstract_patterns) > 5 else "🔴"
        lines.append(f"  {health} {zone.zone_id}: {len(zone.abstract_patterns)} abstract, "
                    f"{len(zone.specific_leaves)} specific leaves")
    return "\n".join(lines)
```

### 9.8 Module Structure

The monitoring subsystem is organized as follows:

```
fracton/
├── monitoring/
│   ├── __init__.py              # Public API exports
│   ├── pac_tree_monitor.py      # Core TreeMetrics, PatternProfile
│   ├── generalization_monitor.py # Zone detection, pattern classification
│   ├── interventions.py         # Training interventions
│   ├── scbf_bridge.py          # SCBF metric integration
│   └── visualization.py        # Tree health visualization
```

Usage:
```python
from fracton.monitoring import (
    PACTreeMonitor,
    GeneralizationZone,
    detect_generalization_zones,
    GeneralizationNurturingTrainer
)

# Attach monitor to model
monitor = PACTreeMonitor(model.pac_tree)

# Training with nurturing
trainer = GeneralizationNurturingTrainer(model, monitor, base_trainer)
for batch in dataloader:
    loss = trainer.training_step(batch)
    
    # Periodic health check
    if step % 100 == 0:
        health = monitor.get_tree_health()
        print(visualize_tree_health(health, monitor.zones))
```

---

## 10. Kronos Integration

Reserved for Kronos temporal coherence integration. See `cip-core/kronos/` for current Kronos specification.

---

This specification provides the foundation for implementing and using the Fracton language. For implementation details, see the architecture documentation.
