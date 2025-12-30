# Fracton API Reference

**Version**: 2.0.0
**Last Updated**: 2024-12-29

This document provides comprehensive API documentation for all Fracton classes and modules.

---

## Table of Contents

1. [KronosMemory](#kronosmemory) - Main memory system
2. [PAC Engine](#pac-engine) - Potential-Actualization Conservation
3. [SEC Operators](#sec-operators) - Symbolic Entropy Collapse
4. [MED Validator](#med-validator) - Macro Emergence Dynamics
5. [Distance Validator](#distance-validator) - E=mc² Validation
6. [Foundation Integration](#foundation-integration) - Integration Layer
7. [Data Classes](#data-classes) - Core data structures
8. [Enumerations](#enumerations) - Node types, relation types, etc.

---

## KronosMemory

**Location**: `fracton/storage/kronos_memory.py`

### Class: `KronosMemory`

Main unified memory system with PAC/SEC/MED theoretical foundations.

#### Constructor

```python
KronosMemory(
    storage_path: Union[str, Path],
    namespace: str = "default",
    device: str = "cpu",
    embedding_model: str = "mini",
    embedding_dim: int = 384,
    backend: str = "chroma"
)
```

**Parameters**:
- `storage_path` (str | Path): Directory for storing data
- `namespace` (str): Namespace for isolating different projects (default: "default")
- `device` (str): Compute device - "cpu" or "cuda" (default: "cpu")
- `embedding_model` (str): Embedding model - "mini", "base", or "large" (default: "mini")
  - "mini": all-MiniLM-L6-v2 (384 dimensions)
  - "base": all-mpnet-base-v2 (768 dimensions)
  - "large": Custom large model
- `embedding_dim` (int): Embedding dimension (default: 384)
- `backend` (str): Storage backend - "chroma", "sqlite", "neo4j", "qdrant" (default: "chroma")

**Example**:
```python
from fracton.storage import KronosMemory

memory = KronosMemory(
    storage_path="./data",
    namespace="research",
    device="cpu",
    embedding_model="mini"
)
```

#### Methods

##### `connect()`

```python
async def connect() -> None
```

Initialize connection to storage backend and embedding model.

**Raises**:
- `RuntimeError`: If connection fails

**Example**:
```python
await memory.connect()
```

##### `close()`

```python
async def close() -> None
```

Close connections and cleanup resources.

**Example**:
```python
await memory.close()
```

##### `create_graph()`

```python
async def create_graph(
    graph_name: str,
    description: str = ""
) -> None
```

Create a new knowledge graph.

**Parameters**:
- `graph_name` (str): Unique name for the graph
- `description` (str): Optional description

**Raises**:
- `ValueError`: If graph already exists

**Example**:
```python
await memory.create_graph("research", "Research papers and ideas")
await memory.create_graph("code", "Code implementations")
```

##### `store()`

```python
async def store(
    content: str,
    graph: str,
    node_type: NodeType,
    parent_id: Optional[str] = None,
    importance: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```

Store a node with automatic PAC validation.

**Parameters**:
- `content` (str): Text content to store
- `graph` (str): Graph name
- `node_type` (NodeType): Type of node (CONCEPT, FACT, PAPER, etc.)
- `parent_id` (str | None): Parent node ID for hierarchical storage (default: None)
- `importance` (float): Importance score 0.0-1.0 (default: 0.5)
- `metadata` (dict | None): Additional metadata (default: None)

**Returns**:
- `str`: Unique node ID

**Validation**:
- If `parent_id` is provided:
  - Validates Fibonacci recursion: Ψ(parent) = Ψ(child1) + Ψ(child2)
  - Computes balance operator Ξ
  - Checks MED universal bounds
  - Validates E=mc² conservation
  - Detects collapse triggers

**Raises**:
- `ValueError`: If parent not found
- `ValueError`: If graph doesn't exist

**Example**:
```python
# Store root node
root_id = await memory.store(
    content="Machine learning",
    graph="research",
    node_type=NodeType.CONCEPT,
    importance=1.0
)

# Store child with PAC validation
child_id = await memory.store(
    content="Machine learning with neural networks",
    graph="research",
    node_type=NodeType.CONCEPT,
    parent_id=root_id,  # PAC validates Fibonacci recursion
    importance=0.8,
    metadata={"author": "Alice", "year": 2024}
)
```

##### `query()`

```python
async def query(
    query_text: str,
    graphs: Optional[List[str]] = None,
    node_types: Optional[List[NodeType]] = None,
    limit: int = 10,
    min_similarity: float = 0.0
) -> List[QueryResult]
```

Query memory with SEC resonance ranking.

**Parameters**:
- `query_text` (str): Query string
- `graphs` (list[str] | None): Graphs to search (default: all graphs)
- `node_types` (list[NodeType] | None): Filter by node types (default: all types)
- `limit` (int): Maximum results to return (default: 10)
- `min_similarity` (float): Minimum similarity threshold 0.0-1.0 (default: 0.0)

**Returns**:
- `list[QueryResult]`: Ranked results with scores

**Ranking Algorithm**:
```python
sec_score = similarity * (1 - entropy) * recency_factor
```

**Example**:
```python
results = await memory.query(
    query_text="neural network architectures",
    graphs=["research", "code"],
    node_types=[NodeType.CONCEPT, NodeType.PAPER],
    limit=10,
    min_similarity=0.5
)

for result in results:
    print(f"[{result.score:.3f}] {result.node.content}")
    print(f"  Similarity: {result.similarity:.3f}")
    print(f"  Turn: {result.node.metadata.get('turn', '?')}")
```

##### `get_foundation_health()`

```python
def get_foundation_health() -> Dict[str, Any]
```

Get real-time theoretical health metrics.

**Returns**:
- `dict`: Health metrics with statistics

**Structure**:
```python
{
    "c_squared": {
        "count": int,
        "mean": float,
        "std": float,
        "latest": float
    },
    "balance_operator": {
        "count": int,
        "mean": float,
        "latest": float,
        "target": 1.0571  # Ξ target
    },
    "duty_cycle": {
        "count": int,
        "mean": float,
        "latest": float,
        "target": 0.618  # φ/(φ+1)
    },
    "med_quality": {
        "count": int,
        "mean": float,
        "latest": float
    },
    "constants": {
        "phi": 1.618033988749895,
        "xi": 1.0571238898,
        "lambda_star": 0.97,
        "duty_cycle": 0.618
    }
}
```

**Example**:
```python
health = memory.get_foundation_health()
print(f"c² (model constant): {health['c_squared']['latest']:.2f}")
print(f"Balance operator Ξ: {health['balance_operator']['latest']:.4f}")
print(f"Duty cycle: {health['duty_cycle']['latest']:.3f}")
print(f"MED quality: {health['med_quality']['latest']:.3f}")

# Check for collapse risk
if health['balance_operator']['latest'] > 1.2:
    print("⚠️  Collapse risk detected!")
elif health['balance_operator']['latest'] < 0.5:
    print("⚠️  Field decay detected!")
```

##### `get_stats()`

```python
async def get_stats() -> Dict[str, Any]
```

Get usage statistics.

**Returns**:
- `dict`: Statistics including node counts, collapses, etc.

**Example**:
```python
stats = await memory.get_stats()
print(f"Total nodes: {stats['total_nodes']}")
print(f"Total graphs: {stats['total_graphs']}")
print(f"Collapse triggers: {stats['collapses']}")
```

---

## PAC Engine

**Location**: `fracton/storage/pac_engine.py`

### Class: `PACEngine`

Potential-Actualization Conservation engine implementing Fibonacci recursion.

#### Constructor

```python
PACEngine(device: str = "cpu")
```

**Parameters**:
- `device` (str): Compute device - "cpu" or "cuda" (default: "cpu")

**Constants**:
- `PHI` = 1.618033988749895 (Golden ratio)
- `XI` = 1.0571238898 (Balance operator target: 1 + π/F₁₀)
- `LAMBDA_STAR` = 0.97 (Potential decay rate)
- `DUTY_CYCLE` = 0.618 (φ/(φ+1), 61.8% attraction)
- `CONSERVATION_TOLERANCE` = 1e-10 (Precision threshold)

#### Methods

##### `compute_potential()`

```python
def compute_potential(
    depth: int,
    amplitude: float = 1.0
) -> float
```

Compute potential at given depth using golden ratio scaling.

**Formula**:
```python
Ψ(depth) = amplitude * φ^(-depth)
```

**Parameters**:
- `depth` (int): Hierarchy depth (0 = root)
- `amplitude` (float): Initial amplitude (default: 1.0)

**Returns**:
- `float`: Potential value

**Example**:
```python
engine = PACEngine()
root_potential = engine.compute_potential(depth=0)  # 1.0
child_potential = engine.compute_potential(depth=1)  # 0.618
grandchild_potential = engine.compute_potential(depth=2)  # 0.382
```

##### `verify_fibonacci_recursion()`

```python
def verify_fibonacci_recursion(
    parent_potential: float,
    child1_potential: float,
    child2_potential: float
) -> Tuple[bool, float]
```

Verify Fibonacci recursion: Ψ(k) = Ψ(k+1) + Ψ(k+2)

**Parameters**:
- `parent_potential` (float): Parent node potential
- `child1_potential` (float): First child potential
- `child2_potential` (float): Second child potential

**Returns**:
- `tuple[bool, float]`: (is_valid, residual)
  - `is_valid`: True if residual < CONSERVATION_TOLERANCE
  - `residual`: Absolute difference from expected

**Example**:
```python
parent_pot = 1.0
child1_pot = 0.618
child2_pot = 0.382

is_valid, residual = engine.verify_fibonacci_recursion(
    parent_pot, child1_pot, child2_pot
)
print(f"Valid: {is_valid}, Residual: {residual:.2e}")
# Output: Valid: True, Residual: 2.22e-16
```

##### `compute_balance_operator()`

```python
def compute_balance_operator(
    node: PACNode,
    children: List[PACNode]
) -> float
```

Compute balance operator Ξ for collapse detection.

**Formula**:
```python
Ξ = 1 + (excess_pressure / parent_potential)
where excess_pressure = Σ(children_potential) - parent_potential
```

**Parameters**:
- `node` (PACNode): Parent node
- `children` (list[PACNode]): Child nodes

**Returns**:
- `float`: Balance operator value

**Interpretation**:
- `Ξ > XI` (1.0571): COLLAPSE risk
- `Ξ < 0.5`: DECAY warning
- `0.5 ≤ Ξ ≤ XI`: STABLE

**Example**:
```python
from fracton.storage.pac_engine import PACNode

parent = PACNode(potential=1.0, depth=0)
child1 = PACNode(potential=0.618, depth=1)
child2 = PACNode(potential=0.382, depth=1)

xi = engine.compute_balance_operator(parent, [child1, child2])
print(f"Balance Ξ: {xi:.4f}")  # Should be close to 1.0
```

##### `check_collapse_trigger()`

```python
def check_collapse_trigger(xi_local: float) -> str
```

Detect collapse/decay based on balance operator.

**Parameters**:
- `xi_local` (float): Local balance operator value

**Returns**:
- `str`: "COLLAPSE", "DECAY", or "STABLE"

**Example**:
```python
status = engine.check_collapse_trigger(xi_local=0.7)
print(status)  # "STABLE"

status = engine.check_collapse_trigger(xi_local=1.3)
print(status)  # "COLLAPSE"

status = engine.check_collapse_trigger(xi_local=0.3)
print(status)  # "DECAY"
```

---

## SEC Operators

**Location**: `fracton/storage/sec_operators.py`

### Class: `SECOperators`

Symbolic Entropy Collapse operators with 4:1 attraction/repulsion balance.

#### Constructor

```python
SECOperators(device: str = "cpu")
```

**Parameters**:
- `device` (str): Compute device - "cpu" or "cuda" (default: "cpu")

**Constants**:
- `PHI` = 1.618033988749895 (Golden ratio)
- `DUTY_CYCLE` = 0.618 (φ/(φ+1))
- `ATTRACTION_CYCLES` = 4
- `REPULSION_CYCLES` = 1
- `BALANCE_RATIO` = 4.0

#### Methods

##### `merge()`

```python
def merge(
    embeddings: List[torch.Tensor]
) -> torch.Tensor
```

Merge operation (⊕): Combine symbolic structures.

**Parameters**:
- `embeddings` (list[Tensor]): Embeddings to merge

**Returns**:
- `Tensor`: Merged embedding (normalized average)

**Example**:
```python
import torch
from fracton.storage.sec_operators import SECOperators

sec = SECOperators()
emb1 = torch.randn(384)
emb2 = torch.randn(384)
emb3 = torch.randn(384)

merged = sec.merge([emb1, emb2, emb3])
print(merged.shape)  # torch.Size([384])
```

##### `branch()`

```python
def branch(
    embedding: torch.Tensor,
    num_branches: int = 2
) -> List[torch.Tensor]
```

Branch operation (⊗): Split into alternatives.

**Parameters**:
- `embedding` (Tensor): Source embedding
- `num_branches` (int): Number of branches to create (default: 2)

**Returns**:
- `list[Tensor]`: List of branched embeddings

**Example**:
```python
source = torch.randn(384)
branches = sec.branch(source, num_branches=3)
print(len(branches))  # 3
```

##### `gradient()`

```python
def gradient(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor
) -> torch.Tensor
```

Gradient operation (δ): Measure symbolic change.

**Parameters**:
- `embedding1` (Tensor): First embedding
- `embedding2` (Tensor): Second embedding

**Returns**:
- `Tensor`: Gradient (difference vector)

**Example**:
```python
emb_before = torch.randn(384)
emb_after = torch.randn(384)

grad = sec.gradient(emb_before, emb_after)
change_magnitude = torch.norm(grad).item()
print(f"Change magnitude: {change_magnitude:.4f}")
```

##### `compute_duty_cycle()`

```python
def compute_duty_cycle(
    attraction_time: float,
    repulsion_time: float
) -> float
```

Compute duty cycle from attraction/repulsion times.

**Formula**:
```python
duty_cycle = attraction_time / (attraction_time + repulsion_time)
```

**Parameters**:
- `attraction_time` (float): Time in attraction phase
- `repulsion_time` (float): Time in repulsion phase

**Returns**:
- `float`: Duty cycle (should be ≈ 0.618 for golden ratio balance)

**Example**:
```python
duty = sec.compute_duty_cycle(attraction_time=4.0, repulsion_time=1.0)
print(f"Duty cycle: {duty:.3f}")  # 0.800

# Golden ratio duty cycle
duty_golden = sec.compute_duty_cycle(
    attraction_time=0.618,
    repulsion_time=0.382
)
print(f"Golden duty cycle: {duty_golden:.3f}")  # 0.618
```

##### `compute_sec_score()`

```python
def compute_sec_score(
    similarity: float,
    entropy: float,
    recency: float = 1.0
) -> float
```

Compute SEC resonance score.

**Formula**:
```python
sec_score = similarity * (1 - entropy) * recency
```

**Parameters**:
- `similarity` (float): Cosine similarity [0, 1]
- `entropy` (float): Shannon entropy [0, 1]
- `recency` (float): Recency factor [0, 1] (default: 1.0)

**Returns**:
- `float`: SEC resonance score

**Example**:
```python
score = sec.compute_sec_score(
    similarity=0.85,
    entropy=0.3,
    recency=0.9
)
print(f"SEC score: {score:.4f}")  # 0.5355
```

---

## MED Validator

**Location**: `fracton/storage/med_validator.py`

### Class: `MEDValidator`

Macro Emergence Dynamics validator enforcing universal bounds.

#### Constructor

```python
MEDValidator()
```

**Constants**:
- `MAX_DEPTH` = 1 (Universal depth bound)
- `MAX_CHILDREN` = 3 (Universal sibling count bound)

#### Methods

##### `validate_structure()`

```python
def validate_structure(
    depth: int,
    num_children: int
) -> Tuple[bool, float, str]
```

Validate structure against MED universal bounds.

**Parameters**:
- `depth` (int): Current depth in hierarchy
- `num_children` (int): Number of children

**Returns**:
- `tuple[bool, float, str]`: (is_valid, quality_score, message)
  - `is_valid`: True if bounds satisfied
  - `quality_score`: 0.0 if invalid, otherwise (depth_compliance * node_compliance)^0.5
  - `message`: Validation message

**Universal Bounds**:
- `depth ≤ 1`
- `num_children ≤ 3`

**Example**:
```python
from fracton.storage.med_validator import MEDValidator

validator = MEDValidator()

# Valid structure
is_valid, quality, msg = validator.validate_structure(depth=1, num_children=2)
print(f"Valid: {is_valid}, Quality: {quality:.2f}")
# Output: Valid: True, Quality: 1.00

# Invalid depth
is_valid, quality, msg = validator.validate_structure(depth=2, num_children=2)
print(f"Valid: {is_valid}, Quality: {quality:.2f}, Message: {msg}")
# Output: Valid: False, Quality: 0.00, Message: Depth exceeds universal bound
```

##### `compute_emergence_quality()`

```python
def compute_emergence_quality(
    depth_compliance: float,
    node_compliance: float
) -> float
```

Compute emergence quality score.

**Formula**:
```python
quality = sqrt(depth_compliance * node_compliance)
```

**Parameters**:
- `depth_compliance` (float): Depth compliance [0, 1]
- `node_compliance` (float): Node count compliance [0, 1]

**Returns**:
- `float`: Quality score [0, 1]

**Example**:
```python
quality = validator.compute_emergence_quality(
    depth_compliance=1.0,
    node_compliance=0.667  # 2 children out of max 3
)
print(f"Quality: {quality:.3f}")  # 0.816
```

---

## Distance Validator

**Location**: `fracton/storage/distance_validator.py`

### Class: `DistanceValidator`

E=mc² distance validator for geometric conservation.

#### Constructor

```python
DistanceValidator(device: str = "cpu")
```

**Parameters**:
- `device` (str): Compute device - "cpu" or "cuda" (default: "cpu")

#### Methods

##### `measure_model_constant()`

```python
def measure_model_constant(
    embeddings: List[torch.Tensor],
    expected_energy: Optional[float] = None
) -> float
```

Measure model-specific c² constant.

**Formula**:
```python
E = ||embedding||²  # L2 norm squared
c² = E / expected_energy (if provided)
```

**Parameters**:
- `embeddings` (list[Tensor]): Sample embeddings
- `expected_energy` (float | None): Expected energy (default: None)

**Returns**:
- `float`: Model constant c²

**Example**:
```python
from fracton.storage.distance_validator import DistanceValidator
import torch

validator = DistanceValidator()

# Measure c² from sample embeddings
embeddings = [torch.randn(384) for _ in range(100)]
c_squared = validator.measure_model_constant(embeddings)
print(f"Model c²: {c_squared:.2f}")  # Typically ~416 for mini model
```

##### `validate_conservation()`

```python
def validate_conservation(
    parent_embedding: torch.Tensor,
    child_embeddings: List[torch.Tensor],
    tolerance: float = 0.1
) -> Tuple[bool, float]
```

Validate E=mc² conservation.

**Formula**:
```python
E_parent ≈ Σ(E_children) + binding_energy
where E = ||embedding||²
```

**Parameters**:
- `parent_embedding` (Tensor): Parent embedding
- `child_embeddings` (list[Tensor]): Child embeddings
- `tolerance` (float): Relative tolerance (default: 0.1 = 10%)

**Returns**:
- `tuple[bool, float]`: (is_conserved, energy_ratio)

**Example**:
```python
parent = torch.randn(384)
child1 = torch.randn(384)
child2 = torch.randn(384)

is_conserved, ratio = validator.validate_conservation(
    parent, [child1, child2], tolerance=0.2
)
print(f"Conserved: {is_conserved}, Ratio: {ratio:.4f}")
```

##### `compute_fractal_dimension()`

```python
def compute_fractal_dimension(
    embeddings: List[torch.Tensor],
    scales: List[float] = [0.1, 0.5, 1.0]
) -> float
```

Compute fractal dimension of embedding space.

**Formula**:
```python
D = log(N) / log(1/r)
```

**Parameters**:
- `embeddings` (list[Tensor]): Embedding samples
- `scales` (list[float]): Length scales for measurement (default: [0.1, 0.5, 1.0])

**Returns**:
- `float`: Fractal dimension

**Example**:
```python
embeddings = [torch.randn(384) for _ in range(1000)]
dimension = validator.compute_fractal_dimension(embeddings)
print(f"Fractal dimension: {dimension:.3f}")
```

---

## Foundation Integration

**Location**: `fracton/storage/foundation_integration.py`

### Class: `FoundationIntegration`

High-level integration layer bridging foundations to KronosMemory.

#### Constructor

```python
FoundationIntegration(device: str = "cpu")
```

**Parameters**:
- `device` (str): Compute device - "cpu" or "cuda" (default: "cpu")

**Components**:
- `pac_engine`: PACEngine instance
- `sec_operators`: SECOperators instance
- `med_validator`: MEDValidator instance
- `distance_validator`: DistanceValidator instance

#### Methods

##### `validate_storage()`

```python
def validate_storage(
    node: PACNode,
    parent: Optional[PACNode] = None,
    siblings: Optional[List[PACNode]] = None
) -> Dict[str, Any]
```

Validate node storage across all foundations.

**Parameters**:
- `node` (PACNode): Node being stored
- `parent` (PACNode | None): Parent node (default: None)
- `siblings` (list[PACNode] | None): Sibling nodes (default: None)

**Returns**:
- `dict`: Validation results

**Structure**:
```python
{
    "pac_valid": bool,
    "pac_residual": float,
    "balance_operator": float,
    "collapse_status": str,  # "STABLE", "COLLAPSE", "DECAY"
    "med_valid": bool,
    "med_quality": float,
    "distance_conserved": bool,
    "energy_ratio": float,
    "c_squared": float,
    "duty_cycle": float,
    "overall_valid": bool
}
```

**Example**:
```python
from fracton.storage.foundation_integration import FoundationIntegration
from fracton.storage.pac_engine import PACNode

integration = FoundationIntegration()

parent_node = PACNode(
    potential=1.0,
    depth=0,
    embedding=torch.randn(384)
)

child_node = PACNode(
    potential=0.618,
    depth=1,
    embedding=torch.randn(384)
)

results = integration.validate_storage(
    node=child_node,
    parent=parent_node
)

print(f"Overall valid: {results['overall_valid']}")
print(f"Balance Ξ: {results['balance_operator']:.4f}")
print(f"MED quality: {results['med_quality']:.3f}")
print(f"Collapse status: {results['collapse_status']}")
```

##### `compute_health_metrics()`

```python
def compute_health_metrics(
    nodes: List[PACNode]
) -> Dict[str, Any]
```

Compute aggregate health metrics for node collection.

**Parameters**:
- `nodes` (list[PACNode]): Nodes to analyze

**Returns**:
- `dict`: Aggregate metrics

**Example**:
```python
nodes = [PACNode(...) for _ in range(100)]
health = integration.compute_health_metrics(nodes)
print(health)
```

---

## Data Classes

### `PACMemoryNode`

**Location**: `fracton/storage/kronos_memory.py`

Memory node with PAC delta storage.

```python
@dataclass
class PACMemoryNode:
    # Identity
    id: str

    # PAC Storage
    embedding: Optional[torch.Tensor]  # Full embedding (reconstructed)
    delta_embedding: Optional[torch.Tensor]  # Delta from parent
    delta_content: str  # Semantic diff
    parent_id: Optional[str]
    children_ids: List[str]

    # PAC Physics
    potential: float = 1.0  # Ψ (decays with depth)
    depth: int = 0  # Hierarchy level

    # Content
    content: str
    node_type: NodeType
    graph: str

    # Metadata
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    importance: float
```

**Properties**:

```python
@property
def metadata(self) -> Dict[str, Any]:
    """Access user metadata."""
    return getattr(self, '_user_metadata', {})
```

### `QueryResult`

Result from SEC-based query.

```python
@dataclass
class QueryResult:
    node: PACMemoryNode
    score: float  # Overall SEC score
    similarity: float  # Cosine similarity [0-1]
    distance: float  # Euclidean distance
```

### `PACNode`

**Location**: `fracton/storage/pac_engine.py`

Internal PAC node representation.

```python
@dataclass
class PACNode:
    potential: float
    depth: int
    embedding: Optional[torch.Tensor] = None
    children: List['PACNode'] = field(default_factory=list)
```

---

## Enumerations

### `NodeType`

**Location**: `fracton/storage/types.py`

```python
class NodeType(str, Enum):
    # Code/Repository
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    COMMIT = "commit"

    # Research/Concepts
    CONCEPT = "concept"
    PAPER = "paper"
    EXPERIMENT = "experiment"
    HYPOTHESIS = "hypothesis"

    # Personal/Context
    GOAL = "goal"
    PREFERENCE = "preference"
    FACT = "fact"

    # Social/Content
    POST = "post"
    THREAD = "thread"
    ANNOUNCEMENT = "announcement"

    # Services/Monitoring
    SERVICE = "service"
    STATUS = "status"
    ALERT = "alert"
```

### `RelationType`

Relationship types for graph connections.

```python
class RelationType(str, Enum):
    # Structural
    CONTAINS = "contains"
    PART_OF = "part_of"

    # Code dependencies
    IMPORTS = "imports"
    CALLS = "calls"
    IMPLEMENTS = "implements"

    # Temporal/Evolution
    EVOLVES_FROM = "evolves_from"
    EVOLVES_TO = "evolves_to"
    PRECEDES = "precedes"
    FOLLOWS = "follows"

    # Semantic
    RELATES_TO = "relates_to"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    VALIDATES = "validates"

    # Cross-domain
    ANNOUNCES = "announces"
    MONITORS = "monitors"
    RESEARCHES = "researches"
    PURSUES = "pursues"
```

---

## Usage Examples

### Complete Workflow

```python
import asyncio
from fracton.storage import KronosMemory, NodeType

async def main():
    # Initialize
    memory = KronosMemory(
        storage_path="./data",
        namespace="research",
        device="cpu",
        embedding_model="mini"
    )

    try:
        # Connect
        await memory.connect()

        # Create graph
        await memory.create_graph("knowledge", "Knowledge base")

        # Store hierarchy
        root = await memory.store(
            content="Artificial Intelligence",
            graph="knowledge",
            node_type=NodeType.CONCEPT,
            importance=1.0
        )

        ml = await memory.store(
            content="Machine Learning subset of AI",
            graph="knowledge",
            node_type=NodeType.CONCEPT,
            parent_id=root,
            importance=0.9
        )

        dl = await memory.store(
            content="Deep Learning subset of ML using neural networks",
            graph="knowledge",
            node_type=NodeType.CONCEPT,
            parent_id=ml,
            importance=0.8,
            metadata={"field": "neural networks", "year": 2012}
        )

        # Query
        results = await memory.query(
            query_text="neural network learning",
            graphs=["knowledge"],
            limit=5
        )

        print("\nQuery Results:")
        for r in results:
            print(f"[{r.score:.3f}] {r.node.content}")
            print(f"  Depth: {r.node.depth}, Potential: {r.node.potential:.3f}")

        # Health check
        health = memory.get_foundation_health()
        print("\nFoundation Health:")
        print(f"Balance Ξ: {health['balance_operator']['latest']:.4f}")
        print(f"Duty cycle: {health['duty_cycle']['latest']:.3f}")
        print(f"MED quality: {health['med_quality']['latest']:.3f}")

        # Stats
        stats = await memory.get_stats()
        print(f"\nTotal nodes: {stats['total_nodes']}")
        print(f"Collapses: {stats['collapses']}")

    finally:
        await memory.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Error Handling

### Common Exceptions

```python
# Parent not found
try:
    await memory.store(
        content="...",
        graph="test",
        node_type=NodeType.CONCEPT,
        parent_id="nonexistent"
    )
except ValueError as e:
    print(f"Error: {e}")  # Parent node not found

# Graph doesn't exist
try:
    await memory.store(
        content="...",
        graph="nonexistent",
        node_type=NodeType.CONCEPT
    )
except ValueError as e:
    print(f"Error: {e}")  # Graph not found
```

### Validation Warnings

Validation warnings are logged but don't raise exceptions:

```python
# PAC conservation residual > 1e-6
# Logged as: "PAC conservation residual 2.3e-5 exceeds tolerance"

# Balance operator < 0.5
# Logged as: "Field decay detected: Ξ=0.45 < 0.5"

# MED bounds violated
# Logged as: "MED quality=0.0, depth=2 exceeds universal bound"
```

---

## Performance Tips

1. **Batch Operations**: Store multiple nodes in sequence for better cache utilization
2. **GPU Acceleration**: Use `device="cuda"` for large-scale queries (10k+ nodes)
3. **Graph Scoping**: Always specify `graphs` parameter in queries to avoid searching all graphs
4. **Importance Scoring**: Use `importance` to help SEC ranking prioritize critical nodes
5. **Metadata Filtering**: Use `node_types` parameter to narrow query scope

---

## See Also

- [KronosMemory Specification](../.spec/kronos-memory.spec.md)
- [Testing Guide](../tests/TESTING_GUIDE.md)
- [Docker Deployment](../DOCKER.md)
- [Examples](../examples/)
