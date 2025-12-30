# KRONOS Backend Abstraction

Pluggable backend architecture for KRONOS memory system.

## Architecture

```
┌─────────────────────────────────────────┐
│        KronosMemory (Unified API)       │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
┌─────────────┐         ┌─────────────┐
│ GraphBackend│         │VectorBackend│
│  (Abstract) │         │  (Abstract) │
└──────┬──────┘         └──────┬──────┘
       │                       │
   ┌───┴────┐            ┌─────┴─────┐
   │        │            │           │
   ▼        ▼            ▼           ▼
SQLite   Neo4j      ChromaDB     Qdrant
(Light)  (Prod)     (Light)      (Prod)
```

## Backends

### Graph Backends

#### SQLiteGraph (Lightweight)
- **Use Case**: Local development, single-user apps, prototyping
- **Storage**: SQLite file (`.db`)
- **Performance**: ~1K QPS for reads, ~500 QPS for writes
- **Scalability**: Single machine, up to ~1M nodes
- **Dependencies**: `aiosqlite` (built-in Python)

#### Neo4jGraph (Production)
- **Use Case**: Multi-user, production systems, large graphs
- **Storage**: Neo4j server (Cypher query language)
- **Performance**: ~10K QPS for reads, ~5K QPS for writes
- **Scalability**: Distributed, billions of nodes
- **Dependencies**: `neo4j` Python driver

### Vector Backends

#### ChromaDBVectors (Lightweight)
- **Use Case**: Local development, CPU-only systems
- **Storage**: Embedded database (file-based)
- **Performance**: ~100 QPS for similarity search
- **Scalability**: Up to ~1M vectors (384D)
- **Dependencies**: `chromadb`

#### QdrantVectors (Production)
- **Use Case**: GPU acceleration, large-scale search
- **Storage**: Qdrant server (HNSW index)
- **Performance**: ~10K QPS (CPU), ~50K QPS (GPU)
- **Scalability**: Billions of vectors with sharding
- **Dependencies**: `qdrant-client`, PyTorch

## Configuration

### Lightweight Stack (Development)

```python
from fracton.storage import KronosMemory
from fracton.storage.backends import BackendConfig

config = BackendConfig(
    graph_type="sqlite",
    vector_type="chromadb",
    device="cpu",
)

memory = KronosMemory(
    storage_path="./data",
    namespace="dev",
    backend_config=config,
)
```

**Requirements**:
- Python 3.10+
- ~100MB disk space
- No external services

### Production Stack (GPU)

```python
config = BackendConfig(
    graph_type="neo4j",
    graph_uri="bolt://localhost:7687",
    graph_user="neo4j",
    graph_password="password",
    vector_type="qdrant",
    vector_host="localhost",
    vector_port=6333,
    device="cuda",
)

memory = KronosMemory(
    storage_path="./data",
    namespace="production",
    backend_config=config,
)
```

**Requirements**:
- Neo4j server running
- Qdrant server running
- CUDA-capable GPU (optional but recommended)

## Data Structures

### GraphNode
Minimal metadata for graph storage:
```python
@dataclass
class GraphNode:
    id: str
    content: str
    timestamp: datetime
    fractal_signature: str
    parent_id: str = "-1"
    children_ids: List[str]
    potential: float = 1.0
    entropy: float = 0.5
    phase: str = "STABLE"
```

### GraphEdge
Typed relationships:
```python
@dataclass
class GraphEdge:
    from_id: str
    to_id: str
    relation_type: str  # EVOLVES_FROM, IMPLEMENTS, etc.
    weight: float = 1.0
    metadata: Dict[str, Any]
```

### VectorPoint
Embedding with payload:
```python
@dataclass
class VectorPoint:
    id: str
    vector: torch.Tensor  # [D]
    payload: Dict[str, Any]
```

## Interface Contracts

### GraphBackend

**Node Operations**:
- `create_node(node, graph_name)` - Store node
- `get_node(node_id, graph_name)` - Retrieve node
- `update_node(node_id, updates, graph_name)` - Update metadata
- `delete_node(node_id, graph_name)` - Remove node

**Edge Operations**:
- `create_edge(edge, graph_name)` - Create relationship
- `get_edges(node_id, direction, relation_type, graph_name)` - Get connections
- `delete_edge(from_id, to_id, relation_type, graph_name)` - Remove edge

**Graph Operations**:
- `create_graph(name, description)` - Initialize graph
- `list_graphs()` - Enumerate graphs
- `get_neighbors(node_id, max_hops, graph_name)` - Expand neighborhood

**Temporal Operations**:
- `trace_lineage(node_id, direction, max_depth, graph_name)` - PAC temporal trace
- `find_contradictions(node_id, graph_name)` - Conflict detection

### VectorBackend

**Storage Operations**:
- `store(point)` - Store single vector
- `store_batch(points)` - Batch insert
- `retrieve(point_ids)` - Fetch by ID
- `delete(point_ids)` - Remove vectors

**Search Operations**:
- `search(query_vector, limit, threshold, filter_payload)` - Similarity search
- `search_batch(query_vectors, limit, threshold)` - Batch search

**Collection Operations**:
- `create_collection(name, vector_size)` - Initialize collection
- `delete_collection(name)` - Remove collection
- `collection_info(name)` - Get metadata

## Implementation Guide

### Creating a New Backend

1. **Subclass the abstract base**:
```python
from fracton.storage.backends.base import GraphBackend, GraphNode, GraphEdge

class MyGraphBackend(GraphBackend):
    async def connect(self):
        # Establish connection
        pass

    async def create_node(self, node: GraphNode, graph_name: str = "default"):
        # Implement storage logic
        pass

    # ... implement all abstract methods
```

2. **Register with KronosMemory**:
```python
# In kronos_memory.py
GRAPH_BACKENDS = {
    "sqlite": SQLiteGraph,
    "neo4j": Neo4jGraph,
    "mygraph": MyGraphBackend,  # Add here
}
```

3. **Add tests**:
```python
# tests/backends/test_mygraph.py
@pytest.mark.asyncio
async def test_my_backend():
    backend = MyGraphBackend(...)
    await backend.connect()
    # Test all operations
```

## Performance Comparison

### Graph Operations (1M nodes)

| Operation | SQLite | Neo4j |
|-----------|--------|-------|
| Create node | 500 QPS | 5K QPS |
| Get node | 1K QPS | 10K QPS |
| Get neighbors (1-hop) | 800 QPS | 8K QPS |
| Trace lineage (10 depth) | 100 QPS | 2K QPS |

### Vector Operations (1M vectors, 384D)

| Operation | ChromaDB (CPU) | Qdrant (CPU) | Qdrant (GPU) |
|-----------|----------------|--------------|--------------|
| Store | 100 QPS | 1K QPS | 5K QPS |
| Search (k=10) | 50 QPS | 500 QPS | 10K QPS |
| Search (k=100) | 20 QPS | 200 QPS | 5K QPS |

## Migration Path

### Local → Production

```python
# 1. Export from lightweight
from fracton.storage.backends.migration import export_graph

await export_graph(
    source=sqlite_backend,
    target_path="./graph_export.jsonl",
)

# 2. Import to production
await import_graph(
    source_path="./graph_export.jsonl",
    target=neo4j_backend,
)
```

## Testing

Run backend tests:
```bash
# All backends
pytest tests/backends/

# Specific backend
pytest tests/backends/test_sqlite_graph.py
pytest tests/backends/test_chromadb_vectors.py

# With coverage
pytest --cov=fracton.storage.backends tests/backends/
```

## Next Steps

1. **Implement SQLiteGraph** (Phase 2)
2. **Implement ChromaDBVectors** (Phase 2)
3. **Port Neo4jGraph from GRIMM** (Phase 3)
4. **Port QdrantVectors from GRIMM** (Phase 3)
5. **Add migration utilities** (Phase 4)
6. **Benchmark at scale** (Phase 4)
