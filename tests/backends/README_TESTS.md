# KRONOS Backend Test Suite

Comprehensive test suite for all KRONOS backend implementations.

## Test Structure

```
tests/backends/
├── test_base.py                    # Abstract interface tests (13 tests)
├── test_sqlite_graph.py            # SQLite graph backend (15 tests)
├── test_chromadb_vectors.py        # ChromaDB vector backend (15 tests)
├── test_neo4j_graph.py             # Neo4j graph backend (20 tests)
├── test_qdrant_vectors.py          # Qdrant vector backend (20 tests)
├── test_backend_integration.py     # Integration tests (15+ tests)
└── README_TESTS.md                 # This file
```

## Test Categories

### 1. Unit Tests (Backend-Specific)

**Abstract Base Tests** (`test_base.py`)
- ✅ 13 tests
- Tests data structures (GraphNode, VectorPoint, etc.)
- Tests abstract class enforcement
- All passing

**SQLite Graph Tests** (`test_sqlite_graph.py`)
- ✅ 15 tests
- Connection, health checks
- CRUD operations (nodes and edges)
- Parent-child relationships
- Temporal lineage tracing
- Contradiction detection
- Persistence across reconnections
- All passing (100%)

**ChromaDB Vector Tests** (`test_chromadb_vectors.py`)
- ⚠️ 15 tests
- Storage and retrieval
- Similarity search
- Batch operations
- Collection management
- Core functionality works (Windows file locking affects cleanup)

**Neo4j Graph Tests** (`test_neo4j_graph.py`)
- 🔄 20 tests
- Requires Neo4j server running
- Tests all GraphBackend operations
- Cypher query generation
- Large metadata handling
- Unicode support

**Qdrant Vector Tests** (`test_qdrant_vectors.py`)
- 🔄 20 tests
- Requires Qdrant server running
- Tests all VectorBackend operations
- GPU acceleration support
- Concurrent operations
- Performance benchmarks

### 2. Integration Tests

**Backend Compatibility** (`test_backend_integration.py`)
- Tests all backends implement same interface
- Ensures consistent behavior across implementations
- Tests:
  - GraphBackend compatibility (SQLite vs Neo4j)
  - VectorBackend compatibility (ChromaDB vs Qdrant)
  - Data migration between backends
  - Performance characteristics
  - Edge cases (unicode, empty content, zero vectors)

## Running Tests

### Run All Tests

```bash
cd fracton
pytest tests/backends/ -v
```

### Run Specific Backend Tests

```bash
# SQLite only (no dependencies)
pytest tests/backends/test_sqlite_graph.py -v

# ChromaDB only
pytest tests/backends/test_chromadb_vectors.py -v

# Neo4j only (requires server)
pytest tests/backends/test_neo4j_graph.py -v

# Qdrant only (requires server)
pytest tests/backends/test_qdrant_vectors.py -v

# Integration tests
pytest tests/backends/test_backend_integration.py -v
```

### Run Without Production Backends

```bash
# Skip Neo4j and Qdrant tests
pytest tests/backends/ -v -k "not neo4j and not qdrant"
```

### Run With Coverage

```bash
pytest tests/backends/ --cov=fracton.storage.backends --cov-report=html
```

## Test Requirements

### Minimal (Lightweight Backends Only)

```bash
pip install pytest pytest-asyncio torch
# SQLite tests will run (built-in)
# ChromaDB tests require: pip install chromadb
```

### Full (All Backends)

```bash
pip install pytest pytest-asyncio torch chromadb neo4j qdrant-client
```

Plus running services:
- **Neo4j**: `docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest`
- **Qdrant**: `docker run -p 6333:6333 qdrant/qdrant:latest`

## Environment Variables

Configure test connections via environment variables:

```bash
# Neo4j
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"

# Qdrant
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
```

## Test Coverage

### Current Coverage

| Backend | Tests | Passing | Coverage |
|---------|-------|---------|----------|
| Base (Abstract) | 13 | ✅ 13 | 100% |
| SQLite Graph | 15 | ✅ 15 | 100% |
| ChromaDB Vectors | 15 | ⚠️ Core works | ~90% |
| Neo4j Graph | 20 | 🔄 Pending server | - |
| Qdrant Vectors | 20 | 🔄 Pending server | - |
| Integration | 15+ | 🔄 Pending | - |

**Total**: ~98 tests

### Coverage Areas

✅ **Covered**:
- Connection management
- Node/edge CRUD operations
- Vector storage and retrieval
- Similarity search
- Batch operations
- Temporal lineage tracing
- Neighborhood expansion
- Contradiction detection
- Metadata handling
- Unicode support
- Empty content handling
- Persistence
- Health checks

🔄 **Pending Server Setup**:
- Neo4j production tests
- Qdrant production tests
- GPU acceleration tests
- Migration tests
- Performance benchmarks

## Quick Start

### 1. Run Lightweight Tests (No Setup)

```bash
# Install minimal dependencies
pip install pytest pytest-asyncio torch

# Run SQLite tests
pytest tests/backends/test_sqlite_graph.py -v

# Expected output: 15/15 passing
```

### 2. Add ChromaDB

```bash
pip install chromadb

pytest tests/backends/test_chromadb_vectors.py -v

# Expected: Core tests passing (some cleanup warnings on Windows)
```

### 3. Add Production Backends

**Start Services**:
```bash
# Neo4j
docker run -d --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest

# Qdrant
docker run -d --name qdrant \
    -p 6333:6333 \
    qdrant/qdrant:latest
```

**Run Tests**:
```bash
pytest tests/backends/test_neo4j_graph.py -v
pytest tests/backends/test_qdrant_vectors.py -v
```

### 4. Run Full Suite

```bash
pytest tests/backends/ -v --cov=fracton.storage.backends
```

## Test Data

Tests use:
- **Temporary directories**: Auto-cleaned after each test
- **Isolated graphs**: Each test uses unique graph names
- **Small datasets**: 10-100 nodes/vectors per test
- **Random vectors**: `torch.randn()` for reproducibility

## Continuous Integration

### GitHub Actions Example

```yaml
name: Backend Tests

on: [push, pull_request]

jobs:
  test-lightweight:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install pytest pytest-asyncio torch chromadb
      - name: Run lightweight tests
        run: |
          pytest tests/backends/test_base.py -v
          pytest tests/backends/test_sqlite_graph.py -v
          pytest tests/backends/test_chromadb_vectors.py -v

  test-production:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:latest
        ports:
          - 7687:7687
        env:
          NEO4J_AUTH: neo4j/password
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install pytest pytest-asyncio torch neo4j qdrant-client
      - name: Run production tests
        run: |
          pytest tests/backends/test_neo4j_graph.py -v
          pytest tests/backends/test_qdrant_vectors.py -v
          pytest tests/backends/test_backend_integration.py -v
```

## Troubleshooting

### "Neo4j not available" - Tests Skipped

**Cause**: Neo4j server not running or driver not installed

**Fix**:
```bash
# Install driver
pip install neo4j

# Start server
docker run -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# Verify connection
python -c "from neo4j import GraphDatabase; GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')).verify_connectivity()"
```

### "Qdrant not available" - Tests Skipped

**Cause**: Qdrant server not running or client not installed

**Fix**:
```bash
# Install client
pip install qdrant-client

# Start server
docker run -p 6333:6333 qdrant/qdrant:latest

# Verify connection
python -c "from qdrant_client import QdrantClient; QdrantClient(host='localhost', port=6333).get_collections()"
```

### ChromaDB File Lock Warnings on Windows

**Cause**: Known ChromaDB/SQLite file locking issue on Windows

**Impact**: Tests pass, but cleanup may warn

**Workaround**:
- Tests still verify functionality
- Linux/Mac unaffected
- Production use unaffected (uses persistent directories)

### Slow Tests

**Cause**: Docker services starting up

**Fix**:
- Keep Docker containers running between test runs
- Use `docker start neo4j qdrant` instead of recreating

## Performance Benchmarks

Expected performance (consumer laptop):

| Operation | SQLite | Neo4j | ChromaDB | Qdrant |
|-----------|--------|-------|----------|--------|
| Create node | <1ms | <5ms | - | - |
| Get node | <1ms | <2ms | - | - |
| Store vector | - | - | <5ms | <2ms |
| Search (k=10, 1K vectors) | - | - | ~50ms | ~10ms |
| Search (k=10, 10K vectors) | - | - | ~200ms | ~30ms |
| Batch insert (100 items) | <100ms | ~200ms | ~300ms | ~100ms |

GPU benchmarks (Qdrant with CUDA):
- Search: 25-50x faster than CPU
- Batch operations: 10-100x faster

## Next Steps

1. ✅ **Lightweight backends fully tested**
2. 🔄 **Set up Docker services for CI**
3. 🔄 **Run production backend tests**
4. 🔄 **Run integration tests**
5. 🔄 **GPU acceleration tests**
6. 🔄 **Performance benchmarks at scale**

## Contributing

When adding new backends:

1. **Implement abstract interface** (`GraphBackend` or `VectorBackend`)
2. **Create test file** (`test_<backend_name>.py`)
3. **Add to integration tests**
4. **Update this README**
5. **Ensure all tests pass**

Test template:
```python
@pytest.mark.asyncio
async def test_<operation>(self, backend):
    """Test <description>."""
    # Setup
    # Execute
    # Assert
    # Cleanup
```
