# Kronos Agent Module Added to Fracton

**Date**: 2025-12-31 21:30
**Commit**: pending
**Type**: engineering

## Summary
Added `kronos_agent` module to the fracton package. This module provides the tool interface for Kronos operations used by LangGraph agents and the Kronos MCP server.

## Changes

### Added
- `fracton/kronos_agent/` - New module with:
  - `tools.py` - `KronosTools` class (1054 lines)
  - `session.py` - Session and context management
  - `config.py` - Configuration settings
  - `agent.py` - LangGraph agent definition
  - `run_agent.py` - Agent runner script
  - `run_pipeline.py` - Pipeline runner script
  - `__init__.py` - Module exports

### Changed
- All imports updated from absolute to relative:
  - `from ..storage import KronosMemory, NodeType, RelationType`
  - `from ..storage import KronosMemory` (in other files)

## KronosTools API

```python
class KronosTools:
    def __init__(self, storage_path, namespace, device, embedding_dim)
    
    async def connect() -> bool
    async def close() -> None
    
    async def scan_repo(
        repo_path: str,
        graph_name: str,
        file_patterns: list[str] = None
    ) -> dict
    
    async def query_graph(
        query: str,
        graphs: list[str] = None,
        limit: int = 10
    ) -> dict
    
    async def list_graphs() -> list[str]
    async def get_stats() -> dict
```

## Integration Points

This module is used by:
1. **Kronos MCP Server** (`dawn-infrastructure/services/kronos/`)
   - Dockerfile installs fracton from GitHub
   - `kronos_server.py` imports `from fracton.kronos_agent.tools`

2. **Local Development** (`internal/kronos_agent/`)
   - Original location, can still be used for local testing
   - Fracton copy is the canonical version for deployment

## Testing
```python
import asyncio
from fracton.kronos_agent.tools import KronosTools
from pathlib import Path

async def test():
    tools = KronosTools(
        storage_path=Path('./kronos_data'),
        namespace='test',
        device='cuda'
    )
    await tools.connect()
    await tools.scan_repo('/path/to/repo', 'repo_graph')
    result = await tools.query_graph('search term', graphs=['repo_graph'])
    await tools.close()

asyncio.run(test())
```

## Related
- `fracton/.spec/kronos-memory.spec.md` - Underlying memory spec
- `dawn-infrastructure/.changelog/20251231_213000_kronos_agent_integration.md` - Server integration
