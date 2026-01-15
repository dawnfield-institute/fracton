# Legacy KRONOS Implementations

This directory contains previous versions of KRONOS for reference and potential reuse.

## v1_rag - RAG-Based Memory System

**Status:** Deprecated (moved to legacy January 2026)

The v1 implementation focused on:
- RAG (Retrieval-Augmented Generation) with semantic similarity
- Delta-only storage for space optimization
- SEC (Symbolic Entropy Collapse) resonance ranking
- PAC engine for storage compression

**Key Files:**
- `kronos_memory.py` (1002 lines) - Main memory interface
- `kronos_backend.py` (437 lines) - FDO-based persistence
- `KRONOS_README.md` - Original documentation

**Reusable Components:**
These foundation components are still in active use:
- `pac_engine.py` - PAC conservation validation (moved to `../pac_engine.py`)
- `sec_operators.py` - SEC operators (moved to `../sec_operators.py`)
- `med_validator.py` - MED validation (moved to `../med_validator.py`)
- `distance_validator.py` - E=mc² distance (moved to `../distance_validator.py`)
- `embeddings.py` - Embedding service (moved to `../embeddings.py`)
- `backends/` - Storage backends (moved to `../backends/`)

**Why Deprecated:**
v1 used flat semantic similarity search, which doesn't capture:
- Conceptual genealogy (what crystallized from what)
- Hierarchical structure (depth of actualization)
- Confluence patterns (how concepts merge)
- Temporal/causal directionality

**Migration to v2:**
KRONOS v2 (current) implements conceptual genealogy trees where:
- Node identity IS the confluence pattern
- Retrieval returns lineage slices (ancestors + descendants + siblings)
- Confidence derived from graph geometry
- PAC conservation applied to concept relationships

See `../` for v2 implementation and `.spec/kronos-memory.spec.md` for architecture.

---

**Date Archived:** January 13, 2026
**Reason:** Architectural shift from flat RAG to conceptual genealogy
**Reuse Status:** Foundation components extracted and still in use
