# Fracton — Claude Code Context

## Identity
Fracton is the Infodynamics SDK and programming language — the computational substrate for Dawn Field Theory. It provides core primitives (RecursiveEngine, MemoryField, PAC regulation), language constructs (@recursive, @entropy_gate), and the KRONOS v2 storage layer. Version 2.1.0.

## Architecture

```
fracton/
├── fracton/                    # Main package (v2.1.0)
│   ├── core/                   # Runtime engine
│   │   ├── recursive_engine.py # RecursiveExecutor, PhysicsRecursiveExecutor
│   │   ├── entropy_dispatch.py # EntropyDispatcher, DispatchConditions
│   │   ├── bifractal_trace.py  # Forward/reverse operation tracking
│   │   ├── memory_field.py     # MemoryField, FieldController
│   │   ├── pac_regulation.py   # PACRegulator (conservation enforcement)
│   │   ├── pac_node.py         # PACNode, PACNodeFactory (delta-only)
│   │   ├── pac_system.py       # PACSystem, TieredCache (v2.0)
│   │   ├── mobius_tensor.py    # MobiusMatrix, Feigenbaum constants (v2.1-2.3)
│   │   └── feigenbaum_mobius.py
│   ├── lang/                   # Language constructs
│   │   ├── decorators.py       # @recursive, @entropy_gate, @tool_binding
│   │   ├── primitives.py       # Physics primitives (klein_gordon, etc.)
│   │   └── context.py          # ExecutionContext
│   ├── field/                  # Field operations
│   │   ├── encoding.py         # Spherical encoding, Klein-Gordon evolution
│   │   ├── rbf_engine.py       # Radial Basis Function engine
│   │   └── qbe_regulator.py    # QBE regulator
│   ├── storage/                # KRONOS v2 layer
│   │   ├── node.py             # KronosNode (identity = confluence)
│   │   ├── edge.py             # KronosEdge, 14 relation types
│   │   ├── graph.py            # KronosGraph (genealogy tree)
│   │   ├── confidence.py       # GeometricConfidence
│   │   ├── response_generator.py # Field-aware personality (6 strategies)
│   │   ├── pac_engine.py       # PAC conservation engine
│   │   ├── sec_operators.py    # SEC merge/branch/gradient operators
│   │   ├── med_validator.py    # MED universal bounds
│   │   └── backends/           # SQLite, Neo4j, ChromaDB, Qdrant
│   ├── physics/                # Conservation & constants
│   │   ├── conservation.py     # PAC/SEC/MED engines
│   │   └── constants.py        # PHI, XI, physical constants
│   ├── monitoring/             # PAC tree monitoring, interventions
│   ├── tools/shadowpuppet/     # Architecture-as-code evolution (v0.4)
│   └── kronos_agent/           # Graph operations & ingestion pipeline
├── tests/                      # 48 test modules, ~637 tests
├── scripts/                    # Ingestion & analysis scripts
├── data/                       # Knowledge graph data
└── examples/                   # Demo scripts
```

## Key APIs (56 public exports)

**Core**: RecursiveExecutor, ExecutionContext, PACNode, PACSystem, TieredCache
**Decorators**: @recursive, @entropy_gate, @tool_binding, @tail_recursive
**Physics**: klein_gordon_evolution, enforce_pac_conservation, field_pattern_matching
**Memory**: MemoryField, PhysicsMemoryField, FieldController
**Storage**: KronosNode, KronosEdge, KronosGraph, GeometricConfidence, FDOSerializer
**PAC**: PACRegulator, validate_pac_conservation, enable_pac_self_regulation

## Consumers

- **dawn-field-theory**: Imports MobiusTensor for Feigenbaum research, GAIA modules use core engine
- **reality-engine**: Imports PAC/Mobius for physics simulations
- **GRIM**: No direct imports (independent)

## Conventions

- PAC conservation: every function preserves Ψ(parent) = Ψ(child1) + Ψ(child2)
- Entropy-driven dispatch: functions activate based on field entropy thresholds
- Tests: `pytest tests/` from repo root
- Installation: `pip install -e .` for development

## Related Repos

- `dawn-field-theory` — physics framework (fracton provides the computational substrate)
- `reality-engine` — simulator (imports fracton for PAC/Mobius)
- `GRIM/mcp/kronos/` — MCP server (separate knowledge graph layer)
- `kronos-vault` — knowledge vault (FDOs reference fracton modules)

## Current State

- v2.1.0 production-ready
- ~637 tests, ~98% passing (1 known minor failure in collapse_trigger)
- ShadowPuppet v0.4 (architecture evolution framework)
- Well-documented: README, ARCHITECTURE.md, SPEC.md (23.9 KB)

## Guardrails

- Do NOT modify PAC conservation invariants without running full test suite
- Do NOT break public API surface (56 exports in __init__.py)
- Always run `pytest tests/` after changes
- Respect the separation: fracton = SDK, reality-engine = physics implementation
