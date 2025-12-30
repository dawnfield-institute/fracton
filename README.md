# Fracton: Infodynamics SDK & Programming Language

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Development Status](https://img.shields.io/badge/status-production-green.svg)](https://github.com/dawnfield-institute/fracton)

---

> **📢 Repository Scope (November 2025)**  
> **Fracton is the SDK/programming language** for infodynamics applications. It provides core primitives (RecursiveEngine, MemoryField, PAC regulation) and language constructs for building infodynamics systems.
>
> **Physics simulations** (Reality Engine, Big Bang, etc.) live in the separate **reality-engine** repository, which imports from Fracton as needed. This keeps responsibilities clean: Fracton = reusable SDK, reality-engine = physics implementation.

---

## Overview

Fracton is a domain-specific SDK and programming language for infodynamics research. It provides the foundational primitives for building systems that involve recursive execution, entropy-driven control flow, field operations, and PAC (Potential-Actualization-Conservation) regulation.

**This is part of the [Dawn Field Theory](https://github.com/dawnfield-institute/dawn-field-theory) ecosystem, extracted as a standalone SDK for easier adoption and development.**

### What Fracton Provides

- **Core Primitives**: RecursiveEngine, MemoryField, PACRegulator
- **Language Constructs**: @recursive, @entropy_gate, Context management
- **Field Operations**: RBFEngine, QBERegulator, initializers
- **Execution Framework**: Bifractal tracing, entropy dispatch
- **GPU Support**: Built-in CUDA acceleration for field operations
- **Theoretical Foundations**: PAC/SEC/MED conservation engines with real-time validation

### Theoretical Foundations (NEW)

Fracton now includes production-ready implementations of Dawn Field Theory's core physics:

**PAC (Potential-Actualization Conservation)**:
- Fibonacci recursion: Ψ(k) = Ψ(k+1) + Ψ(k+2)
- Three-dimensional conservation (value, complexity, effect)
- Balance operator Ξ = 1 + π/F₁₀ for collapse detection
- Validated to 1e-10 precision across 100+ hierarchy levels

**SEC (Symbolic Entropy Collapse)**:
- Merge (⊕), branch (⊗), gradient (δ) operators
- 4:1 attraction/repulsion balance ratio
- Duty cycle equilibrium at φ/(φ+1) ≈ 0.618
- Resonance ranking for semantic queries

**MED (Macro Emergence Dynamics)**:
- Universal bounds: depth(S) ≤ 1, nodes(S) ≤ 3
- Quality scoring for emergent structures
- Validated across 1000+ simulations

**E=mc² Distance Validation**:
- Geometric conservation through Euclidean distances
- Model-specific constants (c² ≈ 416 for llama3.2)
- Amplification and binding energy measurement
- Fractal dimension computation

**Status**: 65/65 tests passing, production-ready with <10% overhead

### What Fracton Does NOT Provide

- Physics simulations (that's [reality-engine](https://github.com/dawnfield-institute/reality-engine))
- Möbius topology implementations (that's reality-engine/substrate)
- ~~SEC/Confluence operators~~ **NOW INCLUDED** in `fracton.storage.sec_operators`

## Installation

```bash
# Install from PyPI (when published)
pip install fracton

# Or install from source
git clone https://github.com/dawnfield-institute/fracton.git
cd fracton
pip install -e .
```

## Quick Start

### KronosMemory with Theoretical Foundations

```python
from fracton.storage import KronosMemory, NodeType

# Initialize memory with real embeddings and theoretical validation
async with KronosMemory(
    storage_path="./data",
    namespace="demo",
    embedding_model="mini",  # Uses sentence-transformers
    device="cuda" if torch.cuda.is_available() else "cpu"
) as memory:
    await memory.connect()
    await memory.create_graph("knowledge")

    # Store hierarchical knowledge with automatic PAC conservation
    root_id = await memory.store(
        content="Machine learning is a branch of AI",
        graph="knowledge",
        node_type=NodeType.CONCEPT,
    )

    child_id = await memory.store(
        content="Deep learning uses neural networks",
        graph="knowledge",
        node_type=NodeType.CONCEPT,
        parent_id=root_id,  # Conservation validated here
    )

    # Query with SEC resonance ranking
    results = await memory.query(
        query_text="neural networks",
        graphs=["knowledge"],
        limit=10,
    )

    # Check theoretical health metrics
    health = memory.get_foundation_health()
    print(f"c² (model constant): {health['c_squared']['latest']:.2f}")
    print(f"Balance operator Ξ: {health['balance_operator']['latest']:.4f}")
    print(f"Duty cycle: {health['duty_cycle']['latest']:.3f}")

    # Get full stats including collapse detection
    stats = await memory.get_stats()
    print(f"Collapse triggers: {stats['collapses']}")
```

### Recursive Field Processing

```python
import fracton

@fracton.recursive
@fracton.entropy_gate(0.5)
def fibonacci_field(memory, context):
    if context.depth <= 1:
        return 1

    a = fracton.recurse(fibonacci_field, memory, context.deeper(1))
    b = fracton.recurse(fibonacci_field, memory, context.deeper(2))
    return a + b

with fracton.memory_field() as field:
    context = fracton.Context(depth=10, entropy=0.8)
    result = fibonacci_field(field, context)
```

## Reality Simulation with Fracton

**NEW**: Fracton now includes the **Möbius module** for reality simulation, where physics emerges from first principles.

```python
from fracton.mobius import RealityEngine

# Create universe simulator
reality = RealityEngine(size=(256, 64), device='cuda')

# Initialize from Big Bang (maximum disequilibrium)
reality.initialize('big_bang')

# Evolve and watch physics emerge
for state in reality.evolve(steps=100000):
    if state['step'] % 1000 == 0:
        print(f"Time: {state['time']:.2f}, Temp: {state['temperature']:.4f}")
        print(f"Disequilibrium: {state['disequilibrium']:.6f}")

# Discover emergent laws
laws = reality.discover_laws(states)
print(f"Discovered: {laws}")
```

**Key Features**:
- **Möbius Topology**: Anti-periodic boundaries create non-orientable substrate
- **PAC Conservation**: Machine-precision (<1e-12) conservation enforcement
- **Thermodynamic Coupling**: Information-energy duality (Landauer principle)
- **Time Emergence**: Time flows from disequilibrium pressure, not imposed
- **Law Discovery**: Automated detection of conservation laws, forces, symmetries

See [Reality Engine Integration Guide](docs/REALITY_ENGINE_INTEGRATION.md) for details.

## Core Philosophy

- **Recursion as First-Class Primitive**: All computation flows through recursive function calls
- **Entropy-Driven Execution**: Functions activate based on entropy thresholds and field pressure
- **Bifractal Traceability**: Every operation maintains forward and reverse traces for analysis and healing
- **Field-Aware Memory**: Shared memory structures that respect entropy and context boundaries
- **Tool Expression**: External systems accessed as contextual expressions rather than static calls

## Language Features

### 1. Recursive Execution Model
```python
@fracton.recursive
def process_field(memory, context):
    if context.entropy < threshold:
        return memory.stable_state()
    
    # Recursive dispatch based on field conditions
    result = fracton.recurse(analyze_patterns, memory, context)
    return fracton.crystallize(result)
```

### 2. Entropy-Gated Dispatch
```python
@fracton.entropy_gate(min_threshold=0.7)
def collapse_dynamics(memory, context):
    # Only executes when entropy exceeds 0.7
    return perform_collapse(memory, context)
```

### 3. Bifractal Memory Management
```python
with fracton.memory_field() as field:
    # Forward trace automatically recorded
    result = recursive_operation(field, context)
    # Reverse trace available for analysis
    trace = field.get_bifractal_trace()
```

### 4. Tool Expression Framework
```python
@fracton.tool_binding
def github_interface(memory, context):
    # Tool accessed based on field context
    return fracton.express_tool('github', context.project_state)
```

## Applications

### GAIA (Recursive Cognition)
- Field-aware symbolic processing
- Collapse dynamics modeling
- Meta-cognitive recursion

### Aletheia (Truth Verification)
- Recursive fact-checking
- Evidence field analysis
- Truth crystallization

### Kronos (Temporal Modeling)
- Recursive causality chains
- Temporal field dynamics
- Event entropy analysis

### Custom Research Models
- Emergent intelligence studies
- Complex systems modeling
- Infodynamics experiments

## Architecture

```
fracton/
├── storage/                 # KronosMemory + Theoretical Foundations
│   ├── kronos_memory.py     # Main memory engine
│   ├── backends/            # Backend implementations (SQLite, ChromaDB, Neo4j, Qdrant)
│   ├── pac_engine.py        # PAC conservation engine
│   ├── sec_operators.py     # SEC collapse dynamics
│   ├── med_validator.py     # MED universal bounds
│   ├── distance_validator.py # E=mc² distance validation
│   └── foundation_integration.py # Integration layer
├── core/                    # Core language runtime
│   ├── recursive_engine.py  # Main execution engine
│   ├── entropy_dispatch.py  # Context-aware function dispatch
│   ├── bifractal_trace.py   # Forward/reverse operation tracing
│   └── memory_field.py      # Shared memory coordination
├── lang/                    # Language constructs
│   ├── decorators.py        # @fracton decorators
│   ├── primitives.py        # Core language primitives
│   ├── context.py           # Execution context management
│   └── compiler.py          # Optional DSL compilation
├── tools/                   # Tool expression framework
│   ├── registry.py          # Tool registration system
│   ├── bindings/            # External system connectors
│   └── expression.py        # Context-aware tool access
├── models/                  # Pre-built model templates
│   ├── gaia.py             # GAIA cognition model
│   ├── aletheia.py         # Truth verification model
│   └── base.py             # Base model class
├── utils/                   # Utilities
│   ├── visualization.py     # Trace and field visualization
│   ├── analysis.py          # Performance and pattern analysis
│   └── debugging.py         # Recursive debugging tools
└── examples/                # Usage examples and tutorials
```

## Getting Started

### Basic Fracton Program
```python
import fracton

# Define a recursive field processor
@fracton.recursive
@fracton.entropy_gate(0.5)
def fibonacci_field(memory, context):
    if context.depth < 2:
        return 1
    
    # Recursive computation with entropy awareness
    a = fracton.recurse(fibonacci_field, memory, context.deeper(1))
    b = fracton.recurse(fibonacci_field, memory, context.deeper(2))
    
    return a + b

# Execute with field context
with fracton.memory_field() as field:
    context = fracton.Context(depth=10, entropy=0.8)
    result = fibonacci_field(field, context)
    
    # Analyze the recursive trace
    trace = field.get_bifractal_trace()
    fracton.visualize_trace(trace)
```

### GAIA Integration Example
```python
import fracton
from fracton.models import gaia

# Define GAIA-specific recursive operations
@fracton.recursive
@fracton.entropy_gate(0.7)
def cognitive_collapse(memory, context):
    # Process symbolic structures
    symbols = memory.get_symbols()
    
    # Recursive pattern analysis
    patterns = fracton.recurse(analyze_patterns, memory, context)
    
    # Crystallize insights
    return gaia.crystallize(patterns, symbols)

# Run GAIA cognition model
model = gaia.GAIAModel()
result = model.run(cognitive_collapse, initial_symbols)
```

## Design Principles

1. **Minimal Syntax**: Clean, expressive syntax for complex recursive operations
2. **Performance**: Optimized for deep recursion and large memory fields
3. **Debuggability**: Rich tracing and visualization for understanding recursive flows
4. **Modularity**: Easy integration with external tools and systems
5. **Research-Oriented**: Designed for experimental exploration of infodynamics

## Development Status

- [ ] Core recursive engine
- [ ] Entropy dispatch system
- [ ] Bifractal tracing
- [ ] Memory field management
- [ ] Tool expression framework
- [ ] GAIA model integration
- [ ] Visualization tools
- [ ] Documentation and examples

## Contributing

Fracton is designed to be a foundational language for infodynamics research. Contributions should focus on:
- Core language features
- Model templates for specific research areas
- Tool bindings for external systems
- Visualization and analysis capabilities

## Dawn Field Theory Ecosystem

Fracton is part of the larger Dawn Field Theory ecosystem:

- **[dawn-field-theory](https://github.com/dawnfield-institute/dawn-field-theory)** - Core theoretical foundation
- **[fracton-sdk](https://github.com/dawnfield-institute/fracton)** - This computational language ⭐
- **[dawn-devkit](https://github.com/dawnfield-institute/dawn-devkit)** - Development tools and templates
- **[dawn-models](https://github.com/dawnfield-institute/dawn-models)** - AI architectures and implementations
- **[cip-core](https://github.com/dawnfield-institute/cip-core)** - Cognition Index Protocol

## Documentation

- **Setup Guide**: [SETUP.md](./SETUP.md)
- **Architecture**: [ARCHITECTURE.md](./ARCHITECTURE.md)
- **Specification**: [SPEC.md](./SPEC.md)
- **Testing Guide**: [tests/TESTING_GUIDE.md](./tests/TESTING_GUIDE.md)
- **Testing Report**: [tests/TESTING_REPORT.md](./tests/TESTING_REPORT.md) - Foundation test results
- **Roadmap**: [ROADMAP.md](./ROADMAP.md)

### Foundation Documentation

- **PAC Engine**: [fracton/storage/pac_engine.py](./fracton/storage/pac_engine.py)
- **SEC Operators**: [fracton/storage/sec_operators.py](./fracton/storage/sec_operators.py)
- **MED Validator**: [fracton/storage/med_validator.py](./fracton/storage/med_validator.py)
- **Distance Validator**: [fracton/storage/distance_validator.py](./fracton/storage/distance_validator.py)
- **Integration Layer**: [fracton/storage/foundation_integration.py](./fracton/storage/foundation_integration.py)

## Development

- **Development Roadmap**: [roadmaps/fracton_roadmap.md](./roadmaps/fracton_roadmap.md)
- **Current Tasks**: [todo/fracton_todo.md](./todo/fracton_todo.md)
- **Contributing**: See contributing guidelines in the main repository
- **Issues**: [GitHub Issues](https://github.com/dawnfield-institute/fracton/issues)

## License

Apache License 2.0 - See LICENSE file for details
