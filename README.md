# Fracton: Infodynamics Computational Modeling Language

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/dawnfield-institute/fracton)

---

> **📢 Project Status Update (October 2025)**  
> Due to increased work commitments over the next 2 months, all roadmaps are temporarily paused. The project is **not frozen**—work continues behind the scenes—but updates and documentation will be delayed. Expected to resume regular updates in December 2025. Thank you for your patience and continued interest.

---

## Overview

Fracton is a domain-specific computational modeling language designed for infodynamics research and recursive field-aware systems. It provides a unified substrate for modeling emergent intelligence, entropy dynamics, and bifractal computation patterns.

**This is part of the [Dawn Field Theory](https://github.com/dawnfield-institute/dawn-field-theory) ecosystem, extracted as a standalone SDK for easier adoption and development.**

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
- **Roadmap**: [ROADMAP.md](./ROADMAP.md)

## Development

- **Development Roadmap**: [roadmaps/fracton_roadmap.md](./roadmaps/fracton_roadmap.md)
- **Current Tasks**: [todo/fracton_todo.md](./todo/fracton_todo.md)
- **Contributing**: See contributing guidelines in the main repository
- **Issues**: [GitHub Issues](https://github.com/dawnfield-institute/fracton/issues)

## License

Apache License 2.0 - See LICENSE file for details
