# Fracton Documentation

**Version**: 2.0.0
**Last Updated**: 2024-12-29

Complete documentation for Fracton - the infodynamics SDK with real theoretical foundations from Dawn Field Theory.

---

## Quick Links

- **New to Fracton?** Start with [Quick Reference](QUICK_REFERENCE.md)
- **Need API details?** See [API Reference](API_REFERENCE.md)
- **Want examples?** Check [Examples](../examples/)
- **Deploying?** Read [Docker Guide](../DOCKER.md)

---

## NEW: Theoretical Foundations (v2.0.0)

### KronosMemory with PAC/SEC/MED
- **[API Reference](API_REFERENCE.md)** - Complete class documentation
- **[Quick Reference](QUICK_REFERENCE.md)** - Cheat sheet and common patterns
- **[KronosMemory Spec](../.spec/kronos-memory.spec.md)** - Full specification
- **[Production Ready](../PRODUCTION_READY.md)** - Release checklist

### Getting Started
- **[Main README](../README.md)** - Project overview, installation, quick start
- **[Setup Guide](../SETUP.md)** - Installation and configuration

### Core Documentation
- **[Architecture](../ARCHITECTURE.md)** - System architecture and design
- **[Language Specification](../SPEC.md)** - Fracton language specification
- **[Roadmap](../ROADMAP.md)** - Development roadmap and milestones
- **[Status](../STATUS.md)** - Current implementation status

### Reality Engine Integration
- **[Integration Summary](REALITY_ENGINE_INTEGRATION_SUMMARY.md)** - Executive summary of Reality Engine integration
- **[Full Integration Guide](REALITY_ENGINE_INTEGRATION.md)** - Complete technical specification
- **[Möbius Quick Start](MOBIUS_QUICKSTART.md)** - Quick start for reality simulation

---

## Documentation Structure

### 1. Overview Documents

#### Main README ([../README.md](../README.md))
- Project overview
- Installation instructions
- Quick start examples
- Core philosophy
- Application domains

#### Status ([../STATUS.md](../STATUS.md))
- Current implementation status
- Completed features
- In-progress work
- Next steps

### 2. Technical Specifications

#### Architecture ([../ARCHITECTURE.md](../ARCHITECTURE.md))
- System architecture
- Module organization
- Data flow
- Integration patterns

#### Language Specification ([../SPEC.md](../SPEC.md))
- Language syntax
- Built-in functions
- Decorators
- Memory model
- Tool expression framework

#### Roadmap ([../ROADMAP.md](../ROADMAP.md))
- Development phases
- Milestones
- Feature timeline
- Integration plans

### 3. Reality Engine Integration (November 2025)

#### Executive Summary ([REALITY_ENGINE_INTEGRATION_SUMMARY.md](REALITY_ENGINE_INTEGRATION_SUMMARY.md))
**Start here for Reality Engine integration overview**

Contents:
- Strategic decision rationale
- Architecture overview
- Key features
- Validation strategy
- Implementation timeline
- Success criteria
- Benefits and risks

#### Full Integration Guide ([REALITY_ENGINE_INTEGRATION.md](REALITY_ENGINE_INTEGRATION.md))
**Complete technical specification**

Contents:
- Detailed architecture
- Möbius substrate design
- Thermodynamic fields
- SEC operator integration
- Time emergence
- Confluence operator
- Complete code examples
- Implementation roadmap

#### Möbius Quick Start ([MOBIUS_QUICKSTART.md](MOBIUS_QUICKSTART.md))
**Quick start guide for reality simulation**

Contents:
- Installation (when available)
- Basic usage examples
- Key concepts
- Integration with Fracton
- Advanced usage
- Troubleshooting

---

## Reading Guide

### For New Users

**Start with**:
1. [Main README](../README.md) - Understand what Fracton is
2. [Setup Guide](../SETUP.md) - Install Fracton
3. Pick your domain:
   - **Infodynamics/Cognition**: Continue with main README examples
   - **Reality Simulation**: Go to [Möbius Quick Start](MOBIUS_QUICKSTART.md)

### For Developers

**Start with**:
1. [Architecture](../ARCHITECTURE.md) - System design
2. [Language Specification](../SPEC.md) - Language details
3. [Status](../STATUS.md) - What's implemented
4. [Roadmap](../ROADMAP.md) - What's planned

### For Reality Simulation Users

**Start with**:
1. [Integration Summary](REALITY_ENGINE_INTEGRATION_SUMMARY.md) - Big picture
2. [Möbius Quick Start](MOBIUS_QUICKSTART.md) - Hands-on examples
3. [Full Integration Guide](REALITY_ENGINE_INTEGRATION.md) - Deep dive

### For Contributors

**Start with**:
1. [Status](../STATUS.md) - Current state
2. [Roadmap](../ROADMAP.md) - Planned work
3. [Architecture](../ARCHITECTURE.md) - System design
4. For Reality Engine: [Integration Summary](REALITY_ENGINE_INTEGRATION_SUMMARY.md)

---

## Key Concepts

### Fracton Core

**Recursion as First-Class Primitive**  
All computation flows through recursive function calls, managed by the RecursiveEngine.

**Entropy-Driven Execution**  
Functions activate based on entropy thresholds, enabling context-aware dispatch.

**Bifractal Traceability**  
Every operation maintains forward and reverse traces for analysis and debugging.

**Field-Aware Memory**  
Shared memory structures that respect entropy boundaries and context isolation.

### Reality Simulation (Möbius Module)

**Möbius Topology**  
Non-orientable substrate with anti-periodic boundaries: f(u+π, v) = -f(u, 1-v)

**PAC Conservation**  
P - A - M = 0 at machine precision (<1e-12), enforced every step.

**Thermodynamic-Information Duality**  
Information and energy are two views of the same field, unified via free energy F = E - TS.

**Time Emergence**  
Time is not fundamental - it emerges from disequilibrium pressure seeking equilibrium.

**Law Discovery**  
Physical laws emerge from geometry + conservation + thermodynamics, detected automatically.

---

## Examples

### Infodynamics Example

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

### Reality Simulation Example

```python
from fracton.mobius import RealityEngine

# Create universe simulator
reality = RealityEngine(size=(128, 32), device='cuda')

# Initialize from Big Bang
reality.initialize('big_bang')

# Evolve and watch physics emerge
for state in reality.evolve(steps=10000):
    if state['step'] % 1000 == 0:
        print(f"Time: {state['time']:.2f}")
        print(f"Temperature: {state['temperature']:.4f}")
        print(f"Disequilibrium: {state['disequilibrium']:.6f}")
```

---

## Contributing

We welcome contributions! Here's how to get started:

1. **Read the documentation** - Understand the architecture
2. **Check the roadmap** - See what's planned
3. **Check the status** - See what's implemented
4. **Pick a task** - Choose something from the roadmap
5. **Submit a PR** - Follow coding standards

For Reality Engine integration specifically, see Phase 0 in the [Roadmap](../ROADMAP.md).

---

## Support

### Questions?
- Check existing documentation
- Review examples in the main README
- Look at the roadmap for planned features

### Found a Bug?
- Check if it's a known issue
- Document steps to reproduce
- Submit with minimal test case

### Feature Request?
- Check if it's on the roadmap
- Explain use case and motivation
- Consider contributing!

---

## KronosMemory Quick Start

### Basic Usage

```python
from fracton.storage import KronosMemory, NodeType

# Initialize
memory = KronosMemory(storage_path="./data", namespace="demo")
await memory.connect()
await memory.create_graph("notes")

# Store
note_id = await memory.store(
    content="Fracton is production ready!",
    graph="notes",
    node_type=NodeType.CONCEPT
)

# Query
results = await memory.query("production ready", graphs=["notes"])
for r in results:
    print(f"[{r.score:.3f}] {r.node.content}")

# Health monitoring
health = memory.get_foundation_health()
print(f"Balance Ξ: {health['balance_operator']['latest']:.4f}")
print(f"Duty cycle: {health['duty_cycle']['latest']:.3f}")
```

### Theoretical Foundations

**PAC (Potential-Actualization Conservation)**:
- Fibonacci recursion: Ψ(k) = Ψ(k+1) + Ψ(k+2)
- Golden ratio scaling: φ = 1.618
- Validated to 1e-10 precision

**SEC (Symbolic Entropy Collapse)**:
- 4:1 attraction/repulsion balance
- Duty cycle: φ/(φ+1) = 0.618
- Resonance ranking

**MED (Macro Emergence Dynamics)**:
- Universal bounds: depth ≤ 1, nodes ≤ 3
- Quality scoring

**E=mc² Distance Validation**:
- Geometric conservation
- Model-specific c² constants

**Status**: ✅ 65/65 tests passing, <10% overhead

---

## Document Updates

| Document | Last Updated | Status |
|----------|-------------|--------|
| API_REFERENCE.md | Dec 29, 2024 | ✅ New (v2.0.0) |
| QUICK_REFERENCE.md | Dec 29, 2024 | ✅ New (v2.0.0) |
| README.md (this file) | Dec 29, 2024 | ✅ Updated |
| ../PRODUCTION_READY.md | Dec 29, 2024 | ✅ New |
| ../.spec/kronos-memory.spec.md | Dec 29, 2024 | ✅ Updated (v2.0.0) |
| ../README.md | Dec 29, 2024 | ✅ Current |
| ../ARCHITECTURE.md | Oct 2025 | ✅ Current |
| ../SPEC.md | Oct 2025 | ✅ Current |
| ../STATUS.md | Nov 4, 2025 | ✅ Current |
| ../ROADMAP.md | Nov 4, 2025 | ✅ Current |
| REALITY_ENGINE_INTEGRATION_SUMMARY.md | Nov 4, 2025 | ✅ Current |
| REALITY_ENGINE_INTEGRATION.md | Nov 4, 2025 | ✅ Current |
| MOBIUS_QUICKSTART.md | Nov 4, 2025 | ✅ Current |

---

## License

Fracton is released under the Apache 2.0 License. See [LICENSE](../LICENSE) for details.

---

**Fracton: The Programming Language for Infodynamics and Reality Simulation** 🌌
