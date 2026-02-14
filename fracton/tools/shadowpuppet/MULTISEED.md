# ShadowPuppet v0.6 - Multi-Seed Architecture

## What's New

Added **multi-seed evolution** for microservice-style architectures. Now you can:

1. **Define multiple seeds** (services/bounded contexts) with explicit dependencies
2. **Evolve them in topological order** (foundations first, dependents second)
3. **Extract public interfaces** from evolved components automatically
4. **Validate cross-seed contracts** to prevent integration bugs
5. **Run integration tests** that span multiple seeds

## Core Concepts

### SeedArchitecture
Represents a complete service or bounded context:
- Has internal components (GrowthGaps)
- Exposes public interfaces to other seeds
- May depend on other seeds' interfaces
- Evolves independently with its own fitness criteria

### SeedConnector
Explicit contract between two seeds:
- Provider seed exposes interfaces
- Consumer seed uses those interfaces
- Validated during evolution to prevent mismatches

### MultiSeedEvolution
Orchestrates evolution across multiple seeds:
- Topologically sorts seeds by dependencies
- Evolves each seed in order
- Passes evolved interfaces to dependent seeds
- Validates cross-seed calls and integration tests

## Example: E-Commerce System

```python
from fracton.tools.shadowpuppet import (
    SeedArchitecture,
    MultiSeedEvolution,
    ProtocolSpec,
    GrowthGap
)

# Seed 1: UserService (no dependencies)
user_seed = SeedArchitecture(
    name="UserService",
    gaps=[GrowthGap(protocol=user_repo_protocol)],
    exposed_interfaces=["UserRepository", "AuthService"],
    dependencies={}
)

# Seed 2: ProductService (no dependencies)
product_seed = SeedArchitecture(
    name="ProductService",
    gaps=[GrowthGap(protocol=product_repo_protocol)],
    exposed_interfaces=["ProductRepository"],
    dependencies={}
)

# Seed 3: OrderService (depends on both)
order_seed = SeedArchitecture(
    name="OrderService",
    gaps=[GrowthGap(protocol=order_processor_protocol)],
    exposed_interfaces=["OrderRepository", "OrderProcessor"],
    dependencies={
        "UserService": ["AuthService"],
        "ProductService": ["ProductRepository"]
    }
)

# Evolve all three
multi_evolution = MultiSeedEvolution([user_seed, product_seed, order_seed])
results = multi_evolution.evolve(max_generations=10, cross_seed_iterations=2)
```

## Output Structure

```
generated/multi_seed/
├── userservice/
│   ├── userrepository.py
│   ├── authservice.py
│   └── interfaces.json
├── productservice/
│   ├── productrepository.py
│   └── interfaces.json
├── orderservice/
│   ├── orderrepository.py
│   ├── orderprocessor.py
│   └── interfaces.json
└── connectors.json  # Cross-seed contracts
```

## Evolution Process

1. **Topological Sort**: `UserService, ProductService → OrderService`
2. **Evolve UserService**: Generate UserRepository + AuthService
3. **Extract Interfaces**: Extract exact method signatures
4. **Evolve ProductService**: Generate ProductRepository
5. **Extract Interfaces**: Extract ProductRepository interface
6. **Evolve OrderService**: 
   - Receives AuthService + ProductRepository interfaces in prompts
   - Generates OrderProcessor that calls those interfaces
   - Validates calls match actual signatures
7. **Cross-Seed Validation**: Check all inter-seed calls are valid
8. **Integration Tests**: Run tests spanning multiple seeds

## Key Benefits

### Architectural
- **Explicit boundaries**: Seeds define clear service boundaries
- **Contract enforcement**: Interface mismatches fail fast
- **Dependency management**: Topological ordering prevents circular deps
- **Modular evolution**: Each seed evolves independently

### Practical
- **Scalable**: Evolve 2 seeds or 20 seeds with same process
- **Testable**: Integration tests validate full system
- **Documentable**: Generated connectors.json shows all contracts
- **Reproducible**: Seed definitions are the source of truth

## Use Cases

### Microservices
Each seed = one microservice with explicit API contracts

### Bounded Contexts (DDD)
Each seed = one domain with clear boundaries

### Plugin Architectures
Core seed + multiple plugin seeds that consume core interfaces

### Layered Systems
Data layer seed → Business logic seed → API layer seed

## What's Next (v0.7?)

- **Contract versioning**: Track interface evolution over time
- **Breaking change detection**: Flag when interfaces change incompatibly
- **Partial re-evolution**: Re-evolve one seed without restarting all
- **Docker/K8s generation**: Generate deployment configs from seeds
- **Service mesh integration**: Generate Istio/Linkerd configs

## Try It

See `examples/ecommerce_multiseed.py` for a complete working example with three seeds:
- UserService (authentication)
- ProductService (catalog + inventory)
- OrderService (order processing, depends on both)

```bash
cd fracton/fracton/tools/shadowpuppet
python -m examples.ecommerce_multiseed
```
