# ShadowPuppet

**Architecture-as-Code Evolution Framework** (v0.2.0)

A model-agnostic framework for evolving software through protocol-driven generation.

## The Metaphor

- **Protocol** is the puppet (structure, joints, constraints)
- **Generator** is the puppeteer (brings it to life)
- **Shadow** is the generated code (the projection that runs)
- **Evolution** selects better puppets (architecture improvement)

## What's New in v0.2

- **Rich Type Signatures** - `TypeAnnotation` for stronger method contracts
- **Genetic Crossover** - Combine methods from high-fitness parents
- **PAC Invariant Validation** - Hard enforcement of conservation laws
- **Targeted Refinement** - Fix specific issues in borderline candidates
- **LLM Semantic Review** - Optional AI-based semantic validation
- **Tournament Selection** - Better parent selection strategy

## Quick Start

```python
from fracton.tools.shadowpuppet import (
    SoftwareEvolution,
    ProtocolSpec,
    GrowthGap,
    MockGenerator,
    TypeAnnotation
)

# Define architecture with rich types
api_protocol = ProtocolSpec(
    name="APIRouter",
    methods=["get", "post", "put", "delete"],
    method_signatures=[
        TypeAnnotation("get", {"path": "str"}, "Response"),
        TypeAnnotation("post", {"path": "str", "body": "Dict"}, "Response", raises=["ValidationError"]),
    ],
    docstring="REST API router with CRUD operations",
    pac_invariants=["All routes return JSON", "Errors use standard HTTP codes"],
    attributes=["routes: Dict[str, Callable]", "middleware: List[Callable]"]
)

# Create evolution with crossover and refinement
from fracton.tools.shadowpuppet.evolution import EvolutionConfig

evolution = SoftwareEvolution(
    generator=MockGenerator(),
    config=EvolutionConfig(
        coherence_threshold=0.65,
        enable_crossover=True,
        enable_refinement=True
    )
)

# Evolve!
results = evolution.grow([GrowthGap(protocol=api_protocol)])

# Get generated code
for component in evolution.components:
    print(f"{component.id}: {component.coherence_score:.3f}")
```

## Key Concepts

### TypeAnnotation (New in v0.2)

Rich type signatures for methods:

```python
TypeAnnotation(
    name="get_user",
    params={"user_id": "str", "include_deleted": "bool = False"},
    returns="Optional[User]",
    raises=["NotFoundError", "ValidationError"],
    async_method=False
)
# Generates: def get_user(self, user_id: str, include_deleted: bool = False) -> Optional[User]
```

### ProtocolSpec

Defines a component's structure:

```python
ProtocolSpec(
    name="UserService",
    methods=["create_user", "get_user", "update_user", "delete_user"],
    method_signatures=[...],  # Optional rich types
    attributes=["users: Dict[str, User]", "email_index: Dict[str, str]"],
    docstring="User management with validation",
    pac_invariants=[
        "User IDs are unique and immutable",
        "Emails are validated before storage",
        "Passwords are never stored in plaintext"  # Will be enforced!
    ]
)
```

### GrowthGap

Identifies what needs to be generated:

```python
GrowthGap(
    protocol=user_protocol,
    parent_components=[existing_component],  # Optional: context for AI
    priority=1.0
)
```

### Generators

Pluggable code generators:

| Generator | Description | Requirements |
|-----------|-------------|--------------|
| `MockGenerator` | Template-based, no AI | None |
| `CopilotGenerator` | GitHub Copilot CLI | `gh copilot` installed |
| `ClaudeGenerator` | Claude API | `ANTHROPIC_API_KEY` |
| `ClaudeCodeGenerator` | Claude Code CLI | `claude` installed |

### Coherence Evaluation

Three-dimensional fitness scoring:

- **Structural** (0-1): Type correctness, method signatures
- **Semantic** (0-1): Logic alignment with docstring/invariants
- **Energetic** (0-1): Efficiency, resource usage

Combined: `fitness = (structural * semantic * energetic) ^ (1/3)`

**PAC Invariant Validation** (New in v0.2):
```python
evaluator = CoherenceEvaluator(
    enforce_invariants=True,  # Hard-fail on violations
    llm_reviewer=ClaudeGenerator()  # Optional semantic review
)
```

### EvolutionConfig

Configure the evolution process:

```python
EvolutionConfig(
    coherence_threshold=0.70,      # Minimum fitness to survive
    reproduction_threshold=0.80,   # Minimum to become parent
    max_population=50,
    candidates_per_gap=3,
    mutation_rate=0.2,
    max_generations=10,
    # New in v0.2
    enable_crossover=True,         # Genetic crossover
    crossover_rate=0.3,            # Probability of crossover
    enable_refinement=True,        # Targeted repair
    refinement_threshold=0.5,      # Score to trigger refinement
    max_refinement_attempts=2
)
```

### Crossover (New in v0.2)

When enabled, evolution can combine methods from two parents:

```python
# Automatic during evolution
evolution = SoftwareEvolution(
    config=EvolutionConfig(enable_crossover=True)
)

# Or manually
child_code = evolution.crossover(parent_a, parent_b, protocol)
```

### Refinement (New in v0.2)

Borderline candidates get targeted repair:

```python
# Automatic during evolution for scores < refinement_threshold
# Or manually
refined = evolution.refine(component, context, ["Fix: passwords not hashed"])

## Examples

### Web Application

```bash
python -m fracton.tools.shadowpuppet.examples.webapp_seed
```

Generates:
- `APIRouter` - REST routing with decorators and middleware
- `UserService` - CRUD with validation and authentication
- `TemplateRenderer` - HTML templates with escaping
- `StaticFileServer` - Static files with MIME types
- `WebApp` - HTTP server orchestrator

### Chatbot

```bash
python -m fracton.tools.shadowpuppet.examples.chatbot_seed
```

Generates:
- `IntentClassifier` - Keyword-based intent detection
- `ResponseGenerator` - Template-based responses
- `ConversationManager` - Session and history management
- `ChatBot` - Main orchestrator with CLI mode

## Architecture

```
shadowpuppet/
├── __init__.py          # Public API
├── protocols.py         # ProtocolSpec, GrowthGap, ComponentOrganism
├── coherence.py         # CoherenceEvaluator
├── evolution.py         # SoftwareEvolution engine
├── genealogy.py         # GenealogyTree for provenance
├── generators/
│   ├── base.py          # CodeGenerator protocol
│   ├── mock.py          # Template-based (no AI)
│   ├── copilot.py       # GitHub Copilot CLI
│   └── claude.py        # Claude API + Claude Code CLI
└── examples/
    ├── webapp_seed.py   # Frontend + API example
    └── chatbot_seed.py  # Chatbot example
```

## Theoretical Foundation

ShadowPuppet implements PAC (Potential-Actualization Conservation) from Dawn Field Theory:

- **Potential** = Protocol specification (what could be)
- **Actualization** = Generated code (what is)
- **Conservation** = Coherence ensures the actualized code preserves the protocol's intent

The evolution process applies SEC (Symbolic Entropy Collapse):
- High-fitness candidates represent lower entropy (more coherent)
- Selection pressure drives the system toward structured solutions

## Custom Generators

Implement the `CodeGenerator` protocol:

```python
from fracton.tools.shadowpuppet import CodeGenerator, GenerationContext

class MyGenerator(CodeGenerator):
    @property
    def name(self) -> str:
        return "my-generator"
    
    def generate(self, context: GenerationContext) -> str:
        # context.protocol - the ProtocolSpec
        # context.parent_code - optional parent context
        # context.temperature - creativity parameter
        return "# Your generated code here"
```

## License

Same as Fracton - see repository root LICENSE.
