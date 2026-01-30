# ShadowPuppet

**Architecture-as-Code Evolution Framework** (v0.3.0)

## The Idea

Code generation is solved. Architecture isn't.

LLMs can write functions, classes, even small systems. But they hit a wall at **architecture**.. the exponential explosion of interactions between components, the judgment calls about where to draw boundaries, the invariants that must hold across an entire system.

ShadowPuppet doesn't try to solve architecture. It provides a **process for navigating architectural space**:

1. **Declare what coherence means** — Protocols, invariants, dependencies, tests
2. **Let AI explore implementations** — Generate candidates that might satisfy those constraints
3. **Evaluate fitness** — Structural correctness, semantic alignment, execution validity
4. **Select survivors** — Only coherent implementations persist

The seed file becomes the artifact. Code becomes a derived projection — regenerable, disposable, always consistent with the spec.

This is **not** "AI writes code for you." This is **tooling for the next abstraction layer** — the one that defines what it means for code to be architecturally coherent.

## The Metaphor

- **Protocol** is the puppet (structure, joints, constraints)
- **Generator** is the puppeteer (brings it to life)
- **Shadow** is the generated code (the projection that runs)
- **Evolution** selects better shadows (implementation improvement)

## What's New in v0.3

- **Dependency Ordering** — Topological sort ensures components generate in correct order
- **Domain Types** — Pass dataclass/type definitions to generators for context
- **Test Suites** — Attach unit tests to gaps; tests run during fitness evaluation
- **ClaudeGenerator Integration** — All seeds now use real Claude API with MockGenerator fallback
- **Improved Coherence Scoring** — 85/15 weighting for early generations prevents premature extinction

### From v0.2
- **Rich Type Signatures** - `TypeAnnotation` for stronger method contracts
- **Genetic Crossover** - Combine methods from high-fitness parents
- **PAC Invariant Validation** - Hard enforcement of conservation laws
- **Targeted Refinement** - Fix specific issues in borderline candidates

## Quick Start

```python
from fracton.tools.shadowpuppet import (
    SoftwareEvolution,
    ProtocolSpec,
    GrowthGap,
    EvolutionConfig,
    TestSuite
)
from fracton.tools.shadowpuppet.generators import ClaudeGenerator, MockGenerator

# Define your domain types
DOMAIN_TYPES = [
    '''
@dataclass
class User:
    id: str
    email: str
    name: str
    '''
]

# Define architecture with dependencies
user_service = ProtocolSpec(
    name="UserService",
    methods=["create_user", "get_user", "update_user", "delete_user"],
    docstring="User management with validation",
    attributes=["users: Dict[str, User]"],
    pac_invariants=[
        "User IDs are unique",
        "Emails are validated before storage"
    ],
    dependencies=[]  # No dependencies
)

api_router = ProtocolSpec(
    name="APIRouter",
    methods=["get", "post", "handle_request"],
    docstring="REST API router",
    attributes=["routes: Dict[str, Callable]", "user_service: UserService"],
    pac_invariants=["All routes return Response objects"],
    dependencies=["UserService"]  # Depends on UserService
)

# Define tests
def test_user_crud(service):
    user = service.create_user("test@example.com", "Test")
    assert service.get_user(user.id) is not None
    return True

# Create gaps with tests and domain context
gaps = [
    GrowthGap(
        protocol=user_service,
        test_suite=TestSuite(unit=[test_user_crud]),
        domain_types=DOMAIN_TYPES
    ),
    GrowthGap(
        protocol=api_router,
        domain_types=DOMAIN_TYPES
    ),
]

# Configure evolution
config = EvolutionConfig(
    coherence_threshold=0.65,
    candidates_per_gap=3,
    max_generations=10
)

# Use Claude with MockGenerator fallback
generator = ClaudeGenerator(
    model="claude-sonnet-4-20250514",
    fallback_generator=MockGenerator()
)

# Evolve!
evolution = SoftwareEvolution(generator=generator, config=config)
results = evolution.grow(gaps)

# Components are generated in dependency order:
# UserService first, then APIRouter (which can reference UserService)
for component in evolution.components:
    print(f"{component.id}: {component.coherence_score:.3f}")

# Save generated code
evolution.save_code(Path("generated/"))
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
    test_suite=TestSuite(unit=[test_func1, test_func2]),  # v0.3: attached tests
    domain_types=["@dataclass\nclass User: ..."],        # v0.3: type context
    parent_components=[existing_component],               # Optional: context for AI
    priority=1.0
)
```

### TestSuite (New in v0.3)

Attach tests that run during fitness evaluation:

```python
def test_user_creation(service):
    """Test must accept instance and return bool."""
    user = service.create_user("test@example.com", "Test User")
    return user is not None and user.email == "test@example.com"

def test_user_uniqueness(service):
    service.create_user("dup@example.com", "First")
    try:
        service.create_user("dup@example.com", "Second")
        return False  # Should have raised
    except ValueError:
        return True

gap = GrowthGap(
    protocol=user_protocol,
    test_suite=TestSuite(
        unit=[test_user_creation, test_user_uniqueness],
        integration=[],  # Future: cross-component tests
        property=[]      # Future: property-based tests
    )
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

### GAIA (PAC/SEC Dynamics)

```bash
python -m fracton.tools.shadowpuppet.examples.gaia_seed
```

An 8-component system modeling information-entropy dynamics:
- `InformationField` - Information density field
- `EntropyField` - Entropy density field  
- `PACAggregator` - Potential-Actualization conservation
- `BalanceOperator` - Field equilibrium (Xi constant)
- `CollapseDetector` - SEC collapse events
- `RecursiveLayer` - Recursive balance feedback
- `StructureEmitter` - Structure crystallization
- `GAIAModel` - Main orchestrator

### Web Application

```bash
python -m fracton.tools.shadowpuppet.examples.webapp_seed
```

Generates:
- `APIRouter` - REST routing with middleware
- `UserService` - CRUD with validation
- `TemplateRenderer` - HTML templates with escaping
- `StaticFileServer` - Static files with MIME types
- `WebApp` - HTTP server orchestrator

### Chatbot

```bash
python -m fracton.tools.shadowpuppet.examples.chatbot_seed
```

Generates:
- `IntentClassifier` - Intent detection with confidence
- `ResponseGenerator` - Template-based responses
- `ConversationManager` - Session and history management
- `ChatBot` - Main orchestrator

## Architecture

```
shadowpuppet/
├── __init__.py          # Public API
├── protocols.py         # ProtocolSpec, GrowthGap, TestSuite, ComponentOrganism
├── coherence.py         # CoherenceEvaluator (structural, semantic, execution)
├── evolution.py         # SoftwareEvolution engine with dependency ordering
├── genealogy.py         # GenealogyTree for provenance tracking
├── generators/
│   ├── base.py          # CodeGenerator protocol, GenerationContext
│   ├── mock.py          # Template-based (no AI, for testing)
│   ├── copilot.py       # GitHub Copilot CLI
│   └── claude.py        # Claude API + Claude Code CLI
└── examples/
    ├── gaia_seed.py     # PAC/SEC dynamics (8 components)
    ├── webapp_seed.py   # Frontend + API (5 components)
    └── chatbot_seed.py  # Chatbot (4 components)
```

## Theoretical Foundation

ShadowPuppet implements concepts from Dawn Field Theory:

### PAC (Potential-Actualization Conservation)
- **Potential** = Protocol specification (what could be)
- **Actualization** = Generated code (what is)
- **Conservation** = Coherence ensures the actualized code preserves the protocol's intent

### SEC (Symbolic Entropy Collapse)
The evolution process applies SEC dynamics:
- **Information gradient** (∇I) = Protocol constraints that must be satisfied
- **Entropy gradient** (∇H) = Random variation in generation
- **Collapse boundary** = Coherence threshold where structure crystallizes

High-fitness candidates represent lower entropy (more coherent structure).
Selection pressure drives the system toward solutions that balance constraint satisfaction with implementation flexibility.

### Why Evolution?
Single-shot generation works for isolated components. But architecture involves **n² interactions** between n components. Evolution provides:
- **Constraint propagation** — Dependencies ensure components see their dependencies' implementations
- **Feedback loops** — Tests validate cross-component behavior
- **Selection pressure** — Only coherent implementations survive

The seed file defines the collapse boundary. Evolution finds what crystallizes within those constraints.

## Custom Generators

Implement the `CodeGenerator` protocol:

```python
from fracton.tools.shadowpuppet.generators import CodeGenerator, GenerationContext

class MyGenerator(CodeGenerator):
    @property
    def name(self) -> str:
        return "my-generator"
    
    def generate(self, context: GenerationContext) -> str:
        # Available context:
        # - context.protocol: ProtocolSpec to implement
        # - context.domain_types: List[str] of type definitions
        # - context.resolved_dependencies: Dict[str, ComponentOrganism]
        # - context.parent_code: Optional parent implementation
        # - context.temperature: Creativity parameter
        
        prompt = self.build_prompt(context)  # Use built-in prompt builder
        # ... call your LLM ...
        return self.extract_code(response)   # Extract code from response
```

## Limitations

- **Scale**: Works well for ~10-20 components. The n² interaction explosion means larger systems need decomposition into subsystems.
- **Novel architecture**: Can only generate what's in LLM training data. Novel patterns require human design.
- **Integration testing**: Currently validates components individually. Cross-component integration is a gap.

## License

Same as Fracton - see repository root LICENSE.
