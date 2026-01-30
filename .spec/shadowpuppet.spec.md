# ShadowPuppet Specification

> **Architecture-as-Code Evolution Framework**

---

## Version

**v0.3 (Dependencies, Domain Types, Test Suites)**

**Status:** Active Development  
**Last Updated:** January 2025

---

## 1. Executive Summary

ShadowPuppet is a model-agnostic framework for evolving software through:
- **Architecture-as-code**: Python protocols define structure
- **Template-guided generation**: Parent code as context
- **Coherence evaluation**: Structural/semantic/energetic fitness
- **Genealogy tracking**: Full provenance trees

### The Metaphor

```
Protocol    = Puppet structure (joints, constraints)
Generator   = Puppeteer (brings it to life)
Shadow      = Generated code (the projection that runs)
Evolution   = Selection for better puppets
```

### Core Role in Fracton

```
ShadowPuppet (Tool)  → Generates code components
    ↓
Fracton (SDK)        → Uses generated components
    ↓
Applications         → Built on generated substrate
```

---

## 2. Requirements

### 2.1 Functional Requirements

- [x] Define software architecture as Python protocols
- [x] Generate implementations from protocol specs
- [x] Evaluate generated code fitness (coherence)
- [x] Track genealogy of generated components
- [x] Support multiple generators (Mock, Copilot, Claude)
- [x] Enforce PAC invariants during generation
- [x] Order generation by dependencies
- [x] Pass domain types to generators
- [x] Attach test suites to gaps
- [ ] LLM-based semantic validation (optional)

### 2.2 Non-Functional Requirements

- [x] Generator-agnostic (works without LLM)
- [x] Testable (comprehensive unit + integration tests)
- [x] Documented (README, examples, docstrings)
- [ ] Performance: Generate 10+ components in <10s (with Mock)

---

## 3. Architecture

### 3.1 Module Structure

```
tools/shadowpuppet/
├── __init__.py          # Public exports
├── protocols.py         # Core data structures
├── coherence.py         # Fitness evaluation
├── evolution.py         # Main evolution engine
├── genealogy.py         # Lineage tracking
├── generators/
│   ├── base.py          # Generator protocol
│   ├── mock.py          # Template-based (no LLM)
│   ├── copilot.py       # GitHub Copilot
│   └── claude.py        # Claude API
└── examples/
    ├── webapp_seed.py   # Web application architecture
    └── chatbot_seed.py  # Chatbot architecture
```

### 3.2 Core Data Structures

#### ProtocolSpec

Defines what to generate:

```python
@dataclass
class ProtocolSpec:
    name: str                          # Class name
    methods: List[str]                 # Required methods
    docstring: str                     # Description
    attributes: List[str]              # Class attributes (name: Type)
    pac_invariants: List[str]          # Conservation laws
    dependencies: List[str]            # Other protocols this depends on
    method_signatures: List[TypeAnnotation]  # Rich method types
```

#### TypeAnnotation

Rich method type information:

```python
@dataclass
class TypeAnnotation:
    name: str
    params: Dict[str, str]      # param_name -> type_hint
    returns: str
    raises: List[str]
    async_method: bool
```

#### GrowthGap

What needs to be filled:

```python
@dataclass
class GrowthGap:
    protocol: ProtocolSpec
    parent_components: List[ComponentOrganism]
    required_coherence: float
    priority: float
    test_suite: Optional[TestSuite]
    domain_types: List[str]     # Type definition source code
```

#### ComponentOrganism

Generated code as organism:

```python
@dataclass
class ComponentOrganism:
    id: str
    protocol_name: str
    code: str
    coherence_score: float      # Overall fitness
    structural_score: float     # Type correctness
    semantic_score: float       # Logic correctness
    energetic_score: float      # Efficiency
    parent_id: Optional[str]
    generation: int
    derivation_path: List[str]
```

### 3.3 Evolution Flow

```
1. ORDER GAPS        → Topological sort by dependencies
2. FOR EACH GAP:
   a. SELECT PARENT  → Tournament selection from population
   b. GENERATE       → Create N candidates (with crossover/mutation)
   c. EVALUATE       → Score coherence (structure/semantic/energy)
   d. REFINE         → Fix borderline candidates
   e. SELECT         → Keep best above threshold
3. TRACK GENEALOGY   → Record lineage
4. CHECK CONVERGENCE → Stop if all gaps filled
```

### 3.4 Dependency Ordering

Uses Kahn's algorithm for topological sort:

```python
def _order_by_dependencies(gaps: List[GrowthGap]) -> List[GrowthGap]:
    # Build dependency graph
    # Start with gaps that have no in-set dependencies
    # Process in order, reducing in-degrees
    # Return sorted list (dependencies first)
```

**Example:**
```
Input order:  [WebApp, APIRouter, UserService]
Dependencies: WebApp -> [APIRouter, UserService]
Output order: [APIRouter, UserService, WebApp]
```

### 3.5 Domain Types

Domain types are source code strings passed to generators:

```python
DOMAIN_TYPES = [
    '''@dataclass
class User:
    id: str
    name: str''',
]

gap = GrowthGap(
    protocol=proto,
    domain_types=DOMAIN_TYPES
)
```

Generators include these in prompts under "DOMAIN TYPES" section.

### 3.6 Test Suites

Test functions attached to gaps for validation:

```python
def test_user_crud(service):
    user = service.create_user("test", "test@example.com", "pass")
    assert user.username == "test"
    return True

gap = GrowthGap(
    protocol=proto,
    test_suite=TestSuite(unit=[test_user_crud])
)
```

Tests are passed to evaluator for fitness scoring.

---

## 4. Generators

### 4.1 Generator Protocol

```python
class CodeGenerator(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @abstractmethod
    def generate(self, context: GenerationContext) -> str: ...
    
    def build_prompt(self, context: GenerationContext) -> str: ...
```

### 4.2 GenerationContext

```python
@dataclass
class GenerationContext:
    protocol: ProtocolSpec
    parent: Optional[ComponentOrganism]
    siblings: List[ComponentOrganism]
    mutation_rate: float
    pac_invariants: List[str]
    extra_instructions: str
    domain_types: List[str]
    resolved_dependencies: Dict[str, ComponentOrganism]
```

### 4.3 MockGenerator

Template-based generator for testing (no LLM required):

- Recognizes 9 component types by name pattern
- Generates working implementations with real logic
- Supports custom templates via `templates` dict

**Supported Patterns:**
- `APIRouter`, `UserService`, `TemplateRenderer`
- `StaticFileServer`, `WebApp`
- `IntentClassifier`, `ResponseGenerator`
- `ConversationManager`, `ChatBot`

---

## 5. Coherence Evaluation

### 5.1 Three Fitness Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| Structural | 40% | Syntax, types, interfaces |
| Semantic | 35% | Logic, invariants, correctness |
| Energetic | 25% | Efficiency, simplicity, elegance |

### 5.2 Invariant Validation

```python
def validate_invariants(component: ComponentOrganism, invariants: List[str]) -> Dict:
    # Pattern-based checking
    # - "unique" → check for ID generation/uniqueness
    # - "validated" → check for validation code
    # - "never empty" → check for empty guards
    # Returns {component_id: [violations]}
```

### 5.3 Hard Mode

When `hard_invariants=True`, any violation sets score to 0.

---

## 6. Evolution Configuration

```python
@dataclass
class EvolutionConfig:
    coherence_threshold: float = 0.70      # Minimum to survive
    reproduction_threshold: float = 0.80   # Template eligibility
    max_population: int = 50
    candidates_per_gap: int = 3
    mutation_rate: float = 0.2
    max_generations: int = 10
    
    # V0.2 features
    enable_crossover: bool = True
    crossover_rate: float = 0.3
    enable_refinement: bool = True
    refinement_threshold: float = 0.5
    max_refinement_attempts: int = 2
```

---

## 7. Genetic Operations

### 7.1 Crossover

AST-based method splicing:

```python
def crossover(parent_a, parent_b, protocol) -> str:
    # Parse both parents
    # Collect methods from each
    # Combine: parent_a methods not in parent_b + parent_b methods not in parent_a
    # Generate new class with combined methods
```

### 7.2 Refinement

Targeted repair for borderline candidates:

```python
def refine(component, context, violations) -> str:
    # Add violations to extra_instructions
    # Request generator to fix specific issues
    # Return refined code
```

### 7.3 Tournament Selection

```python
def _tournament_select(candidates, tournament_size=3):
    # Random sample of tournament_size
    # Return best by coherence_score
```

---

## 8. Examples

### 8.1 WebApp Seed Structure

```
DOMAIN TYPES:
├── Request (method, path, headers, body)
├── Response (status_code, headers, body)
└── User (id, username, email)

PROTOCOLS:
├── APIRouter (get, post, put, delete, handle)
├── UserService (create_user, get_user, update_user, delete_user)
├── TemplateRenderer (render, load_template, escape_html)
├── StaticFileServer (serve, get_mime_type)
└── WebApp (handle_request, start, stop) → depends on [APIRouter, UserService, TemplateRenderer, StaticFileServer]

TEST FUNCTIONS:
├── test_router_returns_response
├── test_router_path_params
├── test_user_service_crud
├── test_template_escapes_html
└── test_static_server_mime_types
```

### 8.2 ChatBot Seed Structure

```
DOMAIN TYPES:
├── Message (role, content, timestamp, metadata)
├── Conversation (id, messages, context)
└── Intent (name, confidence, entities)

PROTOCOLS:
├── IntentClassifier (classify, extract_entities, train)
├── ResponseGenerator (generate, set_template, format_response)
├── ConversationManager (create_conversation, get_conversation, add_message)
└── ChatBot (chat, start_conversation, end_conversation) → depends on [IntentClassifier, ResponseGenerator, ConversationManager]

TEST FUNCTIONS:
├── test_intent_confidence_valid
├── test_intent_unknown_fallback
├── test_response_not_empty
├── test_conversation_message_order
├── test_conversation_context_isolation
└── test_chatbot_returns_response
```

---

## 9. Fibonacci Constraints (MED Principle)

Seeds encode bounded complexity via MED:

```python
'fibonacci_constraints': {
    'max_depth': 2,      # Maximum composition depth
    'max_components': 5  # Maximum component count
}
```

This enforces "all complex flows converge to symbolic patterns with depth ≤ 2 and nodes ≤ 3."

---

## 10. Testing

### 10.1 Test Coverage

| Module | Unit Tests | Integration Tests |
|--------|------------|-------------------|
| protocols | TypeAnnotation, ProtocolSpec, GrowthGap, ComponentOrganism | - |
| coherence | CoherenceEvaluator, invariant validation | - |
| evolution | Config, parent selection, crossover, refinement | Full cycles |
| genealogy | Tree operations, node tracking | - |
| generators | MockGenerator, GenerationContext | Code execution |
| integration | - | E2E seed examples |

### 10.2 Test Categories

```python
test_protocols.py      # Data structure tests
test_coherence.py      # Evaluation tests
test_evolution.py      # Evolution engine + dependency ordering
test_genealogy.py      # Lineage tracking
test_generators.py     # Code generation
test_integration.py    # E2E with seed examples
```

---

## 11. Status

### Completed (v0.3)

- [x] Core protocol definitions
- [x] MockGenerator with 9 patterns
- [x] Coherence evaluation
- [x] Genealogy tracking
- [x] Crossover and refinement
- [x] TypeAnnotation for rich signatures
- [x] Dependency ordering (topological sort)
- [x] Domain types in GenerationContext
- [x] TestSuite integration
- [x] Updated seed examples
- [x] 147 passing tests

### Planned (v0.4)

- [ ] CopilotGenerator implementation
- [ ] ClaudeGenerator implementation
- [ ] LLM semantic validation
- [ ] Performance benchmarks
- [ ] Additional seed examples (CLI, API gateway)

---

## 12. API Reference

### Public Exports

```python
from fracton.tools.shadowpuppet import (
    # Core
    ProtocolSpec,
    GrowthGap,
    ComponentOrganism,
    TestSuite,
    TypeAnnotation,
    
    # Evolution
    SoftwareEvolution,
    EvolutionConfig,
    
    # Evaluation
    CoherenceEvaluator,
    
    # Genealogy
    GenealogyTree,
    
    # Generators
    CodeGenerator,
    MockGenerator,
    GenerationContext,
)
```

---

## 13. Changelog

### v0.3 (January 2025)
- Added `dependencies` field to ProtocolSpec
- Added `domain_types` field to GrowthGap
- Added `resolved_dependencies` to GenerationContext
- Implemented topological sort for dependency ordering
- Added test functions to seed examples
- Added 22 new tests (dependency, domain type, test suite)
- E2E tests for seed examples

### v0.2 (January 2025)
- TypeAnnotation for rich method signatures
- Genetic crossover via AST method splicing
- PAC invariant validation with hard mode
- Targeted refinement for borderline candidates
- Tournament selection
- 125 passing tests

### v0.1 (January 2025)
- Initial implementation
- ProtocolSpec, GrowthGap, ComponentOrganism
- MockGenerator with 9 patterns
- CoherenceEvaluator
- GenealogyTree
- Webapp and chatbot seed examples
