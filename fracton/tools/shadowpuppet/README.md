# ShadowPuppet

**Architecture-as-Code Evolution Framework**

A model-agnostic framework for evolving software through protocol-driven generation.

## The Metaphor

- **Protocol** is the puppet (structure, joints, constraints)
- **Generator** is the puppeteer (brings it to life)
- **Shadow** is the generated code (the projection that runs)
- **Evolution** selects better puppets (architecture improvement)

## Quick Start

```python
from fracton.tools.shadowpuppet import (
    SoftwareEvolution,
    ProtocolSpec,
    GrowthGap,
    MockGenerator
)

# Define architecture as Python protocols
api_protocol = ProtocolSpec(
    name="APIRouter",
    methods=["get", "post", "put", "delete"],
    docstring="REST API router with CRUD operations",
    pac_invariants=["All routes return JSON", "Errors use standard HTTP codes"]
)

# Create evolution with your generator
evolution = SoftwareEvolution(
    generator=MockGenerator(),  # No AI required
    coherence_threshold=0.65
)

# Evolve!
results = evolution.grow([GrowthGap(protocol=api_protocol)])

# Get generated code
for organism in results.population:
    print(f"{organism.id}: {organism.fitness:.3f}")
    print(organism.code)
```

## Key Concepts

### ProtocolSpec

Defines a component's structure:

```python
ProtocolSpec(
    name="UserService",
    methods=["create_user", "get_user", "update_user", "delete_user"],
    attributes=["users", "email_index"],
    docstring="User management with validation",
    pac_invariants=[
        "User IDs are unique and immutable",
        "Emails are validated before storage",
        "Passwords are never stored in plaintext"
    ]
)
```

### GrowthGap

Identifies what needs to be generated:

```python
GrowthGap(
    protocol=user_protocol,
    parent_code=existing_code,  # Optional: context for AI
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
