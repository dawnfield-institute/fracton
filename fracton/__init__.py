"""
Fracton: Infodynamics Computational Modeling Language

Fracton is a domain-specific computational modeling language designed for
infodynamics research and recursive field-aware systems. It provides a
unified substrate for modeling emergent intelligence, entropy dynamics,
and bifractal computation patterns.

v2.1: Recursion as the Mainstay

The top-level fracton namespace exports the recursive core — everything
needed to write recursive, entropy-aware, PAC-conserving computations.
Toolkit modules (physics, field, storage, etc.) are accessed via sub-packages:

    from fracton import recursive, recurse, MemoryField, Context
    from fracton.field import evolve, spherical_encode
    from fracton.physics import PHI, XI
    from fracton.core import PACSystem, PACNode
    from fracton.storage import KronosGraph

Example Usage:
    import fracton

    @fracton.recursive
    def my_function(memory, context):
        if context.depth >= 3:
            return fracton.crystallize(memory.get("data"))
        return fracton.recurse(my_function, memory, context.deeper(1))

    with fracton.memory_field() as field:
        result = my_function(field, fracton.Context(entropy=0.5, depth=0))
"""

import importlib
import warnings

# === Recursive Core ===
# The mainstay: everything needed for recursive, entropy-aware computation.

from .core.physics_plugin import PhysicsPlugin

from .core import (
    # Recursive engine
    RecursiveExecutor, ExecutionContext,
    PhysicsRecursiveExecutor, get_physics_executor, recursive_physics,
    # Entropy dispatch (recursion's activation mechanism)
    EntropyDispatcher, PhysicsEntropyDispatcher, DispatchConditions,
    get_physics_dispatcher,
    # Bifractal tracing (recursion's analysis layer)
    BifractalTrace, TraceAnalysis,
    # Memory fields (recursion's state substrate)
    MemoryField, PhysicsMemoryField, FieldController,
    memory_field,
    # PAC self-regulation (recursion's conservation enforcement)
    PACRegulator, PACRecursiveContext, pac_recursive,
    get_global_pac_regulator, validate_pac_conservation,
    enable_pac_self_regulation, get_system_pac_metrics,
)

# Language constructs (recursion's control flow & decorators)
from .lang import (
    recursive, entropy_gate, tool_binding, tail_recursive,
    recurse, crystallize, branch, merge_contexts,
    Context, create_context,
)

# Version info
__version__ = "2.1.0"
__author__ = "Dawn Field Institute"
__email__ = "info@dawnfield.org"

# Alias for common Context usage
Context = ExecutionContext


# === Public API ===
# Only the recursive core is in __all__. Toolkit modules are accessed
# via sub-packages (fracton.field, fracton.physics, fracton.storage, fracton.core).

__all__ = [
    # Core recursive engine
    "RecursiveExecutor",
    "ExecutionContext",
    "PhysicsRecursiveExecutor",
    "get_physics_executor",
    "recursive_physics",

    # Entropy dispatch
    "EntropyDispatcher",
    "PhysicsEntropyDispatcher",
    "DispatchConditions",
    "get_physics_dispatcher",

    # Bifractal tracing
    "BifractalTrace",
    "TraceAnalysis",

    # Memory fields
    "MemoryField",
    "PhysicsMemoryField",
    "FieldController",
    "memory_field",

    # Decorators
    "recursive",
    "entropy_gate",
    "tool_binding",
    "tail_recursive",

    # Runtime functions
    "recurse",
    "crystallize",
    "branch",
    "merge_contexts",
    "Context",
    "create_context",

    # PAC self-regulation
    "PACRegulator",
    "PACRecursiveContext",
    "pac_recursive",
    "get_global_pac_regulator",
    "validate_pac_conservation",
    "enable_pac_self_regulation",
    "get_system_pac_metrics",

    # Convenience functions
    "initialize_field",
    "analyze_trace",
    "visualize_trace",
]


# === Convenience Functions ===

def initialize_field(capacity: int = 1000, entropy: float = 0.5,
                     field_id: str = None, **kwargs) -> MemoryField:
    """Initialize a memory field with specific parameters."""
    return MemoryField(capacity=capacity, entropy=entropy,
                       field_id=field_id, **kwargs)


def analyze_trace(trace: BifractalTrace) -> TraceAnalysis:
    """Analyze a bifractal trace for patterns and performance insights."""
    return trace.analyze_patterns()


def visualize_trace(trace: BifractalTrace, format: str = "text", **kwargs) -> str:
    """Generate visualization of a bifractal trace."""
    if format == "text":
        return trace.visualize_text(kwargs.get("max_depth", 50))
    elif format == "graph":
        return trace.visualize_graph()
    elif format == "timeline":
        return trace.get_entropy_timeline()
    else:
        raise ValueError(f"Unsupported visualization format: {format}")


# === Backward Compatibility ===
# These names moved to sub-modules in v2.1. Imports still work but emit
# a DeprecationWarning so consumers can migrate at their own pace.

_MOVED_ATTRS = {
    # Physics primitives → fracton.lang
    "physics_primitive": "fracton.lang",
    "conservation_primitive": "fracton.lang",
    "klein_gordon_evolution": "fracton.lang",
    "enforce_pac_conservation": "fracton.lang",
    "calculate_balance_operator": "fracton.lang",
    "field_pattern_matching": "fracton.lang",
    "resonance_field_interaction": "fracton.lang",
    "entropy_driven_collapse": "fracton.lang",
    "cognitive_pattern_extraction": "fracton.lang",
    "superfluid_memory_dynamics": "fracton.lang",
    # PAC-Lazy substrate → fracton.core
    "PACNode": "fracton.core",
    "PACNodeFactory": "fracton.core",
    "PACSystem": "fracton.core",
    "TieredCache": "fracton.core",
    # Storage → fracton.storage
    "KronosNode": "fracton.storage",
    "KronosEdge": "fracton.storage",
    "KronosGraph": "fracton.storage",
    "GeometricConfidence": "fracton.storage",
    "DocumentReference": "fracton.storage",
    "CrystallizationEvent": "fracton.storage",
    "FDOSerializer": "fracton.storage",
    "TemporalIndex": "fracton.storage",
    "EpisodeTracker": "fracton.storage",
    # Factory functions → fracton.core
    "physics_memory_field": "fracton.core",
    "create_physics_engine": "fracton.core",
}


def __getattr__(name):
    if name in _MOVED_ATTRS:
        target_module = _MOVED_ATTRS[name]
        warnings.warn(
            f"fracton.{name} has moved to {target_module}. "
            f"Update your import to: from {target_module} import {name}",
            DeprecationWarning,
            stacklevel=2,
        )
        module = importlib.import_module(target_module)
        return getattr(module, name)
    raise AttributeError(f"module 'fracton' has no attribute {name}")
