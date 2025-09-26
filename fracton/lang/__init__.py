"""
Fracton Language Package

Provides the high-level language constructs for the Fracton computational
modeling language, including decorators, primitives, context management,
and optional DSL compilation.
"""

from .decorators import recursive, entropy_gate, tool_binding, tail_recursive
from .primitives import (
    recurse, crystallize, branch, merge_contexts,
    physics_primitive, conservation_primitive,
    klein_gordon_evolution, enforce_pac_conservation,
    calculate_balance_operator, field_pattern_matching,
    resonance_field_interaction, entropy_driven_collapse,
    cognitive_pattern_extraction, superfluid_memory_dynamics
)
from .context import Context, create_context
from .compiler import compile_fracton_dsl

__version__ = "0.1.0"
__all__ = [
    "recursive",
    "entropy_gate", 
    "tool_binding",
    "tail_recursive",
    "recurse",
    "crystallize",
    "branch",
    "merge_contexts",
    "physics_primitive",
    "conservation_primitive",
    "klein_gordon_evolution",
    "enforce_pac_conservation",
    "calculate_balance_operator",
    "field_pattern_matching",
    "resonance_field_interaction",
    "entropy_driven_collapse",
    "cognitive_pattern_extraction", 
    "superfluid_memory_dynamics",
    "Context",
    "create_context",
    "compile_fracton_dsl"
]
