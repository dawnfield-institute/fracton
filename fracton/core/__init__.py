"""
Fracton Core Package

The core package provides the fundamental runtime engine for the Fracton
computational modeling language, including recursive execution, entropy
dispatch, bifractal tracing, memory field management, and native PAC
self-regulation ensuring f(parent) = Σf(children) conservation.

Enhanced with physics capabilities for GAIA integration including PAC
conservation, Klein-Gordon field evolution, and physics-aware recursion.
"""

from .recursive_engine import (
    RecursiveExecutor, ExecutionContext, get_default_executor,
    PhysicsRecursiveExecutor, get_physics_executor, recursive_physics
)
from .entropy_dispatch import (
    EntropyDispatcher, DispatchConditions, get_default_dispatcher,
    PhysicsEntropyDispatcher, get_physics_dispatcher
)
from .bifractal_trace import BifractalTrace, TraceAnalysis
from .memory_field import (
    MemoryField, FieldController, get_default_controller,
    PhysicsMemoryField, physics_memory_field
)
from .pac_regulation import (
    PACRegulator, PACRecursiveContext, pac_recursive,
    get_global_pac_regulator, validate_pac_conservation,
    enable_pac_self_regulation, get_system_pac_metrics
)

__version__ = "0.1.0"
__all__ = [
    # Base recursive engine
    "RecursiveExecutor",
    "ExecutionContext", 
    "get_default_executor",
    
    # Physics recursive engine
    "PhysicsRecursiveExecutor",
    "get_physics_executor", 
    "recursive_physics",
    
    # Base entropy dispatch
    "EntropyDispatcher",
    "DispatchConditions",
    "get_default_dispatcher",
    
    # Physics entropy dispatch
    "PhysicsEntropyDispatcher",
    "get_physics_dispatcher",
    
    # Bifractal tracing
    "BifractalTrace",
    "TraceAnalysis",
    
    # Base memory fields
    "MemoryField",
    "FieldController",
    "get_default_controller",
    
    # Physics memory fields
    "PhysicsMemoryField",
    "physics_memory_field",
    
    # Native PAC self-regulation
    "PACRegulator",
    "PACRecursiveContext", 
    "pac_recursive",
    "get_global_pac_regulator",
    "validate_pac_conservation",
    "enable_pac_self_regulation",
    "get_system_pac_metrics"
]
