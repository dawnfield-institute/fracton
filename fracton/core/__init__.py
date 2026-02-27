"""
Fracton Core Package

The core package provides the fundamental runtime engine for the Fracton
computational modeling language, including recursive execution, entropy
dispatch, bifractal tracing, memory field management, and native PAC
self-regulation ensuring f(parent) = Σf(children) conservation.

Enhanced with physics capabilities for GAIA integration including PAC
conservation, Klein-Gordon field evolution, and physics-aware recursion.

v2.0: Added PACNode and PACSystem for PAC-Lazy substrate architecture.
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
    PhysicsMemoryField, physics_memory_field, memory_field
)
from .pac_regulation import (
    PACRegulator, PACRecursiveContext, pac_recursive,
    get_global_pac_regulator, validate_pac_conservation,
    enable_pac_self_regulation, get_system_pac_metrics
)
from .pac_node import PACNode, PACNodeFactory
from .pac_system import PACSystem, TieredCache
from .mobius_tensor import (
    MobiusMatrix, MobiusFrame, MobiusStripTensor,
    MobiusFibonacciTensor, MobiusRecursiveTensor,
    cross_ratio, create_fibonacci_mobius, verify_4pi_periodicity,
    PHI, PHI_INV,
    # Neural network components (v2.2)
    MobiusNeuron, MobiusLayer, MobiusRecursiveLayer, MobiusNetwork
)
from .feigenbaum_mobius import (
    # Constants (v2.3)
    DELTA_FEIGENBAUM, ALPHA_FEIGENBAUM, UNIVERSAL_DELTA_Z,
    R_INF_LOGISTIC, R_INF_SINE,
    # Classes
    FibonacciMobius, M10, F as fibonacci,
    # Functions
    compute_delta_self_consistent, compute_universal_delta_z,
    prove_eigenvalue_identity, get_constants_summary,
)

def create_physics_engine(xi_target: float = 1.0571,
                         conservation_strictness: float = 1e-12,
                         field_dimensions: tuple = (32,),
                         enable_pac_regulation: bool = True) -> dict:
    """
    Create a complete physics engine with native PAC self-regulation.

    Returns a configured physics system with recursive executor,
    entropy dispatcher, and physics memory field ready for
    Klein-Gordon evolution and automatic PAC conservation.
    """
    physics_executor = PhysicsRecursiveExecutor(
        xi_target=xi_target,
        conservation_strictness=conservation_strictness,
        pac_regulation=enable_pac_regulation
    )

    physics_dispatcher = PhysicsEntropyDispatcher(
        xi_target=xi_target,
        conservation_strictness=conservation_strictness
    )

    pac_regulator = None
    if enable_pac_regulation:
        pac_regulator = enable_pac_self_regulation()

    physics_executor.set_physics_dispatcher(physics_dispatcher)

    return {
        'executor': physics_executor,
        'dispatcher': physics_dispatcher,
        'memory_field': physics_memory_field(
            capacity=2000,
            entropy=0.5,
            physics_dimensions=field_dimensions,
            conservation_strictness=conservation_strictness,
            xi_target=xi_target
        ),
        'pac_regulator': pac_regulator,
        'xi_target': xi_target,
        'conservation_strictness': conservation_strictness,
        'field_dimensions': field_dimensions,
        'pac_enabled': enable_pac_regulation,
    }


__version__ = "2.3.0"
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
    "get_system_pac_metrics",
    
    # PAC-Lazy substrate (v2.0)
    "PACNode",
    "PACNodeFactory",
    "PACSystem",
    "TieredCache",
    
    # Möbius tensor architecture (v2.1)
    "MobiusMatrix",
    "MobiusFrame",
    "MobiusStripTensor",
    "MobiusFibonacciTensor",
    "MobiusRecursiveTensor",
    "cross_ratio",
    "create_fibonacci_mobius",
    "verify_4pi_periodicity",
    "PHI",
    "PHI_INV",
    
    # Möbius neural network (v2.2)
    "MobiusNeuron",
    "MobiusLayer",
    "MobiusRecursiveLayer",
    "MobiusNetwork",
    
    # Feigenbaum-Möbius structure (v2.3)
    "DELTA_FEIGENBAUM",
    "ALPHA_FEIGENBAUM",
    "UNIVERSAL_DELTA_Z",
    "R_INF_LOGISTIC",
    "R_INF_SINE",
    "FibonacciMobius",
    "M10",
    "fibonacci",
    "compute_delta_self_consistent",
    "compute_universal_delta_z",
    "prove_eigenvalue_identity",
    "get_constants_summary",

    # Factory functions
    "create_physics_engine",
    "memory_field",
]
