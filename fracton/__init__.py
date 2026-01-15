"""
Fracton: Infodynamics Computational Modeling Language

Fracton is a domain-specific computational modeling language designed for 
infodynamics research and recursive field-aware systems. It provides a 
unified substrate for modeling emergent intelligence, entropy dynamics, 
and bifractal computation patterns.

v2.0: PAC-Lazy Substrate Architecture
- Delta-only storage with PACNode
- Tiered caching with PACSystem
- Physics constants from Dawn Field Theory
- Spherical encoding and Klein-Gordon evolution

Main Features:
- Recursive execution as first-class primitive
- Entropy-driven function dispatch
- Bifractal tracing for operation analysis
- Field-aware memory management
- Tool expression framework
- PAC-Lazy substrate for GAIA integration

Example Usage:
    import fracton
    from fracton.physics import PHI, XI
    from fracton.core import PACSystem
    from fracton.field import spherical_encode, evolve
    
    # Create PAC-Lazy substrate
    system = PACSystem(device='cuda')
    
    # Inject a pattern
    field = spherical_encode(token_id=42, dim=64, device='cuda')
    node_id = system.inject(field)
    
    # Evolve the field
    evolved = evolve(field, steps=5)
    
    # Find resonant patterns
    similar = system.find_resonant(evolved, top_k=5)
"""

# Core runtime components with native PAC self-regulation
from .core import (
    RecursiveExecutor, ExecutionContext, 
    EntropyDispatcher, DispatchConditions,
    BifractalTrace, TraceAnalysis,
    MemoryField, FieldController,
    memory_field,
    # Native PAC regulation
    PACRegulator, PACRecursiveContext, pac_recursive,
    get_global_pac_regulator, validate_pac_conservation,
    enable_pac_self_regulation, get_system_pac_metrics,
    # Physics capabilities
    PhysicsRecursiveExecutor, get_physics_executor, recursive_physics,
    PhysicsEntropyDispatcher, get_physics_dispatcher,
    PhysicsMemoryField, physics_memory_field,
    # PAC-Lazy substrate (v2.0)
    PACNode, PACNodeFactory, PACSystem, TieredCache
)

# Language constructs
from .lang import (
    recursive, entropy_gate, tool_binding, tail_recursive,
    recurse, crystallize, branch, merge_contexts,
    Context, create_context,
    # Physics primitives
    physics_primitive, conservation_primitive,
    klein_gordon_evolution, enforce_pac_conservation,
    calculate_balance_operator, field_pattern_matching,
    resonance_field_interaction, entropy_driven_collapse,
    cognitive_pattern_extraction, superfluid_memory_dynamics
)

# Storage module (Kronos v2)
from .storage import (
    KronosNode,
    KronosEdge,
    KronosGraph,
    GeometricConfidence,
    DocumentReference,
    CrystallizationEvent,
    FDOSerializer,
    TemporalIndex,
    EpisodeTracker
)

# Version info
__version__ = "2.1.0"
__author__ = "Dawn Field Institute"
__email__ = "info@dawnfield.org"

# Main API exports
__all__ = [
    # Core execution
    "RecursiveExecutor",
    "ExecutionContext", 
    "Context",
    "create_context",
    
    # Physics execution
    "PhysicsRecursiveExecutor",
    "get_physics_executor",
    
    # PAC-Lazy substrate (v2.0)
    "PACNode",
    "PACNodeFactory",
    "PACSystem",
    "TieredCache",
    
    # Decorators
    "recursive",
    "recursive_physics",
    "entropy_gate",
    "tool_binding", 
    "tail_recursive",
    "physics_primitive",
    "conservation_primitive",
    
    # Runtime functions
    "recurse",
    "crystallize",
    "branch",
    "merge_contexts",
    
    # Physics primitives
    "klein_gordon_evolution",
    "enforce_pac_conservation", 
    "calculate_balance_operator",
    "field_pattern_matching",
    "resonance_field_interaction",
    "entropy_driven_collapse",
    "cognitive_pattern_extraction",
    "superfluid_memory_dynamics",
    
    # Memory management
    "MemoryField",
    "PhysicsMemoryField",
    "FieldController",
    "memory_field",
    "physics_memory_field",
    
    # Dispatch and analysis
    "EntropyDispatcher",
    "PhysicsEntropyDispatcher",
    "get_physics_dispatcher",
    "DispatchConditions",
    "BifractalTrace",
    "TraceAnalysis",
    
    # Native PAC self-regulation
    "PACRegulator",
    "PACRecursiveContext", 
    "pac_recursive",
    "get_global_pac_regulator",
    "validate_pac_conservation",
    "enable_pac_self_regulation",
    "get_system_pac_metrics",
    
    # Storage (Kronos)
    "KronosBackend",
    "FDOSerializer",
    "TemporalIndex",
    "EpisodeTracker",
    
    # Utilities
    "initialize_field",
    "analyze_trace",
    "visualize_trace",
    "express_tool",
    "create_physics_engine"
]


def initialize_field(capacity: int = 1000, entropy: float = 0.5) -> MemoryField:
    """
    Initialize a new memory field with specified capacity and entropy.
    
    Args:
        capacity: Maximum number of items in the field
        entropy: Initial entropy level (0.0 - 1.0)
        
    Returns:
        New MemoryField instance
        
    Example:
        field = fracton.initialize_field(capacity=2000, entropy=0.7)
        field.set("data", my_data)
    """
    from .core.memory_field import get_default_controller
    controller = get_default_controller()
    return controller.create_field(capacity, entropy)


def analyze_trace(trace: BifractalTrace) -> TraceAnalysis:
    """
    Analyze a bifractal trace for patterns and performance insights.
    
    Args:
        trace: BifractalTrace to analyze
        
    Returns:
        TraceAnalysis with comprehensive insights
        
    Example:
        analysis = fracton.analyze_trace(my_trace)
        print(f"Max depth: {analysis.max_depth}")
        print(f"Total calls: {analysis.total_calls}")
    """
    return trace.analyze_patterns()


def visualize_trace(trace: BifractalTrace, format: str = "text", **kwargs) -> str:
    """
    Generate visualization of a bifractal trace.
    
    Args:
        trace: BifractalTrace to visualize
        format: Visualization format ("text", "graph", "timeline")
        **kwargs: Additional visualization options
        
    Returns:
        Visualization string or data structure
        
    Example:
        text_viz = fracton.visualize_trace(my_trace, "text", max_depth=20)
        print(text_viz)
    """
    if format == "text":
        return trace.visualize_text(kwargs.get("max_depth", 50))
    elif format == "graph":
        return trace.visualize_graph()
    elif format == "timeline":
        return trace.get_entropy_timeline()
    else:
        raise ValueError(f"Unsupported visualization format: {format}")


def express_tool(tool_name: str, operation: str = None, *args, **kwargs):
    """
    Express (access) an external tool through the Fracton tool framework.
    
    This function provides context-aware access to external systems based
    on the current execution context and entropy levels.
    
    Args:
        tool_name: Name of the tool to access
        operation: Specific operation to perform
        *args: Operation arguments
        **kwargs: Operation keyword arguments
        
    Returns:
        Result of tool operation
        
    Example:
        # Basic tool expression
        result = fracton.express_tool("github", "create_issue", 
                                    title="Bug report", 
                                    body="Description of the bug")
        
        # Context-sensitive expression
        if context.entropy > 0.8:
            data = fracton.express_tool("database", "experimental_query", query)
        else:
            data = fracton.express_tool("database", "stable_query", query)
    """
    # Tool expression will be implemented in the tools package
    # For now, provide a placeholder that can be extended
    from .tools import get_tool_registry
    
    registry = get_tool_registry()
    tool = registry.get_tool(tool_name)
    
    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found in registry")
    
    if operation:
        return tool.execute(operation, *args, **kwargs)
    else:
        return tool.execute(*args, **kwargs)


# Alias for common Context usage
Context = ExecutionContext


def set_global_config(**kwargs):
    """
    Set global configuration options for Fracton runtime.
    
    Available options:
        max_recursion_depth: Global maximum recursion depth
        enable_tracing: Enable/disable bifractal tracing globally
        default_entropy: Default entropy level for new contexts
        cache_size: Size of various internal caches
        
    Example:
        fracton.set_global_config(
            max_recursion_depth=2000,
            enable_tracing=True,
            default_entropy=0.6
        )
    """
    from .core.recursive_engine import get_default_executor
    from .core.entropy_dispatch import get_default_dispatcher
    
    executor = get_default_executor()
    dispatcher = get_default_dispatcher()
    
    if 'max_recursion_depth' in kwargs:
        executor.max_depth = kwargs['max_recursion_depth']
    
    if 'enable_tracing' in kwargs:
        # This would need to be implemented in the executor
        pass
    
    # Add other configuration options as needed


# Utility functions for tests and convenience
def memory_field(capacity: int = 1000, entropy: float = 0.5, **kwargs) -> MemoryField:
    """Create a new memory field instance."""
    return MemoryField(capacity=capacity, entropy=entropy, **kwargs)

def initialize_field(capacity: int = 1000, entropy: float = 0.5, 
                    field_id: str = None, **kwargs) -> MemoryField:
    """Initialize a memory field with specific parameters."""
    return MemoryField(capacity=capacity, entropy=entropy, 
                      field_id=field_id, **kwargs)

def get_runtime_stats() -> dict:
    """
    Get runtime statistics for the Fracton system.
    
    Returns:
        Dictionary with execution statistics, memory usage, and performance metrics
        
    Example:
        stats = fracton.get_runtime_stats()
        print(f"Total recursive calls: {stats['total_calls']}")
        print(f"Average entropy: {stats['avg_entropy']}")
    """
    from .core.recursive_engine import get_default_executor
    from .core.entropy_dispatch import get_default_dispatcher
    from .core.memory_field import get_default_controller
    
    executor = get_default_executor()
    dispatcher = get_default_dispatcher()
    controller = get_default_controller()
    
    exec_stats = executor.get_execution_stats()
    dispatch_stats = dispatcher.get_dispatch_stats()
    
    return {
        'execution': exec_stats.__dict__ if hasattr(exec_stats, '__dict__') else {},
        'dispatch': dispatch_stats,
        'memory_fields': len(controller.get_all_fields()),
        'version': __version__
    }


def create_physics_engine(xi_target: float = 1.0571, 
                         conservation_strictness: float = 1e-12,
                         field_dimensions: tuple = (32,),
                         enable_pac_regulation: bool = True) -> dict:
    """
    Create a complete physics engine with native PAC self-regulation.
    
    Returns a configured physics system with recursive executor,
    entropy dispatcher, and physics memory field ready for
    Klein-Gordon evolution and automatic PAC conservation.
    
    Args:
        xi_target: Target balance operator value (default: 1.0571)
        conservation_strictness: PAC conservation tolerance
        field_dimensions: Physics field dimensions
        enable_pac_regulation: Enable automatic PAC validation
        
    Returns:
        Dictionary containing configured physics components
        
    Example:
        engine = fracton.create_physics_engine()
        with engine['memory_field'] as physics_memory:
            result = engine['executor'].execute_with_physics(
                my_physics_function, physics_memory, context
            )
    """
    # Create physics components with PAC regulation
    physics_executor = PhysicsRecursiveExecutor(
        xi_target=xi_target,
        conservation_strictness=conservation_strictness,
        pac_regulation=enable_pac_regulation
    )
    
    physics_dispatcher = PhysicsEntropyDispatcher(
        xi_target=xi_target,
        conservation_strictness=conservation_strictness
    )
    
    # Enable global PAC regulation if requested
    pac_regulator = None
    if enable_pac_regulation:
        pac_regulator = enable_pac_self_regulation()
    
    # Link executor and dispatcher
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
        'version': __version__
    }
