"""
Recursive Engine - Core execution engine with Native PAC Self-Regulation

Enhanced recursive execution engine that natively implements
Potential-Actualization Conservation as foundational self-regulation.
Every recursive operation automatically maintains f(parent) = Σf(children).
"""

import time
import uuid
import threading
from collections import deque
from enum import Enum
from typing import Union, Callable
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from contextlib import contextmanager

# Import PAC regulation for native integration
from .pac_regulation import (
    PACRegulator, PACRecursiveContext, pac_recursive,
    get_global_pac_regulator, validate_pac_conservation
)
import threading
from collections import deque


class TrampolineResult(Enum):
    """Enumeration for trampoline execution results."""
    CONTINUE = "continue"
    COMPLETE = "complete"


@dataclass
class Continuation:
    """Represents a continuation in the trampoline execution."""
    func: Callable
    memory: Any
    context: 'ExecutionContext'
    result_type: TrampolineResult = TrampolineResult.CONTINUE


@dataclass
class ExecutionContext:
    """
    Execution context that carries metadata through recursive calls.
    
    Attributes:
        entropy: Current entropy level (0.0 - 1.0)
        depth: Current recursion depth
        trace_id: Unique identifier for this execution trace
        field_state: Field-specific metadata
        parent_context: Reference to the calling context
        metadata: Additional context metadata
    """
    
    def __init__(self, entropy: float = 0.5, depth: int = 0, 
                 trace_id: str = None, field_state: Dict[str, Any] = None,
                 parent_context: 'ExecutionContext' = None, 
                 metadata: Dict[str, Any] = None, experiment: str = None):
        import time
        self.entropy = entropy
        self.depth = depth
        self.trace_id = trace_id or str(uuid.uuid4())
        self.field_state = field_state or {}
        self.parent_context = parent_context
        self.metadata = metadata or {}
        self.experiment = experiment or self.metadata.get('experiment', 'default')
        self.timestamp = time.time()  # Add timestamp for test compatibility
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from metadata (test-compatible interface)."""
        return self.metadata.get(key, default)
    
    def with_data(self, data: Dict[str, Any]) -> 'ExecutionContext':
        """Create a new context with additional data."""
        new_metadata = self.metadata.copy()
        new_metadata.update(data)
        
        return ExecutionContext(
            entropy=self.entropy,
            depth=self.depth,
            trace_id=self.trace_id,
            field_state=self.field_state.copy(),
            parent_context=self.parent_context,
            metadata=new_metadata,
            experiment=self.experiment
        )
    
    def deeper(self, steps: int = 1) -> 'ExecutionContext':
        """Create a new context with increased depth."""
        return ExecutionContext(
            entropy=self.entropy,
            depth=self.depth + steps,
            trace_id=self.trace_id,
            field_state=self.field_state.copy(),
            parent_context=self,
            metadata=self.metadata.copy(),
            experiment=self.experiment
        )
    
    def with_entropy(self, entropy: float) -> 'ExecutionContext':
        """Create a new context with modified entropy."""
        new_context = ExecutionContext(
            entropy=entropy,
            depth=self.depth,
            trace_id=self.trace_id,
            field_state=self.field_state.copy(),
            parent_context=self.parent_context,
            metadata=self.metadata.copy(),
            experiment=self.experiment
        )
        return new_context
    
    def with_metadata(self, **kwargs) -> 'ExecutionContext':
        """Create a new context with additional metadata."""
        new_metadata = self.metadata.copy()
        new_metadata.update(kwargs)
        
        return ExecutionContext(
            entropy=self.entropy,
            depth=self.depth,
            trace_id=self.trace_id,
            field_state=self.field_state.copy(),
            parent_context=self.parent_context,
            metadata=new_metadata,
            experiment=self.experiment
        )


@dataclass
class CallFrame:
    """Represents a single call in the recursive call stack."""
    function: Callable
    context: ExecutionContext
    timestamp: float
    trace_id: str
    

@dataclass
class ExecutionStats:
    """Statistics about recursive execution performance."""
    total_calls: int = 0
    max_depth: int = 0
    total_time: float = 0.0
    entropy_distribution: Dict[str, int] = field(default_factory=dict)
    function_call_counts: Dict[str, int] = field(default_factory=dict)


class StackOverflowError(Exception):
    """Raised when recursion depth exceeds safe limits."""
    
    def __init__(self, depth: int, max_depth: int):
        self.depth = depth
        self.max_depth = max_depth
        super().__init__(f"Stack overflow at depth {depth} (max: {max_depth})")


class EntropyGateError(Exception):
    """Raised when function execution violates entropy gate conditions."""
    
    def __init__(self, current_entropy: float, required_range: tuple):
        self.current_entropy = current_entropy
        self.required_range = required_range
        super().__init__(
            f"Entropy {current_entropy} outside required range {required_range}"
        )


class CallStack:
    """Manages the recursive call stack with entropy awareness."""
    
    def __init__(self, max_depth: int = 1000):
        self.max_depth = max_depth
        self._stack: deque = deque()
        self._lock = threading.Lock()
    
    def push(self, func: Callable, context: ExecutionContext, trace_id: str) -> None:
        """Push a new call frame onto the stack."""
        with self._lock:
            if len(self._stack) >= self.max_depth:
                raise StackOverflowError(len(self._stack), self.max_depth)
            
            frame = CallFrame(
                function=func,
                context=context,
                timestamp=time.time(),
                trace_id=trace_id
            )
            self._stack.append(frame)
    
    def pop(self) -> Optional[CallFrame]:
        """Pop the top call frame from the stack."""
        with self._lock:
            if self._stack:
                return self._stack.pop()
            return None
    
    def current_depth(self) -> int:
        """Get the current stack depth."""
        with self._lock:
            return len(self._stack)
    
    def check_overflow(self) -> bool:
        """Check if the stack is approaching overflow."""
        return self.current_depth() >= self.max_depth * 0.9
    
    def get_trace(self) -> List[CallFrame]:
        """Get a copy of the current call stack."""
        with self._lock:
            return list(self._stack)


class RecursiveExecutor:
    """
    Main execution engine for recursive function calls in Fracton.
    
    Natively implements PAC self-regulation: f(parent) = Σf(children)
    for all recursive operations. Handles entropy gating, stack management,
    tail recursion optimization, and conservation validation.
    """
    
    def __init__(self, max_depth: int = 1000, enable_tail_optimization: bool = True, 
                 entropy_regulation: bool = True, pac_regulation: bool = True):
        self.max_depth = max_depth
        self.enable_tail_optimization = enable_tail_optimization
        self.entropy_regulation = entropy_regulation
        self.pac_regulation = pac_regulation
        self.call_stack = CallStack(max_depth)
        self.stats = ExecutionStats()
        self._entropy_gates: Dict[Callable, tuple] = {}
        self._tail_recursive_functions: set = set()
        self._execution_lock = threading.Lock()
        
        # Native PAC regulation
        self.pac_regulator = get_global_pac_regulator() if pac_regulation else None
    
    def register_entropy_gate(self, func: Callable, min_threshold: float, 
                            max_threshold: float = 1.0) -> None:
        """Register entropy gate conditions for a function."""
        self._entropy_gates[func] = (min_threshold, max_threshold)
    
    def register_tail_recursive(self, func: Callable) -> None:
        """Register a function as tail-recursive for optimization."""
        self._tail_recursive_functions.add(func)
    
    def _check_entropy_gate(self, func: Callable, context: ExecutionContext) -> None:
        """Check if function execution meets entropy gate requirements."""
        if func not in self._entropy_gates:
            return
        
        min_threshold, max_threshold = self._entropy_gates[func]
        
        if not (min_threshold <= context.entropy <= max_threshold):
            raise EntropyGateError(
                context.entropy, 
                (min_threshold, max_threshold)
            )
    
    def _update_stats(self, func: Callable, context: ExecutionContext, 
                     execution_time: float) -> None:
        """Update execution statistics."""
        self.stats.total_calls += 1
        self.stats.max_depth = max(self.stats.max_depth, context.depth)
        self.stats.total_time += execution_time
        
        # Update entropy distribution
        entropy_bucket = f"{int(context.entropy * 10) / 10:.1f}"
        self.stats.entropy_distribution[entropy_bucket] = (
            self.stats.entropy_distribution.get(entropy_bucket, 0) + 1
        )
        
        # Update function call counts
        func_name = getattr(func, '__name__', str(func))
        self.stats.function_call_counts[func_name] = (
            self.stats.function_call_counts.get(func_name, 0) + 1
        )
    
    def get_current_depth(self) -> int:
        """Get current recursion depth."""
        return self.call_stack.current_depth()
    
    def get_stack_size(self) -> int:
        """Get current stack size (alias for get_current_depth for test compatibility)."""
        return self.get_current_depth()
    
    def recursive(self, func: Callable) -> Callable:
        """Decorator to mark a function as recursive (test compatibility)."""
        # For test compatibility - just return the function
        # In practice, the @fracton.recursive decorator should be used
        return func
    
    def _is_tail_recursive_call(self, func: Callable) -> bool:
        """Check if this is a tail-recursive call that can be optimized."""
        return (
            self.enable_tail_optimization and 
            func in self._tail_recursive_functions and
            self.call_stack.current_depth() > 0
        )
    
    def execute(self, func: Callable, memory: Any, context: Union[ExecutionContext, Dict]) -> Any:
        """
        Execute a recursive function with PAC self-regulation and context management.
        
        Automatically validates f(parent) = Σf(children) for all recursive operations.
        
        Args:
            func: The function to execute
            memory: Shared memory field
            context: Execution context (ExecutionContext object or dict)
            
        Returns:
            The result of function execution
            
        Raises:
            StackOverflowError: If recursion depth exceeds limits
            EntropyGateError: If entropy conditions are not met
        """
        # Convert dict context to ExecutionContext if needed
        if isinstance(context, dict):
            exec_context = ExecutionContext(
                entropy=context.get('entropy', 0.5),
                depth=context.get('depth', 0),
                metadata=context
            )
        else:
            exec_context = context
            
        start_time = time.time()
        
        # PAC regulation context
        parent_value = memory if hasattr(memory, '__len__') else getattr(memory, 'total_value', memory)
        operation_name = f"{func.__name__}_depth_{exec_context.depth}"
        
        try:
            with self._execution_lock:
                # Check entropy gate conditions
                self._check_entropy_gate(func, exec_context)
                
                # Check for stack overflow
                if exec_context.depth >= self.max_depth:
                    raise StackOverflowError(exec_context.depth, self.max_depth)
                
                # Execute with PAC regulation if enabled
                if self.pac_regulation and self.pac_regulator:
                    with PACRecursiveContext(self.pac_regulator, operation_name, parent_value) as pac_context:
                        result = self._execute_with_pac_validation(func, memory, exec_context, pac_context)
                        return result
                else:
                    # Handle tail recursion optimization
                    if self._is_tail_recursive_call(func):
                        # Use trampoline for tail recursion
                        return self._execute_trampoline(func, memory, exec_context)
                    else:
                        # Push new frame and execute with trampoline
                        self.call_stack.push(func, exec_context, exec_context.trace_id)
                        try:
                            return self._execute_trampoline(func, memory, exec_context)
                        finally:
                            self.call_stack.pop()
                
        except Exception as e:
            # Add execution context to exception
            if hasattr(e, 'execution_context'):
                e.execution_context = exec_context
            raise
            
    def _execute_with_pac_validation(self, func: Callable, memory: Any, 
                                   exec_context: ExecutionContext, 
                                   pac_context: PACRecursiveContext) -> Any:
        """
        Execute function with PAC conservation validation.
        
        Ensures f(parent) = Σf(children) throughout recursive execution.
        """
        # Handle tail recursion optimization
        if self._is_tail_recursive_call(func):
            # Use trampoline for tail recursion
            result = self._execute_trampoline(func, memory, exec_context)
        else:
            # Push new frame and execute with trampoline
            self.call_stack.push(func, exec_context, exec_context.trace_id)
            try:
                result = self._execute_trampoline(func, memory, exec_context)
            finally:
                self.call_stack.pop()
        
        # Register result with PAC context for validation
        if isinstance(result, (list, tuple)):
            for child in result:
                pac_context.add_child_result(child)
        elif isinstance(result, dict) and 'children' in result:
            for child in result['children']:
                pac_context.add_child_result(child)
        else:
            pac_context.add_child_result(result)
            
        return result
    
    def _execute_trampoline(self, func: Callable, memory: Any, context: 'ExecutionContext') -> Any:
        """
        Execute a function using trampoline-based recursion management.
        
        This prevents deep recursion by converting recursive calls into
        an iterative loop with continuations.
        """
        import threading
        start_time = time.time()
        continuation_queue = deque([Continuation(func, memory, context)])
        result = None
        
        # Mark that we're in trampoline execution
        current_thread = threading.current_thread()
        old_value = getattr(current_thread, '_fracton_in_trampoline', False)
        current_thread._fracton_in_trampoline = True
        
        try:
            while continuation_queue:
                current = continuation_queue.popleft()
                
                try:
                    # Execute the function
                    temp_result = current.func(current.memory, current.context)
                    
                    # Check if result is a continuation (recursive call)
                    if isinstance(temp_result, Continuation):
                        # Add to queue for further processing
                        continuation_queue.append(temp_result)
                    else:
                        # Final result
                        result = temp_result
                        
                except Exception as e:
                    # Add execution context to exception
                    if hasattr(e, 'execution_context'):
                        e.execution_context = current.context
                    raise
        finally:
            # Restore previous thread state
            current_thread._fracton_in_trampoline = old_value
        
        # Update statistics
        execution_time = time.time() - start_time
        self._update_stats(func, context, execution_time)
        
        return result
    
    def get_execution_stats(self) -> ExecutionStats:
        """Get current execution statistics."""
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.stats = ExecutionStats()
    
    @contextmanager
    def execution_context(self, max_depth: int = None):
        """Context manager for temporary execution settings."""
        original_max_depth = self.max_depth
        if max_depth is not None:
            self.max_depth = max_depth
        
        try:
            yield self
        finally:
            self.max_depth = original_max_depth


# Global default executor instance
_default_executor = RecursiveExecutor()


def get_default_executor() -> RecursiveExecutor:
    """Get the default global executor instance."""
    return _default_executor


def set_default_executor(executor: RecursiveExecutor) -> None:
    """Set a new default global executor instance."""
    global _default_executor
    _default_executor = executor


class PhysicsRecursiveExecutor(RecursiveExecutor):
    """
    Physics-aware recursive executor for GAIA integration.
    
    Extends the base RecursiveExecutor with physics state preservation,
    conservation enforcement, and Klein-Gordon field evolution throughout
    recursive execution.
    """
    
    def __init__(self, max_depth: int = 100, xi_target: float = 1.0571,
                 conservation_strictness: float = 1e-12, pac_regulation: bool = True):
        super().__init__(max_depth, pac_regulation=pac_regulation)
        self.xi_target = xi_target
        self.conservation_strictness = conservation_strictness
        self._physics_dispatcher = None
    
    def set_physics_dispatcher(self, dispatcher):
        """Set the physics entropy dispatcher."""
        self._physics_dispatcher = dispatcher
    
    def execute_with_physics(self, func: Callable, memory, context: ExecutionContext,
                           physics_mode: str = "standard") -> Any:
        """
        Execute function with physics state preservation.
        
        Args:
            func: Function to execute recursively
            memory: Physics memory field
            context: Execution context
            physics_mode: Physics execution mode
            
        Returns:
            Function result with physics state maintained
        """
        # Validate physics state before execution
        if hasattr(memory, 'get_physics_metrics'):
            initial_physics = memory.get_physics_metrics()
            initial_conservation = initial_physics.get('conservation_residual', 0.0)
        else:
            initial_physics = None
            initial_conservation = 0.0
        
        # Execute with physics monitoring
        try:
            if physics_mode == "conservative":
                result = self._execute_conservative_physics(func, memory, context)
            elif physics_mode == "exploratory":
                result = self._execute_exploratory_physics(func, memory, context)
            else:
                result = self.execute(func, memory, context)
            
            # Verify physics state after execution
            if hasattr(memory, 'get_physics_metrics'):
                final_physics = memory.get_physics_metrics()
                final_conservation = final_physics.get('conservation_residual', 0.0)
                
                # Enforce conservation if violated
                if abs(final_conservation) > self.conservation_strictness:
                    if hasattr(memory, 'enforce_pac_conservation'):
                        memory.enforce_pac_conservation(self.conservation_strictness)
            
            return result
            
        except Exception as e:
            # Attempt physics state recovery
            if hasattr(memory, 'enforce_pac_conservation'):
                try:
                    memory.enforce_pac_conservation(self.conservation_strictness)
                except:
                    pass  # Recovery failed, but continue
            raise e
    
    def _execute_conservative_physics(self, func: Callable, memory, context: ExecutionContext) -> Any:
        """Execute with strict physics conservation."""
        
        # Pre-execution conservation check
        if hasattr(memory, 'enforce_pac_conservation'):
            memory.enforce_pac_conservation(self.conservation_strictness * 0.1)  # Stricter
        
        # Execute with reduced entropy for stability
        conservative_context = ExecutionContext(
            entropy=min(0.3, context.entropy),  # Cap entropy for stability
            depth=context.depth,
            trace_id=context.trace_id,
            field_state=context.field_state.copy() if context.field_state else None,
            parent_context=context,
            metadata={**context.metadata, 'physics_mode': 'conservative'}
        )
        
        return super().execute(func, memory, conservative_context)
    
    def _execute_exploratory_physics(self, func: Callable, memory, context: ExecutionContext) -> Any:
        """Execute with relaxed constraints for exploration."""
        
        # Allow higher entropy for exploration
        exploratory_context = ExecutionContext(
            entropy=min(0.9, context.entropy * 1.5),  # Boost entropy
            depth=context.depth,
            trace_id=context.trace_id,
            field_state=context.field_state.copy() if context.field_state else None,
            parent_context=context,
            metadata={**context.metadata, 'physics_mode': 'exploratory'}
        )
        
        # Execute with relaxed conservation
        result = super().execute(func, memory, exploratory_context)
        
        # Post-execution stabilization
        if hasattr(memory, 'enforce_pac_conservation'):
            memory.enforce_pac_conservation(self.conservation_strictness * 5)  # Relaxed cleanup
        
        return result
    
    def create_physics_context(self, entropy: float = 0.5, physics_state: Dict = None,
                             conservation_mode: str = "standard") -> ExecutionContext:
        """Create physics-aware execution context."""
        
        field_state = {
            'physics_state': physics_state or {},
            'xi_target': self.xi_target,
            'conservation_strictness': self.conservation_strictness,
            'conservation_mode': conservation_mode
        }
        
        metadata = {
            'physics_enabled': True,
            'conservation_mode': conservation_mode,
            'xi_target': self.xi_target
        }
        
        return ExecutionContext(
            entropy=entropy,
            depth=0,
            trace_id=str(uuid.uuid4()),
            field_state=field_state,
            metadata=metadata
        )
    
    def execute_klein_gordon_recursion(self, func: Callable, memory, 
                                     initial_context: ExecutionContext,
                                     evolution_steps: int = 10,
                                     dt: float = 0.01) -> Any:
        """
        Execute recursive function with Klein-Gordon field evolution.
        
        Integrates Klein-Gordon dynamics into the recursive execution,
        evolving the field state at each recursion level.
        """
        results = []
        current_context = initial_context
        
        for step in range(evolution_steps):
            # Evolve field before recursion step
            if hasattr(memory, 'evolve_klein_gordon'):
                memory.evolve_klein_gordon(dt)
            
            # Create step context
            step_context = ExecutionContext(
                entropy=current_context.entropy,
                depth=step,
                trace_id=current_context.trace_id,
                field_state=current_context.field_state,
                parent_context=current_context,
                metadata={
                    **current_context.metadata,
                    'klein_gordon_step': step,
                    'dt': dt,
                    'evolution_mode': 'integrated'
                }
            )
            
            # Execute recursive step
            try:
                step_result = self.execute_with_physics(func, memory, step_context, 
                                                      physics_mode="standard")
                results.append(step_result)
                
                # Update context for next step
                current_context = step_context
                
            except RecursionError:
                # Recursion limit reached, return accumulated results
                break
        
        return results
    
    def get_physics_stats(self) -> Dict[str, Any]:
        """Get physics execution statistics."""
        base_stats = self.get_stats()
        
        physics_stats = {
            'xi_target': self.xi_target,
            'conservation_strictness': self.conservation_strictness,
            'physics_mode_enabled': True,
            'base_execution_stats': base_stats
        }
        
        return physics_stats


# Physics recursive function decorator
def recursive_physics(conservation_mode: str = "standard", xi_target: float = 1.0571):
    """
    Decorator for physics-aware recursive functions.
    
    Args:
        conservation_mode: "conservative", "standard", or "exploratory"
        xi_target: Target balance operator value
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Get physics executor
            physics_executor = PhysicsRecursiveExecutor(xi_target=xi_target)
            
            # Extract memory and context from args
            if len(args) >= 2:
                memory, context = args[0], args[1]
                
                # Execute with physics awareness
                return physics_executor.execute_with_physics(
                    func, memory, context, physics_mode=conservation_mode
                )
            else:
                # Fallback to standard execution
                return func(*args, **kwargs)
        
        wrapper._is_physics_recursive = True
        wrapper._conservation_mode = conservation_mode
        wrapper._xi_target = xi_target
        return wrapper
    
    return decorator


# Global physics executor instance  
_physics_executor = PhysicsRecursiveExecutor()


def get_physics_executor() -> PhysicsRecursiveExecutor:
    """Get the default global physics executor instance."""
    return _physics_executor


def set_physics_executor(executor: PhysicsRecursiveExecutor) -> None:
    """Set a new default global physics executor instance."""
    global _physics_executor
    _physics_executor = executor
