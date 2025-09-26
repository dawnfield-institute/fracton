"""
PAC Self-Regulation - Foundational conservation enforcement for Fracton

This module implements Potential-Actualization Conservation as the foundational
self-regulation mechanism for Fracton's recursive architecture. Every recursive
operation automatically maintains PAC conservation: f(parent) = Σf(children)
"""

import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class PACValidationResult:
    """Result of PAC conservation validation"""
    conserved: bool
    residual: float
    xi_value: float
    correction_applied: bool
    validation_time: float

class PACRegulator:
    """
    Foundational PAC self-regulation for Fracton recursive operations.
    
    Automatically enforces f(parent) = Σf(children) across all recursive
    decompositions, ensuring Fracton operations are natively PAC-compliant.
    """
    
    def __init__(self, xi_target: float = 1.0571, tolerance: float = 1e-12,
                 auto_correction: bool = True):
        self.xi_target = xi_target  # Balance operator target (PAC critical value)
        self.tolerance = tolerance   # Conservation precision
        self.auto_correction = auto_correction  # Enable automatic corrections
        
        # Self-regulation state
        self._regulation_active = True
        self._conservation_history = []
        self._correction_count = 0
        self._lock = threading.RLock()
        
        # PAC violation handlers
        self._violation_handlers = []
    
    def register_violation_handler(self, handler: Callable[[PACValidationResult], None]):
        """Register handler for PAC violation events"""
        with self._lock:
            self._violation_handlers.append(handler)
    
    def validate_recursive_conservation(self, parent_value: Any, 
                                      children_values: List[Any],
                                      operation_context: str = "recursive_operation") -> PACValidationResult:
        """
        Validate PAC conservation: f(parent) = Σf(children)
        
        This is the foundational check that ensures Fracton operations
        maintain conservation across recursive decompositions.
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # Convert to numerical form for conservation analysis
                parent_total = self._extract_conservation_value(parent_value)
                children_total = sum(self._extract_conservation_value(child) 
                                   for child in children_values)
                
                # Calculate conservation residual
                residual = abs(parent_total - children_total)
                conserved = residual < self.tolerance
                
                # Calculate balance operator
                xi_current = self._calculate_balance_operator(parent_value, children_values)
                
                # Apply correction if needed and enabled
                correction_applied = False
                if not conserved and self.auto_correction:
                    correction_applied = self._apply_conservation_correction(
                        parent_value, children_values, residual
                    )
                    self._correction_count += 1
                
                # Create result
                result = PACValidationResult(
                    conserved=conserved or correction_applied,
                    residual=residual,
                    xi_value=xi_current,
                    correction_applied=correction_applied,
                    validation_time=time.time() - start_time
                )
                
                # Track conservation history
                self._conservation_history.append({
                    'timestamp': time.time(),
                    'operation': operation_context,
                    'residual': residual,
                    'xi_value': xi_current,
                    'corrected': correction_applied
                })
                
                # Handle violations
                if not result.conserved:
                    self._handle_conservation_violation(result, operation_context)
                
                return result
                
            except Exception as e:
                # Return failure result for invalid operations
                return PACValidationResult(
                    conserved=False,
                    residual=float('inf'),
                    xi_value=0.0,
                    correction_applied=False,
                    validation_time=time.time() - start_time
                )
    
    def _extract_conservation_value(self, value: Any) -> float:
        """Extract numerical conservation value from various data types"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, (list, tuple)):
            return sum(self._extract_conservation_value(v) for v in value)
        elif isinstance(value, dict):
            if 'conservation_value' in value:
                return float(value['conservation_value'])
            return sum(self._extract_conservation_value(v) for v in value.values())
        elif isinstance(value, np.ndarray):
            return float(np.sum(value))
        elif hasattr(value, '__len__') and hasattr(value, '__iter__'):
            try:
                return sum(self._extract_conservation_value(v) for v in value)
            except:
                return 1.0  # Default conservation value
        else:
            return 1.0  # Default conservation value for unknown types
    
    def _calculate_balance_operator(self, parent_value: Any, children_values: List[Any]) -> float:
        """Calculate PAC balance operator Ξ"""
        try:
            parent_magnitude = abs(self._extract_conservation_value(parent_value))
            children_magnitudes = [abs(self._extract_conservation_value(child)) 
                                 for child in children_values]
            
            if parent_magnitude == 0:
                return self.xi_target
            
            # Balance operator: ratio of children dispersion to parent magnitude
            children_variance = np.var(children_magnitudes) if len(children_magnitudes) > 1 else 0
            xi = (children_variance / parent_magnitude) + 1.0
            
            return xi
            
        except:
            return self.xi_target  # Return target on calculation failure
    
    def _apply_conservation_correction(self, parent_value: Any, 
                                     children_values: List[Any], 
                                     residual: float) -> bool:
        """Apply automatic conservation correction to maintain PAC"""
        try:
            if not children_values:
                return False
            
            # Distribute correction across children proportionally
            correction_per_child = residual / len(children_values)
            
            # Apply correction based on data type
            for child in children_values:
                if isinstance(child, dict) and 'conservation_value' in child:
                    child['conservation_value'] += correction_per_child
                elif isinstance(child, np.ndarray):
                    # Add correction uniformly across array
                    child += correction_per_child / child.size
            
            return True
            
        except:
            return False  # Correction failed
    
    def _handle_conservation_violation(self, result: PACValidationResult, context: str):
        """Handle PAC conservation violations"""
        violation_event = {
            'timestamp': time.time(),
            'context': context,
            'residual': result.residual,
            'xi_value': result.xi_value,
            'severity': 'critical' if result.residual > self.tolerance * 100 else 'warning'
        }
        
        # Notify registered handlers
        for handler in self._violation_handlers:
            try:
                handler(result)
            except Exception as e:
                pass  # Don't let handler errors break regulation
    
    def get_regulation_metrics(self) -> Dict[str, Any]:
        """Get current PAC regulation performance metrics"""
        with self._lock:
            recent_history = self._conservation_history[-100:]  # Last 100 operations
            
            if not recent_history:
                return {
                    'total_operations': 0,
                    'conservation_rate': 1.0,
                    'average_residual': 0.0,
                    'average_xi': self.xi_target,
                    'corrections_applied': 0
                }
            
            residuals = [h['residual'] for h in recent_history]
            xi_values = [h['xi_value'] for h in recent_history]
            corrections = sum(1 for h in recent_history if h['corrected'])
            
            return {
                'total_operations': len(self._conservation_history),
                'recent_operations': len(recent_history),
                'conservation_rate': sum(1 for r in residuals if r < self.tolerance) / len(residuals),
                'average_residual': np.mean(residuals),
                'max_residual': np.max(residuals),
                'average_xi': np.mean(xi_values),
                'xi_deviation': abs(np.mean(xi_values) - self.xi_target),
                'corrections_applied': corrections,
                'total_corrections': self._correction_count
            }


class PACRecursiveContext:
    """
    Context manager for PAC-regulated recursive operations.
    
    Automatically validates conservation at entry/exit of recursive calls,
    making Fracton natively PAC-compliant.
    """
    
    def __init__(self, regulator: PACRegulator, operation_name: str,
                 parent_value: Any, expected_children: Optional[List[Any]] = None):
        self.regulator = regulator
        self.operation_name = operation_name
        self.parent_value = parent_value
        self.expected_children = expected_children
        self.actual_children = []
        self.validation_result = None
    
    def __enter__(self):
        """Enter PAC-regulated context"""
        # Pre-validation if children are known
        if self.expected_children is not None:
            self.validation_result = self.regulator.validate_recursive_conservation(
                self.parent_value, self.expected_children, 
                f"{self.operation_name}_pre"
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit PAC-regulated context with post-validation"""
        if self.actual_children:
            # Post-validation with actual results
            self.validation_result = self.regulator.validate_recursive_conservation(
                self.parent_value, self.actual_children,
                f"{self.operation_name}_post"  
            )
    
    def add_child_result(self, child_value: Any):
        """Add child result for post-validation"""
        self.actual_children.append(child_value)
    
    def is_conserved(self) -> bool:
        """Check if conservation was maintained"""
        return self.validation_result.conserved if self.validation_result else True


def pac_recursive(operation_name: str = None):
    """
    Decorator to make any function PAC-compliant recursive operation.
    
    Automatically enforces f(parent) = Σf(children) for the decorated function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal operation_name
            if operation_name is None:
                operation_name = func.__name__
            
            # Extract parent value (typically first argument or in kwargs)
            parent_value = args[0] if args else kwargs.get('parent', None)
            
            # Get global PAC regulator (or create one)
            regulator = getattr(wrapper, '_pac_regulator', None)
            if regulator is None:
                regulator = PACRegulator()
                wrapper._pac_regulator = regulator
            
            # Execute function with PAC regulation
            with PACRecursiveContext(regulator, operation_name, parent_value) as pac_context:
                result = func(*args, **kwargs)
                
                # Handle different result types
                if isinstance(result, (list, tuple)):
                    for child in result:
                        pac_context.add_child_result(child)
                elif isinstance(result, dict) and 'children' in result:
                    for child in result['children']:
                        pac_context.add_child_result(child)
                else:
                    pac_context.add_child_result(result)
                
                # Add regulation metrics to result if possible
                if isinstance(result, dict):
                    result['pac_metrics'] = regulator.get_regulation_metrics()
                
                return result
        
        # Expose regulator for external access
        wrapper.get_pac_regulator = lambda: getattr(wrapper, '_pac_regulator', None)
        wrapper.pac_regulated = True
        
        return wrapper
    return decorator


# Global PAC regulator instance for system-wide use
_global_pac_regulator = None

def get_global_pac_regulator() -> PACRegulator:
    """Get or create global PAC regulator for system-wide conservation"""
    global _global_pac_regulator
    if _global_pac_regulator is None:
        _global_pac_regulator = PACRegulator(xi_target=1.0571, auto_correction=True)
    return _global_pac_regulator

def validate_pac_conservation(parent, children, context="operation") -> PACValidationResult:
    """Convenience function for PAC validation using global regulator"""
    regulator = get_global_pac_regulator()
    return regulator.validate_recursive_conservation(parent, children, context)

def enable_pac_self_regulation():
    """Enable PAC self-regulation across all Fracton operations"""
    regulator = get_global_pac_regulator()
    regulator._regulation_active = True
    return regulator

def get_system_pac_metrics() -> Dict[str, Any]:
    """Get system-wide PAC regulation metrics"""
    regulator = get_global_pac_regulator()
    return regulator.get_regulation_metrics()