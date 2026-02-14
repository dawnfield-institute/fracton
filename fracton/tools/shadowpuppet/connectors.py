"""
ShadowPuppet Connectors

Explicit interface contracts between components.

The Problem:
Components generated independently can have interface mismatches:
- ComponentA calls `store.get(id)` 
- ComponentB was generated with `store.fetch(item_id)`

Connectors solve this by:
1. Extracting exact interfaces from provider components
2. Passing exact signatures to consumer generation
3. Validating consumer calls match provider interfaces
4. Failing fast on mismatches (not post-hoc)

Think of Connectors as the "wiring harness" between components.
Each connection point has a defined shape that both sides must match.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set


@dataclass
class MethodSignature:
    """Exact signature of a method."""
    name: str
    params: Dict[str, str]  # param_name -> type_annotation
    returns: str
    is_async: bool = False
    docstring: Optional[str] = None
    
    def to_call_pattern(self) -> str:
        """Generate the expected call pattern."""
        params_str = ", ".join(
            f"{name}: {typ}" for name, typ in self.params.items()
        )
        return f"{self.name}({params_str}) -> {self.returns}"
    
    def matches_call(self, call_name: str, arg_count: int) -> bool:
        """Check if a call matches this signature (loose match)."""
        if call_name != self.name:
            return False
        # Allow for default args
        required_params = len([p for p in self.params if '=' not in str(p)])
        return required_params <= arg_count <= len(self.params)


@dataclass 
class Connector:
    """
    Interface contract between a provider and its consumers.
    
    A Connector defines exactly what interface a provider exposes
    and which consumers are expected to use it.
    
    Example:
        connector = Connector(
            provider="TaskStore",
            interface=[
                MethodSignature("get", {"task_id": "int"}, "Optional[Task]"),
                MethodSignature("add", {"task": "Task"}, "Task"),
            ],
            consumers=["TaskManager", "TaskApp"]
        )
    """
    provider: str
    interface: List[MethodSignature]
    consumers: List[str] = field(default_factory=list)
    
    def get_method(self, name: str) -> Optional[MethodSignature]:
        """Get method signature by name."""
        for method in self.interface:
            if method.name == name:
                return method
        return None
    
    def to_prompt_context(self) -> str:
        """Generate prompt context showing available interface."""
        lines = [f"{self.provider} interface (use exactly these signatures):"]
        for method in self.interface:
            lines.append(f"  - {method.to_call_pattern()}")
        return "\n".join(lines)


class InterfaceExtractor:
    """
    Extracts actual interface from generated component code.
    
    Parses the AST to get exact method signatures, not just
    what we asked for in ProtocolSpec.
    """
    
    def extract(self, code: str, class_name: str) -> List[MethodSignature]:
        """
        Extract method signatures from generated code.
        
        Args:
            code: Generated Python code
            class_name: Name of class to extract from
            
        Returns:
            List of actual method signatures
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        
        signatures = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sig = self._extract_method(item)
                        if sig and not sig.name.startswith('_'):
                            signatures.append(sig)
        
        return signatures
    
    def _extract_method(self, node: ast.FunctionDef) -> Optional[MethodSignature]:
        """Extract signature from a method node."""
        name = node.name
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        # Extract parameters (skip self)
        params = {}
        for arg in node.args.args:
            if arg.arg == 'self':
                continue
            type_hint = self._get_annotation(arg.annotation)
            params[arg.arg] = type_hint
        
        # Handle default values - mark with '='
        defaults_offset = len(node.args.args) - len(node.args.defaults)
        for i, default in enumerate(node.args.defaults):
            param_idx = defaults_offset + i
            if param_idx < len(node.args.args):
                param_name = node.args.args[param_idx].arg
                if param_name in params:
                    params[param_name] = f"{params[param_name]} = ..."
        
        # Extract return type
        returns = self._get_annotation(node.returns)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        return MethodSignature(
            name=name,
            params=params,
            returns=returns,
            is_async=is_async,
            docstring=docstring
        )
    
    def _get_annotation(self, node: Optional[ast.expr]) -> str:
        """Convert AST annotation to string."""
        if node is None:
            return "Any"
        
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Subscript):
            # Handle generics like Optional[Task], List[str]
            if isinstance(node.value, ast.Name):
                base = node.value.id
                if isinstance(node.slice, ast.Name):
                    inner = node.slice.id
                elif isinstance(node.slice, ast.Tuple):
                    inner = ", ".join(self._get_annotation(e) for e in node.slice.elts)
                else:
                    inner = self._get_annotation(node.slice)
                return f"{base}[{inner}]"
        elif isinstance(node, ast.Attribute):
            return f"{self._get_annotation(node.value)}.{node.attr}"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union types: X | Y
            left = self._get_annotation(node.left)
            right = self._get_annotation(node.right)
            return f"{left} | {right}"
        
        # Fallback: try to unparse
        try:
            return ast.unparse(node)
        except:
            return "Any"


class CallValidator:
    """
    Validates that consumer code calls match provider interfaces.
    
    Parses consumer code to find calls to dependency objects,
    then checks those calls match the connector signatures.
    """
    
    def __init__(self):
        self.extractor = InterfaceExtractor()
    
    def validate(
        self,
        consumer_code: str,
        connectors: List[Connector]
    ) -> Tuple[bool, List[str]]:
        """
        Validate consumer code against connector interfaces.
        
        Args:
            consumer_code: Generated consumer code
            connectors: List of dependency connectors
            
        Returns:
            (is_valid, list_of_violations)
        """
        try:
            tree = ast.parse(consumer_code)
        except SyntaxError:
            return False, ["Syntax error in consumer code"]
        
        violations = []
        
        # Build a map of provider -> methods
        provider_methods = {}
        for connector in connectors:
            provider_methods[connector.provider.lower()] = {
                m.name: m for m in connector.interface
            }
        
        # Find all method calls on attributes (e.g., self.store.get(...))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                violation = self._check_call(node, provider_methods)
                if violation:
                    violations.append(violation)
        
        return len(violations) == 0, violations
    
    def _check_call(
        self,
        call: ast.Call,
        provider_methods: Dict[str, Dict[str, MethodSignature]]
    ) -> Optional[str]:
        """Check a single call against known interfaces."""
        
        # We're looking for calls like: self.store.get(...)
        if not isinstance(call.func, ast.Attribute):
            return None
        
        method_name = call.func.attr
        
        # Get the object being called on
        obj = call.func.value
        if isinstance(obj, ast.Attribute):
            # self.store.get -> provider might be "store"
            provider_attr = obj.attr.lower()
        elif isinstance(obj, ast.Name):
            # store.get -> provider is "store"
            provider_attr = obj.id.lower()
        else:
            return None
        
        # Check if this is a known provider
        for provider_name, methods in provider_methods.items():
            # Match by attribute name containing provider name
            # e.g., "store" matches "TaskStore" provider
            if provider_name in provider_attr or provider_attr in provider_name:
                if method_name in methods:
                    # Method exists, check args
                    sig = methods[method_name]
                    actual_args = len(call.args) + len(call.keywords)
                    if not sig.matches_call(method_name, actual_args):
                        return (
                            f"Call {method_name}() has {actual_args} args, "
                            f"but {provider_name}.{method_name} expects "
                            f"{len(sig.params)} params"
                        )
                else:
                    # Method doesn't exist on provider
                    available = list(methods.keys())
                    return (
                        f"Call to {provider_attr}.{method_name}() but "
                        f"{provider_name} only has: {available}"
                    )
        
        return None


class ConnectorRegistry:
    """
    Registry of all connectors in a system.
    
    Tracks which components provide what interfaces,
    and which components consume them.
    """
    
    def __init__(self):
        self.connectors: Dict[str, Connector] = {}  # provider_name -> Connector
        self.extractor = InterfaceExtractor()
        self.validator = CallValidator()
    
    def register_provider(
        self,
        provider_name: str,
        code: str,
        consumers: List[str] = None
    ) -> Connector:
        """
        Register a provider component and extract its interface.
        
        Args:
            provider_name: Name of the provider class
            code: Generated code for the provider
            consumers: List of consumer names
            
        Returns:
            The created Connector
        """
        interface = self.extractor.extract(code, provider_name)
        
        connector = Connector(
            provider=provider_name,
            interface=interface,
            consumers=consumers or []
        )
        
        self.connectors[provider_name] = connector
        return connector
    
    def get_connectors_for_consumer(self, consumer_name: str) -> List[Connector]:
        """Get all connectors where this component is a consumer."""
        return [
            c for c in self.connectors.values()
            if consumer_name in c.consumers
        ]
    
    def validate_consumer(
        self,
        consumer_name: str,
        code: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate consumer code against its dependency connectors.
        
        Args:
            consumer_name: Name of the consumer component
            code: Generated consumer code
            
        Returns:
            (is_valid, list_of_violations)
        """
        connectors = self.get_connectors_for_consumer(consumer_name)
        if not connectors:
            return True, []  # No dependencies to validate
        
        return self.validator.validate(code, connectors)
    
    def get_dependency_context(self, consumer_name: str) -> str:
        """
        Generate prompt context for a consumer's dependencies.
        
        Returns formatted interface descriptions for all
        dependencies the consumer should use.
        """
        connectors = self.get_connectors_for_consumer(consumer_name)
        if not connectors:
            return ""
        
        parts = ["DEPENDENCY INTERFACES (use exactly these signatures):"]
        parts.append("")
        
        for connector in connectors:
            parts.append(connector.to_prompt_context())
            parts.append("")
        
        parts.append("IMPORTANT: Your code MUST call these methods with the exact")
        parts.append("signatures shown above. Method names and parameter types must match.")
        parts.append("")
        
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry for checkpointing."""
        return {
            name: {
                'provider': c.provider,
                'consumers': c.consumers,
                'interface': [
                    {
                        'name': m.name,
                        'params': m.params,
                        'returns': m.returns,
                        'is_async': m.is_async
                    }
                    for m in c.interface
                ]
            }
            for name, c in self.connectors.items()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConnectorRegistry':
        """Deserialize registry from checkpoint."""
        registry = cls()
        for name, cdata in data.items():
            interface = [
                MethodSignature(
                    name=m['name'],
                    params=m['params'],
                    returns=m['returns'],
                    is_async=m.get('is_async', False)
                )
                for m in cdata['interface']
            ]
            registry.connectors[name] = Connector(
                provider=cdata['provider'],
                interface=interface,
                consumers=cdata['consumers']
            )
        return registry


def build_connectors_from_gaps(gaps: List['GrowthGap']) -> Dict[str, List[str]]:
    """
    Build a provider->consumers map from gap dependencies.
    
    Args:
        gaps: List of GrowthGaps with protocol dependencies
        
    Returns:
        Dict mapping provider name to list of consumer names
    """
    from .protocols import GrowthGap
    
    consumers_map: Dict[str, List[str]] = {}
    
    for gap in gaps:
        consumer = gap.protocol.name
        for dep_name in gap.protocol.dependencies:
            if dep_name not in consumers_map:
                consumers_map[dep_name] = []
            if consumer not in consumers_map[dep_name]:
                consumers_map[dep_name].append(consumer)
    
    return consumers_map
