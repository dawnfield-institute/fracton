"""
ShadowPuppet Protocol Definitions

Core data structures for architecture-as-code:
- ProtocolSpec: Defines what to generate (the puppet structure)
- GrowthGap: Represents a missing component in the system
- ComponentOrganism: A generated code component (the shadow)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable


@dataclass
class ProtocolSpec:
    """
    Architecture-as-code protocol specification.
    
    Defines the contract that generated code must fulfill.
    Think of this as the puppet's skeleton - the structure
    that constrains and enables movement.
    
    Attributes:
        name: Protocol/class name to generate
        methods: Required method names
        docstring: Description and purpose
        type_signature: Optional type hints as string
        pac_invariants: Conservation laws that must hold
        attributes: Required class attributes
        dependencies: Other protocols this depends on
    
    Example:
        ProtocolSpec(
            name="APIRouter",
            methods=["get", "post", "put", "delete"],
            docstring="REST API router with CRUD operations",
            pac_invariants=[
                "All routes return JSON responses",
                "Errors use standard HTTP status codes"
            ],
            attributes=["routes", "middleware"],
            dependencies=["RequestHandler", "ResponseSerializer"]
        )
    """
    name: str
    methods: List[str]
    docstring: str
    type_signature: str = ""
    pac_invariants: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_prompt_context(self) -> str:
        """Convert to prompt-friendly format for generators."""
        lines = [
            f"Protocol: {self.name}",
            f"Description: {self.docstring}",
            f"Methods: {', '.join(self.methods)}",
        ]
        
        if self.attributes:
            lines.append(f"Attributes: {', '.join(self.attributes)}")
        
        if self.type_signature:
            lines.append(f"Type Signature:\n{self.type_signature}")
            
        if self.pac_invariants:
            lines.append("PAC Invariants (must be preserved):")
            for inv in self.pac_invariants:
                lines.append(f"  - {inv}")
                
        if self.dependencies:
            lines.append(f"Dependencies: {', '.join(self.dependencies)}")
            
        return '\n'.join(lines)


@dataclass
class TestSuite:
    """
    Test functions for evaluating generated components.
    
    Attributes:
        unit: Isolated component tests
        integration: Component interaction tests
        e2e: Full system tests
    """
    unit: List[Callable] = field(default_factory=list)
    integration: List[Callable] = field(default_factory=list)
    e2e: List[Callable] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List[Callable]]:
        """Convert to dictionary format for evaluator."""
        result = {}
        if self.unit:
            result['unit'] = self.unit
        if self.integration:
            result['integration'] = self.integration
        if self.e2e:
            result['e2e'] = self.e2e
        return result


@dataclass
class GrowthGap:
    """
    A gap in the system that needs to be filled.
    
    Like an energy field attracting new components,
    a GrowthGap represents missing functionality that
    evolution will attempt to fill.
    
    Attributes:
        protocol: The specification to implement
        parent_components: Existing components for template context
        required_coherence: Minimum coherence to survive
        priority: Higher priority gaps are filled first
        test_suite: Tests to validate generated code
        context: Additional context for generation
    """
    protocol: ProtocolSpec
    parent_components: List['ComponentOrganism'] = field(default_factory=list)
    required_coherence: float = 0.70
    priority: float = 1.0
    test_suite: Optional[TestSuite] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def get_test_dict(self) -> Dict[str, List[Callable]]:
        """Get test suite as dictionary."""
        if self.test_suite:
            return self.test_suite.to_dict()
        return {}


@dataclass
class ComponentOrganism:
    """
    A software component as a digital organism.
    
    The "shadow" cast by the puppet - the actual generated
    code that implements a protocol.
    
    Attributes:
        id: Unique identifier
        protocol_name: Which protocol this implements
        code: Generated implementation
        
        # Fitness metrics
        coherence_score: Overall fitness (0.0-1.0)
        structural_score: Type/interface correctness
        semantic_score: Logical correctness
        energetic_score: Efficiency/simplicity
        
        # Genealogy
        parent_id: Parent component ID (if any)
        generation: Which generation born in
        age: How many generations survived
        derivation_path: Full lineage
        
        # Resources
        integration_energy: Accumulated "energy" from integration
        reuse_count: How many times used as template
    """
    id: str
    protocol_name: str
    code: str
    
    # Fitness metrics
    coherence_score: float = 0.0
    structural_score: float = 0.0
    semantic_score: float = 0.0
    energetic_score: float = 0.0
    
    # Genealogy
    parent_id: Optional[str] = None
    generation: int = 0
    age: int = 0
    derivation_path: List[str] = field(default_factory=list)
    confluence_pattern: Dict[str, float] = field(default_factory=dict)
    
    # Resources
    integration_energy: float = 100.0
    reuse_count: int = 0
    
    # Metadata
    generator_used: str = "unknown"
    generation_time: float = 0.0
    
    def __len__(self) -> int:
        """Length is code length."""
        return len(self.code)
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"{self.id} (gen {self.generation}): "
            f"coherence={self.coherence_score:.3f} "
            f"[S:{self.structural_score:.2f} "
            f"Se:{self.semantic_score:.2f} "
            f"E:{self.energetic_score:.2f}]"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'protocol_name': self.protocol_name,
            'code': self.code,
            'coherence_score': self.coherence_score,
            'structural_score': self.structural_score,
            'semantic_score': self.semantic_score,
            'energetic_score': self.energetic_score,
            'parent_id': self.parent_id,
            'generation': self.generation,
            'age': self.age,
            'derivation_path': self.derivation_path,
            'confluence_pattern': self.confluence_pattern,
            'integration_energy': self.integration_energy,
            'reuse_count': self.reuse_count,
            'generator_used': self.generator_used,
            'generation_time': self.generation_time,
        }
