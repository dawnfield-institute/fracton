"""
ShadowPuppet Base Generator Protocol

Defines the interface that all code generators must implement.
This is the "puppeteer" abstraction - any system that can
generate code from a protocol specification.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..protocols import ProtocolSpec, ComponentOrganism


@dataclass
class GenerationContext:
    """
    Context provided to generators for code generation.
    
    Attributes:
        protocol: What to generate
        parent: Parent component for template guidance
        siblings: Related components for context
        mutation_rate: How much to vary from parent (0.0-1.0)
        pac_invariants: Conservation laws to preserve
        extra_instructions: Additional generation instructions
    """
    protocol: ProtocolSpec
    parent: Optional[ComponentOrganism] = None
    siblings: List[ComponentOrganism] = None
    mutation_rate: float = 0.1
    pac_invariants: List[str] = None
    extra_instructions: str = ""
    
    def __post_init__(self):
        if self.siblings is None:
            self.siblings = []
        if self.pac_invariants is None:
            self.pac_invariants = []


class CodeGenerator(ABC):
    """
    Abstract base class for code generators.
    
    Any code generation system (Copilot, Claude, local models, etc.)
    can be used by implementing this interface.
    
    The key insight: template-guided generation uses parent code
    as context, allowing generated code to inherit structure
    while introducing controlled variation.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Generator name for logging/tracking."""
        ...
    
    @abstractmethod
    def generate(self, context: GenerationContext) -> str:
        """
        Generate code implementing the protocol.
        
        Args:
            context: Generation context with protocol, parent, etc.
            
        Returns:
            Generated Python code as string
            
        Raises:
            GenerationError: If generation fails
        """
        ...
    
    def build_prompt(self, context: GenerationContext) -> str:
        """
        Build the generation prompt from context.
        
        This is a shared implementation that generators can override
        or use directly. Implements template-guided prompting.
        """
        parts = []
        
        # Protocol specification
        parts.append("Implement this Python class:\n")
        parts.append(context.protocol.to_prompt_context())
        parts.append("")
        
        # PAC invariants
        if context.pac_invariants:
            parts.append("PAC INVARIANTS (must be preserved):")
            for inv in context.pac_invariants:
                parts.append(f"  - {inv}")
            parts.append("")
        
        # Template guidance from parent
        if context.parent:
            parts.append("TEMPLATE CONTEXT (use as structural guide):")
            parts.append(f"Parent component: {context.parent.protocol_name}")
            parts.append(f"Parent code structure:\n```python\n{context.parent.code[:800]}\n```")
            parts.append(f"Generate similar structure but for {context.protocol.name}")
            parts.append("")
            
            # Mutation directive
            mutation_directive = self._get_mutation_directive(context.mutation_rate)
            if mutation_directive:
                parts.append(mutation_directive)
                parts.append("")
        
        # Sibling context
        if context.siblings:
            parts.append("SIBLING COMPONENTS (must integrate with):")
            for sibling in context.siblings[:2]:  # Limit context size
                parts.append(f"- {sibling.protocol_name}:")
                parts.append(f"```python\n{sibling.code[:300]}...\n```")
            parts.append("")
        
        # Extra instructions
        if context.extra_instructions:
            parts.append(f"ADDITIONAL INSTRUCTIONS:\n{context.extra_instructions}")
            parts.append("")
        
        # Final instruction
        parts.append("Return ONLY the complete implementation in Python.")
        parts.append("Include proper docstrings, type hints, and error handling.")
        
        return '\n'.join(parts)
    
    def _get_mutation_directive(self, mutation_rate: float) -> Optional[str]:
        """Get mutation directive based on rate."""
        import random
        
        if mutation_rate < 0.1:
            return None
        
        elif mutation_rate < 0.3:
            mutations = [
                "VARIATION: Use different variable names but keep the same algorithm",
                "VARIATION: Add more detailed docstrings and comments",
                "VARIATION: Reorder the methods but keep same implementations"
            ]
        
        elif mutation_rate < 0.5:
            mutations = [
                "VARIATION: Use a different algorithm (iterative vs recursive, loop vs comprehension)",
                "VARIATION: Use different data structures (dict instead of list, set instead of array)",
                "VARIATION: Add comprehensive error handling with try/except blocks",
                "VARIATION: Optimize for performance (reduce loops, use caching)"
            ]
        
        else:
            mutations = [
                "VARIATION: Completely different implementation approach",
                "VARIATION: Minimize complexity - make it as simple as possible",
                "VARIATION: Maximize robustness - add validation and edge case handling",
                "VARIATION: Use different design patterns (functional vs OOP)"
            ]
        
        return random.choice(mutations)
    
    def extract_code(self, response: str) -> str:
        """Extract Python code from response."""
        # Try to find code blocks
        if '```python' in response:
            code = response.split('```python')[1].split('```')[0]
        elif '```' in response:
            code = response.split('```')[1].split('```')[0]
        else:
            code = response
        
        return code.strip()


class GenerationError(Exception):
    """Raised when code generation fails."""
    pass
