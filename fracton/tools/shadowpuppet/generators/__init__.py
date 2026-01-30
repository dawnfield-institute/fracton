"""
ShadowPuppet Code Generators

Pluggable code generation backends. Any LLM or code generator
can be used - just implement the CodeGenerator protocol.

Available generators:
- CopilotGenerator: GitHub Copilot CLI integration
- ClaudeGenerator: Anthropic Claude API
- MockGenerator: Template-based fallback (no AI required)
"""

from .base import CodeGenerator, GenerationContext
from .copilot import CopilotGenerator
from .claude import ClaudeGenerator
from .mock import MockGenerator

__all__ = [
    'CodeGenerator',
    'GenerationContext',
    'CopilotGenerator', 
    'ClaudeGenerator',
    'MockGenerator',
]
