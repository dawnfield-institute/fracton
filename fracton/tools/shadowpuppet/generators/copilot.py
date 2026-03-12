"""
ShadowPuppet GitHub Copilot Generator

Integrates with GitHub Copilot CLI for code generation.
Requires the Copilot CLI to be installed and authenticated.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional

from .base import CodeGenerator, GenerationContext, GenerationError


class CopilotGenerator(CodeGenerator):
    """
    GitHub Copilot CLI code generator.
    
    Uses the Copilot CLI to generate code from prompts.
    Requires:
    - GitHub Copilot subscription
    - Copilot CLI installed (npm install -g @githubnext/github-copilot-cli)
    - Authentication configured
    
    Example:
        generator = CopilotGenerator()
        code = generator.generate(GenerationContext(protocol=my_protocol))
    """
    
    def __init__(
        self,
        cli_path: Optional[str] = None,
        timeout: int = 60,
        fallback_generator: Optional[CodeGenerator] = None
    ):
        """
        Initialize Copilot generator.
        
        Args:
            cli_path: Path to Copilot CLI (default: find in PATH)
            timeout: Generation timeout in seconds
            fallback_generator: Fallback if Copilot fails
        """
        self.cli_path = cli_path or self._find_copilot_cli()
        self.timeout = timeout
        self.fallback_generator = fallback_generator
    
    @property
    def name(self) -> str:
        return "copilot"
    
    def _find_copilot_cli(self) -> str:
        """Find Copilot CLI in PATH."""
        # Try common names
        for name in ['github-copilot-cli', 'copilot', 'gh copilot']:
            path = shutil.which(name)
            if path:
                return path
        
        # Default to 'copilot' and let it fail if not found
        return 'copilot'
    
    def generate(self, context: GenerationContext) -> str:
        """Generate code using Copilot CLI."""
        prompt = self.build_prompt(context)
        
        try:
            result = subprocess.run(
                [self.cli_path, '--prompt', prompt, '--allow-all'],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                raise GenerationError(f"Copilot CLI failed: {result.stderr}")
            
            code = self.extract_code(result.stdout)
            
            if not code or len(code) < 20:
                raise GenerationError("Copilot returned empty or minimal code")
            
            return code
            
        except subprocess.TimeoutExpired:
            raise GenerationError(f"Copilot generation timed out after {self.timeout}s")
            
        except FileNotFoundError:
            if self.fallback_generator:
                print(f"[!] Copilot CLI not found, using fallback")
                return self.fallback_generator.generate(context)
            raise GenerationError(
                f"Copilot CLI not found at '{self.cli_path}'. "
                "Install with: npm install -g @githubnext/github-copilot-cli"
            )
        
        except Exception as e:
            if self.fallback_generator:
                print(f"[!] Copilot failed ({e}), using fallback")
                return self.fallback_generator.generate(context)
            raise GenerationError(f"Copilot generation failed: {e}")


class CopilotChatGenerator(CodeGenerator):
    """
    GitHub Copilot Chat integration (VS Code extension API).
    
    For use within VS Code with the Copilot Chat extension.
    This is a placeholder - actual integration would use the
    VS Code extension API.
    """
    
    def __init__(self):
        self._fallback = None
    
    @property
    def name(self) -> str:
        return "copilot-chat"
    
    def generate(self, context: GenerationContext) -> str:
        """
        Generate code via Copilot Chat.
        
        Note: This requires VS Code extension integration.
        For standalone use, prefer CopilotGenerator (CLI).
        """
        # This would integrate with VS Code Copilot Chat API
        # For now, raise informative error
        raise GenerationError(
            "CopilotChatGenerator requires VS Code extension integration. "
            "Use CopilotGenerator for CLI-based generation."
        )
