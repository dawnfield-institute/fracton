"""
ShadowPuppet Claude Generator

Integrates with Anthropic's Claude API for code generation.
Supports both Claude API and Claude Code (via CLI).
"""

import os
import json
import subprocess
from typing import Optional

from .base import CodeGenerator, GenerationContext, GenerationError


class ClaudeGenerator(CodeGenerator):
    """
    Anthropic Claude API code generator.
    
    Uses the Anthropic Python SDK to generate code.
    Requires:
    - anthropic package installed (pip install anthropic)
    - ANTHROPIC_API_KEY environment variable set
    
    Example:
        generator = ClaudeGenerator(model="claude-sonnet-4-20250514")
        code = generator.generate(GenerationContext(protocol=my_protocol))
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        fallback_generator: Optional[CodeGenerator] = None
    ):
        """
        Initialize Claude generator.
        
        Args:
            model: Claude model to use
            api_key: API key (default: from ANTHROPIC_API_KEY env)
            max_tokens: Maximum tokens in response
            temperature: Generation temperature (lower = more deterministic)
            fallback_generator: Fallback if Claude fails
        """
        self.model = model
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fallback_generator = fallback_generator
        
        self._client = None
    
    @property
    def name(self) -> str:
        return f"claude-{self.model.split('-')[1]}"
    
    def _get_client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise GenerationError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
        return self._client
    
    def generate(self, context: GenerationContext) -> str:
        """Generate code using Claude API."""
        prompt = self.build_prompt(context)
        
        try:
            client = self._get_client()
            
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                system="You are an expert Python developer. Generate clean, well-documented code that follows best practices. Return only the code, no explanations."
            )
            
            response_text = message.content[0].text
            code = self.extract_code(response_text)
            
            if not code or len(code) < 20:
                raise GenerationError("Claude returned empty or minimal code")
            
            return code
            
        except Exception as e:
            if self.fallback_generator:
                print(f"[!] Claude failed ({e}), using fallback")
                return self.fallback_generator.generate(context)
            raise GenerationError(f"Claude generation failed: {e}")


class ClaudeCodeGenerator(CodeGenerator):
    """
    Claude Code CLI generator.
    
    Uses the Claude Code CLI tool for generation.
    Requires Claude Code to be installed and configured.
    """
    
    def __init__(
        self,
        cli_path: Optional[str] = None,
        timeout: int = 120,
        fallback_generator: Optional[CodeGenerator] = None
    ):
        """
        Initialize Claude Code generator.
        
        Args:
            cli_path: Path to Claude Code CLI (auto-detected if None)
            timeout: Generation timeout in seconds
            fallback_generator: Fallback if Claude Code fails
        """
        # Auto-detect CLI path based on OS
        if cli_path is None:
            import platform
            if platform.system() == 'Windows':
                # On Windows, use claude.cmd from npm
                self.cli_path = 'claude.cmd'
            else:
                self.cli_path = 'claude'
        else:
            self.cli_path = cli_path
        self.timeout = timeout
        self.fallback_generator = fallback_generator
    
    @property
    def name(self) -> str:
        return "claude-code"
    
    def generate(self, context: GenerationContext) -> str:
        """Generate code using Claude Code CLI."""
        prompt = self.build_prompt(context)
        
        try:
            # Claude Code CLI command - use shell=True on Windows for .cmd files
            import platform
            use_shell = platform.system() == 'Windows'
            
            result = subprocess.run(
                [self.cli_path, '--print', '-p', prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                shell=use_shell
            )
            
            if result.returncode != 0:
                raise GenerationError(f"Claude Code failed: {result.stderr}")
            
            code = self.extract_code(result.stdout)
            
            if not code or len(code) < 20:
                raise GenerationError("Claude Code returned empty or minimal code")
            
            return code
            
        except subprocess.TimeoutExpired:
            raise GenerationError(f"Claude Code timed out after {self.timeout}s")
            
        except FileNotFoundError:
            if self.fallback_generator:
                print(f"[!] Claude Code not found, using fallback")
                return self.fallback_generator.generate(context)
            raise GenerationError(
                f"Claude Code not found at '{self.cli_path}'. "
                "Install Claude Code CLI first."
            )
        
        except Exception as e:
            if self.fallback_generator:
                print(f"[!] Claude Code failed ({e}), using fallback")
                return self.fallback_generator.generate(context)
            raise GenerationError(f"Claude Code generation failed: {e}")
