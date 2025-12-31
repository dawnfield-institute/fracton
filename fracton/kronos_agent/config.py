"""
Configuration for Kronos Agent.

Loads API keys and settings from environment or grimm's .env.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load from grimm's .env if available
GRIMM_ENV = Path(__file__).parent.parent.parent / "grimm" / ".env"
if GRIMM_ENV.exists():
    load_dotenv(GRIMM_ENV)

# Also try local .env
LOCAL_ENV = Path(__file__).parent / ".env"
if LOCAL_ENV.exists():
    load_dotenv(LOCAL_ENV, override=True)


@dataclass
class AgentConfig:
    """Agent configuration."""
    
    # Anthropic
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    # Use claude-sonnet-4-20250514 - the valid model name
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("GRIM_MAX_TOKENS", "4096"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("GRIM_TEMPERATURE", "0.7"))
    )
    
    # Kronos storage
    storage_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("KRONOS_STORAGE_PATH", "./data/kronos")
        )
    )
    namespace: str = "kronos_agent"
    device: str = "cuda"  # GPU acceleration
    embedding_dim: int = 384
    
    # MCP Server
    mcp_port: int = 8765
    mcp_host: str = "localhost"
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return True


# Global config instance
config = AgentConfig()
