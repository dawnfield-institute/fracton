"""
Agent State Definitions for LangGraph.

Defines the state schema that flows through the agent graph.
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import operator


class AgentPhase(str, Enum):
    """Current phase of the agent."""
    IDLE = "idle"
    SCANNING = "scanning"
    ANALYZING = "analyzing"
    UPDATING = "updating"
    RESPONDING = "responding"
    ERROR = "error"


class MessageRole(str, Enum):
    """Message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    """A message in the conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None


class AgentState(TypedDict):
    """
    State that flows through the LangGraph agent.
    
    Annotated fields with operator.add are accumulated across steps.
    """
    
    # Conversation
    messages: Annotated[List[Dict[str, Any]], operator.add]
    
    # Current phase
    phase: AgentPhase
    
    # Current task
    current_task: Optional[str]
    
    # Repo context
    repo_path: Optional[str]
    repo_scanned: bool
    
    # Knowledge graph state
    graphs_available: List[str]
    active_graph: Optional[str]
    nodes_created: int
    
    # Query context
    last_query: Optional[str]
    last_results: List[Dict[str, Any]]
    
    # Error tracking
    error: Optional[str]
    
    # Tool results accumulator
    tool_outputs: Annotated[List[Dict[str, Any]], operator.add]


def create_initial_state() -> AgentState:
    """Create initial agent state."""
    return AgentState(
        messages=[],
        phase=AgentPhase.IDLE,
        current_task=None,
        repo_path=None,
        repo_scanned=False,
        graphs_available=[],
        active_graph=None,
        nodes_created=0,
        last_query=None,
        last_results=[],
        error=None,
        tool_outputs=[],
    )
