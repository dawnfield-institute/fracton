"""
LangGraph Agent with Kronos Tools.

Uses Anthropic Claude with tool calling to manage knowledge graphs.
"""

# Suppress TensorFlow/oneDNN noise before imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import asyncio
import json
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path

from anthropic import Anthropic

from .config import config
from .state import AgentState, AgentPhase, create_initial_state
from .tools import KronosTools, TOOL_DEFINITIONS
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class KronosAgent:
    """
    LangGraph-style agent using Anthropic Claude with Kronos tools.
    
    Implements a simple agentic loop:
    1. Receive user message
    2. Call Claude with tools
    3. Execute tool calls
    4. Return results to Claude
    5. Repeat until done
    """
    
    def __init__(
        self,
        storage_path: Path = None,
        namespace: str = "agent",
    ):
        # Validate config
        config.validate()
        
        # Anthropic client
        self.client = Anthropic(api_key=config.anthropic_api_key)
        self.model = config.model
        self.max_tokens = config.max_tokens
        
        # Kronos tools
        self.tools = KronosTools(
            storage_path=storage_path or config.storage_path,
            namespace=namespace,
            device=config.device,
            embedding_dim=config.embedding_dim,
        )
        
        # State
        self.state = create_initial_state()
        self.conversation_history: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize agent (connect to Kronos)."""
        await self.tools.connect()
        logger.info("Kronos Agent initialized")
        
    async def shutdown(self):
        """Shutdown agent."""
        await self.tools.close()
        logger.info("Kronos Agent shutdown")
    
    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a tool and return the result."""
        
        logger.info(f"Executing tool: {tool_name}")
        
        if tool_name == "scan_repo":
            return await self.tools.scan_repo(**tool_input)
        elif tool_name == "query_graph":
            return await self.tools.query_graph(**tool_input)
        elif tool_name == "ingest_diff":
            return await self.tools.ingest_diff(**tool_input)
        elif tool_name == "trace_lineage":
            return await self.tools.trace_lineage(**tool_input)
        elif tool_name == "get_context":
            return await self.tools.get_context(**tool_input)
        elif tool_name == "list_graphs":
            return await self.tools.list_graphs()
        elif tool_name == "get_stats":
            return await self.tools.get_stats()
        elif tool_name == "add_source":
            return await self.tools.add_source(**tool_input)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.
        
        Implements the agentic loop with tool calling.
        """
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        
        # Agentic loop
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Call Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=self.conversation_history,
            )
            
            # Check stop reason
            if response.stop_reason == "end_turn":
                # Extract text response
                text_content = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text_content += block.text
                
                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content,
                })
                
                return text_content
            
            elif response.stop_reason == "tool_use":
                # Extract tool calls
                tool_calls = []
                text_parts = []
                
                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                    elif hasattr(block, "text"):
                        text_parts.append(block.text)
                
                # Add assistant message with tool calls
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content,
                })
                
                # Execute tools and collect results
                tool_results = []
                for call in tool_calls:
                    result = await self._execute_tool(call["name"], call["input"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": json.dumps(result, default=str),
                    })
                
                # Add tool results to history
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results,
                })
            
            else:
                # Unknown stop reason
                logger.warning(f"Unknown stop reason: {response.stop_reason}")
                break
        
        return "I've reached my iteration limit. Please try a simpler request."
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.state = create_initial_state()


async def create_agent(
    storage_path: Path = None,
    namespace: str = "agent",
) -> KronosAgent:
    """Create and initialize a Kronos agent."""
    agent = KronosAgent(storage_path=storage_path, namespace=namespace)
    await agent.initialize()
    return agent


# ============================================================================
# CLI Interface
# ============================================================================

async def main():
    """Interactive CLI for the Kronos Agent."""
    import sys
    import time
    
    start_time = time.perf_counter()
    
    # Use existing database with all ingested repos
    storage = Path(__file__).parent.parent / "kronos_dawn_models"
    
    print("=" * 60)
    print("Kronos Agent - Knowledge Graph Assistant")
    print("=" * 60)
    print(f"\nStorage: {storage}")
    print("Loading embedding model in background...")
    
    # Start agent creation in background
    agent_task = asyncio.create_task(
        create_agent(storage_path=storage, namespace="dawn_models")
    )
    
    # Show spinner while loading
    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    while not agent_task.done():
        elapsed = time.perf_counter() - start_time
        print(f"\r{spinner[i % len(spinner)]} Initializing... ({elapsed:.1f}s)", end="", flush=True)
        i += 1
        await asyncio.sleep(0.1)
    
    try:
        agent = await agent_task
    except ValueError as e:
        print(f"\n\nError: {e}")
        print("Make sure ANTHROPIC_API_KEY is set in grimm/.env")
        sys.exit(1)
    
    # Warm-up query to prime CUDA kernels
    await agent.tools.query_graph("warmup", limit=1)
    
    # Show available graphs
    graphs = await agent.tools.list_graphs()
    total_time = time.perf_counter() - start_time
    total_nodes = sum(g['node_count'] for g in graphs['graphs'])
    print(f"\r✓ Ready in {total_time:.1f}s! {len(graphs['graphs'])} graphs, {total_nodes} nodes:")
    for g in graphs['graphs']:
        print(f"  • {g['name']}: {g['node_count']} nodes")
    
    print("\nType 'quit' to exit, 'reset' to clear history.\n")
    
    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                break
            
            if user_input.lower() == "reset":
                agent.reset_conversation()
                print("Conversation reset.")
                continue
            
            print("\nAgent: ", end="", flush=True)
            
            try:
                response = await agent.chat(user_input)
                print(response)
            except Exception as e:
                print(f"\nError: {e}")
                logger.exception("Chat error")
    
    finally:
        await agent.shutdown()
        print("\nGoodbye!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
