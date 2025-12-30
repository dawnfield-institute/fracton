"""
Tool-Based Agentic Chatbot with KronosMemory

This example shows how to build an agent that uses KronosMemory as a tool,
rather than trying to stuff context into prompts. The agent decides when and
how to query memory based on the conversation.

Key Pattern:
- Define memory operations as tools
- Let the LLM decide which tools to use
- Memory system is a black box - agent doesn't need to understand PAC/SEC/MED
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

from fracton.storage import KronosMemory, NodeType


class MemoryAgent:
    """
    Agentic chatbot with tool-based memory access.

    The agent has access to memory tools and decides when to use them.
    Memory system internals (PAC/SEC/MED) are abstracted away.
    """

    def __init__(
        self,
        storage_path: str = "./data/agent_chatbot",
        device: str = "cpu",
        llm_provider: str = "anthropic",
        llm_model: str = "claude-3-haiku-20240307",
        llm_api_key: Optional[str] = None,
    ):
        self.storage_path = Path(storage_path)
        self.device = device
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key

        self.memory: Optional[KronosMemory] = None
        self.conversation_id: Optional[str] = None
        self.turn_count = 0

    async def initialize(self):
        """Initialize memory system."""
        print("Initializing Memory Agent...")

        self.memory = KronosMemory(
            storage_path=self.storage_path,
            namespace="agent",
            device=self.device,
            embedding_model="mini",
        )

        await self.memory.connect()
        await self.memory.create_graph("conversations")

        # Create conversation marker
        self.conversation_id = await self.memory.store(
            content=f"Conversation started at {datetime.now().isoformat()}",
            graph="conversations",
            node_type=NodeType.CONCEPT,
            metadata={"type": "conversation_start"},
        )

        print(f"Agent ready (conversation: {self.conversation_id[:8]}...)")
        print()

    # ========================================================================
    # Memory Tools - These are exposed to the agent
    # ========================================================================

    async def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memory using semantic similarity.

        Args:
            query: What to search for
            limit: Max results

        Returns:
            List of relevant memories with content and metadata
        """
        results = await self.memory.query(
            query_text=query,
            graphs=["conversations"],
            limit=limit,
        )

        memories = []
        for r in results:
            memories.append({
                "content": r.node.content,
                "role": r.node.metadata.get("role", "system"),
                "turn": r.node.metadata.get("turn", 0),
                "relevance": r.score,
            })

        return memories

    async def store_message(self, content: str, role: str) -> str:
        """
        Store a message in memory.

        Args:
            content: Message content
            role: 'user' or 'assistant'

        Returns:
            Message ID
        """
        msg_id = await self.memory.store(
            content=content,
            graph="conversations",
            node_type=NodeType.FACT if role == "user" else NodeType.CONCEPT,
            parent_id=self.conversation_id,
            metadata={
                "role": role,
                "turn": self.turn_count,
                "timestamp": datetime.now().isoformat(),
            },
        )
        return msg_id

    async def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history (chronological).

        Args:
            limit: How many recent messages

        Returns:
            Recent messages in chronological order
        """
        # Query for recent messages
        results = await self.memory.query(
            query_text="conversation history",  # Generic query
            graphs=["conversations"],
            limit=limit * 2,  # Get more to filter
        )

        # Filter and sort by turn
        messages = []
        for r in results:
            if r.node.metadata.get("role") in ["user", "assistant"]:
                messages.append({
                    "content": r.node.content,
                    "role": r.node.metadata.get("role"),
                    "turn": r.node.metadata.get("turn", 0),
                })

        # Sort by turn (chronological)
        messages.sort(key=lambda x: x["turn"], reverse=True)
        return messages[:limit]

    # ========================================================================
    # Agent Logic - Decides when to use tools
    # ========================================================================

    async def chat(self, user_message: str) -> str:
        """
        Process a message using tool-based agent pattern.

        The agent decides:
        1. Should I search for relevant memories?
        2. Should I get recent conversation history?
        3. How should I synthesize the response?
        """
        self.turn_count += 1

        print(f"\n{'='*60}")
        print(f"Turn {self.turn_count}")
        print(f"{'='*60}")
        print(f"\nUser: {user_message}")

        # Store user message
        await self.store_message(user_message, "user")

        # Let the agent decide what tools to use
        response = await self._agent_response(user_message)

        print(f"\nAssistant: {response}")

        # Store assistant response
        await self.store_message(response, "assistant")

        return response

    async def _agent_response(self, user_message: str) -> str:
        """
        Generate response using Anthropic with tool calling.

        The LLM decides which memory tools to call based on the query.
        """
        if self.llm_provider != "anthropic":
            return self._simple_response(user_message)

        try:
            import anthropic

            client = anthropic.Anthropic(
                api_key=self.llm_api_key or os.getenv("ANTHROPIC_API_KEY")
            )

            # Define tools available to the agent
            tools = [
                {
                    "name": "search_memory",
                    "description": "Search your memory for relevant past conversation using semantic similarity. Use this when the user asks about something they mentioned before, or when you need context about previous topics.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to search for (e.g., 'user name', 'physics discussion', 'what user likes')"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_recent_messages",
                    "description": "Get recent conversation history in chronological order. Use this to understand the flow of the current conversation.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "How many recent messages to retrieve (default 10)",
                                "default": 10
                            }
                        },
                        "required": []
                    }
                }
            ]

            # Initial message to agent
            messages = [
                {
                    "role": "user",
                    "content": user_message
                }
            ]

            # Agent loop - let it use tools
            max_iterations = 3
            for iteration in range(max_iterations):
                response = client.messages.create(
                    model=self.llm_model,
                    max_tokens=2048,
                    tools=tools,
                    messages=messages,
                    system="""You are a helpful AI assistant with access to a persistent memory system.

You have two tools available:
1. search_memory - Search for relevant past conversations semantically
2. get_recent_messages - Get recent chronological conversation history

Use these tools when:
- User asks about something they told you before (search_memory)
- User asks "what do you remember" or "what did I say" (search_memory)
- You need context about the current conversation flow (get_recent_messages)

When you have the information you need, respond directly to the user without explaining which tools you used."""
                )

                # Check if agent wants to use tools
                if response.stop_reason == "tool_use":
                    # Agent is calling tools
                    tool_results = []

                    for content_block in response.content:
                        if content_block.type == "tool_use":
                            tool_name = content_block.name
                            tool_input = content_block.input

                            print(f"\n[Agent using tool: {tool_name}]")

                            # Execute the tool
                            if tool_name == "search_memory":
                                result = await self.search_memory(
                                    query=tool_input["query"],
                                    limit=tool_input.get("limit", 5)
                                )
                                print(f"  Found {len(result)} memories")
                            elif tool_name == "get_recent_messages":
                                result = await self.get_recent_messages(
                                    limit=tool_input.get("limit", 10)
                                )
                                print(f"  Retrieved {len(result)} recent messages")
                            else:
                                result = {"error": "Unknown tool"}

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": str(result)
                            })

                    # Add agent's message and tool results to conversation
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})

                    # Continue loop - agent will use tool results to respond

                else:
                    # Agent has final response
                    final_response = ""
                    for content_block in response.content:
                        if hasattr(content_block, "text"):
                            final_response += content_block.text

                    return final_response

            return "I'm sorry, I couldn't complete that request."

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            return self._simple_response(user_message)

    def _simple_response(self, user_message: str) -> str:
        """Fallback response without LLM."""
        return f"Echo: {user_message}"

    async def close(self):
        """Close memory connection."""
        if self.memory:
            await self.memory.close()
            print("\nAgent closed.")


async def main():
    """Run the agent chatbot."""
    print("""
==============================================================
          Tool-Based Memory Agent Demo
  Powered by KronosMemory (PAC/SEC/MED Foundations)
==============================================================

The agent has access to memory tools and decides when to use them.
Type 'quit' to exit, 'help' for commands.
    """)

    # Create agent
    agent = MemoryAgent(
        storage_path=os.getenv("FRACTON_DATA_DIR", "./data/agent_chatbot"),
        device=os.getenv("DEVICE", "cpu"),
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        llm_model=os.getenv("LLM_MODEL", "claude-3-haiku-20240307"),
        llm_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    try:
        await agent.initialize()

        print("Type your messages below:\n")

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == "help":
                    print("""
Commands:
  quit     - Exit the chat
  help     - Show this help

The agent will automatically search memory when needed.
                    """)
                    continue

                # Process message
                await agent.chat(user_input)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
