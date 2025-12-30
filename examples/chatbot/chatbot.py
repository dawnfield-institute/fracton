"""
Agentic Chatbot with KronosMemory

A conversational AI agent that uses Fracton's KronosMemory with full
PAC/SEC/MED theoretical foundations for context-aware responses.

Features:
- Long-term memory with conservation validation
- SEC resonance ranking for context retrieval
- Real-time foundation health monitoring
- Collapse detection and warnings
- Multi-turn conversation with hierarchy
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env file if it exists
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

from fracton.storage import KronosMemory, NodeType


class AgenticChatbot:
    """
    Agentic chatbot with PAC/SEC/MED memory foundations.

    Uses KronosMemory for:
    - Storing conversation history with parent-child relationships
    - Retrieving relevant context via SEC resonance ranking
    - Monitoring theoretical health (c², Ξ, duty cycle)
    - Detecting collapse triggers
    """

    def __init__(
        self,
        storage_path: str = "./data/chatbot",
        namespace: str = "chatbot",
        device: str = "cpu",
        embedding_model: str = "mini",
        llm_provider: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        """
        Initialize chatbot.

        Args:
            storage_path: Path for KronosMemory storage
            namespace: Namespace for memory
            device: 'cpu' or 'cuda'
            embedding_model: 'mini' or 'base' for sentence-transformers
            llm_provider: 'openai' or 'anthropic' (None for mock responses)
            llm_api_key: API key for LLM provider
            llm_model: Model name (e.g., 'claude-3-haiku-20240307')
        """
        self.storage_path = Path(storage_path)
        self.namespace = namespace
        self.device = device
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model

        self.memory: Optional[KronosMemory] = None
        self.conversation_root: Optional[str] = None
        self.last_message_id: Optional[str] = None
        self.turn_count = 0

    async def initialize(self):
        """Initialize memory and conversation."""
        print("🚀 Initializing Agentic Chatbot...")

        # Create KronosMemory
        self.memory = KronosMemory(
            storage_path=self.storage_path,
            namespace=self.namespace,
            device=self.device,
            embedding_model=self.embedding_model,
        )

        await self.memory.connect()

        # Create conversation graph
        await self.memory.create_graph("conversations")

        # Create conversation root
        self.conversation_root = await self.memory.store(
            content=f"Conversation started at {datetime.now().isoformat()}",
            graph="conversations",
            node_type=NodeType.CONCEPT,
            metadata={"type": "conversation_root", "started_at": datetime.now().isoformat()},
        )

        print(f"✅ Chatbot initialized")
        print(f"   Device: {self.device}")
        print(f"   Embedding model: {self.embedding_model}")
        print(f"   LLM provider: {self.llm_provider or 'mock'}")
        if self.llm_provider:
            print(f"   LLM model: {self.llm_model or 'default'}")
        print(f"   Storage: {self.storage_path}")
        print(f"   Conversation root: {self.conversation_root[:8]}...")

        # Show initial health
        await self._show_health()
        print()

    async def chat(self, user_message: str) -> str:
        """
        Process user message and generate response.

        Args:
            user_message: User's message

        Returns:
            Chatbot's response
        """
        self.turn_count += 1
        print(f"\n{'='*60}")
        print(f"Turn {self.turn_count}")
        print(f"{'='*60}")

        # Store user message
        print(f"\n💬 User: {user_message}")
        user_msg_id = await self._store_message(
            content=user_message,
            role="user",
            parent_id=self.last_message_id or self.conversation_root,
        )

        # Retrieve relevant context using SEC resonance
        all_context = await self._retrieve_context(user_message, limit=20)

        # Filter out the current user message (it will match itself with high score)
        # Keep top 10 for better conversation understanding
        context = [ctx for ctx in all_context if ctx['content'] != user_message][:10]

        print(f"\n🔍 Retrieved {len(context)} relevant context items:")
        for i, ctx in enumerate(context, 1):
            turn_info = f"Turn {ctx['turn']}" if ctx.get('turn') != '?' else ""
            print(f"   {i}. [{ctx['score']:.3f}] {turn_info} {ctx['content'][:50]}...")

        # Generate response (using LLM or mock)
        response = await self._generate_response(user_message, context)

        print(f"\n🤖 Bot: {response}")

        # Store bot response
        bot_msg_id = await self._store_message(
            content=response,
            role="assistant",
            parent_id=user_msg_id,
        )

        self.last_message_id = bot_msg_id

        # Show health metrics periodically
        if self.turn_count % 5 == 0:
            await self._show_health()

        return response

    async def _store_message(
        self,
        content: str,
        role: str,
        parent_id: str,
    ) -> str:
        """Store message with conservation validation."""
        msg_id = await self.memory.store(
            content=content,
            graph="conversations",
            node_type=NodeType.FACT if role == "user" else NodeType.CONCEPT,
            parent_id=parent_id,
            metadata={
                "role": role,
                "timestamp": datetime.now().isoformat(),
                "turn": self.turn_count,
            },
        )

        return msg_id

    async def _retrieve_context(
        self,
        query: str,
        limit: int = 15,  # Increased from 5 to 15 for better context
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context using SEC resonance ranking."""
        results = await self.memory.query(
            query_text=query,
            graphs=["conversations"],
            limit=limit,
        )

        context = []
        for result in results:
            # Parse timestamp for relative time
            timestamp = result.node.metadata.get("timestamp", "")
            turn = result.node.metadata.get("turn", "?")

            context.append({
                "content": result.node.content,
                "score": result.score,
                "similarity": result.similarity,
                "role": result.node.metadata.get("role", "unknown"),
                "timestamp": timestamp,
                "turn": turn,
            })

        return context

    async def _generate_response(
        self,
        user_message: str,
        context: List[Dict[str, Any]],
    ) -> str:
        """Generate response using LLM or mock."""
        if self.llm_provider == "openai":
            return await self._generate_openai(user_message, context)
        elif self.llm_provider == "anthropic":
            return await self._generate_anthropic(user_message, context)
        else:
            return self._generate_mock(user_message, context)

    async def _generate_openai(
        self,
        user_message: str,
        context: List[Dict[str, Any]],
    ) -> str:
        """Generate response using OpenAI API."""
        try:
            import openai

            openai.api_key = self.llm_api_key or os.getenv("OPENAI_API_KEY")

            # Build context string
            context_str = "\n".join([
                f"[{ctx['role']}]: {ctx['content']}"
                for ctx in context
            ])

            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant with memory powered by PAC/SEC/MED theoretical foundations. Use the provided context to give relevant responses."
                    },
                    {
                        "role": "system",
                        "content": f"Relevant context:\n{context_str}"
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                temperature=0.7,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"⚠️  OpenAI API error: {e}")
            return self._generate_mock(user_message, context)

    async def _generate_anthropic(
        self,
        user_message: str,
        context: List[Dict[str, Any]],
    ) -> str:
        """Generate response using Anthropic API."""
        try:
            import anthropic

            client = anthropic.Anthropic(
                api_key=self.llm_api_key or os.getenv("ANTHROPIC_API_KEY")
            )

            # Build context string from conversation history with turn numbers
            context_lines = []
            for ctx in context:
                turn_marker = f"[Turn {ctx['turn']}]" if ctx.get('turn') != '?' else ""
                role = ctx['role'].capitalize()
                context_lines.append(f"{turn_marker} {role}: {ctx['content']}")
            context_str = "\n".join(context_lines)

            # Use configured model or default to Haiku
            model = self.llm_model or os.getenv("LLM_MODEL", "claude-3-haiku-20240307")

            system_prompt = f"""You are a helpful AI assistant with a persistent memory system powered by PAC/SEC/MED theoretical foundations from Dawn Field Theory.

IMPORTANT: You have access to our ACTUAL conversation history below. This is not hypothetical context - these are real messages from our ongoing conversation, retrieved using SEC (Symbolic Entropy Collapse) resonance ranking based on semantic relevance to the current query.

Your memory works by:
1. Storing every message with PAC (Potential-Actualization Conservation) validation
2. Retrieving the most relevant past messages using golden ratio resonance ranking
3. Presenting them to you ordered by relevance score (not chronological order)

CONVERSATION HISTORY (ranked by relevance to current query):
{context_str if context_str else "(No previous conversation history yet)"}

When answering:
- Treat this history as YOUR memory of our conversation
- Reference specific things the user told you when relevant
- Don't say "I don't have memory" or "I can't recall" if the information is in the history above
- Be confident about facts you can see in the conversation history
- If asked what you remember, summarize what you can see in the history"""

            message = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            )

            return message.content[0].text

        except Exception as e:
            print(f"⚠️  Anthropic API error: {e}")
            return self._generate_mock(user_message, context)

    def _generate_mock(
        self,
        user_message: str,
        context: List[Dict[str, Any]],
    ) -> str:
        """Generate mock response (for testing without LLM API)."""
        # Simple rule-based responses for demo
        msg_lower = user_message.lower()

        if "hello" in msg_lower or "hi" in msg_lower:
            return "Hello! I'm an agentic chatbot powered by Fracton's KronosMemory with PAC/SEC/MED foundations. How can I help you today?"

        elif "memory" in msg_lower or "remember" in msg_lower:
            if context:
                return f"I remember we discussed: {context[0]['content'][:100]}... My memory uses PAC conservation and SEC resonance ranking!"
            else:
                return "I don't have any relevant memories yet, but I'm storing everything we discuss using theoretical foundations from Dawn Field Theory!"

        elif "health" in msg_lower or "status" in msg_lower:
            return "My theoretical health is monitored in real-time! Check the console for c², balance operator Ξ, and duty cycle metrics."

        elif "collapse" in msg_lower:
            return "Collapse detection is active! The balance operator Ξ monitors for instability. When Ξ > 1.0571, a collapse is triggered."

        elif len(context) > 0:
            return f"Based on our conversation, I recall: '{context[0]['content'][:80]}...' Is there something specific you'd like to know more about?"

        else:
            return f"I understand you said: '{user_message}'. This is stored in my memory with full PAC conservation validation. What would you like to discuss?"

    async def _show_health(self):
        """Display foundation health metrics."""
        health = self.memory.get_foundation_health()
        stats = await self.memory.get_stats()

        print(f"\n{'─'*60}")
        print("📊 Foundation Health Metrics")
        print(f"{'─'*60}")

        # c² (model constant)
        if health['c_squared']['count'] > 0:
            c2 = health['c_squared']['latest']
            c2_mean = health['c_squared']['mean']
            print(f"   c² (model constant):    {c2:.3f} (mean: {c2_mean:.3f})")
        else:
            print(f"   c² (model constant):    No data yet")

        # Balance operator Ξ
        if health['balance_operator']['count'] > 0:
            xi = health['balance_operator']['latest']
            xi_target = health['constants']['xi']
            status = "STABLE"
            if xi > xi_target:
                status = "⚠️  COLLAPSE"
            elif xi < 0.9514:
                status = "⚠️  DECAY"
            print(f"   Balance operator Ξ:     {xi:.4f} (target: {xi_target:.4f}) [{status}]")
        else:
            print(f"   Balance operator Ξ:     No data yet")

        # Duty cycle
        if health['duty_cycle']['count'] > 0:
            duty = health['duty_cycle']['latest']
            duty_target = health['constants']['duty_cycle']
            print(f"   Duty cycle:             {duty:.3f} (target: {duty_target:.3f})")
        else:
            print(f"   Duty cycle:             No data yet")

        # Stats
        print(f"   Total nodes:            {stats['total_nodes']}")
        print(f"   Collapse triggers:      {stats['collapses']}")
        print(f"{'─'*60}")

    async def close(self):
        """Close memory connection."""
        if self.memory:
            await self.memory.close()
            print("\n👋 Chatbot closed. Memory saved.")


async def main():
    """Run chatbot demo."""
    print("""
==============================================================
              Fracton Agentic Chatbot Demo
  Powered by KronosMemory with PAC/SEC/MED Foundations
==============================================================
    """)

    # Configuration from environment
    device = os.getenv("DEVICE", "cpu")
    llm_provider = os.getenv("LLM_PROVIDER")
    llm_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    llm_model = os.getenv("LLM_MODEL", "claude-3-haiku-20240307")

    # Create chatbot
    chatbot = AgenticChatbot(
        storage_path=os.getenv("FRACTON_DATA_DIR", "./data/chatbot"),
        device=device,
        embedding_model="mini",
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )

    try:
        await chatbot.initialize()

        print("\n💡 Type your messages (or 'quit' to exit, 'health' for metrics)\n")

        # Interactive loop
        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == "health":
                    await chatbot._show_health()
                    continue

                # Process message
                response = await chatbot.chat(user_input)

            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

    finally:
        await chatbot.close()


if __name__ == "__main__":
    asyncio.run(main())
