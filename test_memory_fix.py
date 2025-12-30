"""Quick test to verify memory is working correctly"""

import asyncio
import os
import sys
from pathlib import Path

# Add fracton to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Import after env is loaded
from examples.chatbot.chatbot import AgenticChatbot


async def main():
    print("Testing Memory Fix")
    print("="*60)

    chatbot = AgenticChatbot(
        storage_path=Path("./data/memory_test"),
        namespace="test",
        device="cpu",
        embedding_model="mini",
        llm_provider="anthropic",
        llm_api_key=os.getenv("ANTHROPIC_API_KEY"),
        llm_model="claude-3-haiku-20240307",
    )

    try:
        await chatbot.initialize()

        # Test conversation
        print("\n" + "="*60)
        await chatbot.chat("My name is Alice")

        print("\n" + "="*60)
        await chatbot.chat("I love quantum physics")

        print("\n" + "="*60)
        await chatbot.chat("What's my name and what do I love?")

    finally:
        await chatbot.close()


if __name__ == "__main__":
    asyncio.run(main())
