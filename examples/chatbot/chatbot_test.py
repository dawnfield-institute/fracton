"""
Simple chatbot test without emojis for Windows compatibility
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

from fracton.storage import KronosMemory, NodeType


async def main():
    print("="*60)
    print("Fracton Agentic Chatbot Test")
    print("="*60)

    # Configuration
    device = os.getenv("DEVICE", "cpu")
    llm_provider = os.getenv("LLM_PROVIDER")
    llm_model = os.getenv("LLM_MODEL", "claude-3-haiku-20240307")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  LLM Provider: {llm_provider}")
    print(f"  LLM Model: {llm_model}")
    print(f"  API Key: {'***' + anthropic_key[-10:] if anthropic_key else 'not set'}")

    print("\n[*] Initializing KronosMemory...")

    memory = KronosMemory(
        storage_path=Path("./data/chatbot"),
        namespace="chatbot_test",
        device=device,
        embedding_model="mini",
    )

    try:
        await memory.connect()
        await memory.create_graph("conversations")

        print("[+] Memory initialized successfully!")

        # Create conversation root
        print("\n[*] Creating conversation root...")
        root_id = await memory.store(
            content=f"Test conversation started at {datetime.now().isoformat()}",
            graph="conversations",
            node_type=NodeType.CONCEPT,
            metadata={"type": "conversation_root"},
        )
        print(f"[+] Root created: {root_id[:8]}...")

        # Show initial health
        health = memory.get_foundation_health()
        stats = await memory.get_stats()
        print(f"\n[*] Initial Health Metrics:")
        print(f"    Total nodes: {stats['total_nodes']}")
        print(f"    Collapses: {stats.get('collapses', 0)}")

        # Test messages
        messages = [
            "Hello! This is a test of the memory system.",
            "Can you tell me about PAC conservation?",
            "What is the balance operator?",
        ]

        last_id = root_id

        for i, msg in enumerate(messages, 1):
            print(f"\n{'='*60}")
            print(f"Turn {i}: Storing message")
            print(f"{'='*60}")
            print(f"Message: {msg}")

            # Store user message
            user_id = await memory.store(
                content=msg,
                graph="conversations",
                node_type=NodeType.FACT,
                parent_id=last_id,
                metadata={"role": "user", "turn": i},
            )
            print(f"[+] User message stored: {user_id[:8]}...")

            # Retrieve context
            print("[*] Retrieving relevant context...")
            results = await memory.query(
                query_text=msg,
                graphs=["conversations"],
                limit=3,
            )

            print(f"[+] Retrieved {len(results)} context items:")
            for j, result in enumerate(results, 1):
                print(f"    {j}. [score={result.score:.3f}] {result.node.content[:50]}...")

            # Generate response with Anthropic
            if llm_provider == "anthropic" and anthropic_key:
                print("[*] Generating response with Anthropic...")
                try:
                    import anthropic

                    client = anthropic.Anthropic(api_key=anthropic_key)

                    context_str = "\n".join([
                        f"[{getattr(r.node, 'metadata', {}).get('role', 'system') if hasattr(r.node, 'metadata') else 'system'}]: {r.node.content}"
                        for r in results
                    ])

                    response = client.messages.create(
                        model=llm_model,
                        max_tokens=512,
                        system=f"You are a helpful assistant with memory powered by PAC/SEC/MED theoretical foundations.\n\nContext:\n{context_str}",
                        messages=[{"role": "user", "content": msg}]
                    )

                    bot_response = response.content[0].text
                    print(f"[+] Anthropic response:\n{bot_response}")

                except Exception as e:
                    bot_response = f"Error calling Anthropic: {e}"
                    print(f"[!] {bot_response}")
            else:
                bot_response = f"Mock response to: {msg}"
                print(f"[+] Mock response: {bot_response}")

            # Store bot response
            bot_id = await memory.store(
                content=bot_response,
                graph="conversations",
                node_type=NodeType.CONCEPT,
                parent_id=user_id,
                metadata={"role": "assistant", "turn": i},
            )
            print(f"[+] Bot response stored: {bot_id[:8]}...")

            last_id = bot_id

            # Show health every turn
            health = memory.get_foundation_health()
            stats = await memory.get_stats()

            if health['c_squared']['count'] > 0:
                c2 = health['c_squared']['latest']
                print(f"\n[*] Health Update:")
                print(f"    c^2: {c2:.3f}")

            if health['balance_operator']['count'] > 0:
                xi = health['balance_operator']['latest']
                xi_target = health['constants']['xi']
                status = "STABLE"
                if xi > xi_target:
                    status = "COLLAPSE"
                elif xi < 0.5:  # Updated threshold (values near 0.618 are normal)
                    status = "DECAY"
                print(f"    Balance Xi: {xi:.4f} [{status}]")

            if health['duty_cycle']['count'] > 0:
                duty = health['duty_cycle']['latest']
                print(f"    Duty cycle: {duty:.3f}")

            print(f"    Total nodes: {stats['total_nodes']}")
            print(f"    Collapses: {stats.get('collapses', 0)}")

        # Final summary
        print(f"\n{'='*60}")
        print("Test Complete!")
        print(f"{'='*60}")

        health = memory.get_foundation_health()
        stats = await memory.get_stats()

        print(f"\nFinal Statistics:")
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Total collapses: {stats.get('collapses', 0)}")

        if health['c_squared']['count'] > 0:
            print(f"\n  c^2 (model constant):")
            print(f"    Count: {health['c_squared']['count']}")
            print(f"    Mean: {health['c_squared']['mean']:.3f}")
            print(f"    Latest: {health['c_squared']['latest']:.3f}")

        if health['balance_operator']['count'] > 0:
            print(f"\n  Balance operator Xi:")
            print(f"    Count: {health['balance_operator']['count']}")
            print(f"    Mean: {health['balance_operator']['mean']:.4f}")
            print(f"    Latest: {health['balance_operator']['latest']:.4f}")
            print(f"    Target: {health['constants']['xi']:.4f}")

        if health['duty_cycle']['count'] > 0:
            print(f"\n  Duty cycle:")
            print(f"    Count: {health['duty_cycle']['count']}")
            print(f"    Mean: {health['duty_cycle']['mean']:.3f}")
            print(f"    Latest: {health['duty_cycle']['latest']:.3f}")
            if 'duty_cycle' in health['constants']:
                print(f"    Target: {health['constants']['duty_cycle']:.3f}")

    finally:
        await memory.close()
        print("\n[*] Memory closed.")


if __name__ == "__main__":
    asyncio.run(main())
