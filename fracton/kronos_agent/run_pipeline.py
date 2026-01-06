#!/usr/bin/env python3
"""
Run the 2-Agent Pipeline interactively.

SearchAgent → SessionGraph → SynthesisAgent → Response with citations
"""

# Suppress TensorFlow noise
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from .pipeline import AgentPipeline


async def main():
    """Interactive CLI for the 2-Agent Pipeline."""
    start_time = time.perf_counter()
    
    storage = Path(__file__).parent.parent / "kronos_dawn_models"
    
    print("=" * 60)
    print("Kronos 2-Agent Pipeline")
    print("SearchAgent → SessionGraph → SynthesisAgent")
    print("=" * 60)
    print(f"\nStorage: {storage}")
    print("Loading...")
    
    # Initialize pipeline with spinner
    pipeline = AgentPipeline(storage_path=storage, namespace="dawn_models")
    
    init_task = asyncio.create_task(pipeline.initialize())
    
    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    while not init_task.done():
        elapsed = time.perf_counter() - start_time
        print(f"\r{spinner[i % len(spinner)]} Initializing... ({elapsed:.1f}s)", end="", flush=True)
        i += 1
        await asyncio.sleep(0.1)
    
    await init_task
    
    # Warm-up query
    await pipeline.tools.query_graph("warmup", limit=1)
    
    # Show stats
    graphs = await pipeline.list_graphs()
    total_time = time.perf_counter() - start_time
    total_nodes = sum(g['node_count'] for g in graphs['graphs'])
    
    print(f"\r✓ Ready in {total_time:.1f}s! {len(graphs['graphs'])} graphs, {total_nodes} nodes:")
    for g in graphs['graphs']:
        print(f"  • {g['name']}: {g['node_count']} nodes")
    
    print("\n" + "=" * 60)
    print("Pipeline: Query → Search → SessionGraph → Synthesis")
    print("All responses will include source citations.")
    print("=" * 60)
    print("\nType 'quit' to exit.\n")
    
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
            
            print("\n[Searching...]", end="", flush=True)
            
            try:
                result = await pipeline.query(user_input)
                
                # Clear "Searching..." and show timing
                timings = result.get("timings", {})
                session = result.get("session", {})
                
                print(f"\r[Search: {timings.get('search', 0)*1000:.0f}ms | "
                      f"Sources: {session.get('chunk_count', 0)} | "
                      f"Synthesis: {timings.get('synthesis', 0):.1f}s]")
                print()
                print(result.get("response", "No response"))
                
            except Exception as e:
                print(f"\nError: {e}")
                logging.exception("Pipeline error")
    
    finally:
        await pipeline.shutdown()
        print("\nGoodbye!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
