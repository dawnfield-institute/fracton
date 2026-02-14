"""
Quick test to verify cross-seed dependency context is being passed.

This inspects what instructions OrderProcessor receives during generation.
"""

import sys
from pathlib import Path

# Mock the generation to capture prompts
captured_prompts = []

original_generate = None

def capture_generate(self, context):
    """Capture the prompt before calling original generate."""
    from fracton.tools.shadowpuppet.generators import GenerationContext
    
    # Build the prompt to see what's being sent
    prompt = self.build_prompt(context)
    
    captured_prompts.append({
        'protocol': context.protocol.name,
        'prompt': prompt,
        'extra_instructions': context.extra_instructions
    })
    
    # Call original
    return original_generate(self, context)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
    
    from fracton.tools.shadowpuppet import SeedArchitecture, MultiSeedEvolution, ProtocolSpec, GrowthGap
    from fracton.tools.shadowpuppet.generators import MockGenerator
    from fracton.tools.shadowpuppet.evolution import EvolutionConfig
    
    # Patch MockGenerator to capture prompts
    original_generate = MockGenerator.generate
    MockGenerator.generate = capture_generate
    
    # Simple 2-seed setup
    user_repo = ProtocolSpec(
        name="UserRepo",
        methods=["get_user"],
        docstring="User storage",
        dependencies=[]
    )
    
    order_proc = ProtocolSpec(
        name="OrderProc",
        methods=["create_order"],
        docstring="Order processing",
        attributes=["user_repo: UserRepo"],
        dependencies=["UserRepo"]
    )
    
    user_seed = SeedArchitecture(
        name="UserService",
        gaps=[GrowthGap(protocol=user_repo)],
        exposed_interfaces=["UserRepo"]
    )
    
    order_seed = SeedArchitecture(
        name="OrderService",
        gaps=[GrowthGap(protocol=order_proc)],
        dependencies={"UserService": ["UserRepo"]}
    )
    
    # Evolve
    multi = MultiSeedEvolution(
        seeds=[user_seed, order_seed],
        generator=MockGenerator(),
        global_config=EvolutionConfig(coherence_threshold=0.60, candidates_per_gap=1)
    )
    
    results = multi.evolve(max_generations=1)
    
    # Show what OrderProc received
    print("\n" + "="*70)
    print("DEPENDENCY CONTEXT VERIFICATION")
    print("="*70)
    
    for item in captured_prompts:
        if item['protocol'] == 'OrderProc':
            print(f"\nProtocol: {item['protocol']}")
            print(f"\nExtra Instructions Received:")
            print(item['extra_instructions'])
            print("\n" + "-"*70)
            
            if "UserRepo" in item['extra_instructions']:
                print("✅ SUCCESS: OrderProc received UserRepo interface context!")
            else:
                print("❌ FAIL: OrderProc did NOT receive dependency context")
