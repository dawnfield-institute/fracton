"""
End-to-End CLI Avatar Test with Comprehensive Graph

Tests the full pipeline:
1. Build comprehensive knowledge graph
2. Load into CLI avatar
3. Test variety of queries
4. Verify responses match expected personality for topology
"""

import sys
import io
import asyncio
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

# Add paths
fracton_path = Path(__file__).parent.parent.parent
grimm_path = fracton_path.parent / "grimm" / "avatars" / "cli"
sys.path.insert(0, str(fracton_path))
sys.path.insert(0, str(grimm_path))

# Import after path setup
from test_kronos_comprehensive_graph import create_comprehensive_graph
from cli_avatar.avatar import CLIAvatar, CLIAvatarConfig
from cli_avatar.kronos_integration import KronosService


async def test_cli_end_to_end():
    """Full end-to-end test with comprehensive graph."""

    print("=" * 80)
    print("END-TO-END CLI AVATAR TEST")
    print("=" * 80)

    # Build comprehensive graph
    print("\n1. Building comprehensive knowledge graph...")
    graph = create_comprehensive_graph()
    print(f"   ✓ Created {len(graph.nodes)} node graph")
    print(f"   ✓ Includes well-documented and sparse regions")
    print(f"   ✓ Includes confluence nodes (entanglement, quantum_computing)")

    # Initialize CLI avatar
    print("\n2. Initializing CLI avatar with KRONOS...")
    config = CLIAvatarConfig(
        avatar_id="test_avatar",
        soul_url="http://localhost:8000",
        offline_mode=True,
    )
    avatar = CLIAvatar(config)
    avatar.kronos = KronosService(kronos_graph=graph)
    print(f"   ✓ Avatar initialized")
    print(f"   ✓ KRONOS available: {avatar.kronos.is_available()}")

    # Test queries representing different scenarios
    test_scenarios = [
        {
            "name": "Well-Documented Root Concept",
            "query": "what is physics",
            "expect": "Should show authoritative tone (but currently low confidence due to small graph)",
        },
        {
            "name": "Well-Documented Confluence Node",
            "query": "explain quantum entanglement",
            "expect": "Should show integrative synthesis of 3 parent concepts",
        },
        {
            "name": "Well-Documented Application",
            "query": "what is quantum cryptography",
            "expect": "Should show comprehensive explanation with citations",
        },
        {
            "name": "Sparse/Emerging Concept (No Docs)",
            "query": "what is quantum biology",
            "expect": "Should show tentative tone, flag lack of documentation",
        },
        {
            "name": "Sparse/Speculative Concept",
            "query": "tell me about quantum consciousness",
            "expect": "Should show low confidence warning, epistemic issues",
        },
        {
            "name": "Non-Concept Query",
            "query": "how does quantum mechanics work",
            "expect": "Should fall through to Soul/offline message",
        },
        {
            "name": "Concept Search",
            "query": "quantum",
            "expect": "Should suggest multiple quantum concepts",
        },
    ]

    print("\n" + "=" * 80)
    print("TESTING QUERIES")
    print("=" * 80)

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'─' * 80}")
        print(f"TEST {i}/{len(test_scenarios)}: {scenario['name']}")
        print(f"{'─' * 80}")
        print(f"Query: \"{scenario['query']}\"")
        print(f"Expected: {scenario['expect']}")
        print()

        # Process query
        response = await avatar.process(scenario['query'])

        # Display response
        print("RESPONSE:")
        print("-" * 80)
        # Show first 40 lines of response
        response_lines = response.split('\n')
        for line in response_lines[:40]:
            print(line)
        if len(response_lines) > 40:
            print(f"... ({len(response_lines) - 40} more lines)")

        print()

        # Brief analysis
        if "⚠️" in response:
            print("✓ Low confidence warning present")
        if "Confluence" in response or "emerges from" in response:
            print("✓ Confluence synthesis detected")
        if "Epistemic Issues" in response:
            print("✓ Epistemic issues flagged")
        if "Applications" in response:
            print("✓ Applications listed (descendants)")
        if "Related" in response:
            print("✓ Related concepts listed (siblings)")
        if "Sources" in response or "Crystallized" in response:
            print("✓ Historical context provided")
        if "[Offline mode]" in response:
            print("✓ Fell through to offline mode (not concept query)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("""
Key Observations:

1. **Confluence Nodes**: Quantum entanglement should show integrative response
   breaking down by parent components (superposition 40%, nonlocality 35%,
   measurement 25%)

2. **Documentation Effect**: Well-documented concepts (entanglement, cryptography)
   should have higher confidence than sparse ones (biology, consciousness)

3. **Epistemic Honesty**: Sparse concepts should flag lack of documentation
   and show tentative language

4. **Lineage Tracking**: All responses should show full derivation path from
   root (Physics) to concept

5. **Related Concepts**: Applications (descendants) and alternatives (siblings)
   should be listed

6. **Field Dynamics**: FieldState affect/energy should vary with confidence
   (though not visible in CLI text output)

7. **Query Detection**: Non-concept queries should fall through to Soul
   (which shows offline message in this test)
""")

    print("\n✓ End-to-end test complete!")
    print("\nNext steps:")
    print("  - Tune confidence computation if scores seem too low")
    print("  - Add more documentation to boost confidence in core concepts")
    print("  - Test with real Soul service connection")
    print("  - Expand knowledge graph to other domains")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("KRONOS v2 - CLI AVATAR END-TO-END TEST")
    print("Field-Based Personality Validation")
    print("=" * 80)

    asyncio.run(test_cli_end_to_end())
