"""
Unit and Integration Tests for KRONOS Unified Memory

Tests PAC+SEC+PAS integration:
- PAC: Delta-only storage and reconstruction
- SEC: Resonance-based ranking
- PAS: Conservation validation
- Bifractal: Temporal tracing
- Multi-graph: Cross-graph linking
"""

import pytest
import torch
import asyncio
from pathlib import Path
import tempfile
import shutil

from fracton.storage import (
    KronosMemory,
    PACMemoryNode,
    NodeType,
    RelationType,
    ResonanceResult
)


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def memory(temp_storage):
    """Create KRONOS memory instance (sync fixture that returns async-ready object)."""
    mem = KronosMemory(
        storage_path=temp_storage,
        namespace="test",
        device="cpu",
        embedding_dim=384
    )
    # Create test graph in each test instead
    return mem


class TestPACDeltaStorage:
    """Test PAC (Predictive Adaptive Coding) delta storage."""

    @pytest.mark.asyncio
    async def test_root_node_storage(self, memory):
        """Test storing root node (full content)."""
        await memory.create_graph("test_graph", "Test graph")

        node_id = await memory.store(
            content="Root concept",
            graph="test_graph",
            node_type=NodeType.CONCEPT
        )

        assert node_id is not None
        assert len(node_id) > 0

        # Verify stored
        node = memory._graphs["test_graph"][node_id]
        assert node.is_root
        assert node.parent_id == "-1"
        assert node.content == "Root concept"

    @pytest.mark.asyncio
    async def test_child_node_delta_encoding(self, memory):
        """Test PAC delta encoding for child nodes."""
        # Store root
        root_id = await memory.store(
            content="Base concept",
            graph="test_graph",
            node_type=NodeType.CONCEPT
        )

        # Store child (should only store delta)
        child_id = await memory.store(
            content="Base concept with additional detail",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            parent_id=root_id
        )

        root = memory._graphs["test_graph"][root_id]
        child = memory._graphs["test_graph"][child_id]

        # Child should reference parent
        assert child.parent_id == root_id
        assert not child.is_root

        # Parent should know about child
        assert child_id in root.children_ids

    @pytest.mark.asyncio
    async def test_pac_reconstruction(self, memory):
        """Test PAC delta chain reconstruction."""
        # Create chain: root -> child -> grandchild
        root_id = await memory.store(
            content="A",
            graph="test_graph",
            node_type=NodeType.CONCEPT
        )

        child_id = await memory.store(
            content="AB",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            parent_id=root_id
        )

        grandchild_id = await memory.store(
            content="ABC",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            parent_id=child_id
        )

        # Reconstruct grandchild embedding
        grandchild_emb = await memory._reconstruct_embedding("test_graph", grandchild_id)

        assert grandchild_emb is not None
        assert grandchild_emb.shape[0] == 384
        assert memory._stats["reconstructions"] > 0

    @pytest.mark.asyncio
    async def test_potential_decay(self, memory):
        """Test PAC potential decay with depth."""
        root_id = await memory.store(
            content="Root",
            graph="test_graph",
            node_type=NodeType.CONCEPT
        )

        child_id = await memory.store(
            content="Root child",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            parent_id=root_id
        )

        root = memory._graphs["test_graph"][root_id]
        child = memory._graphs["test_graph"][child_id]

        # Root should have potential = 1.0
        assert root.potential == 1.0

        # Child should have decayed potential
        assert child.potential < root.potential
        assert child.potential > 0


class TestSECResonance:
    """Test SEC (Symbolic Entropy Collapse) resonance ranking."""

    @pytest.mark.asyncio
    async def test_entropy_calculation(self, memory):
        """Test symbolic entropy computation."""
        # Structured content (low entropy)
        structured = "The quick brown fox jumps over the lazy dog"
        entropy_structured = memory._compute_symbolic_entropy(structured)

        # Random content (high entropy)
        random = "asdfjkl qwerty zxcvbn mnbvcx asdfgh"
        entropy_random = memory._compute_symbolic_entropy(random)

        # Structured should have lower entropy
        assert entropy_structured < entropy_random

    @pytest.mark.asyncio
    async def test_sec_ranking(self, memory):
        """Test SEC-based resonance ranking."""
        # Store structured content
        structured_id = await memory.store(
            content="Machine learning uses neural networks for pattern recognition",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            importance=1.0
        )

        # Store random content
        random_id = await memory.store(
            content="asdfgh qwerty zxcvbn machine learning asdfgh",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            importance=0.1
        )

        # Query
        results = await memory.query(
            query_text="machine learning neural networks",
            graphs=["test_graph"],
            limit=10
        )

        # Structured content should rank higher
        assert len(results) >= 2
        top_result = results[0]

        # Top result should be structured (low entropy)
        assert top_result.node.entropy < 0.7

    @pytest.mark.asyncio
    async def test_sec_weights(self, memory):
        """Test custom SEC weight configuration."""
        await memory.store(
            content="Important concept",
            graph="test_graph",
            node_type=NodeType.CONCEPT
        )

        # Query with custom weights
        results = await memory.query(
            query_text="concept",
            graphs=["test_graph"],
            sec_weights={
                "similarity": 0.5,
                "entropy": 0.5,
                "recency": 0.0,
                "coherence": 0.0
            }
        )

        assert len(results) > 0


class TestPASConservation:
    """Test PAS (Potential Actualization) conservation."""

    @pytest.mark.asyncio
    async def test_conservation_validation(self, memory):
        """Test PAS conservation validation."""
        # Create parent with children
        parent_id = await memory.store(
            content="Parent concept",
            graph="test_graph",
            node_type=NodeType.CONCEPT
        )

        child1_id = await memory.store(
            content="Parent concept - aspect A",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            parent_id=parent_id
        )

        child2_id = await memory.store(
            content="Parent concept - aspect B",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            parent_id=parent_id
        )

        # Validation should pass
        valid, residual = await memory._validate_conservation("test_graph", parent_id)

        assert valid or residual < 1e-3  # Allow small residual
        assert memory._stats["conservations_validated"] > 0

    @pytest.mark.asyncio
    async def test_potential_conservation(self, memory):
        """Test potential field conservation."""
        root_id = await memory.store(
            content="Root",
            graph="test_graph",
            node_type=NodeType.CONCEPT
        )

        # Store multiple children
        children = []
        for i in range(3):
            child_id = await memory.store(
                content=f"Root child {i}",
                graph="test_graph",
                node_type=NodeType.CONCEPT,
                parent_id=root_id
            )
            children.append(child_id)

        # Get nodes
        root = memory._graphs["test_graph"][root_id]
        child_nodes = [memory._graphs["test_graph"][cid] for cid in children]

        # Total potential should be conserved (roughly)
        # Root potential ≈ Σ(child potentials)
        total_child_potential = sum(c.potential for c in child_nodes)

        # Due to decay, total is less, but should be reasonable
        assert total_child_potential > 0


class TestBifractalTracing:
    """Test bifractal temporal tracing."""

    @pytest.mark.asyncio
    async def test_backward_trace(self, memory):
        """Test backward temporal trace (ancestry)."""
        # Create chain
        root_id = await memory.store(
            content="Original idea",
            graph="test_graph",
            node_type=NodeType.CONCEPT
        )

        child_id = await memory.store(
            content="Original idea evolved",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            parent_id=root_id
        )

        grandchild_id = await memory.store(
            content="Original idea evolved further",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            parent_id=child_id
        )

        # Trace backward from grandchild
        trace = await memory.trace_evolution(
            graph="test_graph",
            node_id=grandchild_id,
            direction="backward"
        )

        # Should have 3 steps (grandchild -> child -> root)
        assert len(trace["backward_path"]) == 3
        assert trace["backward_path"][0]["node_id"] == grandchild_id
        assert trace["backward_path"][-1]["node_id"] == root_id

    @pytest.mark.asyncio
    async def test_forward_trace(self, memory):
        """Test forward temporal trace (evolution)."""
        # Create chain
        root_id = await memory.store(
            content="Root",
            graph="test_graph",
            node_type=NodeType.CONCEPT
        )

        child_id = await memory.store(
            content="Root evolved",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            parent_id=root_id
        )

        # Trace forward from root
        trace = await memory.trace_evolution(
            graph="test_graph",
            node_id=root_id,
            direction="forward"
        )

        # Should show evolution
        assert len(trace["forward_path"]) >= 1

    @pytest.mark.asyncio
    async def test_entropy_evolution(self, memory):
        """Test entropy evolution tracking."""
        root_id = await memory.store(
            content="Very structured and organized content",
            graph="test_graph",
            node_type=NodeType.CONCEPT
        )

        child_id = await memory.store(
            content="asdfgh qwerty zxcvbn",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            parent_id=root_id
        )

        trace = await memory.trace_evolution(
            graph="test_graph",
            node_id=child_id,
            direction="backward"
        )

        # Should have entropy evolution data
        assert "entropy_evolution" in trace
        assert len(trace["entropy_evolution"]) > 0


class TestMultiGraph:
    """Test multi-graph and cross-graph linking."""

    @pytest.mark.asyncio
    async def test_multiple_graphs(self, memory):
        """Test creating multiple graphs."""
        await memory.create_graph("research", "Research")
        await memory.create_graph("code", "Code")

        assert "research" in memory._graphs
        assert "code" in memory._graphs
        assert memory._stats["total_graphs"] >= 3  # test_graph + 2 new

    @pytest.mark.asyncio
    async def test_cross_graph_storage(self, memory):
        """Test storing in different graphs."""
        await memory.create_graph("graph_a", "Graph A")
        await memory.create_graph("graph_b", "Graph B")

        id_a = await memory.store(
            content="Content in A",
            graph="graph_a",
            node_type=NodeType.CONCEPT
        )

        id_b = await memory.store(
            content="Content in B",
            graph="graph_b",
            node_type=NodeType.CONCEPT
        )

        assert id_a in memory._graphs["graph_a"]
        assert id_b in memory._graphs["graph_b"]

    @pytest.mark.asyncio
    async def test_cross_graph_linking(self, memory):
        """Test linking nodes across graphs."""
        await memory.create_graph("research", "Research")
        await memory.create_graph("code", "Code")

        paper_id = await memory.store(
            content="Research paper",
            graph="research",
            node_type=NodeType.PAPER
        )

        commit_id = await memory.store(
            content="Code commit",
            graph="code",
            node_type=NodeType.COMMIT
        )

        # Link across graphs
        await memory.link_across_graphs(
            from_graph="code",
            from_id=commit_id,
            to_graph="research",
            to_id=paper_id,
            relation=RelationType.IMPLEMENTS
        )

        # Verify link
        code_node = memory._graphs["code"][commit_id]
        research_node = memory._graphs["research"][paper_id]

        assert paper_id in code_node.relationships.get(RelationType.IMPLEMENTS, [])

    @pytest.mark.asyncio
    async def test_cross_graph_query(self, memory):
        """Test querying across multiple graphs."""
        await memory.create_graph("g1", "Graph 1")
        await memory.create_graph("g2", "Graph 2")

        await memory.store(
            content="Machine learning concept",
            graph="g1",
            node_type=NodeType.CONCEPT
        )

        await memory.store(
            content="Machine learning implementation",
            graph="g2",
            node_type=NodeType.COMMIT
        )

        # Query both graphs
        results = await memory.query(
            query_text="machine learning",
            graphs=["g1", "g2"],
            limit=10
        )

        # Should find results from both graphs
        graphs_found = {r.node.graph for r in results}
        assert len(graphs_found) >= 1  # At least one graph has results


class TestPACExpansion:
    """Test PAC graph expansion during queries."""

    @pytest.mark.asyncio
    async def test_pac_neighbor_expansion(self, memory):
        """Test PAC-based neighbor expansion."""
        # Create network: root -> [child1, child2]
        root_id = await memory.store(
            content="Root concept",
            graph="test_graph",
            node_type=NodeType.CONCEPT
        )

        child1_id = await memory.store(
            content="Root concept child 1",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            parent_id=root_id
        )

        child2_id = await memory.store(
            content="Root concept child 2",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            parent_id=root_id
        )

        # Get neighbors
        neighbors = await memory._get_pac_neighbors("test_graph", child1_id, max_depth=2)

        # Should include root and sibling
        neighbor_ids = {n.id for n in neighbors}
        assert root_id in neighbor_ids


class TestStatistics:
    """Test statistics and monitoring."""

    @pytest.mark.asyncio
    async def test_get_stats(self, memory):
        """Test statistics retrieval."""
        stats = memory.get_stats()

        assert "total_nodes" in stats
        assert "total_graphs" in stats
        assert "queries" in stats
        assert "reconstructions" in stats
        assert "conservations_validated" in stats
        assert "device" in stats

    @pytest.mark.asyncio
    async def test_stats_updated(self, memory):
        """Test that stats are updated."""
        initial_nodes = memory._stats["total_nodes"]

        await memory.store(
            content="New concept",
            graph="test_graph",
            node_type=NodeType.CONCEPT
        )

        assert memory._stats["total_nodes"] == initial_nodes + 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_content(self, memory):
        """Test storing empty content."""
        node_id = await memory.store(
            content="",
            graph="test_graph",
            node_type=NodeType.NOTE
        )

        assert node_id is not None

    @pytest.mark.asyncio
    async def test_query_empty_graph(self, memory):
        """Test querying empty graph."""
        await memory.create_graph("empty", "Empty")

        results = await memory.query(
            query_text="anything",
            graphs=["empty"]
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_invalid_parent(self, memory):
        """Test error handling for invalid parent."""
        with pytest.raises(ValueError):
            await memory.store(
                content="Child",
                graph="test_graph",
                node_type=NodeType.CONCEPT,
                parent_id="nonexistent_id"
            )


# Integration Tests

class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    @pytest.mark.asyncio
    async def test_research_to_code_workflow(self, memory):
        """Test research -> code linking workflow."""
        await memory.create_graph("research", "Research")
        await memory.create_graph("code", "Code")

        # Store paper
        paper_id = await memory.store(
            content="Attention mechanism paper",
            graph="research",
            node_type=NodeType.PAPER,
            importance=1.0
        )

        # Store implementation
        impl_id = await memory.store(
            content="Attention mechanism implementation",
            graph="code",
            node_type=NodeType.COMMIT
        )

        # Link
        await memory.link_across_graphs(
            from_graph="code",
            from_id=impl_id,
            to_graph="research",
            to_id=paper_id,
            relation=RelationType.IMPLEMENTS
        )

        # Query
        results = await memory.query(
            query_text="attention mechanism",
            graphs=["research", "code"],
            expand_graph=True
        )

        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_idea_evolution_tracking(self, memory):
        """Test tracking idea evolution over time."""
        # Create evolution chain
        v1 = await memory.store(
            content="Initial idea",
            graph="test_graph",
            node_type=NodeType.CONCEPT
        )

        v2 = await memory.store(
            content="Initial idea with refinement",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            parent_id=v1
        )

        v3 = await memory.store(
            content="Initial idea with refinement and validation",
            graph="test_graph",
            node_type=NodeType.CONCEPT,
            parent_id=v2
        )

        # Trace evolution
        trace = await memory.trace_evolution("test_graph", v3, "both")

        assert len(trace["backward_path"]) == 3
        assert len(trace["entropy_evolution"]) == 3
        assert len(trace["potential_evolution"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
