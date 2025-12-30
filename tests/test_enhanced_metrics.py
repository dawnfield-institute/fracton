"""
Test enhanced metrics tracking in KronosMemory.

Validates:
- Balance operator Ξ tracking
- Duty cycle monitoring
- Collapse detection
- SEC resonance in queries
"""

import pytest
import pytest_asyncio

from fracton.storage.kronos_memory import KronosMemory, NodeType


@pytest_asyncio.fixture
async def memory(tmp_path):
    """Create KronosMemory instance."""
    mem = KronosMemory(
        storage_path=tmp_path,
        namespace="test",
        device="cpu",
        embedding_dim=128,
        embedding_model="mini",
    )
    await mem.connect()
    await mem.create_graph("test")
    yield mem
    await mem.close()


class TestEnhancedMetrics:
    """Test enhanced foundation metrics."""

    @pytest.mark.asyncio
    async def test_balance_operator_tracking(self, memory):
        """Test that balance operator Ξ is tracked"""
        # Create hierarchy
        parent_id = await memory.store(
            content="Parent for balance test",
            graph="test",
            node_type=NodeType.CONCEPT,
        )

        child1_id = await memory.store(
            content="Child 1",
            graph="test",
            node_type=NodeType.CONCEPT,
            parent_id=parent_id,
        )

        child2_id = await memory.store(
            content="Child 2",
            graph="test",
            node_type=NodeType.CONCEPT,
            parent_id=parent_id,
        )

        # Check balance operator tracked
        health = memory.get_foundation_health()
        assert health["balance_operator"]["count"] > 0
        assert health["balance_operator"]["latest"] is not None

        # Balance operator should be around 1.0 (stable)
        xi = health["balance_operator"]["latest"]
        print(f"\nBalance operator Ξ: {xi:.4f}")
        print(f"  Target: {memory.foundation.constants.XI:.4f}")
        print(f"  Status: {'COLLAPSE' if xi > memory.foundation.constants.XI else 'STABLE'}")

    @pytest.mark.asyncio
    async def test_duty_cycle_tracking(self, memory):
        """Test that SEC duty cycle is tracked"""
        # Create multiple hierarchies
        for i in range(3):
            parent_id = await memory.store(
                content=f"Parent {i}",
                graph="test",
                node_type=NodeType.CONCEPT,
            )

            await memory.store(
                content=f"Child of {i}",
                graph="test",
                node_type=NodeType.CONCEPT,
                parent_id=parent_id,
            )

        # Check duty cycle tracked
        health = memory.get_foundation_health()
        assert health["duty_cycle"]["count"] > 0

        # Duty cycle should be around 0.618 (golden ratio)
        duty = health["duty_cycle"]["latest"]
        target = memory.foundation.constants.DUTY_CYCLE

        print(f"\nDuty cycle: {duty:.3f}")
        print(f"  Target (φ/(φ+1)): {target:.3f}")

    @pytest.mark.asyncio
    async def test_all_metrics_collected(self, memory):
        """Test that all health metrics are collected together"""
        # Create hierarchy
        parent_id = await memory.store(
            content="Complete metrics test",
            graph="test",
            node_type=NodeType.CONCEPT,
        )

        for i in range(2):
            await memory.store(
                content=f"Child {i}",
                graph="test",
                node_type=NodeType.CONCEPT,
                parent_id=parent_id,
            )

        # Get all metrics
        health = memory.get_foundation_health()

        # All should have data
        assert health["c_squared"]["count"] > 0
        assert health["balance_operator"]["count"] > 0
        assert health["duty_cycle"]["count"] > 0

        # Print summary
        print(f"\nComplete metrics:")
        print(f"  c²: {health['c_squared']['latest']:.2f}")
        print(f"  Ξ:  {health['balance_operator']['latest']:.4f}")
        print(f"  Duty: {health['duty_cycle']['latest']:.3f}")

    @pytest.mark.asyncio
    async def test_collapse_detection(self, memory):
        """Test collapse detection in stats"""
        # Create some hierarchies
        for i in range(5):
            parent_id = await memory.store(
                content=f"Test parent {i}",
                graph="test",
                node_type=NodeType.CONCEPT,
            )

            await memory.store(
                content=f"Test child {i}",
                graph="test",
                node_type=NodeType.CONCEPT,
                parent_id=parent_id,
            )

        # Check collapse count
        stats = await memory.get_stats()
        collapses = stats["collapses"]

        print(f"\nCollapse triggers detected: {collapses}")

        # Collapse count should be tracked (may be 0 if no collapses)
        assert isinstance(collapses, int)
        assert collapses >= 0

    @pytest.mark.asyncio
    async def test_sec_resonance_in_query(self, memory):
        """Test that queries use SEC resonance ranking"""
        # Store some nodes at different depths
        root_id = await memory.store(
            content="Database systems",
            graph="test",
            node_type=NodeType.CONCEPT,
        )

        child_id = await memory.store(
            content="Relational databases use SQL",
            graph="test",
            node_type=NodeType.CONCEPT,
            parent_id=root_id,
        )

        grandchild_id = await memory.store(
            content="MySQL is a relational database",
            graph="test",
            node_type=NodeType.CONCEPT,
            parent_id=child_id,
        )

        # Query should rank by resonance
        results = await memory.query(
            query_text="database",
            graphs=["test"],
            limit=10,
        )

        assert len(results) > 0

        # Results should have resonance scores
        for i, result in enumerate(results):
            print(f"\n  [{i}] {result.node.content[:50]}")
            print(f"      Score: {result.score:.4f}")
            print(f"      Similarity: {result.similarity:.4f}")
            print(f"      Depth: {result.node.path.count('/') if result.node.path else 0}")

    @pytest.mark.asyncio
    async def test_health_in_stats(self, memory):
        """Test that foundation health appears in stats"""
        # Store a hierarchy
        parent_id = await memory.store(
            content="Stats test",
            graph="test",
            node_type=NodeType.CONCEPT,
        )

        await memory.store(
            content="Stats child",
            graph="test",
            node_type=NodeType.CONCEPT,
            parent_id=parent_id,
        )

        # Get stats
        stats = await memory.get_stats()

        # Should have foundation_health
        assert "foundation_health" in stats
        assert "c_squared" in stats["foundation_health"]
        assert "balance_operator" in stats["foundation_health"]
        assert "duty_cycle" in stats["foundation_health"]
        assert "constants" in stats["foundation_health"]

        print(f"\nFoundation health in stats:")
        print(f"  {stats['foundation_health']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
