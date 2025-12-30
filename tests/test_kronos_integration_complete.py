"""
Test complete integration of foundations with KronosMemory.

Validates:
- Conservation tracking during storage
- Health metrics collection
- c² measurement
- Balance operator monitoring
"""

import pytest
import pytest_asyncio
from pathlib import Path

from fracton.storage.kronos_memory import KronosMemory, NodeType


@pytest_asyncio.fixture
async def memory(tmp_path):
    """Create KronosMemory instance with foundations."""
    mem = KronosMemory(
        storage_path=tmp_path,
        namespace="test",
        device="cpu",
        embedding_dim=128,  # Smaller for speed
        embedding_model="mini",
    )
    await mem.connect()
    await mem.create_graph("test")
    yield mem
    await mem.close()


class TestFoundationIntegration:
    """Test foundation integration with KronosMemory."""

    @pytest.mark.asyncio
    async def test_conservation_tracking(self, memory):
        """Test that conservation is tracked during storage"""
        # Store parent
        parent_id = await memory.store(
            content="Parent node for conservation test",
            graph="test",
            node_type=NodeType.CONCEPT,
        )

        # Store children
        child1_id = await memory.store(
            content="First child node",
            graph="test",
            node_type=NodeType.CONCEPT,
            parent_id=parent_id,
        )

        child2_id = await memory.store(
            content="Second child node",
            graph="test",
            node_type=NodeType.CONCEPT,
            parent_id=parent_id,
        )

        # Check health metrics
        health = memory.get_foundation_health()

        # Should have c² measurements
        assert health["c_squared"]["count"] > 0
        assert health["c_squared"]["latest"] is not None

        # Should have constants
        assert health["constants"]["phi"] is not None
        assert health["constants"]["xi"] is not None

        print(f"\nHealth Metrics:")
        print(f"  c²: {health['c_squared']}")
        print(f"  constants: {health['constants']}")

    @pytest.mark.asyncio
    async def test_health_metrics_in_stats(self, memory):
        """Test health metrics appear in get_stats()"""
        # Store some nodes
        await memory.store(
            content="Test node 1",
            graph="test",
            node_type=NodeType.CONCEPT,
        )

        await memory.store(
            content="Test node 2",
            graph="test",
            node_type=NodeType.CONCEPT,
        )

        # Get stats
        stats = await memory.get_stats()

        # Should include foundation_health
        assert "foundation_health" in stats
        assert "c_squared" in stats["foundation_health"]
        assert "constants" in stats["foundation_health"]

        print(f"\nStats:")
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Foundation health: {stats['foundation_health']}")

    @pytest.mark.asyncio
    async def test_c_squared_measurement(self, memory):
        """Test c² (model constant) measurement"""
        # Create hierarchy
        root_id = await memory.store(
            content="Machine learning is a subset of AI",
            graph="test",
            node_type=NodeType.CONCEPT,
        )

        child1_id = await memory.store(
            content="Supervised learning uses labeled data",
            graph="test",
            node_type=NodeType.CONCEPT,
            parent_id=root_id,
        )

        child2_id = await memory.store(
            content="Unsupervised learning finds patterns",
            graph="test",
            node_type=NodeType.CONCEPT,
            parent_id=root_id,
        )

        # Get c² measurement
        health = memory.get_foundation_health()
        c_squared = health["c_squared"]

        assert c_squared["count"] > 0
        assert c_squared["latest"] > 0

        # For real embeddings, c² should be > 1
        # For hash-based, c² ≈ 1
        print(f"\nModel constant c²: {c_squared['latest']:.2f}")
        print(f"  Mean: {c_squared['mean']:.2f}")
        print(f"  Std: {c_squared['std']:.2f}")

    @pytest.mark.asyncio
    async def test_multiple_hierarchies(self, memory):
        """Test health metrics across multiple hierarchies"""
        # Create multiple parent-child relationships
        for i in range(5):
            parent_id = await memory.store(
                content=f"Parent concept {i}",
                graph="test",
                node_type=NodeType.CONCEPT,
            )

            for j in range(2):
                await memory.store(
                    content=f"Child {j} of parent {i}",
                    graph="test",
                    node_type=NodeType.CONCEPT,
                    parent_id=parent_id,
                )

        # Should have 5 c² measurements (one per parent with children)
        health = memory.get_foundation_health()
        assert health["c_squared"]["count"] >= 5

        print(f"\nMultiple hierarchies:")
        print(f"  c² measurements: {health['c_squared']['count']}")
        print(f"  c² range: [{health['c_squared']['min']:.2f}, {health['c_squared']['max']:.2f}]")

    @pytest.mark.asyncio
    async def test_foundation_initialization(self, memory):
        """Test that foundation is properly initialized"""
        assert memory.foundation is not None
        assert memory.foundation.pac_engine is not None
        assert memory.foundation.sec_operators is not None
        assert memory.foundation.med_validator is not None
        assert memory.foundation.distance_validator is not None

        # Check constants
        constants = memory.foundation.constants
        assert abs(constants.PHI - 1.618033988749895) < 1e-10
        assert abs(constants.XI - 1.0571238898) < 1e-5  # π/55 precision

        print(f"\nFoundation constants:")
        print(f"  PHI: {constants.PHI}")
        print(f"  XI: {constants.XI}")
        print(f"  LAMBDA_STAR: {constants.LAMBDA_STAR}")
        print(f"  DUTY_CYCLE: {constants.DUTY_CYCLE}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
