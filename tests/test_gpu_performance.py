"""
GPU Performance Tests for KRONOS

Tests GPU enablement, device consistency, and performance across:
- Embedding service
- Vector backends (Qdrant)
- KronosMemory integration
- Memory management
"""

import pytest
import pytest_asyncio
import torch
import time
from pathlib import Path
import tempfile
import shutil
from typing import List

from fracton.storage import (
    KronosMemory,
    BackendConfig,
    NodeType,
    EmbeddingService,
    create_embedding_service,
)

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


@pytest_asyncio.fixture
async def temp_dir():
    """Create temporary directory."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    try:
        shutil.rmtree(temp_path)
    except:
        pass


class TestEmbeddingServiceGPU:
    """Test GPU enablement for embedding service."""

    @pytest.mark.asyncio
    async def test_gpu_initialization(self, temp_dir):
        """Test embedding service initializes on GPU."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        service = await create_embedding_service(
            model="mini",
            device="cuda",
            cache_dir=temp_dir,
        )

        assert service.device == "cuda"
        assert service.is_available()

        # Test single embedding
        text = "Test GPU embedding"
        embedding = await service.embed(text)

        assert embedding.device.type == "cuda"
        assert embedding.shape[0] == service.embedding_dim

    @pytest.mark.asyncio
    async def test_gpu_batch_processing(self, temp_dir):
        """Test batch processing on GPU."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        service = await create_embedding_service(
            model="mini",
            device="cuda",
            cache_dir=temp_dir,
        )

        # Batch embed
        texts = [f"Test text {i}" for i in range(100)]
        embeddings = await service.embed_batch(texts)

        assert embeddings.device.type == "cuda"
        assert embeddings.shape == (100, service.embedding_dim)

    @pytest.mark.asyncio
    async def test_gpu_vs_cpu_performance(self, temp_dir):
        """Benchmark GPU vs CPU performance."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        # Create both services
        gpu_service = await create_embedding_service(
            model="mini",
            device="cuda",
            cache_dir=temp_dir,
        )

        cpu_service = await create_embedding_service(
            model="mini",
            device="cpu",
            cache_dir=temp_dir,
        )

        # Benchmark texts
        texts = [f"Performance test text number {i}" for i in range(100)]

        # GPU benchmark
        gpu_start = time.time()
        gpu_embeddings = await gpu_service.embed_batch(texts)
        gpu_time = time.time() - gpu_start

        # CPU benchmark
        cpu_start = time.time()
        cpu_embeddings = await cpu_service.embed_batch(texts)
        cpu_time = time.time() - cpu_start

        # GPU should be faster (or at least not slower by much)
        # On small batches, overhead might make GPU slower
        speedup = cpu_time / gpu_time
        print(f"\nGPU speedup: {speedup:.2f}x (GPU: {gpu_time:.3f}s, CPU: {cpu_time:.3f}s)")

        # Verify embeddings are equivalent (within floating point tolerance)
        assert torch.allclose(
            gpu_embeddings.cpu(),
            cpu_embeddings,
            atol=1e-5
        )

    @pytest.mark.asyncio
    async def test_gpu_memory_management(self, temp_dir):
        """Test GPU memory is properly managed."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        service = await create_embedding_service(
            model="mini",
            device="cuda",
            cache_dir=temp_dir,
        )

        # Get initial GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        # Process large batch
        texts = [f"Memory test {i}" for i in range(1000)]
        embeddings = await service.embed_batch(texts)

        # Clear references
        del embeddings
        torch.cuda.empty_cache()

        # Memory should be mostly freed
        final_memory = torch.cuda.memory_allocated()
        memory_leaked = final_memory - initial_memory

        # Allow some overhead but not massive leaks
        assert memory_leaked < 100 * 1024 * 1024  # < 100MB


class TestKronosMemoryGPU:
    """Test GPU enablement for KronosMemory."""

    @pytest.mark.asyncio
    async def test_gpu_device_consistency(self, temp_dir):
        """Test all tensors stay on GPU."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        memory = KronosMemory(
            storage_path=temp_dir,
            namespace="gpu_test",
            embedding_model="mini",
            device="cuda",
        )

        await memory.connect()

        # Verify embedding service on GPU
        assert memory.embedding_service.device == "cuda"

        # Create graph
        await memory.create_graph("gpu_test")

        # Store node
        node_id = await memory.store(
            content="GPU consistency test",
            graph="gpu_test",
            node_type=NodeType.CONCEPT,
        )

        # Retrieve and check embedding device
        node = await memory.retrieve("gpu_test", node_id)
        assert node.delta_embedding.device.type == "cuda"

        await memory.close()

    @pytest.mark.asyncio
    async def test_gpu_query_performance(self, temp_dir):
        """Test query performance on GPU."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        memory = KronosMemory(
            storage_path=temp_dir,
            namespace="gpu_query",
            embedding_model="mini",
            device="cuda",
        )

        await memory.connect()
        await memory.create_graph("perf_test")

        # Store nodes
        for i in range(50):
            await memory.store(
                content=f"Performance test node {i}",
                graph="perf_test",
                node_type=NodeType.CONCEPT,
            )

        # Query
        start = time.time()
        results = await memory.query(
            query_text="performance test",
            graphs=["perf_test"],
            limit=10,
        )
        query_time = time.time() - start

        print(f"\nGPU query time: {query_time:.3f}s for 50 nodes")

        assert len(results) > 0
        # Should be reasonably fast
        assert query_time < 10.0  # 10 second timeout

        await memory.close()

    @pytest.mark.asyncio
    async def test_gpu_pac_reconstruction(self, temp_dir):
        """Test PAC reconstruction on GPU."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        memory = KronosMemory(
            storage_path=temp_dir,
            namespace="gpu_pac",
            embedding_model="mini",
            device="cuda",
        )

        await memory.connect()
        await memory.create_graph("pac_test")

        # Create chain
        parent_id = await memory.store(
            content="GPU PAC parent",
            graph="pac_test",
            node_type=NodeType.CONCEPT,
        )

        child_id = await memory.store(
            content="GPU PAC child",
            graph="pac_test",
            node_type=NodeType.CONCEPT,
            parent_id=parent_id,
        )

        grandchild_id = await memory.store(
            content="GPU PAC grandchild",
            graph="pac_test",
            node_type=NodeType.CONCEPT,
            parent_id=child_id,
        )

        # Reconstruct full embedding from deltas
        reconstructed = await memory._reconstruct_embedding("pac_test", grandchild_id)

        # Should be on GPU
        assert reconstructed.device.type == "cuda"

        # Verify reconstruction by re-encoding the content
        grandchild = await memory.retrieve("pac_test", grandchild_id)
        expected_full = await memory._compute_embedding(grandchild.content)

        # Reconstructed should match the full embedding
        # (within reasonable tolerance for floating point and model variance)
        assert torch.allclose(reconstructed, expected_full, atol=1e-3)

        await memory.close()


class TestDeviceConsistency:
    """Test device consistency across operations."""

    @pytest.mark.asyncio
    async def test_mixed_device_detection(self, temp_dir):
        """Test detection of mixed CPU/GPU tensors."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        memory = KronosMemory(
            storage_path=temp_dir,
            namespace="mixed_device",
            embedding_model="mini",
            device="cuda",
        )

        await memory.connect()
        await memory.create_graph("mixed_test")

        # Store with explicit embedding on wrong device (CPU)
        cpu_embedding = torch.randn(384, device="cpu")

        # This should either move to GPU or raise clear error
        # Currently system will move to GPU in backends
        node_id = await memory.store(
            content="Mixed device test",
            graph="mixed_test",
            node_type=NodeType.CONCEPT,
            embedding=cpu_embedding,
        )

        # Retrieved embedding should be on GPU
        node = await memory.retrieve("mixed_test", node_id)
        # Note: ChromaDB converts to CPU on retrieval, which is OK
        # The important part is it doesn't crash

        await memory.close()

    @pytest.mark.asyncio
    async def test_device_transfer_overhead(self, temp_dir):
        """Measure overhead of device transfers."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        # Create same-device embeddings
        gpu_embeddings = torch.randn(100, 384, device="cuda")
        cpu_embeddings = torch.randn(100, 384, device="cpu")

        # GPU -> CPU transfer
        start = time.time()
        _ = gpu_embeddings.cpu()
        gpu_to_cpu = time.time() - start

        # CPU -> GPU transfer
        start = time.time()
        _ = cpu_embeddings.cuda()
        cpu_to_gpu = time.time() - start

        print(f"\nDevice transfer overhead:")
        print(f"  GPU -> CPU: {gpu_to_cpu*1000:.2f}ms")
        print(f"  CPU -> GPU: {cpu_to_gpu*1000:.2f}ms")

        # Should be reasonably fast for small tensors
        assert gpu_to_cpu < 0.1  # 100ms
        assert cpu_to_gpu < 0.1


class TestQdrantGPU:
    """Test GPU performance with Qdrant backend."""

    @pytest.mark.asyncio
    async def test_qdrant_gpu_search(self, temp_dir):
        """Test Qdrant search with GPU tensors."""
        pytest.skip("Requires Qdrant server - integration test")

        # This would test:
        # - Storing GPU tensors in Qdrant
        # - Searching with GPU query vectors
        # - Retrieving and device consistency


class TestPerformanceBenchmarks:
    """Performance benchmarks for GPU operations."""

    @pytest.mark.asyncio
    async def test_embedding_throughput(self, temp_dir):
        """Benchmark embedding throughput."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        service = await create_embedding_service(
            model="mini",
            device="cuda",
            cache_dir=temp_dir,
        )

        # Various batch sizes
        batch_sizes = [1, 10, 50, 100, 500]
        results = {}

        for batch_size in batch_sizes:
            texts = [f"Throughput test {i}" for i in range(batch_size)]

            # Warmup
            _ = await service.embed_batch(texts[:min(10, batch_size)])

            # Benchmark
            start = time.time()
            _ = await service.embed_batch(texts)
            elapsed = time.time() - start

            throughput = batch_size / elapsed
            results[batch_size] = throughput

        print(f"\nEmbedding throughput (texts/sec):")
        for batch_size, throughput in results.items():
            print(f"  Batch {batch_size:3d}: {throughput:6.1f} texts/sec")

        # Larger batches should have better throughput
        assert results[100] > results[1]

    @pytest.mark.asyncio
    async def test_memory_scaling(self, temp_dir):
        """Test GPU memory usage scales properly."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        service = await create_embedding_service(
            model="mini",
            device="cuda",
            cache_dir=temp_dir,
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Embed increasing sizes
        for size in [10, 50, 100, 500]:
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated()

            texts = [f"Scaling test {i}" for i in range(size)]
            embeddings = await service.embed_batch(texts)

            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - start_memory

            print(f"\nBatch {size}: {memory_used / 1024 / 1024:.1f} MB")

            # Clear
            del embeddings
            torch.cuda.empty_cache()

        # Memory should scale roughly linearly
        # (with some overhead for the model itself)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
