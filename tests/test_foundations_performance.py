"""
Performance profiling tests for PAC/SEC/MED foundations.

Benchmarks:
- Computation speed
- Memory usage
- Scalability
- GPU performance (if available)
"""

import pytest
import torch
import time
import tracemalloc
from typing import List

from fracton.storage.pac_engine import (
    PACConservationEngine,
    PACNode,
    PACConstants,
)
from fracton.storage.sec_operators import SECOperators
from fracton.storage.med_validator import MEDValidator
from fracton.storage.distance_validator import DistanceValidator
from fracton.storage.foundation_integration import FoundationIntegration


class TestPACPerformance:
    """Performance tests for PAC engine."""

    def test_potential_computation_speed(self, benchmark):
        """Benchmark potential computation speed"""
        engine = PACConservationEngine()

        def compute_potentials():
            return [engine.compute_potential(k) for k in range(100)]

        result = benchmark(compute_potentials)
        assert len(result) == 100

    def test_fibonacci_verification_speed(self, benchmark):
        """Benchmark Fibonacci verification speed"""
        engine = PACConservationEngine()

        potentials = [(
            engine.compute_potential(k),
            engine.compute_potential(k + 1),
            engine.compute_potential(k + 2),
        ) for k in range(100)]

        def verify_all():
            return [
                engine.verify_fibonacci_recursion(p, c1, c2)
                for p, c1, c2 in potentials
            ]

        results = benchmark(verify_all)
        assert len(results) == 100

    def test_value_conservation_speed(self, benchmark):
        """Benchmark value conservation verification"""
        engine = PACConservationEngine()

        parent = torch.randn(384)
        children = [torch.randn(384) for _ in range(10)]

        result = benchmark(
            engine.verify_value_conservation,
            parent,
            children,
        )

        assert isinstance(result[0], bool)

    def test_full_conservation_speed(self, benchmark):
        """Benchmark full 3D conservation verification"""
        engine = PACConservationEngine()

        parent = PACNode(
            value_embedding=torch.randn(384),
            complexity_vector=torch.randn(384),
            effect_cone=torch.randn(384),
            potential=1.0,
            depth=0,
        )

        children = [
            PACNode(
                value_embedding=torch.randn(384),
                complexity_vector=torch.randn(384),
                effect_cone=torch.randn(384),
                potential=0.618,
                depth=1,
            ),
            PACNode(
                value_embedding=torch.randn(384),
                complexity_vector=torch.randn(384),
                effect_cone=torch.randn(384),
                potential=0.382,
                depth=2,
            ),
        ]

        result = benchmark(
            engine.verify_full_conservation,
            parent,
            children,
        )

        assert "fibonacci" in result

    def test_balance_operator_speed(self, benchmark):
        """Benchmark balance operator computation"""
        engine = PACConservationEngine()

        parent = PACNode(
            value_embedding=torch.randn(384),
            complexity_vector=torch.randn(384),
            effect_cone=torch.randn(384),
            potential=1.0,
            depth=0,
        )

        children = [
            PACNode(
                value_embedding=torch.randn(384),
                complexity_vector=torch.randn(384),
                effect_cone=torch.randn(384),
                potential=0.6,
                depth=1,
            ),
        ]

        result = benchmark(
            engine.compute_balance_operator,
            parent,
            children,
        )

        assert isinstance(result, float)


class TestSECPerformance:
    """Performance tests for SEC operators."""

    def test_merge_speed(self, benchmark):
        """Benchmark merge operator speed"""
        operators = SECOperators()

        node1 = PACNode(
            value_embedding=torch.randn(384),
            complexity_vector=torch.randn(384),
            effect_cone=torch.randn(384),
            potential=1.0,
            depth=0,
        )

        node2 = PACNode(
            value_embedding=torch.randn(384),
            complexity_vector=torch.randn(384),
            effect_cone=torch.randn(384),
            potential=1.0,
            depth=0,
        )

        merged, sec_state = benchmark(operators.merge, node1, node2)

        assert merged.potential == 2.0

    def test_branch_speed(self, benchmark):
        """Benchmark branch operator speed"""
        operators = SECOperators()

        parent = PACNode(
            value_embedding=torch.randn(384),
            complexity_vector=torch.randn(384),
            effect_cone=torch.randn(384),
            potential=1.0,
            depth=0,
        )

        context = torch.randn(384)

        child1, child2 = benchmark(operators.branch, parent, context)

        assert child1.depth == 1
        assert child2.depth == 2

    def test_gradient_detection_speed(self, benchmark):
        """Benchmark gradient detection speed"""
        operators = SECOperators()

        node = PACNode(
            value_embedding=torch.randn(384),
            complexity_vector=torch.randn(384),
            effect_cone=torch.randn(384),
            potential=1.0,
            depth=0,
        )

        neighbors = [
            PACNode(
                value_embedding=torch.randn(384),
                complexity_vector=torch.randn(384),
                effect_cone=torch.randn(384),
                potential=1.0,
                depth=0,
            )
            for _ in range(10)
        ]

        magnitude, direction = benchmark(
            operators.detect_gradient,
            node,
            neighbors,
        )

        assert magnitude >= 0

    def test_duty_cycle_speed(self, benchmark):
        """Benchmark duty cycle computation"""
        operators = SECOperators()

        history = ["attraction", "repulsion"] * 500

        result = benchmark(operators.compute_duty_cycle, history)

        assert 0 <= result <= 1


class TestMEDPerformance:
    """Performance tests for MED validator."""

    def test_depth_validation_speed(self, benchmark):
        """Benchmark depth validation speed"""
        validator = MEDValidator(strict_mode=False)

        nodes = [
            PACNode(
                value_embedding=torch.randn(384),
                complexity_vector=torch.randn(384),
                effect_cone=torch.randn(384),
                potential=1.0,
                depth=i,
            )
            for i in range(10)
        ]

        result = benchmark(
            validator.validate_depth,
            nodes,
            "benchmark",
        )

        assert isinstance(result, bool)

    def test_node_count_validation_speed(self, benchmark):
        """Benchmark node count validation speed"""
        validator = MEDValidator(strict_mode=False)

        nodes = [
            PACNode(
                value_embedding=torch.randn(384),
                complexity_vector=torch.randn(384),
                effect_cone=torch.randn(384),
                potential=1.0,
                depth=0,
            )
            for _ in range(10)
        ]

        result = benchmark(
            validator.validate_node_count,
            nodes,
            "benchmark",
        )

        assert isinstance(result, bool)

    def test_full_validation_speed(self, benchmark):
        """Benchmark full MED validation speed"""
        validator = MEDValidator(strict_mode=False)

        nodes = [
            PACNode(
                value_embedding=torch.randn(384),
                complexity_vector=torch.randn(384),
                effect_cone=torch.randn(384),
                potential=1.0,
                depth=i % 2,
            )
            for i in range(3)
        ]

        result = benchmark(
            validator.validate_structure,
            nodes,
            "benchmark",
        )

        assert isinstance(result, bool)


class TestDistancePerformance:
    """Performance tests for distance validator."""

    def test_energy_conservation_speed(self, benchmark):
        """Benchmark energy conservation validation"""
        validator = DistanceValidator()

        parent = torch.randn(384)
        children = [torch.randn(384) for _ in range(10)]

        metrics = benchmark(
            validator.validate_energy_conservation,
            parent,
            children,
        )

        assert metrics.parent_energy >= 0

    def test_fractal_dimension_speed(self, benchmark):
        """Benchmark fractal dimension computation"""
        validator = DistanceValidator()

        embeddings_by_level = [
            [torch.randn(384) for _ in range(2**i)]
            for i in range(5)
        ]

        metrics = benchmark(
            validator.compute_fractal_dimension,
            embeddings_by_level,
        )

        assert metrics.depth == 5

    def test_amplification_measurement_speed(self, benchmark):
        """Benchmark amplification measurement"""
        validator = DistanceValidator()

        parent = torch.randn(384)
        children = [torch.randn(384) for _ in range(10)]

        amplification, interpretation = benchmark(
            validator.measure_amplification,
            parent,
            children,
        )

        assert isinstance(amplification, float)


class TestIntegrationPerformance:
    """Performance tests for integration layer."""

    def test_node_creation_speed(self, benchmark):
        """Benchmark PAC node creation"""
        integration = FoundationIntegration(embedding_dim=384)

        embedding = torch.randn(384)

        node = benchmark(
            integration.create_pac_node_from_embedding,
            embedding=embedding,
            content="Test node",
            depth=0,
        )

        assert node.depth == 0

    def test_conservation_verification_speed(self, benchmark):
        """Benchmark full conservation verification"""
        integration = FoundationIntegration(embedding_dim=384)

        parent = PACNode(
            value_embedding=torch.randn(384),
            complexity_vector=torch.randn(384),
            effect_cone=torch.randn(384),
            potential=1.0,
            depth=0,
        )

        children = [
            PACNode(
                value_embedding=torch.randn(384),
                complexity_vector=torch.randn(384),
                effect_cone=torch.randn(384),
                potential=0.618,
                depth=1,
            ),
            PACNode(
                value_embedding=torch.randn(384),
                complexity_vector=torch.randn(384),
                effect_cone=torch.randn(384),
                potential=0.382,
                depth=2,
            ),
        ]

        metrics = benchmark(
            integration.verify_conservation,
            parent,
            children,
        )

        assert metrics.balance_operator > 0


class TestScalability:
    """Test scalability with increasing sizes."""

    def test_embedding_dimension_scaling(self):
        """Test performance across embedding dimensions"""
        engine = PACConservationEngine()

        results = {}
        for dim in [64, 128, 256, 512, 1024, 2048]:
            parent = torch.randn(dim)
            children = [torch.randn(dim) for _ in range(2)]

            start = time.time()
            for _ in range(100):
                engine.verify_value_conservation(parent, children)
            elapsed = time.time() - start

            results[dim] = elapsed

        # Should scale roughly linearly
        print(f"\nEmbedding dimension scaling:")
        for dim, elapsed in results.items():
            print(f"  {dim:4d}D: {elapsed:.4f}s")

    def test_node_count_scaling(self):
        """Test performance with increasing node counts"""
        validator = MEDValidator(strict_mode=False)

        results = {}
        for count in [1, 5, 10, 50, 100, 500]:
            nodes = [
                PACNode(
                    value_embedding=torch.randn(384),
                    complexity_vector=torch.randn(384),
                    effect_cone=torch.randn(384),
                    potential=1.0,
                    depth=0,
                )
                for _ in range(count)
            ]

            start = time.time()
            for _ in range(100):
                validator.validate_structure(nodes)
            elapsed = time.time() - start

            results[count] = elapsed

        print(f"\nNode count scaling:")
        for count, elapsed in results.items():
            print(f"  {count:3d} nodes: {elapsed:.4f}s")

    def test_hierarchy_depth_scaling(self):
        """Test performance with increasing hierarchy depth"""
        engine = PACConservationEngine()

        results = {}
        for depth in [10, 50, 100, 200, 500]:
            start = time.time()
            for k in range(depth):
                parent = engine.compute_potential(k)
                child1 = engine.compute_potential(k + 1)
                child2 = engine.compute_potential(k + 2)
                engine.verify_fibonacci_recursion(parent, child1, child2)
            elapsed = time.time() - start

            results[depth] = elapsed

        print(f"\nHierarchy depth scaling:")
        for depth, elapsed in results.items():
            print(f"  {depth:3d} levels: {elapsed:.4f}s")


class TestMemoryUsage:
    """Test memory consumption."""

    def test_pac_node_memory(self):
        """Test memory usage of PAC nodes"""
        tracemalloc.start()

        nodes = []
        for i in range(1000):
            nodes.append(
                PACNode(
                    value_embedding=torch.randn(384),
                    complexity_vector=torch.randn(384),
                    effect_cone=torch.randn(384),
                    potential=1.0,
                    depth=i,
                )
            )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\n1000 PAC nodes memory:")
        print(f"  Current: {current / 1024 / 1024:.2f} MB")
        print(f"  Peak: {peak / 1024 / 1024:.2f} MB")
        print(f"  Per node: {peak / 1000 / 1024:.2f} KB")

    def test_integration_memory(self):
        """Test memory usage of integration layer"""
        tracemalloc.start()

        integration = FoundationIntegration(embedding_dim=384)

        # Create many nodes
        for i in range(1000):
            embedding = torch.randn(384)
            integration.create_pac_node_from_embedding(
                embedding=embedding,
                content=f"Node {i}",
                depth=i % 10,
            )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\nIntegration layer memory:")
        print(f"  Current: {current / 1024 / 1024:.2f} MB")
        print(f"  Peak: {peak / 1024 / 1024:.2f} MB")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUPerformance:
    """GPU performance tests."""

    def test_conservation_cpu_vs_gpu(self):
        """Compare CPU vs GPU performance"""
        # CPU
        engine_cpu = PACConservationEngine(device="cpu")
        parent_cpu = torch.randn(384)
        children_cpu = [torch.randn(384) for _ in range(2)]

        start = time.time()
        for _ in range(100):
            engine_cpu.verify_value_conservation(parent_cpu, children_cpu)
        cpu_time = time.time() - start

        # GPU
        engine_gpu = PACConservationEngine(device="cuda")
        parent_gpu = torch.randn(384, device="cuda")
        children_gpu = [torch.randn(384, device="cuda") for _ in range(2)]

        start = time.time()
        for _ in range(100):
            engine_gpu.verify_value_conservation(parent_gpu, children_gpu)
        gpu_time = time.time() - start

        print(f"\nCPU vs GPU (100 iterations):")
        print(f"  CPU: {cpu_time:.4f}s")
        print(f"  GPU: {gpu_time:.4f}s")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
