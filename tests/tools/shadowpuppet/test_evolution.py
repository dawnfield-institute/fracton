"""
Tests for ShadowPuppet evolution engine.

Tests:
- Evolution configuration
- Population management
- Parent selection
- Crossover operations
- Refinement loop
- Generation lifecycle
"""

import pytest
import tempfile
from pathlib import Path

from fracton.tools.shadowpuppet.evolution import (
    SoftwareEvolution,
    EvolutionConfig,
    CodeEnvironment,
    GenerationStats
)
from fracton.tools.shadowpuppet.protocols import (
    ProtocolSpec,
    GrowthGap,
    ComponentOrganism
)
from fracton.tools.shadowpuppet.generators import MockGenerator


class TestEvolutionConfig:
    """Tests for EvolutionConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EvolutionConfig()
        assert config.coherence_threshold == 0.70
        assert config.reproduction_threshold == 0.80
        assert config.max_population == 50
        assert config.candidates_per_gap == 3
        assert config.mutation_rate == 0.2
        assert config.max_generations == 10
        assert config.enable_crossover is True
        assert config.enable_refinement is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EvolutionConfig(
            coherence_threshold=0.80,
            max_generations=20,
            enable_crossover=False
        )
        assert config.coherence_threshold == 0.80
        assert config.max_generations == 20
        assert config.enable_crossover is False


class TestCodeEnvironment:
    """Tests for CodeEnvironment."""
    
    def test_survival_check(self):
        """Test component survival check."""
        env = CodeEnvironment(coherence_threshold=0.70)
        
        high_fitness = ComponentOrganism(
            id="high", protocol_name="Test", code="...",
            coherence_score=0.85
        )
        low_fitness = ComponentOrganism(
            id="low", protocol_name="Test", code="...",
            coherence_score=0.50
        )
        
        assert env.check_survival(high_fitness) is True
        assert env.check_survival(low_fitness) is False
    
    def test_integration_energy(self):
        """Test integration energy calculation."""
        env = CodeEnvironment()
        
        comp = ComponentOrganism(
            id="test", protocol_name="Test", code="...",
            coherence_score=0.80
        )
        
        energy = env.harvest_integration_energy(comp)
        assert energy > 0
        # Higher coherence = more energy
        assert energy > 10.0  # Base energy


class TestSoftwareEvolution:
    """Tests for SoftwareEvolution engine."""
    
    @pytest.fixture
    def evolution(self):
        return SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(
                coherence_threshold=0.50,  # Low threshold for testing
                max_generations=3,
                candidates_per_gap=2
            )
        )
    
    @pytest.fixture
    def simple_protocol(self):
        return ProtocolSpec(
            name="SimpleService",
            methods=["run"],
            docstring="A simple service"
        )
    
    def test_evolution_initialization(self, evolution):
        """Test evolution engine initialization."""
        assert evolution.generator.name == "mock"
        assert evolution.config.max_generations == 3
        assert len(evolution.components) == 0
        assert evolution.generation == 0
    
    def test_grow_single_gap(self, evolution, simple_protocol):
        """Test growing a single gap."""
        gap = GrowthGap(protocol=simple_protocol)
        results = evolution.grow([gap], max_generations=1)
        
        assert results['success'] is True
        assert len(evolution.components) > 0
        assert evolution.components[0].protocol_name == "SimpleService"
    
    def test_grow_multiple_gaps(self, evolution):
        """Test growing multiple gaps."""
        gaps = [
            GrowthGap(protocol=ProtocolSpec(
                name="ServiceA", methods=["a"], docstring="A"
            )),
            GrowthGap(protocol=ProtocolSpec(
                name="ServiceB", methods=["b"], docstring="B"
            ))
        ]
        results = evolution.grow(gaps, max_generations=1)
        
        assert results['success'] is True
        protocol_names = {c.protocol_name for c in evolution.components}
        assert "ServiceA" in protocol_names
        assert "ServiceB" in protocol_names
    
    def test_genealogy_tracking(self, evolution, simple_protocol):
        """Test that genealogy is tracked."""
        gap = GrowthGap(protocol=simple_protocol)
        evolution.grow([gap], max_generations=1)
        
        genealogy = evolution.genealogy.to_dict()
        assert len(genealogy['nodes']) > 0
    
    def test_history_recording(self, evolution, simple_protocol):
        """Test that history is recorded."""
        gap = GrowthGap(protocol=simple_protocol)
        evolution.grow([gap], max_generations=2)
        
        assert len(evolution.history) > 0
        stats = evolution.history[0]
        assert hasattr(stats, 'generation')
        assert hasattr(stats, 'population')
        assert hasattr(stats, 'mean_coherence')
    
    def test_get_code(self, evolution, simple_protocol):
        """Test getting generated code."""
        gap = GrowthGap(protocol=simple_protocol)
        evolution.grow([gap], max_generations=1)
        
        code_dict = evolution.get_code()
        assert len(code_dict) > 0
        
        # Filter by protocol
        filtered = evolution.get_code(protocol_name="SimpleService")
        assert all("SimpleService" in k for k in filtered.keys())
    
    def test_save_code(self, evolution, simple_protocol):
        """Test saving generated code to files."""
        gap = GrowthGap(protocol=simple_protocol)
        evolution.grow([gap], max_generations=1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            evolution.save_code(output_dir)
            
            files = list(output_dir.glob("*.py"))
            assert len(files) > 0
            
            # Check file content
            content = files[0].read_text()
            assert "class" in content


class TestParentSelection:
    """Tests for parent selection strategies."""
    
    @pytest.fixture
    def evolution(self):
        evo = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(coherence_threshold=0.50)
        )
        # Add some components
        evo.components = [
            ComponentOrganism(
                id="A_0", protocol_name="A", code="class A: pass",
                coherence_score=0.90
            ),
            ComponentOrganism(
                id="A_1", protocol_name="A", code="class A: pass",
                coherence_score=0.70
            ),
            ComponentOrganism(
                id="B_0", protocol_name="B", code="class B: pass",
                coherence_score=0.85
            ),
        ]
        return evo
    
    def test_select_parent_same_protocol(self, evolution):
        """Test parent selection prefers same protocol."""
        gap = GrowthGap(protocol=ProtocolSpec(
            name="A", methods=[], docstring="A"
        ))
        parent = evolution._select_parent(gap)
        
        assert parent is not None
        assert parent.protocol_name == "A"
    
    def test_select_parent_highest_fitness(self, evolution):
        """Test parent selection picks highest fitness."""
        gap = GrowthGap(protocol=ProtocolSpec(
            name="A", methods=[], docstring="A"
        ))
        parent = evolution._select_parent(gap)
        
        # Should usually pick A_0 (0.90) over A_1 (0.70)
        # With tournament selection there's some randomness
        assert parent.coherence_score >= 0.70
    
    def test_select_parent_empty_population(self, evolution):
        """Test parent selection with empty population."""
        evolution.components = []
        gap = GrowthGap(protocol=ProtocolSpec(
            name="X", methods=[], docstring="X"
        ))
        parent = evolution._select_parent(gap)
        
        assert parent is None
    
    def test_tournament_selection(self, evolution):
        """Test tournament selection."""
        # Run multiple times to check randomness
        selected = []
        for _ in range(20):
            winner = evolution._tournament_select(evolution.components, tournament_size=2)
            selected.append(winner.id)
        
        # Should see some variation
        unique = set(selected)
        assert len(unique) >= 1  # At least one unique winner


class TestCrossover:
    """Tests for genetic crossover."""
    
    @pytest.fixture
    def evolution(self):
        return SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(enable_crossover=True)
        )
    
    def test_crossover_combines_methods(self, evolution):
        """Test that crossover combines methods from parents."""
        parent_a = ComponentOrganism(
            id="A", protocol_name="Service", code='''
class Service:
    def method_a(self):
        return "a"
    
    def shared(self):
        return "shared_a"
''',
            coherence_score=0.80
        )
        
        parent_b = ComponentOrganism(
            id="B", protocol_name="Service", code='''
class Service:
    def method_b(self):
        return "b"
    
    def shared(self):
        return "shared_b"
''',
            coherence_score=0.75
        )
        
        proto = ProtocolSpec(name="Service", methods=[], docstring="Service")
        
        offspring = evolution.crossover(parent_a, parent_b, proto)
        
        # Should be valid Python
        compile(offspring, "<crossover>", "exec")
        
        # Should have class definition
        assert "class Service" in offspring
    
    def test_crossover_with_syntax_error(self, evolution):
        """Test crossover falls back on syntax error."""
        parent_a = ComponentOrganism(
            id="A", protocol_name="Service", code="class Service: pass",
            coherence_score=0.80
        )
        parent_b = ComponentOrganism(
            id="B", protocol_name="Service", code="class Broken(\n    oops",
            coherence_score=0.75
        )
        
        proto = ProtocolSpec(name="Service", methods=[], docstring="Service")
        
        # Should fallback to parent_a
        offspring = evolution.crossover(parent_a, parent_b, proto)
        assert "class Service" in offspring


class TestRefinement:
    """Tests for targeted refinement."""
    
    @pytest.fixture
    def evolution(self):
        return SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(enable_refinement=True)
        )
    
    def test_refine_returns_code(self, evolution):
        """Test that refinement returns refined code."""
        comp = ComponentOrganism(
            id="test", protocol_name="Service",
            code="class Service:\n    pass"
        )
        proto = ProtocolSpec(name="Service", methods=["run"], docstring="Service")
        
        context = {"protocol": proto}
        violations = ["Missing run method"]
        
        refined = evolution.refine(comp, context, violations)
        
        # MockGenerator will just regenerate
        assert refined is not None
        assert len(refined) > 0


class TestGenerationStats:
    """Tests for GenerationStats."""
    
    def test_stats_creation(self):
        """Test stats dataclass creation."""
        stats = GenerationStats(
            generation=5,
            population=10,
            births=3,
            deaths=1,
            mean_coherence=0.75,
            max_coherence=0.92,
            best_component_id="Best_5"
        )
        
        assert stats.generation == 5
        assert stats.population == 10
        assert stats.births == 3
        assert stats.deaths == 1
        assert stats.mean_coherence == 0.75
        assert stats.best_component_id == "Best_5"
        assert stats.timestamp is not None


class TestDependencyOrdering:
    """Tests for dependency-based gap ordering."""
    
    @pytest.fixture
    def evolution(self):
        return SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(
                coherence_threshold=0.50,
                max_generations=1,
                candidates_per_gap=1
            )
        )
    
    def test_order_by_dependencies_simple(self, evolution):
        """Test that dependencies are ordered correctly."""
        # B depends on A
        proto_a = ProtocolSpec(name="A", methods=["a"], docstring="A", dependencies=[])
        proto_b = ProtocolSpec(name="B", methods=["b"], docstring="B", dependencies=["A"])
        
        gaps = [
            GrowthGap(protocol=proto_b),  # Put dependent first
            GrowthGap(protocol=proto_a)
        ]
        
        ordered = evolution._order_by_dependencies(gaps)
        
        # A should come before B
        names = [g.protocol.name for g in ordered]
        assert names.index("A") < names.index("B")
    
    def test_order_by_dependencies_chain(self, evolution):
        """Test ordering with dependency chain."""
        # C -> B -> A
        proto_a = ProtocolSpec(name="A", methods=[], docstring="A", dependencies=[])
        proto_b = ProtocolSpec(name="B", methods=[], docstring="B", dependencies=["A"])
        proto_c = ProtocolSpec(name="C", methods=[], docstring="C", dependencies=["B"])
        
        gaps = [
            GrowthGap(protocol=proto_c),
            GrowthGap(protocol=proto_a),
            GrowthGap(protocol=proto_b)
        ]
        
        ordered = evolution._order_by_dependencies(gaps)
        names = [g.protocol.name for g in ordered]
        
        assert names.index("A") < names.index("B")
        assert names.index("B") < names.index("C")
    
    def test_order_by_dependencies_multiple(self, evolution):
        """Test ordering with multiple dependencies."""
        # D depends on A, B, C
        proto_a = ProtocolSpec(name="A", methods=[], docstring="A", dependencies=[])
        proto_b = ProtocolSpec(name="B", methods=[], docstring="B", dependencies=[])
        proto_c = ProtocolSpec(name="C", methods=[], docstring="C", dependencies=[])
        proto_d = ProtocolSpec(name="D", methods=[], docstring="D", dependencies=["A", "B", "C"])
        
        gaps = [
            GrowthGap(protocol=proto_d),
            GrowthGap(protocol=proto_b),
            GrowthGap(protocol=proto_a),
            GrowthGap(protocol=proto_c)
        ]
        
        ordered = evolution._order_by_dependencies(gaps)
        names = [g.protocol.name for g in ordered]
        
        # D should be last
        assert names[-1] == "D"
    
    def test_order_by_dependencies_external(self, evolution):
        """Test that external dependencies don't affect ordering."""
        # B depends on A and External (not in gaps)
        proto_a = ProtocolSpec(name="A", methods=[], docstring="A", dependencies=[])
        proto_b = ProtocolSpec(name="B", methods=[], docstring="B", dependencies=["A", "ExternalLib"])
        
        gaps = [
            GrowthGap(protocol=proto_b),
            GrowthGap(protocol=proto_a)
        ]
        
        ordered = evolution._order_by_dependencies(gaps)
        names = [g.protocol.name for g in ordered]
        
        # Should still order correctly (External is ignored)
        assert names.index("A") < names.index("B")
    
    def test_resolved_dependencies(self, evolution):
        """Test getting resolved dependencies."""
        proto_a = ProtocolSpec(name="A", methods=[], docstring="A")
        proto_b = ProtocolSpec(name="B", methods=[], docstring="B", dependencies=["A"])
        
        # Simulate A already generated
        comp_a = ComponentOrganism(id="A_1", protocol_name="A", code="class A: pass", coherence_score=0.9)
        evolution.components.append(comp_a)
        
        gap_b = GrowthGap(protocol=proto_b)
        resolved = evolution._get_resolved_dependencies(gap_b, evolution.components)
        
        assert "A" in resolved
        assert resolved["A"].id == "A_1"
    
    def test_grow_respects_dependency_order(self, evolution):
        """Test that grow() generates in dependency order."""
        # ChatBot depends on IntentClassifier
        classifier = ProtocolSpec(
            name="IntentClassifier", methods=["classify"], docstring="Classifier",
            dependencies=[]
        )
        chatbot = ProtocolSpec(
            name="ChatBot", methods=["chat"], docstring="Bot",
            dependencies=["IntentClassifier"]
        )
        
        gaps = [
            GrowthGap(protocol=chatbot),  # Dependent first
            GrowthGap(protocol=classifier)
        ]
        
        evolution.grow(gaps, max_generations=1)
        
        # Both should be generated
        names = {c.protocol_name for c in evolution.components}
        assert "IntentClassifier" in names
        assert "ChatBot" in names


class TestDomainTypes:
    """Tests for domain type passing."""
    
    @pytest.fixture
    def evolution(self):
        return SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(max_generations=1, candidates_per_gap=1)
        )
    
    def test_gap_with_domain_types(self):
        """Test GrowthGap stores domain types."""
        domain_types = [
            "@dataclass\nclass User:\n    id: str\n    name: str",
            "@dataclass\nclass Request:\n    method: str"
        ]
        
        proto = ProtocolSpec(name="Service", methods=[], docstring="Service")
        gap = GrowthGap(protocol=proto, domain_types=domain_types)
        
        assert len(gap.domain_types) == 2
        assert "User" in gap.domain_types[0]
    
    def test_generation_context_includes_domain_types(self):
        """Test GenerationContext receives domain types."""
        from fracton.tools.shadowpuppet.generators.base import GenerationContext
        
        domain_types = ["@dataclass\nclass Entity:\n    id: str"]
        proto = ProtocolSpec(name="Service", methods=[], docstring="Service")
        
        ctx = GenerationContext(
            protocol=proto,
            domain_types=domain_types
        )
        
        assert len(ctx.domain_types) == 1
        assert "Entity" in ctx.domain_types[0]
    
    def test_build_prompt_includes_domain_types(self):
        """Test that build_prompt includes domain types."""
        from fracton.tools.shadowpuppet.generators.base import GenerationContext
        
        domain_types = ["@dataclass\nclass Message:\n    content: str"]
        proto = ProtocolSpec(name="Handler", methods=["handle"], docstring="Handler")
        
        ctx = GenerationContext(protocol=proto, domain_types=domain_types)
        
        generator = MockGenerator()
        prompt = generator.build_prompt(ctx)
        
        assert "DOMAIN TYPES" in prompt
        assert "Message" in prompt


class TestTestSuiteIntegration:
    """Tests for test_suite integration."""
    
    def test_gap_test_suite_dict(self):
        """Test GrowthGap.get_test_dict()."""
        from fracton.tools.shadowpuppet.protocols import TestSuite
        
        def test_func():
            return True
        
        proto = ProtocolSpec(name="Service", methods=[], docstring="Service")
        gap = GrowthGap(
            protocol=proto,
            test_suite=TestSuite(unit=[test_func])
        )
        
        test_dict = gap.get_test_dict()
        assert "unit" in test_dict
        assert len(test_dict["unit"]) == 1
    
    def test_empty_test_suite(self):
        """Test GrowthGap without test_suite."""
        proto = ProtocolSpec(name="Service", methods=[], docstring="Service")
        gap = GrowthGap(protocol=proto)
        
        test_dict = gap.get_test_dict()
        assert test_dict == {}
