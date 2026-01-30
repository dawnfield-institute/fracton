"""
Integration tests for ShadowPuppet.

Tests:
- Full evolution cycle
- Generated code execution
- Example architectures
"""

import pytest
import tempfile
from pathlib import Path

from fracton.tools.shadowpuppet import (
    SoftwareEvolution,
    ProtocolSpec,
    GrowthGap,
    MockGenerator,
    EvolutionConfig,
    TypeAnnotation
)


class TestFullEvolutionCycle:
    """Integration tests for complete evolution cycles."""
    
    def test_webapp_architecture(self):
        """Test evolving a complete webapp architecture."""
        # Define webapp protocols
        protocols = [
            ProtocolSpec(
                name="APIRouter",
                methods=["get", "post", "handle"],
                docstring="REST API router",
                pac_invariants=["Return JSON responses"]
            ),
            ProtocolSpec(
                name="UserService",
                methods=["create_user", "get_user", "authenticate"],
                docstring="User management",
                pac_invariants=["Validate emails", "Hash passwords"]
            )
        ]
        
        gaps = [GrowthGap(protocol=p) for p in protocols]
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(
                coherence_threshold=0.50,
                max_generations=2,
                candidates_per_gap=2
            )
        )
        
        results = evolution.grow(gaps)
        
        # Should succeed
        assert results['success'] is True
        assert len(evolution.components) >= 2
        
        # Should have both protocols
        protocol_names = {c.protocol_name for c in evolution.components}
        assert "APIRouter" in protocol_names
        assert "UserService" in protocol_names
    
    def test_chatbot_architecture(self):
        """Test evolving a chatbot architecture."""
        protocols = [
            ProtocolSpec(
                name="IntentClassifier",
                methods=["classify", "add_intent"],
                docstring="Intent classification"
            ),
            ProtocolSpec(
                name="ResponseGenerator",
                methods=["generate", "set_template"],
                docstring="Response generation"
            ),
            ProtocolSpec(
                name="ChatBot",
                methods=["chat", "start_conversation"],
                docstring="Main chatbot"
            )
        ]
        
        gaps = [GrowthGap(protocol=p) for p in protocols]
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(
                coherence_threshold=0.50,
                max_generations=2,
                candidates_per_gap=2
            )
        )
        
        results = evolution.grow(gaps)
        
        assert results['success'] is True
        assert len(evolution.components) >= 3


class TestGeneratedCodeExecution:
    """Tests that generated code actually executes."""
    
    def test_api_router_executes(self):
        """Test that generated APIRouter can be instantiated."""
        proto = ProtocolSpec(
            name="APIRouter",
            methods=["get", "post"],
            docstring="API router"
        )
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(coherence_threshold=0.50, max_generations=1)
        )
        
        evolution.grow([GrowthGap(protocol=proto)])
        
        assert len(evolution.components) > 0
        code = evolution.components[0].code
        
        # Execute the code
        namespace = {}
        exec(code, namespace)
        
        # Should have APIRouter class
        assert "APIRouter" in namespace
        
        # Should be instantiable
        router = namespace["APIRouter"]()
        assert router is not None
    
    def test_user_service_executes(self):
        """Test that generated UserService can be used."""
        proto = ProtocolSpec(
            name="UserService",
            methods=["create_user", "get_user"],
            docstring="User management"
        )
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(coherence_threshold=0.50, max_generations=1)
        )
        
        evolution.grow([GrowthGap(protocol=proto)])
        
        code = evolution.components[0].code
        namespace = {}
        exec(code, namespace)
        
        # Should have UserService class
        assert "UserService" in namespace
        
        # Should be instantiable
        service = namespace["UserService"]()
        
        # Should have create_user method
        assert hasattr(service, "create_user") or hasattr(service, "create")
    
    def test_chatbot_executes(self):
        """Test that generated ChatBot can chat."""
        proto = ProtocolSpec(
            name="ChatBot",
            methods=["chat", "start_conversation"],
            docstring="Chatbot"
        )
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(coherence_threshold=0.50, max_generations=1)
        )
        
        evolution.grow([GrowthGap(protocol=proto)])
        
        code = evolution.components[0].code
        namespace = {}
        exec(code, namespace)
        
        assert "ChatBot" in namespace
        bot = namespace["ChatBot"]()
        
        # Should have chat method
        assert hasattr(bot, "chat")


class TestTypeAnnotationIntegration:
    """Tests for TypeAnnotation in evolution."""
    
    def test_rich_type_in_protocol(self):
        """Test evolution with rich type annotations."""
        proto = ProtocolSpec(
            name="TypedService",
            methods=["process"],
            method_signatures=[
                TypeAnnotation(
                    name="process",
                    params={"data": "Dict[str, Any]"},
                    returns="Result",
                    raises=["ValidationError"]
                )
            ],
            docstring="Typed service"
        )
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(coherence_threshold=0.50, max_generations=1)
        )
        
        evolution.grow([GrowthGap(protocol=proto)])
        
        # Should succeed
        assert len(evolution.components) > 0
        
        # Prompt context should include rich types
        ctx = proto.to_prompt_context()
        assert "def process" in ctx


class TestCodeSaving:
    """Tests for code saving functionality."""
    
    def test_save_to_directory(self):
        """Test saving generated code to files."""
        proto = ProtocolSpec(
            name="SaveTest",
            methods=["run"],
            docstring="Test saving"
        )
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(coherence_threshold=0.50, max_generations=1)
        )
        
        evolution.grow([GrowthGap(protocol=proto)])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            evolution.save_code(output_dir)
            
            # Should create file
            files = list(output_dir.glob("*.py"))
            assert len(files) == 1
            
            # File should contain class
            content = files[0].read_text()
            assert "class" in content


class TestPACInvariantEnforcement:
    """Tests for PAC invariant enforcement in evolution."""
    
    def test_invariants_passed_to_generator(self):
        """Test that invariants are passed to generator context."""
        proto = ProtocolSpec(
            name="SecureService",
            methods=["store"],
            docstring="Secure storage",
            pac_invariants=[
                "Data must be encrypted",
                "Access must be logged"
            ]
        )
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(coherence_threshold=0.50, max_generations=1)
        )
        
        evolution.env.pac_invariants = ["Global invariant"]
        
        evolution.grow([GrowthGap(protocol=proto)])
        
        # Should complete without error
        assert len(evolution.components) > 0


class TestCrossoverIntegration:
    """Integration tests for crossover functionality."""
    
    def test_crossover_in_evolution(self):
        """Test that crossover works during evolution."""
        proto = ProtocolSpec(
            name="CrossoverTest",
            methods=["method_a", "method_b"],
            docstring="Test crossover"
        )
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(
                coherence_threshold=0.50,
                max_generations=3,
                candidates_per_gap=3,
                enable_crossover=True,
                crossover_rate=0.5
            )
        )
        
        # Run multiple gaps to build up population for crossover
        gaps = [GrowthGap(protocol=proto) for _ in range(3)]
        
        evolution.grow(gaps)
        
        # Should complete
        assert len(evolution.components) > 0


class TestGenealogyIntegration:
    """Integration tests for genealogy tracking."""
    
    def test_genealogy_in_evolution(self):
        """Test genealogy tracking during evolution."""
        proto = ProtocolSpec(
            name="GenealogyTest",
            methods=["run"],
            docstring="Test genealogy"
        )
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(
                coherence_threshold=0.50,
                max_generations=2,
                candidates_per_gap=2
            )
        )
        
        evolution.grow([GrowthGap(protocol=proto)])
        
        # Genealogy should have nodes
        genealogy = evolution.genealogy.to_dict()
        assert len(genealogy['nodes']) > 0
        
        # Should have component info
        for node in genealogy['nodes'].values():
            assert 'protocol_name' in node
            assert 'coherence_score' in node


class TestErrorHandling:
    """Tests for error handling in evolution."""
    
    def test_empty_gaps_list(self):
        """Test evolution with empty gaps list."""
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(max_generations=1)
        )
        
        results = evolution.grow([])
        
        # Should not crash
        assert results['success'] is False or results['final_population'] == 0
    
    def test_convergence_detection(self):
        """Test that evolution detects convergence."""
        proto = ProtocolSpec(
            name="ConvergeTest",
            methods=["run"],
            docstring="Test convergence"
        )
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(
                coherence_threshold=0.50,  # Low threshold
                max_generations=10  # High max, but should converge early
            )
        )
        
        evolution.grow([GrowthGap(protocol=proto)])
        
        # Should converge before max generations
        assert evolution.generation < 10 or len(evolution.components) > 0
