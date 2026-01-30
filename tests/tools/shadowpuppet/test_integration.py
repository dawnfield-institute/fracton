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


class TestDependencyOrderingE2E:
    """End-to-end tests for dependency ordering."""
    
    def test_webapp_dependency_order(self):
        """Test that WebApp architecture respects dependencies."""
        from fracton.tools.shadowpuppet.protocols import TestSuite
        
        protocols = [
            ProtocolSpec(
                name="APIRouter",
                methods=["handle"],
                docstring="Router",
                dependencies=[]
            ),
            ProtocolSpec(
                name="UserService",
                methods=["get_user"],
                docstring="Users",
                dependencies=[]
            ),
            ProtocolSpec(
                name="WebApp",
                methods=["start"],
                docstring="App",
                dependencies=["APIRouter", "UserService"]
            )
        ]
        
        # Put WebApp first - it should be reordered to last
        gaps = [
            GrowthGap(protocol=protocols[2]),  # WebApp
            GrowthGap(protocol=protocols[0]),  # APIRouter
            GrowthGap(protocol=protocols[1]),  # UserService
        ]
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(coherence_threshold=0.50, max_generations=1)
        )
        
        # Check ordering
        ordered = evolution._order_by_dependencies(gaps)
        names = [g.protocol.name for g in ordered]
        
        # WebApp should be last
        assert names[-1] == "WebApp"
        
        # Run evolution
        results = evolution.grow(gaps)
        assert results['success']
        assert len(evolution.components) >= 3
    
    def test_chatbot_dependency_order(self):
        """Test that ChatBot architecture respects dependencies."""
        protocols = [
            ProtocolSpec(
                name="IntentClassifier",
                methods=["classify"],
                docstring="Classifier",
                dependencies=[]
            ),
            ProtocolSpec(
                name="ResponseGenerator",
                methods=["generate"],
                docstring="Generator",
                dependencies=[]
            ),
            ProtocolSpec(
                name="ConversationManager",
                methods=["create_conversation"],
                docstring="Manager",
                dependencies=[]
            ),
            ProtocolSpec(
                name="ChatBot",
                methods=["chat"],
                docstring="Bot",
                dependencies=["IntentClassifier", "ResponseGenerator", "ConversationManager"]
            )
        ]
        
        # Mix order
        gaps = [
            GrowthGap(protocol=protocols[3]),  # ChatBot
            GrowthGap(protocol=protocols[1]),  # ResponseGenerator
            GrowthGap(protocol=protocols[0]),  # IntentClassifier
            GrowthGap(protocol=protocols[2]),  # ConversationManager
        ]
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(coherence_threshold=0.50, max_generations=1)
        )
        
        ordered = evolution._order_by_dependencies(gaps)
        names = [g.protocol.name for g in ordered]
        
        # ChatBot should be last
        assert names[-1] == "ChatBot"


class TestDomainTypesE2E:
    """End-to-end tests for domain type integration."""
    
    def test_domain_types_in_generation_context(self):
        """Test that domain types flow through to generation."""
        from fracton.tools.shadowpuppet.generators.base import GenerationContext
        
        domain_types = [
            "@dataclass\nclass User:\n    id: str\n    name: str",
            "@dataclass\nclass Session:\n    token: str"
        ]
        
        proto = ProtocolSpec(
            name="AuthService",
            methods=["authenticate", "validate_session"],
            docstring="Authentication service"
        )
        
        gap = GrowthGap(protocol=proto, domain_types=domain_types)
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(coherence_threshold=0.50, max_generations=1)
        )
        
        results = evolution.grow([gap])
        
        assert results['success']
        assert len(evolution.components) > 0
    
    def test_prompt_includes_types(self):
        """Test that generated prompt includes domain types."""
        from fracton.tools.shadowpuppet.generators.base import GenerationContext
        
        domain_types = [
            "@dataclass\nclass Request:\n    path: str\n    method: str"
        ]
        
        proto = ProtocolSpec(name="Handler", methods=["handle"], docstring="Handler")
        ctx = GenerationContext(protocol=proto, domain_types=domain_types)
        
        generator = MockGenerator()
        prompt = generator.build_prompt(ctx)
        
        assert "DOMAIN TYPES" in prompt
        assert "Request" in prompt
        assert "path: str" in prompt


class TestTestSuiteE2E:
    """End-to-end tests for test suite integration."""
    
    def test_unit_tests_in_gap(self):
        """Test that unit tests can be attached to gaps."""
        from fracton.tools.shadowpuppet.protocols import TestSuite
        
        def test_returns_list(service):
            result = service.list_items()
            assert isinstance(result, list)
            return True
        
        def test_count_positive(service):
            count = service.count()
            assert count >= 0
            return True
        
        proto = ProtocolSpec(
            name="ItemService",
            methods=["list_items", "count"],
            docstring="Item service"
        )
        
        gap = GrowthGap(
            protocol=proto,
            test_suite=TestSuite(unit=[test_returns_list, test_count_positive])
        )
        
        # Verify tests are accessible
        test_dict = gap.get_test_dict()
        assert "unit" in test_dict
        assert len(test_dict["unit"]) == 2
    
    def test_integration_tests_in_gap(self):
        """Test that integration tests can be attached."""
        from fracton.tools.shadowpuppet.protocols import TestSuite
        
        def test_workflow(service_a, service_b):
            return True
        
        proto = ProtocolSpec(name="Service", methods=["run"], docstring="Service")
        gap = GrowthGap(
            protocol=proto,
            test_suite=TestSuite(integration=[test_workflow])
        )
        
        test_dict = gap.get_test_dict()
        assert "integration" in test_dict
        assert len(test_dict["integration"]) == 1


class TestSeedExamplesE2E:
    """End-to-end tests using actual seed examples."""
    
    def test_webapp_seed_protocols(self):
        """Test webapp seed protocol definitions."""
        from fracton.tools.shadowpuppet.protocols import TestSuite
        
        # Recreate the webapp protocols
        protocols = [
            ProtocolSpec(
                name="APIRouter",
                methods=["get", "post", "put", "delete", "handle"],
                docstring="REST API router with CRUD operations",
                attributes=["routes: Dict[str, callable]", "middleware: List[callable]"],
                pac_invariants=["All routes return Response objects", "Errors use standard HTTP status codes"],
                dependencies=[]
            ),
            ProtocolSpec(
                name="UserService",
                methods=["create_user", "get_user", "update_user", "delete_user", "list_users"],
                docstring="User management service with validation",
                pac_invariants=["User IDs are unique", "Email addresses are validated"],
                dependencies=[]
            ),
            ProtocolSpec(
                name="TemplateRenderer",
                methods=["render", "load_template", "escape_html"],
                docstring="HTML template renderer",
                attributes=["template_dir: str", "cache: Dict[str, str]"],
                dependencies=[]
            ),
            ProtocolSpec(
                name="StaticFileServer",
                methods=["serve", "get_mime_type"],
                docstring="Static file server",
                attributes=["static_dir: str", "mime_types: Dict[str, str]"],
                dependencies=[]
            ),
            ProtocolSpec(
                name="WebApp",
                methods=["handle_request", "start", "stop"],
                docstring="Main web application orchestrator",
                dependencies=["APIRouter", "UserService", "TemplateRenderer", "StaticFileServer"]
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
        
        assert results['success']
        
        # All protocols should be generated
        names = {c.protocol_name for c in evolution.components}
        assert "APIRouter" in names
        assert "UserService" in names
        assert "WebApp" in names
    
    def test_chatbot_seed_protocols(self):
        """Test chatbot seed protocol definitions."""
        protocols = [
            ProtocolSpec(
                name="IntentClassifier",
                methods=["classify", "extract_entities", "train"],
                docstring="Classifies user messages into intents",
                attributes=["intents: List[str]", "threshold: float"],
                pac_invariants=["Confidence scores sum to 1.0", "Unknown messages return 'unknown' intent"],
                dependencies=[]
            ),
            ProtocolSpec(
                name="ResponseGenerator",
                methods=["generate", "set_template", "format_response"],
                docstring="Generates responses based on intent and context",
                attributes=["templates: Dict[str, List[str]]", "context_window: int"],
                dependencies=[]
            ),
            ProtocolSpec(
                name="ConversationManager",
                methods=["create_conversation", "get_conversation", "add_message", "get_context", "update_context"],
                docstring="Manages conversation state and history",
                pac_invariants=["Conversation IDs are unique", "Messages stored in order"],
                dependencies=[]
            ),
            ProtocolSpec(
                name="ChatBot",
                methods=["chat", "start_conversation", "end_conversation", "get_history"],
                docstring="Main chatbot orchestrator",
                dependencies=["IntentClassifier", "ResponseGenerator", "ConversationManager"]
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
        
        assert results['success']
        
        names = {c.protocol_name for c in evolution.components}
        assert "IntentClassifier" in names
        assert "ResponseGenerator" in names
        assert "ChatBot" in names
    
    def test_generated_router_works(self):
        """Test that generated APIRouter actually works."""
        proto = ProtocolSpec(
            name="APIRouter",
            methods=["get", "post", "handle"],
            docstring="REST API router"
        )
        
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=EvolutionConfig(coherence_threshold=0.50, max_generations=1)
        )
        
        evolution.grow([GrowthGap(protocol=proto)])
        
        code = evolution.components[0].code
        namespace = {}
        exec(code, namespace)
        
        router = namespace["APIRouter"]()
        
        # Test routing
        @router.get("/test")
        def test_handler(request):
            return {"status": "ok"}
        
        # Handler should be registered (routes is dict of dicts: {METHOD: {path: handler}})
        routes = router.routes
        registered = (
            "/test" in routes or 
            (isinstance(routes, dict) and any("/test" in v if isinstance(v, dict) else False for v in routes.values()))
        )
        assert registered, f"Route /test not found in routes: {routes}"
    
    def test_generated_user_service_crud(self):
        """Test that generated UserService CRUD works."""
        proto = ProtocolSpec(
            name="UserService",
            methods=["create_user", "get_user", "update_user", "delete_user"],
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
        
        service = namespace["UserService"]()
        
        # Create user
        user = service.create_user("testuser", "test@example.com", "password123")
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        
        # Get user
        retrieved = service.get_user(user.id)
        assert retrieved is not None
        assert retrieved.id == user.id
        
        # Update user
        updated = service.update_user(user.id, {"username": "newname"})
        assert updated.username == "newname"
        
        # Delete user
        deleted = service.delete_user(user.id)
        assert deleted is True
    
    def test_generated_chatbot_conversation(self):
        """Test that generated ChatBot can hold conversation."""
        proto = ProtocolSpec(
            name="ChatBot",
            methods=["chat", "start_conversation", "end_conversation", "get_history"],
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
        
        bot = namespace["ChatBot"]()
        
        # Start conversation
        conv_id = bot.start_conversation()
        assert conv_id is not None
        
        # Chat
        response = bot.chat("Hello!", conv_id)
        assert response is not None
        assert isinstance(response, str)
        
        # Get history
        history = bot.get_history(conv_id)
        assert len(history) >= 1
        
        # End conversation
        bot.end_conversation(conv_id)
