"""
Tests for ShadowPuppet generators module.

Tests:
- MockGenerator code generation
- GenerationContext building
- Prompt construction
- Code extraction
"""

import pytest
from fracton.tools.shadowpuppet.generators import MockGenerator
from fracton.tools.shadowpuppet.generators.base import (
    CodeGenerator,
    GenerationContext,
    GenerationError
)
from fracton.tools.shadowpuppet.protocols import (
    ProtocolSpec,
    ComponentOrganism
)


class TestGenerationContext:
    """Tests for GenerationContext dataclass."""
    
    def test_basic_context(self):
        """Test basic context creation."""
        proto = ProtocolSpec(
            name="Service",
            methods=["run"],
            docstring="A service"
        )
        ctx = GenerationContext(protocol=proto)
        
        assert ctx.protocol.name == "Service"
        assert ctx.parent is None
        assert ctx.siblings == []
        assert ctx.mutation_rate == 0.1
        assert ctx.pac_invariants == []
    
    def test_context_with_parent(self):
        """Test context with parent component."""
        proto = ProtocolSpec(name="Child", methods=[], docstring="Child")
        parent = ComponentOrganism(
            id="Parent_0",
            protocol_name="Parent",
            code="class Parent: pass"
        )
        
        ctx = GenerationContext(protocol=proto, parent=parent)
        assert ctx.parent.id == "Parent_0"
    
    def test_context_with_invariants(self):
        """Test context with PAC invariants."""
        proto = ProtocolSpec(name="Test", methods=[], docstring="Test")
        ctx = GenerationContext(
            protocol=proto,
            pac_invariants=["Rule 1", "Rule 2"]
        )
        assert len(ctx.pac_invariants) == 2


class TestMockGenerator:
    """Tests for MockGenerator."""
    
    @pytest.fixture
    def generator(self):
        return MockGenerator()
    
    def test_generator_name(self, generator):
        """Test generator name property."""
        assert generator.name == "mock"
    
    def test_generate_api_router(self, generator):
        """Test API router generation."""
        proto = ProtocolSpec(
            name="APIRouter",
            methods=["get", "post"],
            docstring="REST router"
        )
        ctx = GenerationContext(protocol=proto)
        code = generator.generate(ctx)
        
        assert "class APIRouter" in code
        assert "def get" in code or "def handle" in code
        assert len(code) > 100
    
    def test_generate_user_service(self, generator):
        """Test user service generation."""
        proto = ProtocolSpec(
            name="UserService",
            methods=["create_user", "get_user"],
            docstring="User management"
        )
        ctx = GenerationContext(protocol=proto)
        code = generator.generate(ctx)
        
        assert "class UserService" in code
        assert len(code) > 100
    
    def test_generate_chatbot(self, generator):
        """Test chatbot generation."""
        proto = ProtocolSpec(
            name="ChatBot",
            methods=["chat", "start_conversation"],
            docstring="Chatbot"
        )
        ctx = GenerationContext(protocol=proto)
        code = generator.generate(ctx)
        
        assert "class ChatBot" in code
        assert len(code) > 100
    
    def test_generate_intent_classifier(self, generator):
        """Test intent classifier generation."""
        proto = ProtocolSpec(
            name="IntentClassifier",
            methods=["classify"],
            docstring="Intent detection"
        )
        ctx = GenerationContext(protocol=proto)
        code = generator.generate(ctx)
        
        assert "class IntentClassifier" in code
    
    def test_generate_template_renderer(self, generator):
        """Test template renderer generation."""
        proto = ProtocolSpec(
            name="TemplateRenderer",
            methods=["render"],
            docstring="HTML templates"
        )
        ctx = GenerationContext(protocol=proto)
        code = generator.generate(ctx)
        
        assert "class TemplateRenderer" in code
    
    def test_generate_static_file_server(self, generator):
        """Test static file server generation."""
        proto = ProtocolSpec(
            name="StaticFileServer",
            methods=["serve"],
            docstring="Static files"
        )
        ctx = GenerationContext(protocol=proto)
        code = generator.generate(ctx)
        
        assert "class StaticFileServer" in code
    
    def test_generate_conversation_manager(self, generator):
        """Test conversation manager generation."""
        proto = ProtocolSpec(
            name="ConversationManager",
            methods=["create_conversation", "add_message"],
            docstring="Manage conversations"
        )
        ctx = GenerationContext(protocol=proto)
        code = generator.generate(ctx)
        
        assert "class ConversationManager" in code
    
    def test_generate_response_generator(self, generator):
        """Test response generator generation."""
        proto = ProtocolSpec(
            name="ResponseGenerator",
            methods=["generate"],
            docstring="Generate responses"
        )
        ctx = GenerationContext(protocol=proto)
        code = generator.generate(ctx)
        
        assert "class ResponseGenerator" in code
    
    def test_generate_webapp(self, generator):
        """Test webapp generation."""
        proto = ProtocolSpec(
            name="WebApp",
            methods=["start", "handle_request"],
            docstring="Web application"
        )
        ctx = GenerationContext(protocol=proto)
        code = generator.generate(ctx)
        
        assert "class WebApp" in code
    
    def test_generate_generic(self, generator):
        """Test generic unknown protocol generation."""
        proto = ProtocolSpec(
            name="CustomWidget",
            methods=["create", "update", "delete", "list"],
            docstring="Custom widget manager",
            attributes=["widgets"]
        )
        ctx = GenerationContext(protocol=proto)
        code = generator.generate(ctx)
        
        assert "class CustomWidget" in code
        assert "def create" in code or "def __init__" in code
    
    def test_custom_template(self, generator):
        """Test custom template override."""
        custom_code = "class Custom:\n    pass"
        generator.templates["MyCustom"] = custom_code
        
        proto = ProtocolSpec(
            name="MyCustom",
            methods=[],
            docstring="Custom"
        )
        ctx = GenerationContext(protocol=proto)
        code = generator.generate(ctx)
        
        assert code == custom_code
    
    def test_generated_code_is_valid_python(self, generator):
        """Test that generated code is valid Python."""
        protocols = [
            ProtocolSpec(name="APIRouter", methods=["get"], docstring="API"),
            ProtocolSpec(name="UserService", methods=["create"], docstring="Users"),
            ProtocolSpec(name="ChatBot", methods=["chat"], docstring="Chat"),
            ProtocolSpec(name="Generic", methods=["do_something"], docstring="Generic"),
        ]
        
        for proto in protocols:
            ctx = GenerationContext(protocol=proto)
            code = generator.generate(ctx)
            
            # Should parse without error
            try:
                compile(code, f"<{proto.name}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Generated invalid Python for {proto.name}: {e}")


class TestCodeGeneratorBase:
    """Tests for CodeGenerator base class methods."""
    
    @pytest.fixture
    def generator(self):
        return MockGenerator()
    
    def test_build_prompt_basic(self, generator):
        """Test basic prompt building."""
        proto = ProtocolSpec(
            name="TestService",
            methods=["process"],
            docstring="A test service"
        )
        ctx = GenerationContext(protocol=proto)
        prompt = generator.build_prompt(ctx)
        
        assert "Protocol: TestService" in prompt
        assert "Description: A test service" in prompt
        assert "process" in prompt
    
    def test_build_prompt_with_invariants(self, generator):
        """Test prompt with PAC invariants."""
        proto = ProtocolSpec(
            name="Secure",
            methods=["save"],
            docstring="Secure storage"
        )
        ctx = GenerationContext(
            protocol=proto,
            pac_invariants=["Data must be encrypted", "Audit all access"]
        )
        prompt = generator.build_prompt(ctx)
        
        assert "PAC INVARIANTS" in prompt
        assert "Data must be encrypted" in prompt
        assert "Audit all access" in prompt
    
    def test_build_prompt_with_parent(self, generator):
        """Test prompt with parent template."""
        proto = ProtocolSpec(name="Child", methods=[], docstring="Child")
        parent = ComponentOrganism(
            id="Parent_0",
            protocol_name="Parent",
            code="class Parent:\n    def method(self):\n        return 42"
        )
        ctx = GenerationContext(protocol=proto, parent=parent)
        prompt = generator.build_prompt(ctx)
        
        assert "TEMPLATE CONTEXT" in prompt
        assert "Parent component: Parent" in prompt
        assert "class Parent" in prompt
    
    def test_extract_code_from_markdown(self, generator):
        """Test code extraction from markdown blocks."""
        response = '''
Here's the implementation:

```python
class Service:
    def run(self):
        pass
```

This should work!
'''
        code = generator.extract_code(response)
        assert "class Service" in code
        assert "Here's the implementation" not in code
    
    def test_extract_code_from_plain(self, generator):
        """Test code extraction from plain text."""
        response = '''class Service:
    def run(self):
        pass'''
        code = generator.extract_code(response)
        assert "class Service" in code
    
    def test_extract_code_generic_block(self, generator):
        """Test code extraction from generic code block."""
        response = '''
```
class Service:
    pass
```
'''
        code = generator.extract_code(response)
        assert "class Service" in code


class TestRandomVariationGenerator:
    """Tests for RandomVariationGenerator."""
    
    def test_variation_generator_exists(self):
        """Test that variation generator can be imported."""
        from fracton.tools.shadowpuppet.generators.mock import RandomVariationGenerator
        gen = RandomVariationGenerator(variation_rate=0.5)
        assert gen.name == "mock-random"
