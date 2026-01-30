"""
Tests for ShadowPuppet protocols module.

Tests:
- TypeAnnotation signature generation
- ProtocolSpec creation and serialization
- GrowthGap configuration
- ComponentOrganism lifecycle
- TestSuite composition
"""

import pytest
from fracton.tools.shadowpuppet.protocols import (
    TypeAnnotation,
    ProtocolSpec,
    GrowthGap,
    ComponentOrganism,
    TestSuite
)


class TestTypeAnnotation:
    """Tests for TypeAnnotation dataclass."""
    
    def test_basic_signature(self):
        """Test basic method signature generation."""
        ta = TypeAnnotation(
            name="get_user",
            params={"user_id": "str"},
            returns="User"
        )
        sig = ta.to_signature()
        assert sig == "def get_user(self, user_id: str) -> User"
    
    def test_multiple_params(self):
        """Test signature with multiple parameters."""
        ta = TypeAnnotation(
            name="create_user",
            params={"name": "str", "email": "str", "age": "int"},
            returns="User"
        )
        sig = ta.to_signature()
        assert "name: str" in sig
        assert "email: str" in sig
        assert "age: int" in sig
        assert "-> User" in sig
    
    def test_default_params(self):
        """Test signature with default parameter values."""
        ta = TypeAnnotation(
            name="list_users",
            params={"limit": "int = 100", "offset": "int = 0"},
            returns="List[User]"
        )
        sig = ta.to_signature()
        assert "limit: int = 100" in sig
        assert "offset: int = 0" in sig
    
    def test_async_method(self):
        """Test async method signature."""
        ta = TypeAnnotation(
            name="fetch_data",
            params={"url": "str"},
            returns="Dict[str, Any]",
            async_method=True
        )
        sig = ta.to_signature()
        assert sig.startswith("async def")
    
    def test_raises_list(self):
        """Test raises annotation storage."""
        ta = TypeAnnotation(
            name="validate",
            params={"data": "Dict"},
            returns="bool",
            raises=["ValueError", "TypeError"]
        )
        assert "ValueError" in ta.raises
        assert "TypeError" in ta.raises
    
    def test_empty_params(self):
        """Test method with no parameters."""
        ta = TypeAnnotation(
            name="reset",
            params={},
            returns="None"
        )
        sig = ta.to_signature()
        assert "def reset(self, ) -> None" in sig or "def reset(self,) -> None" in sig


class TestProtocolSpec:
    """Tests for ProtocolSpec dataclass."""
    
    def test_basic_creation(self):
        """Test basic protocol creation."""
        proto = ProtocolSpec(
            name="UserService",
            methods=["create", "read", "update", "delete"],
            docstring="CRUD operations for users"
        )
        assert proto.name == "UserService"
        assert len(proto.methods) == 4
        assert proto.docstring == "CRUD operations for users"
    
    def test_with_invariants(self):
        """Test protocol with PAC invariants."""
        proto = ProtocolSpec(
            name="AuthService",
            methods=["login", "logout"],
            docstring="Authentication service",
            pac_invariants=[
                "Passwords are hashed",
                "Sessions expire after 24h"
            ]
        )
        assert len(proto.pac_invariants) == 2
        assert "Passwords are hashed" in proto.pac_invariants
    
    def test_with_method_signatures(self):
        """Test protocol with rich type signatures."""
        proto = ProtocolSpec(
            name="DataStore",
            methods=["get", "set"],
            method_signatures=[
                TypeAnnotation("get", {"key": "str"}, "Optional[Any]"),
                TypeAnnotation("set", {"key": "str", "value": "Any"}, "None")
            ],
            docstring="Key-value store"
        )
        assert len(proto.method_signatures) == 2
    
    def test_to_prompt_context_basic(self):
        """Test prompt context generation."""
        proto = ProtocolSpec(
            name="Calculator",
            methods=["add", "subtract"],
            docstring="Basic math operations"
        )
        ctx = proto.to_prompt_context()
        assert "Protocol: Calculator" in ctx
        assert "Description: Basic math operations" in ctx
        assert "add" in ctx
        assert "subtract" in ctx
    
    def test_to_prompt_context_with_signatures(self):
        """Test prompt context with rich signatures."""
        proto = ProtocolSpec(
            name="API",
            methods=["get"],
            method_signatures=[
                TypeAnnotation("get", {"path": "str"}, "Response", raises=["NotFoundError"])
            ],
            docstring="REST API"
        )
        ctx = proto.to_prompt_context()
        assert "Methods (with signatures)" in ctx
        assert "def get(self, path: str) -> Response" in ctx
        assert "Raises: NotFoundError" in ctx
    
    def test_to_prompt_context_with_invariants(self):
        """Test prompt context includes enforced invariants."""
        proto = ProtocolSpec(
            name="SecureStore",
            methods=["save"],
            docstring="Secure storage",
            pac_invariants=["Data is encrypted"]
        )
        ctx = proto.to_prompt_context()
        assert "PAC Invariants (MUST be enforced" in ctx
        assert "Data is encrypted" in ctx
    
    def test_get_attribute_types_simple(self):
        """Test attribute type parsing - simple names."""
        proto = ProtocolSpec(
            name="Test",
            methods=[],
            docstring="Test",
            attributes=["data", "cache"]
        )
        types = proto.get_attribute_types()
        assert types["data"] == "Any"
        assert types["cache"] == "Any"
    
    def test_get_attribute_types_with_hints(self):
        """Test attribute type parsing - with type hints."""
        proto = ProtocolSpec(
            name="Test",
            methods=[],
            docstring="Test",
            attributes=["data: Dict[str, Any]", "count: int"]
        )
        types = proto.get_attribute_types()
        assert types["data"] == "Dict[str, Any]"
        assert types["count"] == "int"
    
    def test_dependencies(self):
        """Test protocol dependencies."""
        proto = ProtocolSpec(
            name="UserController",
            methods=["handle"],
            docstring="User controller",
            dependencies=["UserService", "AuthService"]
        )
        assert "UserService" in proto.dependencies
        assert "AuthService" in proto.dependencies


class TestGrowthGap:
    """Tests for GrowthGap dataclass."""
    
    def test_basic_gap(self):
        """Test basic gap creation."""
        proto = ProtocolSpec(
            name="Service",
            methods=["run"],
            docstring="A service"
        )
        gap = GrowthGap(protocol=proto)
        assert gap.protocol.name == "Service"
        assert gap.required_coherence == 0.70
        assert gap.priority == 1.0
    
    def test_gap_with_test_suite(self):
        """Test gap with test suite."""
        proto = ProtocolSpec(name="Test", methods=[], docstring="Test")
        
        def unit_test(comp, ctx):
            return True
        
        suite = TestSuite(unit=[unit_test])
        gap = GrowthGap(protocol=proto, test_suite=suite)
        
        test_dict = gap.get_test_dict()
        assert "unit" in test_dict
        assert len(test_dict["unit"]) == 1
    
    def test_gap_with_context(self):
        """Test gap with additional context."""
        proto = ProtocolSpec(name="Test", methods=[], docstring="Test")
        gap = GrowthGap(
            protocol=proto,
            context={"environment": "production", "version": "2.0"}
        )
        assert gap.context["environment"] == "production"


class TestComponentOrganism:
    """Tests for ComponentOrganism dataclass."""
    
    def test_basic_creation(self):
        """Test basic component creation."""
        comp = ComponentOrganism(
            id="TestService_0",
            protocol_name="TestService",
            code="class TestService:\n    pass"
        )
        assert comp.id == "TestService_0"
        assert comp.coherence_score == 0.0
        assert comp.generation == 0
    
    def test_len_is_code_length(self):
        """Test __len__ returns code length."""
        code = "class Test:\n    def method(self):\n        pass"
        comp = ComponentOrganism(
            id="Test_0",
            protocol_name="Test",
            code=code
        )
        assert len(comp) == len(code)
    
    def test_summary(self):
        """Test human-readable summary."""
        comp = ComponentOrganism(
            id="API_5",
            protocol_name="API",
            code="...",
            coherence_score=0.85,
            structural_score=0.90,
            semantic_score=0.80,
            energetic_score=0.85,
            generation=3
        )
        summary = comp.summary()
        assert "API_5" in summary
        assert "gen 3" in summary
        assert "0.85" in summary
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        comp = ComponentOrganism(
            id="Service_1",
            protocol_name="Service",
            code="class Service: pass",
            coherence_score=0.75,
            parent_id="Service_0",
            generation=1,
            generator_used="mock"
        )
        d = comp.to_dict()
        assert d["id"] == "Service_1"
        assert d["coherence_score"] == 0.75
        assert d["parent_id"] == "Service_0"
        assert d["generator_used"] == "mock"
    
    def test_derivation_path(self):
        """Test derivation path tracking."""
        comp = ComponentOrganism(
            id="Child_0",
            protocol_name="Child",
            code="...",
            derivation_path=["GrandParent", "Parent", "Child"]
        )
        assert len(comp.derivation_path) == 3
        assert comp.derivation_path[-1] == "Child"


class TestTestSuite:
    """Tests for TestSuite dataclass."""
    
    def test_empty_suite(self):
        """Test empty test suite."""
        suite = TestSuite()
        d = suite.to_dict()
        assert d == {}
    
    def test_unit_tests_only(self):
        """Test suite with only unit tests."""
        def test1(c, ctx): return True
        def test2(c, ctx): return True
        
        suite = TestSuite(unit=[test1, test2])
        d = suite.to_dict()
        assert "unit" in d
        assert len(d["unit"]) == 2
        assert "integration" not in d
    
    def test_full_suite(self):
        """Test suite with all test types."""
        def unit(c, ctx): return True
        def integration(c, ctx): return True
        def e2e(c, ctx): return True
        
        suite = TestSuite(
            unit=[unit],
            integration=[integration],
            e2e=[e2e]
        )
        d = suite.to_dict()
        assert len(d) == 3
        assert "unit" in d
        assert "integration" in d
        assert "e2e" in d
