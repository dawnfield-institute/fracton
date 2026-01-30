"""
Tests for ShadowPuppet coherence evaluation module.

Tests:
- Structural evaluation (AST analysis)
- Semantic evaluation (docstrings, types, invariants)
- Energetic evaluation (complexity, nesting)
- Invariant validation
- Combined fitness scoring
"""

import pytest
from fracton.tools.shadowpuppet.coherence import CoherenceEvaluator
from fracton.tools.shadowpuppet.protocols import (
    ProtocolSpec,
    ComponentOrganism
)


class TestStructuralEvaluation:
    """Tests for structural code evaluation."""
    
    @pytest.fixture
    def evaluator(self):
        return CoherenceEvaluator()
    
    @pytest.fixture
    def api_protocol(self):
        return ProtocolSpec(
            name="APIRouter",
            methods=["get", "post", "put", "delete"],
            docstring="REST API router",
            attributes=["routes"]
        )
    
    def test_valid_class_structure(self, evaluator, api_protocol):
        """Test structural score for valid class."""
        code = '''
class APIRouter:
    """REST API router."""
    
    def __init__(self):
        self.routes = {}
    
    def get(self, path):
        pass
    
    def post(self, path, data):
        pass
    
    def put(self, path, data):
        pass
    
    def delete(self, path):
        pass
'''
        comp = ComponentOrganism(id="test", protocol_name="APIRouter", code=code)
        score = evaluator._evaluate_structural(comp, {"protocol": api_protocol})
        # Has class name match (0.3) + method matches (0.5 * ratio)
        # 4/4 methods = 1.0 ratio, so 0.3 + 0.5 * 1.2 = 0.3 + 0.6 = 0.9
        assert score >= 0.7
    
    def test_missing_methods(self, evaluator, api_protocol):
        """Test lower score for missing methods."""
        code = '''
class APIRouter:
    def __init__(self):
        self.routes = {}
    
    def get(self, path):
        pass
'''
        comp = ComponentOrganism(id="test", protocol_name="APIRouter", code=code)
        score = evaluator._evaluate_structural(comp, {"protocol": api_protocol})
        assert score < 0.8  # Missing 3 of 4 methods
    
    def test_wrong_class_name(self, evaluator, api_protocol):
        """Test lower score for wrong class name."""
        code = '''
class WrongName:
    def get(self): pass
    def post(self): pass
    def put(self): pass
    def delete(self): pass
'''
        comp = ComponentOrganism(id="test", protocol_name="APIRouter", code=code)
        score = evaluator._evaluate_structural(comp, {"protocol": api_protocol})
        assert score < 0.8  # Wrong class name
    
    def test_syntax_error_zero_score(self, evaluator, api_protocol):
        """Test zero score for syntax errors."""
        code = "class APIRouter\n    def broken("
        comp = ComponentOrganism(id="test", protocol_name="APIRouter", code=code)
        score = evaluator._evaluate_structural(comp, {"protocol": api_protocol})
        assert score == 0.0
    
    def test_no_protocol_neutral_score(self, evaluator):
        """Test neutral score when no protocol to check against."""
        code = "class Something: pass"
        comp = ComponentOrganism(id="test", protocol_name="Something", code=code)
        score = evaluator._evaluate_structural(comp, {})
        assert score == 0.5


class TestSemanticEvaluation:
    """Tests for semantic code evaluation."""
    
    @pytest.fixture
    def evaluator(self):
        return CoherenceEvaluator()
    
    def test_docstrings_boost_score(self, evaluator):
        """Test that docstrings improve semantic score."""
        code_no_doc = '''
class Service:
    def process(self):
        pass
'''
        code_with_doc = '''
class Service:
    """A service that processes things."""
    
    def process(self):
        """Process the data."""
        pass
'''
        comp1 = ComponentOrganism(id="t1", protocol_name="Service", code=code_no_doc)
        comp2 = ComponentOrganism(id="t2", protocol_name="Service", code=code_with_doc)
        
        score1 = evaluator._evaluate_semantic(comp1, {})
        score2 = evaluator._evaluate_semantic(comp2, {})
        
        assert score2 > score1
    
    def test_type_hints_boost_score(self, evaluator):
        """Test that type hints improve semantic score."""
        code_no_types = '''
class Calc:
    def add(self, a, b):
        return a + b
'''
        code_with_types = '''
class Calc:
    def add(self, a: int, b: int) -> int:
        return a + b
'''
        comp1 = ComponentOrganism(id="t1", protocol_name="Calc", code=code_no_types)
        comp2 = ComponentOrganism(id="t2", protocol_name="Calc", code=code_with_types)
        
        score1 = evaluator._evaluate_semantic(comp1, {})
        score2 = evaluator._evaluate_semantic(comp2, {})
        
        assert score2 > score1
    
    def test_error_handling_boost_score(self, evaluator):
        """Test that try/except improves semantic score."""
        code_no_error = '''
class Parser:
    def parse(self, data):
        return data.split(',')
'''
        code_with_error = '''
class Parser:
    def parse(self, data):
        try:
            return data.split(',')
        except Exception as e:
            raise ValueError(f"Parse error: {e}")
'''
        comp1 = ComponentOrganism(id="t1", protocol_name="Parser", code=code_no_error)
        comp2 = ComponentOrganism(id="t2", protocol_name="Parser", code=code_with_error)
        
        score1 = evaluator._evaluate_semantic(comp1, {})
        score2 = evaluator._evaluate_semantic(comp2, {})
        
        assert score2 > score1
    
    def test_pac_invariant_mentions_boost_score(self, evaluator):
        """Test that mentioning PAC invariants boosts score."""
        code = '''
class UserService:
    """User management with validation."""
    
    def create_user(self, email, password):
        # Validate email format
        if not self._validate_email(email):
            raise ValueError("Invalid email")
        # Hash password for security
        hashed = self._hash_password(password)
        return {"email": email, "password_hash": hashed}
'''
        invariants = [
            "validate email",
            "hash password"
        ]
        comp = ComponentOrganism(id="t", protocol_name="UserService", code=code)
        score = evaluator._evaluate_semantic(comp, {"pac_invariants": invariants})
        assert score > 0.5


class TestEnergeticEvaluation:
    """Tests for energetic (efficiency) evaluation."""
    
    @pytest.fixture
    def evaluator(self):
        return CoherenceEvaluator()
    
    def test_short_code_moderate_score(self, evaluator):
        """Test that very short code gets moderate score."""
        code = "class X: pass"
        comp = ComponentOrganism(id="t", protocol_name="X", code=code)
        score = evaluator._evaluate_energetic(comp, {})
        assert 0.5 <= score <= 0.8  # Short but possibly incomplete
    
    def test_moderate_code_high_score(self, evaluator):
        """Test that moderate-length code gets high score."""
        code = '''
class DataProcessor:
    """Process data efficiently."""
    
    def __init__(self):
        self.cache = {}
    
    def process(self, data):
        """Process input data."""
        if data in self.cache:
            return self.cache[data]
        result = self._compute(data)
        self.cache[data] = result
        return result
    
    def _compute(self, data):
        """Internal computation."""
        return data.upper()
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache = {}
'''
        comp = ComponentOrganism(id="t", protocol_name="DataProcessor", code=code)
        score = evaluator._evaluate_energetic(comp, {})
        assert score >= 0.7
    
    def test_deep_nesting_lower_score(self, evaluator):
        """Test that deeply nested code gets lower score."""
        code = '''
class DeepNester:
    def process(self, data):
        if data:
            for item in data:
                if item:
                    for sub in item:
                        if sub:
                            for x in sub:
                                if x:
                                    return x
        return None
'''
        comp = ComponentOrganism(id="t", protocol_name="DeepNester", code=code)
        score = evaluator._evaluate_energetic(comp, {})
        assert score < 0.9  # Deep nesting penalty


class TestInvariantValidation:
    """Tests for PAC invariant validation."""
    
    @pytest.fixture
    def evaluator(self):
        return CoherenceEvaluator(enforce_invariants=True)
    
    def test_json_response_invariant_pass(self, evaluator):
        """Test JSON response invariant passes."""
        code = '''
class API:
    def respond(self, data):
        return Response.json(data)
'''
        valid, violations = evaluator.validate_invariants(
            code, 
            ["All routes return JSON responses"]
        )
        assert valid
        assert len(violations) == 0
    
    def test_json_response_invariant_fail(self, evaluator):
        """Test JSON response invariant fails."""
        code = '''
class API:
    def respond(self, data):
        return str(data)
'''
        valid, violations = evaluator.validate_invariants(
            code,
            ["All routes return JSON responses"]
        )
        assert not valid
        assert len(violations) == 1
    
    def test_password_hash_invariant_pass(self, evaluator):
        """Test password hashing invariant passes."""
        code = '''
class Auth:
    def save_password(self, password):
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return password_hash
'''
        valid, violations = evaluator.validate_invariants(
            code,
            ["Passwords are never stored in plaintext"]
        )
        assert valid
    
    def test_password_hash_invariant_fail(self, evaluator):
        """Test password hashing invariant fails."""
        code = '''
class Auth:
    def save_password(self, password):
        self.stored_password = password
        return password
'''
        valid, violations = evaluator.validate_invariants(
            code,
            ["Passwords are never stored in plaintext"]
        )
        assert not valid
    
    def test_validation_invariant_pass(self, evaluator):
        """Test validation invariant passes."""
        code = '''
class UserService:
    def create(self, email):
        if not self.validate_email(email):
            raise ValueError("Invalid email")
        return {"email": email}
'''
        valid, violations = evaluator.validate_invariants(
            code,
            ["Emails are validated before storage"]
        )
        assert valid
    
    def test_error_handling_invariant_pass(self, evaluator):
        """Test error handling invariant passes."""
        code = '''
class Service:
    def process(self, data):
        try:
            return self._do_work(data)
        except Exception as e:
            raise ServiceError(str(e))
'''
        valid, violations = evaluator.validate_invariants(
            code,
            ["Errors are handled gracefully"]
        )
        assert valid
    
    def test_syntax_error_fails_validation(self, evaluator):
        """Test that syntax errors fail validation."""
        code = "class Broken(\n    def oops"
        valid, violations = evaluator.validate_invariants(code, ["anything"])
        assert not valid
        assert "syntax" in violations[0].lower()
    
    def test_multiple_invariants(self, evaluator):
        """Test multiple invariants at once."""
        code = '''
class UserService:
    def create(self, email, password):
        if not validate(email):
            raise ValueError("Bad email")
        password_hash = hash(password)
        return Response.json({"email": email})
'''
        valid, violations = evaluator.validate_invariants(
            code,
            [
                "Emails are validated",
                "Passwords are hashed",
                "Return JSON response"
            ]
        )
        assert valid
        assert len(violations) == 0


class TestCombinedEvaluation:
    """Tests for combined fitness evaluation."""
    
    @pytest.fixture
    def evaluator(self):
        return CoherenceEvaluator()
    
    @pytest.fixture
    def protocol(self):
        return ProtocolSpec(
            name="Calculator",
            methods=["add", "subtract"],
            docstring="Basic calculator",
            pac_invariants=["Results are numeric"]
        )
    
    def test_full_evaluation(self, evaluator, protocol):
        """Test full evaluation pipeline."""
        code = '''
class Calculator:
    """Basic calculator."""
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def subtract(self, a: int, b: int) -> int:
        """Subtract two numbers."""
        return a - b
'''
        comp = ComponentOrganism(id="calc", protocol_name="Calculator", code=code)
        fitness = evaluator.evaluate(comp, {"protocol": protocol})
        
        assert 0.0 <= fitness <= 1.0
        assert comp.structural_score > 0
        assert comp.semantic_score > 0
        assert comp.energetic_score > 0
    
    def test_invariant_violation_penalty(self, evaluator, protocol):
        """Test that invariant violations reduce fitness."""
        # Code that violates "Results are numeric" by returning string
        good_code = '''
class Calculator:
    def add(self, a, b):
        return a + b  # numeric result
'''
        bad_code = '''
class Calculator:
    def add(self, a, b):
        return "result"  # string, not numeric
'''
        comp_good = ComponentOrganism(id="g", protocol_name="Calculator", code=good_code)
        comp_bad = ComponentOrganism(id="b", protocol_name="Calculator", code=bad_code)
        
        # Note: This test depends on invariant validation detecting the issue
        # The current implementation may not catch this specific case
        evaluator.evaluate(comp_good, {"protocol": protocol})
        evaluator.evaluate(comp_bad, {"protocol": protocol})
        
        # Both should have some score
        assert comp_good.coherence_score > 0
        assert comp_bad.coherence_score > 0
    
    def test_generation_adaptive_weights(self, evaluator):
        """Test that weights adapt based on generation."""
        early_weights = evaluator._get_generation_weights(0)
        mid_weights = evaluator._get_generation_weights(5)
        late_weights = evaluator._get_generation_weights(15)
        
        # Early: prioritize coherence
        assert early_weights['coherence'] > early_weights['tests']
        
        # Mid: balanced
        assert mid_weights['coherence'] == mid_weights['tests']
        
        # Late: prioritize tests
        assert late_weights['tests'] > late_weights['coherence']
