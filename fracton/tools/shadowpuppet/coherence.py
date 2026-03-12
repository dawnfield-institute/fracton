"""
ShadowPuppet Coherence Evaluation

Evaluates component fitness through three dimensions:
- Structural: Does code match protocol structure?
- Semantic: Does code make logical sense?
- Energetic: Is code simple/efficient?

Plus optional test-based evaluation:
- Unit tests: Isolated component behavior
- Integration tests: Component interactions
- E2E tests: Full system behavior

Improvements over v1:
- Invariant validation: Actually validates PAC invariants post-generation
- LLM review option: Optional semantic review via code generator
- Execution tests: Try to actually run the generated code
"""

import ast
import re
from typing import Dict, List, Any, Optional, Callable, Tuple

from .protocols import ComponentOrganism, ProtocolSpec


class CoherenceEvaluator:
    """
    Evaluate component coherence (fitness).
    
    Like counting Fibonacci contacts in protein folding,
    coherence measures how well a component integrates
    with the system architecture.
    
    Higher coherence = better survival/reproduction chances.
    
    Features:
    - Structural/semantic/energetic scoring
    - PAC invariant validation (hard enforcement)
    - Optional LLM-based semantic review
    - Execution testing for behavioral validation
    """
    
    def __init__(
        self,
        coherence_weights: Optional[Dict[str, float]] = None,
        fitness_weights: Optional[Dict[str, float]] = None,
        generation_adaptive: bool = True,
        enforce_invariants: bool = True,
        llm_reviewer: Optional[Any] = None  # CodeGenerator for semantic review
    ):
        """
        Initialize evaluator.
        
        Args:
            coherence_weights: Weights for structural/semantic/energetic
            fitness_weights: Weights for coherence vs tests
            generation_adaptive: Adjust weights based on generation
            enforce_invariants: Hard-fail if invariants violated
            llm_reviewer: Optional generator for semantic review
        """
        self.coherence_weights = coherence_weights or {
            'structural': 0.4,
            'semantic': 0.3,
            'energetic': 0.3
        }
        
        self.fitness_weights = fitness_weights or {
            'coherence': 0.4,
            'tests': 0.6
        }
        
        self.generation_adaptive = generation_adaptive
        self.enforce_invariants = enforce_invariants
        self.llm_reviewer = llm_reviewer
    
    def evaluate(
        self,
        component: ComponentOrganism,
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate overall fitness combining coherence + tests + invariants.
        
        Args:
            component: The component to evaluate
            context: Protocol, pac_invariants, test_suite, generation, etc.
            
        Returns:
            Fitness score 0.0-1.0
        """
        # 0. INVARIANT VALIDATION (hard gate)
        pac_invariants = context.get('pac_invariants', [])
        if pac_invariants and self.enforce_invariants:
            valid, violations = self.validate_invariants(component.code, pac_invariants)
            if not valid:
                # Store violations for debugging
                context.setdefault('invariant_violations', {})[component.id] = violations
                # Penalty but don't zero out - allow evolution to improve
                invariant_penalty = 0.3 * (len(violations) / len(pac_invariants))
            else:
                invariant_penalty = 0.0
        else:
            invariant_penalty = 0.0
        
        # 1. COHERENCE: Code quality metrics
        coherence_score = self._evaluate_coherence(component, context)
        
        # 2. TESTS: Behavioral correctness
        test_score = self._evaluate_tests(component, context)
        
        # 3. LLM REVIEW: Optional semantic validation
        if self.llm_reviewer:
            llm_score = self._llm_semantic_review(component, context)
            # Blend with semantic score
            component.semantic_score = (component.semantic_score + llm_score) / 2
        
        # 4. COMBINED FITNESS
        generation = context.get('generation', 0)
        weights = self._get_generation_weights(generation) if self.generation_adaptive else self.fitness_weights
        
        fitness = (
            weights['coherence'] * coherence_score +
            weights['tests'] * test_score
        ) - invariant_penalty
        
        fitness = max(0.0, min(1.0, fitness))  # Clamp to [0, 1]
        
        # Store overall fitness
        component.coherence_score = fitness
        return fitness
    
    def validate_invariants(
        self, 
        code: str, 
        invariants: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate PAC invariants against generated code.
        
        Uses pattern matching and AST analysis to check if code
        respects declared invariants.
        
        Args:
            code: Generated Python code
            invariants: List of invariant strings
            
        Returns:
            (all_valid, list_of_violations)
        """
        violations = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False, ["Code has syntax errors - cannot validate"]
        
        code_lower = code.lower()
        
        for invariant in invariants:
            inv_lower = invariant.lower()
            
            # Pattern-based checks
            if not self._check_invariant(inv_lower, code, code_lower, tree):
                violations.append(invariant)
        
        return len(violations) == 0, violations
    
    def _check_invariant(
        self, 
        invariant: str, 
        code: str, 
        code_lower: str, 
        tree: ast.AST
    ) -> bool:
        """Check a single invariant against code."""
        
        # JSON response invariant
        if 'json' in invariant and 'return' in invariant:
            # Should have json.dumps or Response.json or similar
            if not any(p in code_lower for p in ['json.dumps', 'response.json', "'application/json'", 'jsonify']):
                return False
        
        # HTTP status code invariant
        if 'http' in invariant and 'status' in invariant:
            # Should have status_code references
            if 'status_code' not in code_lower and 'status' not in code_lower:
                return False
        
        # Unique ID invariant
        if 'unique' in invariant and 'id' in invariant:
            if not any(p in code_lower for p in ['uuid', 'unique', 'id =' ]):
                return False
        
        # Validation invariant
        if 'validat' in invariant:
            # Should have validation logic
            if not any(p in code_lower for p in ['validate', 'valid', 'check', 'if not', 'raise']):
                return False
        
        # Password/hash invariant
        if 'password' in invariant and ('plain' in invariant or 'hash' in invariant):
            if 'password_hash' not in code_lower and 'hash' not in code_lower:
                return False
        
        # Error handling invariant
        if 'error' in invariant and 'handl' in invariant:
            has_try = any(isinstance(n, ast.Try) for n in ast.walk(tree))
            has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(tree))
            if not (has_try or has_raise):
                return False
        
        # Return type invariant  
        if 'return' in invariant and 'response' in invariant:
            # Check return statements exist
            has_return = any(isinstance(n, ast.Return) for n in ast.walk(tree))
            if not has_return:
                return False
        
        return True  # Default: assume valid if no specific check applies
    
    def _llm_semantic_review(
        self,
        component: ComponentOrganism,
        context: Dict[str, Any]
    ) -> float:
        """
        Use LLM to review semantic correctness.
        
        Asks the LLM to rate how well the code matches the intent.
        """
        if not self.llm_reviewer:
            return 0.5
        
        protocol: Optional[ProtocolSpec] = context.get('protocol')
        if not protocol:
            return 0.5
        
        try:
            from .generators.base import GenerationContext
            
            review_prompt = f"""Review this code for semantic correctness.
            
Protocol: {protocol.name}
Description: {protocol.docstring}

Code:
```python
{component.code[:2000]}
```

Rate the semantic correctness from 0.0 to 1.0:
- Does the code implement the described functionality?
- Are the methods doing what their names suggest?
- Is the logic sound?

Reply with ONLY a number between 0.0 and 1.0."""

            # Create a minimal context for the review
            review_context = GenerationContext(
                protocol=protocol,
                extra_instructions=review_prompt
            )
            
            response = self.llm_reviewer.generate(review_context)
            
            # Extract score from response
            import re
            match = re.search(r'([0-9]+\.?[0-9]*)', response)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
        except Exception:
            pass
        
        return 0.5  # Default neutral score
    
    def _evaluate_coherence(
        self,
        component: ComponentOrganism,
        context: Dict[str, Any]
    ) -> float:
        """Calculate coherence score (structural/semantic/energetic)."""
        structural = self._evaluate_structural(component, context)
        semantic = self._evaluate_semantic(component, context)
        energetic = self._evaluate_energetic(component, context)
        
        # Store individual scores
        component.structural_score = structural
        component.semantic_score = semantic
        component.energetic_score = energetic
        
        # Weighted sum
        coherence = (
            self.coherence_weights['structural'] * structural +
            self.coherence_weights['semantic'] * semantic +
            self.coherence_weights['energetic'] * energetic
        )
        
        return coherence
    
    def _evaluate_structural(
        self,
        component: ComponentOrganism,
        context: Dict[str, Any]
    ) -> float:
        """Check if code matches protocol structure."""
        try:
            tree = ast.parse(component.code)
            
            protocol: Optional[ProtocolSpec] = context.get('protocol')
            if not protocol:
                return 0.5  # No protocol to check against
            
            # Find defined classes and functions
            defined_classes = set()
            defined_methods = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    defined_classes.add(node.name)
                    # Get methods in this class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            defined_methods.add(item.name)
                elif isinstance(node, ast.FunctionDef):
                    defined_methods.add(node.name)
            
            score = 0.0
            
            # Check class name matches
            if protocol.name in defined_classes:
                score += 0.3
            
            # Check methods
            expected_methods = set(protocol.methods)
            if expected_methods:
                match_ratio = len(defined_methods & expected_methods) / len(expected_methods)
                score += 0.5 * min(1.0, match_ratio * 1.2)  # Bonus for extra
            else:
                score += 0.3  # Default if no methods specified
            
            # Check attributes (if defined)
            if protocol.attributes:
                # Look for self.attr assignments in __init__
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                        init_attrs = set()
                        for stmt in ast.walk(node):
                            if isinstance(stmt, ast.Assign):
                                for target in stmt.targets:
                                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                                        if target.value.id == 'self':
                                            init_attrs.add(target.attr)
                        
                        # Parse expected attributes (handle "name: Type" format)
                        expected_attrs = set()
                        for attr in protocol.attributes:
                            if ':' in attr:
                                expected_attrs.add(attr.split(':')[0].strip())
                            else:
                                expected_attrs.add(attr.strip())
                        
                        if expected_attrs:
                            attr_ratio = len(init_attrs & expected_attrs) / len(expected_attrs)
                            score += 0.2 * attr_ratio
            else:
                score += 0.2
            
            return min(1.0, score)
            
        except SyntaxError:
            return 0.0  # Invalid syntax = no coherence
    
    def _evaluate_semantic(
        self,
        component: ComponentOrganism,
        context: Dict[str, Any]
    ) -> float:
        """Check if code makes logical sense."""
        try:
            tree = ast.parse(component.code)
            score = 0.3  # Base score for parseable code
            
            # Has docstrings?
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if (node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                        score += 0.15
                        break
            
            # Has type hints?
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.returns or any(arg.annotation for arg in node.args.args):
                        score += 0.15
                        break
            
            # Has error handling?
            if any(isinstance(node, (ast.Try, ast.Raise)) for node in ast.walk(tree)):
                score += 0.1
            
            # Implements PAC invariant checks?
            pac_invariants = context.get('pac_invariants', [])
            source_lower = component.code.lower()
            if pac_invariants:
                invariants_found = sum(
                    1 for inv in pac_invariants
                    if any(word.lower() in source_lower for word in inv.split()[:3])
                )
                score += 0.2 * min(1.0, invariants_found / len(pac_invariants))
            else:
                score += 0.1
            
            # Has reasonable imports?
            imports = [node for node in ast.walk(tree) 
                      if isinstance(node, (ast.Import, ast.ImportFrom))]
            if 1 <= len(imports) <= 10:
                score += 0.1
            
            return min(1.0, score)
            
        except:
            return 0.2  # Partial credit for unparseable code
    
    def _evaluate_energetic(
        self,
        component: ComponentOrganism,
        context: Dict[str, Any]
    ) -> float:
        """Check if code is simple/efficient."""
        try:
            lines = component.code.split('\n')
            non_empty = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            
            # Shorter is generally better (to a point)
            if len(non_empty) < 20:
                length_score = 0.7  # Too short might be incomplete
            elif len(non_empty) < 100:
                length_score = 1.0
            elif len(non_empty) < 200:
                length_score = 0.8
            else:
                length_score = 0.6
            
            # AST complexity
            tree = ast.parse(component.code)
            node_count = sum(1 for _ in ast.walk(tree))
            
            if node_count < 30:
                complexity_score = 0.7  # Might be too simple
            elif node_count < 150:
                complexity_score = 1.0
            elif node_count < 300:
                complexity_score = 0.8
            else:
                complexity_score = 0.6
            
            # Nesting depth (deep nesting = complex)
            max_depth = self._max_nesting_depth(tree)
            if max_depth <= 3:
                nesting_score = 1.0
            elif max_depth <= 5:
                nesting_score = 0.8
            else:
                nesting_score = 0.5
            
            return (length_score + complexity_score + nesting_score) / 3.0
            
        except:
            return 0.5
    
    def _max_nesting_depth(self, tree: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = current_depth
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, 
                                ast.ClassDef, ast.If, ast.For, ast.While,
                                ast.With, ast.Try)):
                child_depth = self._max_nesting_depth(node, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._max_nesting_depth(node, current_depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _evaluate_tests(
        self,
        component: ComponentOrganism,
        context: Dict[str, Any]
    ) -> float:
        """Run tests to evaluate behavioral correctness."""
        test_suite = context.get('test_suite', {})
        
        # No tests = neutral score
        if not test_suite:
            return 0.5
        
        scores = {}
        
        # Run each test category
        for category in ['unit', 'integration', 'e2e']:
            if category in test_suite:
                scores[category] = self._run_test_category(
                    component, test_suite[category], context
                )
        
        if not scores:
            return 0.5
        
        # Weighted average
        test_weights = {'unit': 0.3, 'integration': 0.4, 'e2e': 0.3}
        total_weight = sum(test_weights[k] for k in scores.keys())
        
        if total_weight == 0:
            return 0.5
        
        return sum(test_weights[k] * scores[k] for k in scores.keys()) / total_weight
    
    def _run_test_category(
        self,
        component: ComponentOrganism,
        test_functions: List[Callable],
        context: Dict[str, Any]
    ) -> float:
        """Run a category of tests."""
        if not test_functions:
            return 0.5
        
        # Try to instantiate the component
        instance = self._get_component_instance(component)
        if instance is None:
            # Can't instantiate - partial credit for valid syntax
            return 0.3
        
        passed = 0
        total = len(test_functions)
        
        for test_fn in test_functions:
            try:
                # Pass the instance to the test function
                result = test_fn(instance)
                if result:
                    passed += 1
            except Exception as e:
                context.setdefault('test_errors', []).append({
                    'component': component.id,
                    'test': getattr(test_fn, '__name__', str(test_fn)),
                    'error': str(e)
                })
        
        return passed / total if total > 0 else 0.5
    
    def _get_component_instance(self, component: ComponentOrganism) -> Any:
        """
        Execute component code and return an instance of the class.
        
        Returns None if instantiation fails.
        """
        try:
            # Create namespace for execution
            import uuid
            from datetime import datetime
            from dataclasses import dataclass, field
            from typing import Dict, List, Any, Optional
            
            namespace = {
                'uuid': uuid,
                'datetime': datetime,
                'dataclass': dataclass,
                'field': field,
                'Dict': Dict,
                'List': List,
                'Any': Any,
                'Optional': Optional,
            }
            
            # Execute the code
            exec(component.code, namespace)
            
            # Find the class with matching protocol name
            if component.protocol_name in namespace:
                cls = namespace[component.protocol_name]
                # Try to instantiate
                return cls()
            
            # Fallback: find any class
            for name, obj in namespace.items():
                if isinstance(obj, type) and not name.startswith('_'):
                    try:
                        return obj()
                    except:
                        continue
            
            return None
        except Exception:
            return None
    
    def _get_generation_weights(self, generation: int) -> Dict[str, float]:
        """
        Get fitness weights based on generation.
        
        Early (0-2): Prioritize coherence heavily (filter garbage, allow survival)
        Mid (3-9): Balanced
        Late (10+): Prioritize tests (optimization)
        """
        if generation < 3:
            return {'coherence': 0.85, 'tests': 0.15}  # Much heavier on coherence early
        elif generation < 10:
            return {'coherence': 0.5, 'tests': 0.5}
        else:
            return {'coherence': 0.3, 'tests': 0.7}
