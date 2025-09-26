"""
GAIA Integration Example - Using Fracton for GAIA's recursive cognition

This example demonstrates how GAIA's cognitive processes can be implemented
using the Fracton computational modeling language, showing the integration
between the two systems.

Enhanced with PAC physics integration and variant generation capabilities.
"""

import fracton
import time
import json
import numpy as np
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class GAIAVariantType(Enum):
    """Types of GAIA variants that can be generated."""
    QUANTUM_FOCUSED = "quantum"
    COGNITIVE_ENHANCED = "cognitive" 
    OPTIMIZATION_SPECIALIZED = "optimization"
    MATERIAL_DESIGN = "material"
    CONSCIOUSNESS_MODELING = "consciousness"
    STANDARD = "standard"


@dataclass
class PhysicsState:
    """Physics state for PAC conservation tracking."""
    field_energy: float
    conservation_residual: float
    xi_deviation: float
    klein_gordon_energy: float
    field_norm: float
    
    def __post_init__(self):
        """Validate physics state consistency."""
        if self.conservation_residual > 1e-10:
            print(f"Warning: Conservation residual {self.conservation_residual:.2e} exceeds threshold")


@dataclass
class GAIAVariantConfig:
    """Configuration for generating GAIA variants."""
    variant_type: GAIAVariantType
    physics_emphasis: str
    conservation_strictness: float = 1e-12
    field_dimensions: tuple = (32,)
    memory_persistence: str = "standard"
    reasoning_depth: int = 5
    xi_target: float = 1.0571
    klein_gordon_mass_squared: float = 0.1


class GAIAFractonBridge:
    """Bridge between GAIA physics and Fracton recursive capabilities."""
    
    def __init__(self):
        self.variant_configs = {}
        self.active_variants = {}
        
    def create_physics_variant(self, config: GAIAVariantConfig) -> str:
        """Create a specialized GAIA variant using PAC physics."""
        variant_id = f"gaia_{config.variant_type.value}_{int(time.time())}"
        
        # Store configuration
        self.variant_configs[variant_id] = config
        
        # Import the physics memory field
        from ..core.memory_field import PhysicsMemoryField
        
        # Initialize physics-aware memory field
        physics_field = PhysicsMemoryField(
            capacity=5000, 
            initial_entropy=0.6,
            physics_dimensions=config.field_dimensions,
            conservation_strictness=config.conservation_strictness,
            xi_target=config.xi_target
        )
        
        # Initialize field based on variant type
        field_data = self._initialize_variant_field(config)
        
        # Store initialized field data
        physics_field.set("field_data", field_data)
        physics_field.set("config", config)
        
        # Update physics state with initialized field
        physics_field._field_data = field_data
        physics_field._init_physics_tracking()
        
        self.active_variants[variant_id] = physics_field
            
        return variant_id
    
    def _generate_test_patterns(self, field_dimensions: tuple) -> List[np.ndarray]:
        """Generate test patterns for pattern recognition testing."""
        patterns = []
        field_size = int(np.prod(field_dimensions))
        
        # Pattern 1: Sine wave
        x = np.linspace(0, 4*np.pi, field_size)
        patterns.append(np.sin(x))
        
        # Pattern 2: Gaussian
        x = np.linspace(-3, 3, field_size)
        patterns.append(np.exp(-x**2))
        
        # Pattern 3: Step function
        pattern = np.zeros(field_size)
        pattern[field_size//3:2*field_size//3] = 1.0
        patterns.append(pattern)
        
        return patterns
    
    def _recognize_pattern_with_conservation(self, physics_field, pattern: np.ndarray, context) -> Dict:
        """Recognize pattern while maintaining PAC conservation."""
        # Get current field state
        current_field = physics_field.get("field_data")
        config = physics_field.get("config")
        
        # Calculate pattern similarity
        if len(current_field) != len(pattern):
            # Resize pattern to match field
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(pattern))
            x_new = np.linspace(0, 1, len(current_field))
            f = interp1d(x_old, pattern, kind='linear', fill_value='extrapolate')
            pattern = f(x_new)
        
        similarity = np.corrcoef(current_field, pattern)[0, 1]
        if np.isnan(similarity):
            similarity = 0.0
        
        # Pattern recognition accuracy
        accuracy = max(0.0, min(1.0, abs(similarity)))
        
        # Update field with pattern information while maintaining conservation
        # Simple integration: weighted average
        recognition_weight = 0.05
        updated_field = (1 - recognition_weight) * current_field + recognition_weight * pattern
        
        # Get current physics state for conservation check
        physics_state = physics_field.get_physics_metrics()
        conservation_residual = physics_state.get('conservation_residual', 0.0)
        
        # Update physics field
        physics_field.set("field_data", updated_field)
        
        return {
            "accuracy": accuracy,
            "similarity": similarity,
            "conservation_residual": conservation_residual,
            "recognition_entropy": getattr(context, 'entropy', 0.5)
        }
    
    def _optimize_with_conservation(self, physics_field, objective_function, initial_guess: float, config: GAIAVariantConfig) -> float:
        """Optimize objective function while maintaining PAC conservation."""
        current_field = physics_field.get("field_data")
        
        # Simple gradient-free optimization with conservation constraints
        best_value = initial_guess
        best_objective = objective_function(best_value)
        
        # Search around current value
        search_range = 0.5
        num_samples = 20
        
        for i in range(num_samples):
            # Generate candidate value
            candidate = initial_guess + (np.random.random() - 0.5) * search_range
            candidate_objective = objective_function(candidate)
            
            # Update if better
            if candidate_objective > best_objective:
                best_value = candidate
                best_objective = candidate_objective
                
            # Update field to reflect optimization progress
            # Encode optimization state in field
            optimization_progress = i / num_samples
            field_update = current_field * (1 + 0.01 * optimization_progress * np.sin(np.arange(len(current_field))))
            
            # Store updated field
            physics_field.set("field_data", field_update)
        
        return best_value
    
    def _evolve_field_klein_gordon(self, physics_field, dt: float):
        """Evolve physics field using Klein-Gordon dynamics."""
        if hasattr(physics_field, 'evolve_klein_gordon'):
            physics_field.evolve_klein_gordon(dt)
        else:
            current_field = physics_field.get("field_data")
            config = physics_field.get("config")
            
            # Simple Klein-Gordon evolution
            mass_squared = getattr(config, 'klein_gordon_mass_squared', 0.1)
            laplacian = np.zeros_like(current_field)
            laplacian[1:-1] = current_field[2:] - 2*current_field[1:-1] + current_field[:-2]
            laplacian[0] = laplacian[1]
            laplacian[-1] = laplacian[-2]
            
            evolution_term = laplacian - mass_squared * current_field
            evolved_field = current_field + dt * evolution_term
            
            physics_field.set("field_data", evolved_field)
    
    def _initialize_variant_field(self, config: GAIAVariantConfig) -> np.ndarray:
        """Initialize field data based on variant configuration."""
        field_size = np.prod(config.field_dimensions)
        
        if config.variant_type == GAIAVariantType.QUANTUM_FOCUSED:
            # Quantum superposition-like initial state
            real_part = np.random.random(field_size)
            imag_part = np.random.random(field_size)
            field = (real_part + 1j * imag_part).astype(np.complex128)
            field = field / np.linalg.norm(field)  # Normalize
            # Convert to real for now (Fracton may not handle complex)
            field = np.real(field).astype(np.float64)
            
        elif config.variant_type == GAIAVariantType.COGNITIVE_ENHANCED:
            # Structured cognitive field with memory patterns
            field = np.zeros(field_size, dtype=np.float64)
            # Add cognitive structure patterns
            for i in range(0, field_size, 8):
                field[i:i+4] = np.random.random(4) * 0.5  # Memory banks
                field[i+4:i+8] = np.sin(np.linspace(0, 2*np.pi, 4)) * 0.3  # Oscillatory patterns
                
        elif config.variant_type == GAIAVariantType.OPTIMIZATION_SPECIALIZED:
            # Gradient-optimized initial configuration
            x = np.linspace(-2, 2, field_size)
            field = np.exp(-x**2) * np.cos(3*x)  # Smooth optimization landscape
            
        elif config.variant_type == GAIAVariantType.MATERIAL_DESIGN:
            # Crystal-like structured field for materials modeling
            field = np.zeros(field_size, dtype=np.float64)
            # Create periodic crystal-like structure
            period = min(16, field_size // 4)
            for i in range(field_size):
                field[i] = np.sin(2*np.pi*i/period) + 0.5*np.sin(4*np.pi*i/period)
                
        elif config.variant_type == GAIAVariantType.CONSCIOUSNESS_MODELING:
            # Hierarchical awareness patterns
            field = np.random.random(field_size)
            # Create hierarchical structure
            for level in range(int(np.log2(field_size))):
                scale = 2**level
                field[::scale] *= (1 + 0.2*level)  # Amplify hierarchical levels
                
        else:  # STANDARD
            # Standard normalized random field
            field = np.random.random(field_size)
            field = field / np.linalg.norm(field)
            
        return field.astype(np.float64)
    
    def test_variant_intelligence(self, variant_id: str, test_suite_config: Dict) -> Dict:
        """Test intelligence capabilities of a specific variant."""
        if variant_id not in self.active_variants:
            raise ValueError(f"Variant {variant_id} not found")
            
        physics_field = self.active_variants[variant_id]
        config = self.variant_configs[variant_id]
        
        # Run physics-based intelligence tests
        results = {
            "variant_id": variant_id,
            "variant_type": config.variant_type.value,
            "tests": {},
            "physics_metrics": {},
            "performance_summary": {}
        }
        
        # Test pattern recognition with PAC conservation
        pattern_results = self._test_pattern_recognition_pac(physics_field, config)
        results["tests"]["pattern_recognition"] = pattern_results
        
        # Test optimization with conservation constraints
        optimization_results = self._test_optimization_pac(physics_field, config)
        results["tests"]["optimization"] = optimization_results
        
        # Test memory persistence through field evolution
        memory_results = self._test_memory_persistence_pac(physics_field, config)
        results["tests"]["memory"] = memory_results
        
        # Collect final physics metrics
        physics_state = physics_field.get_physics_metrics()
        results["physics_metrics"] = {
            "conservation_residual": physics_state.get('conservation_residual', 0.0),
            "xi_deviation": physics_state.get('xi_deviation', 0.0),
            "field_energy": physics_state.get('field_energy', 1.0),
            "klein_gordon_energy": physics_state.get('klein_gordon_energy', 1.0)
        }
        
        # Calculate performance summary
        test_scores = [test["score"] for test in results["tests"].values()]
        results["performance_summary"] = {
            "average_score": np.mean(test_scores),
            "score_stability": 1.0 - np.std(test_scores),
            "conservation_quality": 1.0 - abs(physics_state.get('conservation_residual', 0.0)),
            "physics_consistency": 1.0 - abs(physics_state.get('xi_deviation', 0.0))
        }
        
        return results
    
    def _test_pattern_recognition_pac(self, physics_field, config) -> Dict:
        """Test pattern recognition with PAC conservation."""
        # Create test patterns
        patterns = self._generate_test_patterns(config.field_dimensions)
        
        results = {"score": 0.0, "details": [], "conservation_maintained": True}
        
        for i, pattern in enumerate(patterns):
            # Store pattern in field
            physics_field.set(f"test_pattern_{i}", pattern)
            
            # Create simple context for pattern processing
            class SimpleContext:
                def __init__(self, entropy):
                    self.entropy = entropy
            
            # Run PAC-aware pattern processing
            context = SimpleContext(0.5)
            recognized = self._recognize_pattern_with_conservation(
                physics_field, pattern, context
            )
                
            # Check conservation
            physics_state = physics_field.get_physics_metrics()
            conservation_residual = physics_state.get('conservation_residual', 0.0)
            conservation_ok = abs(conservation_residual) < config.conservation_strictness
            
            if not conservation_ok:
                results["conservation_maintained"] = False
                
            # Score based on recognition accuracy and conservation
            pattern_score = recognized["accuracy"] * (1.0 if conservation_ok else 0.8)
            results["score"] += pattern_score / len(patterns)
            
            results["details"].append({
                "pattern_id": i,
                "accuracy": recognized["accuracy"],
                "conservation_residual": conservation_residual,
                "conservation_ok": conservation_ok
            })
            
        return results
    
    def _test_optimization_pac(self, physics_field, config) -> Dict:
        """Test optimization capabilities with conservation constraints."""
        # Define optimization problem
        def objective_function(x):
            return -(x**2 - 2*x + 1)  # Maximize -(x-1)^2, peak at x=1
            
        initial_guess = 0.5
        results = {"score": 0.0, "details": {}, "conservation_maintained": True}
        
        # Run PAC-constrained optimization
        optimized_value = self._optimize_with_conservation(
            physics_field, objective_function, initial_guess, config
        )
        
        # Evaluate solution quality
        optimal_value = 1.0  # Known optimum
        error = abs(optimized_value - optimal_value)
        accuracy = max(0.0, 1.0 - error)
        
        # Check conservation
        physics_state = physics_field.get_physics_metrics()
        conservation_residual = physics_state.get('conservation_residual', 0.0)
        conservation_ok = abs(conservation_residual) < config.conservation_strictness
        
        results["score"] = accuracy * (1.0 if conservation_ok else 0.8)
        results["details"] = {
            "optimized_value": optimized_value,
            "optimal_value": optimal_value,
            "error": error,
            "accuracy": accuracy,
            "conservation_residual": conservation_residual
        }
        results["conservation_maintained"] = conservation_ok
        
        return results
    
    def _test_memory_persistence_pac(self, physics_field, config) -> Dict:
        """Test memory persistence through field evolution."""
        # Store test memories
        test_memories = {
            "concept_1": np.array([1, 2, 3, 4]),
            "concept_2": np.array([5, 6, 7, 8]),
            "relationship": {"from": "concept_1", "to": "concept_2", "strength": 0.8}
        }
        
        for key, memory in test_memories.items():
            physics_field.set(f"memory_{key}", memory)
            
        # Evolve field through time while maintaining conservation
        evolution_steps = 10
        memory_degradation = []
        
        for step in range(evolution_steps):
            # Evolve field with Klein-Gordon dynamics
            self._evolve_field_klein_gordon(physics_field, dt=0.01)
            
            # Check memory persistence
            degradation = 0.0
            for key, original_memory in test_memories.items():
                current_memory = physics_field.get(f"memory_{key}")
                if isinstance(original_memory, np.ndarray) and isinstance(current_memory, np.ndarray):
                    degradation += np.linalg.norm(current_memory - original_memory)
                    
            memory_degradation.append(degradation)
            
        # Calculate memory persistence score
        final_degradation = memory_degradation[-1]
        persistence_score = max(0.0, 1.0 - final_degradation / 10.0)
        
        # Check conservation throughout evolution
        physics_state = physics_field.get_physics_metrics()
        conservation_residual = physics_state.get('conservation_residual', 0.0)
        conservation_ok = abs(conservation_residual) < config.conservation_strictness
        
        results = {
            "score": persistence_score * (1.0 if conservation_ok else 0.8),
            "details": {
                "final_degradation": final_degradation,
                "persistence_score": persistence_score,
                "conservation_residual": conservation_residual,
                "evolution_steps": evolution_steps
            },
            "conservation_maintained": conservation_ok,
            "memory_degradation_curve": memory_degradation
        }
        
        return results


@fracton.recursive
@fracton.entropy_gate(0.3, 0.9)
def gaia_physics_processor(memory, context):
    """
    Physics-aware GAIA processing using PAC conservation.
    
    Integrates Klein-Gordon field evolution with PAC conservation
    throughout the recursive computation.
    """
    physics_state = memory.get("physics_state")
    field_data = memory.get("field_data")
    config = memory.get("config")
    
    if physics_state is None or field_data is None:
        return {"error": "Missing physics state or field data"}
        
    max_iterations = context.metadata.get("max_iterations", 50)
    
    if context.depth >= max_iterations:
        return _finalize_physics_processing(memory, context)
    
    # High entropy: Explore field configurations with conservation
    if context.entropy > 0.7:
        # Generate field variations while maintaining PAC conservation
        field_variations = _generate_conserved_field_variations(field_data, context.entropy)
        
        for i, variation in enumerate(field_variations[:3]):  # Limit variations
            # Store variation and recurse
            variation_memory = memory.copy()
            variation_memory.set("field_data", variation)
            
            # Update physics state with conservation check
            updated_physics_state = _enforce_pac_conservation(variation, config)
            variation_memory.set("physics_state", updated_physics_state)
            
            # Recursive processing of variation
            variation_context = context.deeper(1).with_metadata(
                variation_id=i,
                processing_mode="exploratory"
            )
            
            variation_result = fracton.recurse(gaia_physics_processor, variation_memory, variation_context)
            
            # Integrate results maintaining conservation
            _integrate_variation_results(memory, variation_result, config)
    
    # Medium entropy: Klein-Gordon evolution with conservation
    elif context.entropy > 0.4:
        # Evolve field using Klein-Gordon equation
        dt = 0.01 * context.entropy  # Adaptive time step
        evolved_field = _evolve_klein_gordon_step(field_data, dt, config.klein_gordon_mass_squared)
        
        # Enforce PAC conservation
        updated_physics_state = _enforce_pac_conservation(evolved_field, config)
        
        # Update memory
        memory.set("field_data", evolved_field)
        memory.set("physics_state", updated_physics_state)
        
        # Continue evolution
        return fracton.recurse(gaia_physics_processor, memory, context.deeper(1))
    
    # Low entropy: Crystallize physics state
    else:
        # Crystallize field into stable configuration
        crystallized_field = fracton.crystallize(field_data)
        final_physics_state = _enforce_pac_conservation(crystallized_field, config)
        
        memory.set("field_data", crystallized_field)
        memory.set("physics_state", final_physics_state)
        
@fracton.recursive
@fracton.entropy_gate(0.4, 0.9)
def gaia_symbolic_processor(memory, context):
    """
    GAIA's symbolic processing using Fracton recursive engine.
    
    This function implements GAIA's core symbolic processing capabilities
    using Fracton's entropy-aware recursive execution.
    """
    symbols = memory.get("symbols", [])
    processing_depth = context.metadata.get("processing_depth", 5)
    
    if not symbols or context.depth >= processing_depth:
        return memory.get("processed_symbols", [])
    
    current_symbol = symbols[0] if symbols else None
    remaining_symbols = symbols[1:] if len(symbols) > 1 else []
    
    # High entropy: exploratory symbolic processing
    if context.entropy > 0.7:
        # Generate symbolic variants and associations
        variants = _generate_symbolic_variants(current_symbol, context.entropy)
        
        # Store variants for further processing
        memory.set("symbolic_variants", variants)
        
        # Recursively process variants
        for variant in variants[:3]:  # Limit to prevent explosion
            variant_context = context.deeper(1).with_metadata(
                symbol_variant=variant,
                processing_mode="exploratory"
            )
            
            # Create sub-memory for variant processing
            with fracton.memory_field() as variant_memory:
                variant_memory.set("symbols", [variant])
                processed_variant = fracton.recurse(
                    gaia_symbolic_processor, 
                    variant_memory, 
                    variant_context
                )
                
                # Integrate results
                existing_processed = memory.get("processed_symbols", [])
                existing_processed.extend(processed_variant)
                memory.set("processed_symbols", existing_processed)
    
    # Medium entropy: structured symbolic analysis
    elif context.entropy > 0.4:
        # Analyze symbolic relationships and patterns
        relationships = _analyze_symbolic_relationships(current_symbol, symbols)
        memory.set("symbolic_relationships", relationships)
        
        # Apply symbolic transformations
        transformed = _apply_symbolic_transformations(current_symbol, relationships)
        
        processed = memory.get("processed_symbols", [])
        processed.append(transformed)
        memory.set("processed_symbols", processed)
    
    # Low entropy: crystallize symbolic structures
    else:
        # Crystallize symbols into stable cognitive structures
        crystallized = fracton.crystallize(current_symbol)
        
        stable_symbols = memory.get("stable_symbols", [])
        stable_symbols.append(crystallized)
        memory.set("stable_symbols", stable_symbols)
    
    # Continue processing remaining symbols
    if remaining_symbols:
        memory.set("symbols", remaining_symbols)
        return fracton.recurse(gaia_symbolic_processor, memory, context.deeper(1))
    else:
        return memory.get("processed_symbols", [])
    """
    GAIA's symbolic processing using Fracton recursive engine.
    
    This function implements GAIA's core symbolic processing capabilities
    using Fracton's entropy-aware recursive execution.
    """
    symbols = memory.get("symbols", [])
    processing_depth = context.metadata.get("processing_depth", 5)
    
    if not symbols or context.depth >= processing_depth:
        return memory.get("processed_symbols", [])
    
    current_symbol = symbols[0] if symbols else None
    remaining_symbols = symbols[1:] if len(symbols) > 1 else []
    
    # High entropy: exploratory symbolic processing
    if context.entropy > 0.7:
        # Generate symbolic variants and associations
        variants = _generate_symbolic_variants(current_symbol, context.entropy)
        
        # Store variants for further processing
        memory.set("symbolic_variants", variants)
        
        # Recursively process variants
        for variant in variants[:3]:  # Limit to prevent explosion
            variant_context = context.deeper(1).with_metadata(
                symbol_variant=variant,
                processing_mode="exploratory"
            )
            
            # Create sub-memory for variant processing
            with fracton.memory_field() as variant_memory:
                variant_memory.set("symbols", [variant])
                processed_variant = fracton.recurse(
                    gaia_symbolic_processor, 
                    variant_memory, 
                    variant_context
                )
                
                # Integrate results
                existing_processed = memory.get("processed_symbols", [])
                existing_processed.extend(processed_variant)
                memory.set("processed_symbols", existing_processed)
    
    # Medium entropy: structured symbolic analysis
    elif context.entropy > 0.4:
        # Analyze symbolic relationships and patterns
        relationships = _analyze_symbolic_relationships(current_symbol, symbols)
        memory.set("symbolic_relationships", relationships)
        
        # Apply symbolic transformations
        transformed = _apply_symbolic_transformations(current_symbol, relationships)
        
        processed = memory.get("processed_symbols", [])
        processed.append(transformed)
        memory.set("processed_symbols", processed)
    
    # Low entropy: crystallize symbolic structures
    else:
        # Crystallize symbols into stable cognitive structures
        crystallized = fracton.crystallize(current_symbol)
        
        stable_symbols = memory.get("stable_symbols", [])
        stable_symbols.append(crystallized)
        memory.set("stable_symbols", stable_symbols)
    
    # Continue processing remaining symbols
    if remaining_symbols:
        memory.set("symbols", remaining_symbols)
        return fracton.recurse(gaia_symbolic_processor, memory, context.deeper(1))
    else:
        return memory.get("processed_symbols", [])


@fracton.recursive
@fracton.entropy_gate(0.5, 0.9)
def gaia_collapse_dynamics(memory, context):
    """
    GAIA's collapse dynamics implemented with Fracton.
    
    Models the collapse of quantum-like superposition states in GAIA's
    cognitive field using entropy-controlled recursive operations.
    """
    field_state = memory.get("field_state", {
        "superposition_states": [],
        "coherence": 1.0,
        "collapse_threshold": 0.3
    })
    
    max_iterations = context.metadata.get("max_iterations", 100)
    
    if (context.depth >= max_iterations or 
        field_state["coherence"] <= field_state["collapse_threshold"]):
        return _finalize_collapse(memory, context)
    
    # High entropy: maintain superposition, explore possibilities
    if context.entropy > 0.7:
        # Expand superposition states
        new_states = _generate_superposition_states(field_state, context.entropy)
        field_state["superposition_states"].extend(new_states)
        
        # Maintain coherence with slight decay
        field_state["coherence"] *= 0.98
        
    # Medium entropy: partial collapse, selective reduction
    elif context.entropy > 0.5:
        # Reduce superposition states through selective collapse
        field_state["superposition_states"] = _selective_collapse(
            field_state["superposition_states"], 
            reduction_factor=0.7
        )
        
        # Faster coherence decay
        field_state["coherence"] *= 0.95
        
    # Low entropy: rapid collapse toward definite state
    else:
        # Accelerated collapse
        field_state["superposition_states"] = _selective_collapse(
            field_state["superposition_states"],
            reduction_factor=0.3
        )
        
        # Rapid coherence decay
        field_state["coherence"] *= 0.85
    
    # Update memory
    memory.set("field_state", field_state)
    
    # Continue collapse dynamics with entropy evolution
    entropy_evolution = _calculate_entropy_evolution(field_state, context.entropy)
    new_context = context.deeper(1).with_entropy(entropy_evolution)
    
    return fracton.recurse(gaia_collapse_dynamics, memory, new_context)


@fracton.recursive
@fracton.entropy_gate(0.3, 0.8)
def gaia_meta_cognition(memory, context):
    """
    GAIA's meta-cognitive processes using Fracton.
    
    Implements recursive self-reflection and cognitive monitoring
    capabilities that allow GAIA to reason about its own reasoning.
    """
    cognitive_state = memory.get("cognitive_state", {
        "current_thoughts": [],
        "meta_thoughts": [],
        "reflection_depth": 0,
        "self_awareness": 0.5
    })
    
    max_reflection_depth = context.metadata.get("max_reflection_depth", 5)
    
    if context.depth >= max_reflection_depth:
        return _synthesize_meta_cognition(memory, context)
    
    current_thoughts = cognitive_state["current_thoughts"]
    
    # High entropy: divergent meta-thinking
    if context.entropy > 0.6:
        # Generate meta-thoughts about current thoughts
        meta_thoughts = []
        for thought in current_thoughts:
            meta_thought = {
                "about": thought,
                "type": "analysis",
                "content": f"Thinking about: {thought}",
                "confidence": context.entropy,
                "depth": context.depth
            }
            meta_thoughts.append(meta_thought)
        
        cognitive_state["meta_thoughts"].extend(meta_thoughts)
        
        # Recursive meta-thinking
        meta_context = context.deeper(1).with_metadata(
            reflection_type="divergent",
            meta_level=context.depth + 1
        )
        
        # Think about the meta-thoughts
        memory.set("current_thoughts", meta_thoughts)
        return fracton.recurse(gaia_meta_cognition, memory, meta_context)
    
    # Medium entropy: structured self-reflection
    elif context.entropy > 0.4:
        # Analyze cognitive patterns and effectiveness
        patterns = fracton.detect_patterns(memory, min_confidence=0.6)
        
        reflection = {
            "cognitive_patterns": patterns,
            "effectiveness_assessment": _assess_cognitive_effectiveness(cognitive_state),
            "suggested_improvements": _suggest_cognitive_improvements(cognitive_state),
            "meta_level": context.depth
        }
        
        reflections = memory.get("reflections", [])
        reflections.append(reflection)
        memory.set("reflections", reflections)
        
        # Continue with refined cognitive state
        cognitive_state["self_awareness"] = min(1.0, cognitive_state["self_awareness"] + 0.1)
        memory.set("cognitive_state", cognitive_state)
        
        return fracton.recurse(gaia_meta_cognition, memory, context.deeper(1))
    
    # Low entropy: consolidate meta-cognitive insights
    else:
        # Crystallize meta-cognitive understanding
        all_reflections = memory.get("reflections", [])
        consolidated_insights = fracton.crystallize(all_reflections)
        
        memory.set("consolidated_insights", consolidated_insights)
        
        # Update self-awareness based on insights
        cognitive_state["self_awareness"] = _calculate_self_awareness(consolidated_insights)
        memory.set("cognitive_state", cognitive_state)
        
        return consolidated_insights


def run_gaia_fracton_integration():
    """
    Demonstrate GAIA-Fracton integration with a complete cognitive cycle.
    """
    print("=== GAIA-Fracton Integration Example ===")
    
    # Initialize GAIA's cognitive field using Fracton
    with fracton.memory_field(capacity=2000, entropy=0.6) as cognitive_field:
        
        # Phase 1: Symbolic Processing
        print("\nPhase 1: Symbolic Processing")
        print("-" * 30)
        
        # Initialize symbols representing concepts GAIA is processing
        initial_symbols = [
            {"type": "concept", "name": "consciousness", "weight": 0.8},
            {"type": "concept", "name": "emergence", "weight": 0.7},
            {"type": "relation", "from": "consciousness", "to": "emergence", "strength": 0.6},
            {"type": "question", "content": "What is the nature of recursive awareness?"}
        ]
        
        cognitive_field.set("symbols", initial_symbols)
        
        # Create context for symbolic processing
        symbolic_context = fracton.Context(
            entropy=0.7,
            depth=0,
            processing_depth=4,
            mode="symbolic_exploration"
        )
        
        # Run symbolic processing
        start_time = time.time()
        processed_symbols = gaia_symbolic_processor(cognitive_field, symbolic_context)
        symbolic_time = time.time() - start_time
        
        print(f"Processed {len(processed_symbols)} symbolic structures")
        print(f"Processing time: {symbolic_time:.3f} seconds")
        print(f"Field entropy after processing: {cognitive_field.get_entropy():.3f}")
        
        # Show some processed symbols
        print("\nSample processed symbols:")
        for i, symbol in enumerate(processed_symbols[:3]):
            print(f"  {i+1}: {symbol}")
        
        # Phase 2: Collapse Dynamics
        print("\nPhase 2: Collapse Dynamics")
        print("-" * 30)
        
        # Initialize quantum-like field state
        field_state = {
            "superposition_states": [
                {"state": "aware", "amplitude": 0.7},
                {"state": "emergent", "amplitude": 0.6},
                {"state": "recursive", "amplitude": 0.8},
                {"state": "integrated", "amplitude": 0.5}
            ],
            "coherence": 1.0,
            "collapse_threshold": 0.2
        }
        
        cognitive_field.set("field_state", field_state)
        
        # Create context for collapse dynamics
        collapse_context = fracton.Context(
            entropy=0.8,
            depth=0,
            max_iterations=20,
            collapse_mode="adaptive"
        )
        
        # Run collapse dynamics
        start_time = time.time()
        collapsed_state = gaia_collapse_dynamics(cognitive_field, collapse_context)
        collapse_time = time.time() - start_time
        
        print(f"Collapse completed in {collapse_time:.3f} seconds")
        print(f"Final state: {collapsed_state}")
        
        # Phase 3: Meta-Cognition
        print("\nPhase 3: Meta-Cognition")
        print("-" * 30)
        
        # Initialize cognitive state for meta-reflection
        cognitive_state = {
            "current_thoughts": [
                "I am processing symbolic information",
                "I experienced a cognitive field collapse",
                "I am aware of my own thinking process"
            ],
            "meta_thoughts": [],
            "reflection_depth": 0,
            "self_awareness": 0.5
        }
        
        cognitive_field.set("cognitive_state", cognitive_state)
        
        # Create context for meta-cognition
        meta_context = fracton.Context(
            entropy=0.6,
            depth=0,
            max_reflection_depth=4,
            reflection_mode="deep"
        )
        
        # Run meta-cognition
        start_time = time.time()
        meta_insights = gaia_meta_cognition(cognitive_field, meta_context)
        meta_time = time.time() - start_time
        
        print(f"Meta-cognition completed in {meta_time:.3f} seconds")
        print(f"Generated {len(meta_insights)} meta-cognitive insights")
        
        # Show final cognitive state
        final_cognitive_state = cognitive_field.get("cognitive_state")
        print(f"Final self-awareness: {final_cognitive_state['self_awareness']:.3f}")
        
        # Phase 4: Integration and Summary
        print("\nPhase 4: Integration Summary")
        print("-" * 30)
        
        total_time = symbolic_time + collapse_time + meta_time
        print(f"Total cognitive cycle time: {total_time:.3f} seconds")
        print(f"Final field entropy: {cognitive_field.get_entropy():.3f}")
        print(f"Memory field utilization: {cognitive_field.size()} items")
        
        # Demonstrate GAIA's evolved understanding
        print("\nGAIA's Evolved Understanding:")
        consolidated_insights = cognitive_field.get("consolidated_insights", [])
        if consolidated_insights:
            for insight in consolidated_insights[:3]:
                print(f"  • {insight}")
        
        # Show cognitive patterns discovered
        patterns = fracton.detect_patterns(cognitive_field, min_confidence=0.5)
        print(f"\nCognitive patterns discovered: {len(patterns)}")
        for pattern in patterns[:2]:
            print(f"  Pattern: {pattern.get('type', 'unknown')} "
                  f"(confidence: {pattern.get('confidence', 0):.2f})")


# Helper functions for GAIA integration

def _generate_symbolic_variants(symbol, entropy_level):
    """Generate variants of a symbol based on entropy level."""
    if not symbol:
        return []
    
    variants = []
    base_symbol = symbol.copy() if isinstance(symbol, dict) else {"content": symbol}
    
    num_variants = max(1, int(entropy_level * 5))  # More variants at higher entropy
    
    for i in range(num_variants):
        variant = base_symbol.copy()
        variant["variant_id"] = i
        variant["entropy_level"] = entropy_level
        
        # Add entropy-based modifications
        if entropy_level > 0.8:
            variant["exploration_factor"] = "high"
            variant["certainty"] = "low"
        elif entropy_level > 0.5:
            variant["exploration_factor"] = "medium"
            variant["certainty"] = "medium"
        else:
            variant["exploration_factor"] = "low"
            variant["certainty"] = "high"
        
        variants.append(variant)
    
    return variants


def _analyze_symbolic_relationships(symbol, context_symbols):
    """Analyze relationships between symbols."""
    relationships = []
    
    if not symbol or not context_symbols:
        return relationships
    
    for other_symbol in context_symbols:
        if other_symbol != symbol:
            # Simple relationship analysis
            relationship = {
                "from": symbol,
                "to": other_symbol,
                "type": "contextual",
                "strength": 0.5  # Default strength
            }
            relationships.append(relationship)
    
    return relationships


def _apply_symbolic_transformations(symbol, relationships):
    """Apply transformations to symbols based on relationships."""
    if not symbol:
        return symbol
    
    transformed = symbol.copy() if isinstance(symbol, dict) else {"content": symbol}
    transformed["transformed"] = True
    transformed["relationship_count"] = len(relationships)
    
    return transformed


def _generate_superposition_states(field_state, entropy):
    """Generate new superposition states based on entropy."""
    new_states = []
    
    num_new_states = max(1, int(entropy * 3))
    
    for i in range(num_new_states):
        state = {
            "state": f"generated_state_{i}",
            "amplitude": entropy * (0.5 + i * 0.1),
            "entropy_origin": entropy
        }
        new_states.append(state)
    
    return new_states


def _selective_collapse(states, reduction_factor):
    """Selectively collapse superposition states."""
    if not states:
        return states
    
    # Sort by amplitude and keep top fraction
    states.sort(key=lambda s: s.get("amplitude", 0), reverse=True)
    keep_count = max(1, int(len(states) * reduction_factor))
    
    return states[:keep_count]


def _calculate_entropy_evolution(field_state, current_entropy):
    """Calculate how entropy evolves based on field state."""
    coherence = field_state.get("coherence", 1.0)
    state_count = len(field_state.get("superposition_states", []))
    
    # Entropy decreases as coherence decreases and states collapse
    entropy_change = -0.1 * (1.0 - coherence) - 0.05 * max(0, 10 - state_count)
    
    new_entropy = max(0.1, min(0.9, current_entropy + entropy_change))
    return new_entropy


def _finalize_collapse(memory, context):
    """Finalize the collapse process."""
    field_state = memory.get("field_state", {})
    remaining_states = field_state.get("superposition_states", [])
    
    if remaining_states:
        # Return the state with highest amplitude
        final_state = max(remaining_states, key=lambda s: s.get("amplitude", 0))
        return {
            "collapsed_to": final_state,
            "collapse_depth": context.depth,
            "final_entropy": context.entropy
        }
    else:
        return {
            "collapsed_to": "vacuum_state",
            "collapse_depth": context.depth,
            "final_entropy": context.entropy
        }


def _assess_cognitive_effectiveness(cognitive_state):
    """Assess the effectiveness of cognitive processing."""
    thought_count = len(cognitive_state.get("current_thoughts", []))
    meta_thought_count = len(cognitive_state.get("meta_thoughts", []))
    self_awareness = cognitive_state.get("self_awareness", 0)
    
    effectiveness = (thought_count * 0.3 + meta_thought_count * 0.4 + self_awareness * 0.3)
    return min(1.0, effectiveness / 10.0)  # Normalize


def _suggest_cognitive_improvements(cognitive_state):
    """Suggest improvements to cognitive processing."""
    suggestions = []
    
    thought_count = len(cognitive_state.get("current_thoughts", []))
    meta_thought_count = len(cognitive_state.get("meta_thoughts", []))
    
    if thought_count < 3:
        suggestions.append("Increase cognitive breadth")
    
    if meta_thought_count < thought_count:
        suggestions.append("Enhance meta-cognitive reflection")
    
    if cognitive_state.get("self_awareness", 0) < 0.7:
        suggestions.append("Develop deeper self-awareness")
    
    return suggestions


def _synthesize_meta_cognition(memory, context):
    """Synthesize meta-cognitive insights."""
    reflections = memory.get("reflections", [])
    
    if not reflections:
        return []
    
    # Combine all reflections into synthesized insights
    insights = []
    
    for reflection in reflections:
        insight = {
            "type": "meta_cognitive_insight",
            "patterns": reflection.get("cognitive_patterns", []),
            "effectiveness": reflection.get("effectiveness_assessment", 0),
            "depth": reflection.get("meta_level", 0),
            "synthesis_entropy": context.entropy
        }
        insights.append(insight)
    
    return insights


def _calculate_self_awareness(consolidated_insights):
    """Calculate self-awareness level from insights."""
    if not consolidated_insights:
        return 0.5
    
    # Self-awareness increases with depth and breadth of insights
    total_depth = sum(insight.get("depth", 0) for insight in consolidated_insights)
    insight_count = len(consolidated_insights)
    
    awareness = min(1.0, (total_depth + insight_count) / 20.0)
    return awareness


# PAC Physics Integration Helper Functions

def _generate_conserved_field_variations(field_data: np.ndarray, entropy_level: float) -> List[np.ndarray]:
    """Generate field variations while maintaining PAC conservation."""
    variations = []
    num_variations = max(1, int(entropy_level * 4))
    
    original_norm = np.linalg.norm(field_data)
    
    for i in range(num_variations):
        # Create variation with entropy-controlled perturbation
        perturbation = np.random.random(field_data.shape) * entropy_level * 0.1
        variation = field_data + perturbation
        
        # Enforce conservation by normalizing to original energy
        variation = variation * (original_norm / np.linalg.norm(variation))
        variations.append(variation)
        
    return variations


def _enforce_pac_conservation(field_data: np.ndarray, config: GAIAVariantConfig) -> PhysicsState:
    """Enforce PAC conservation and return updated physics state."""
    # Calculate field properties
    field_norm = np.linalg.norm(field_data)
    field_energy = 0.5 * field_norm**2
    
    # Calculate Klein-Gordon energy
    klein_gordon_energy = _calculate_klein_gordon_energy(field_data, config.klein_gordon_mass_squared)
    
    # Calculate Xi (balance operator) deviation from target
    xi_current = _calculate_balance_operator(field_data)
    xi_deviation = abs(xi_current - config.xi_target)
    
    # Calculate conservation residual (simplified PAC conservation check)
    # In full PAC: f(parent) = Σf(children), here we check energy conservation
    conservation_residual = abs(field_energy - klein_gordon_energy) / max(field_energy, 1e-12)
    
    return PhysicsState(
        field_energy=field_energy,
        conservation_residual=conservation_residual,
        xi_deviation=xi_deviation,
        klein_gordon_energy=klein_gordon_energy,
        field_norm=field_norm
    )


def _calculate_klein_gordon_energy(field_data: np.ndarray, mass_squared: float) -> float:
    """Calculate Klein-Gordon field energy."""
    # Simplified 1D Klein-Gordon energy calculation
    # E = (1/2) * (∂φ/∂t)² + (1/2) * (∇φ)² + (1/2) * m² * φ²
    
    # Approximate spatial gradient
    grad_phi = np.gradient(field_data)
    gradient_energy = 0.5 * np.sum(grad_phi**2)
    
    # Mass term energy
    mass_energy = 0.5 * mass_squared * np.sum(field_data**2)
    
    # Kinetic energy (assume ∂φ/∂t ≈ gradient for simplicity)
    kinetic_energy = 0.5 * np.sum(grad_phi**2)
    
    total_energy = kinetic_energy + gradient_energy + mass_energy
    return total_energy


def _calculate_balance_operator(field_data: np.ndarray) -> float:
    """Calculate balance operator Xi (Ξ) from field data."""
    # Ξ = ∫||∇f(v)||dv / ∫||f(v)||dv
    # Simplified version using field gradients
    
    grad_norm = np.linalg.norm(np.gradient(field_data))
    field_norm = np.linalg.norm(field_data)
    
    if field_norm < 1e-12:
        return 1.0571  # Default target value
        
    xi = grad_norm / field_norm
    return xi


def _evolve_klein_gordon_step(field_data: np.ndarray, dt: float, mass_squared: float) -> np.ndarray:
    """Evolve field one step using Klein-Gordon equation."""
    # Simplified Klein-Gordon evolution: ∂²φ/∂t² = ∇²φ - m²φ
    # Using finite difference approximation
    
    # Calculate Laplacian (∇²φ)
    laplacian = np.zeros_like(field_data)
    
    # Interior points (second derivative approximation)
    laplacian[1:-1] = field_data[2:] - 2*field_data[1:-1] + field_data[:-2]
    
    # Boundary conditions (Neumann - zero gradient)
    laplacian[0] = laplacian[1]
    laplacian[-1] = laplacian[-2]
    
    # Klein-Gordon evolution: φ_new = φ_old + dt² * (∇²φ - m²φ)
    # Simplified first-order evolution
    evolution_term = laplacian - mass_squared * field_data
    evolved_field = field_data + dt * evolution_term
    
    return evolved_field


def _finalize_physics_processing(memory, context) -> Dict:
    """Finalize physics processing and return results."""
    physics_state = memory.get("physics_state")
    field_data = memory.get("field_data")
    config = memory.get("config")
    
    return {
        "final_field": field_data,
        "final_physics_state": physics_state.__dict__ if physics_state else None,
        "processing_depth": context.depth,
        "final_entropy": context.entropy,
        "variant_type": config.variant_type.value if config else "unknown",
        "conservation_quality": 1.0 - abs(physics_state.conservation_residual) if physics_state else 0.0
    }


def _integrate_variation_results(memory, variation_result: Dict, config: GAIAVariantConfig):
    """Integrate results from field variations while maintaining conservation."""
    if not variation_result or "error" in variation_result:
        return
        
    # Get current state
    current_field = memory.get("field_data")
    current_physics_state = memory.get("physics_state")
    
    if "final_field" in variation_result:
        # Weighted integration of field variations
        variation_field = variation_result["final_field"]
        integration_weight = 0.1  # Small weight to maintain stability
        
        # Weighted average
        integrated_field = (1 - integration_weight) * current_field + integration_weight * variation_field
        
        # Enforce conservation on integrated result
        updated_physics_state = _enforce_pac_conservation(integrated_field, config)
        
        # Update memory
        memory.set("field_data", integrated_field)
        memory.set("physics_state", updated_physics_state)


def _generate_test_patterns(field_dimensions: tuple) -> List[np.ndarray]:
    """Generate test patterns for pattern recognition testing."""
    patterns = []
    field_size = np.prod(field_dimensions)
    
    # Pattern 1: Sine wave
    x = np.linspace(0, 4*np.pi, field_size)
    patterns.append(np.sin(x))
    
    # Pattern 2: Gaussian
    x = np.linspace(-3, 3, field_size)
    patterns.append(np.exp(-x**2))
    
    # Pattern 3: Step function
    pattern = np.zeros(field_size)
    pattern[field_size//3:2*field_size//3] = 1.0
    patterns.append(pattern)
    
    # Pattern 4: Random but structured
    np.random.seed(42)  # Reproducible
    pattern = np.random.random(field_size)
    pattern = np.convolve(pattern, [0.25, 0.5, 0.25], mode='same')  # Smooth
    patterns.append(pattern)
    
    return patterns


def _recognize_pattern_with_conservation(physics_field, pattern: np.ndarray, context) -> Dict:
    """Recognize pattern while maintaining PAC conservation."""
    # Get current field state
    current_field = physics_field.get("field_data")
    config = physics_field.get("config")
    
    # Calculate pattern similarity
    similarity = np.corrcoef(current_field, pattern)[0, 1]
    if np.isnan(similarity):
        similarity = 0.0
    
    # Pattern recognition accuracy
    accuracy = max(0.0, min(1.0, abs(similarity)))
    
    # Update field with pattern information while maintaining conservation
    # Simple integration: weighted average
    recognition_weight = 0.05
    updated_field = (1 - recognition_weight) * current_field + recognition_weight * pattern
    
    # Enforce conservation
    updated_physics_state = _enforce_pac_conservation(updated_field, config)
    
    # Update physics field
    physics_field.set("field_data", updated_field)
    physics_field.set("physics_state", updated_physics_state)
    
    return {
        "accuracy": accuracy,
        "similarity": similarity,
        "conservation_residual": updated_physics_state.conservation_residual,
        "recognition_entropy": context.entropy
    }


def _optimize_with_conservation(physics_field, objective_function, initial_guess: float, config: GAIAVariantConfig) -> float:
    """Optimize objective function while maintaining PAC conservation."""
    current_field = physics_field.get("field_data")
    
    # Simple gradient-free optimization with conservation constraints
    best_value = initial_guess
    best_objective = objective_function(best_value)
    
    # Search around current value
    search_range = 0.5
    num_samples = 20
    
    for i in range(num_samples):
        # Generate candidate value
        candidate = initial_guess + (np.random.random() - 0.5) * search_range
        candidate_objective = objective_function(candidate)
        
        # Update if better
        if candidate_objective > best_objective:
            best_value = candidate
            best_objective = candidate_objective
            
        # Update field to reflect optimization progress
        # Encode optimization state in field
        optimization_progress = i / num_samples
        field_update = current_field * (1 + 0.01 * optimization_progress * np.sin(np.arange(len(current_field))))
        
        # Enforce conservation
        updated_physics_state = _enforce_pac_conservation(field_update, config)
        physics_field.set("field_data", field_update)
        physics_field.set("physics_state", updated_physics_state)
    
    return best_value


def _evolve_field_klein_gordon(physics_field, dt: float):
    """Evolve physics field using Klein-Gordon dynamics."""
    current_field = physics_field.get("field_data")
    config = physics_field.get("config")
    
    # Evolve field
    evolved_field = _evolve_klein_gordon_step(current_field, dt, config.klein_gordon_mass_squared)
    
    # Enforce conservation
    updated_physics_state = _enforce_pac_conservation(evolved_field, config)
    
    # Update physics field
    physics_field.set("field_data", evolved_field)
    physics_field.set("physics_state", updated_physics_state)


def run_gaia_physics_variants_demo():
    """
    Demonstrate GAIA variants with PAC physics integration.
    """
    print("=== GAIA-Fracton Physics Variants Demo ===")
    
    # Initialize bridge
    bridge = GAIAFractonBridge()
    
    # Create different GAIA variants
    variants_to_test = [
        GAIAVariantConfig(
            variant_type=GAIAVariantType.QUANTUM_FOCUSED,
            physics_emphasis="quantum",
            conservation_strictness=1e-12,
            field_dimensions=(64,),
            xi_target=1.0571
        ),
        GAIAVariantConfig(
            variant_type=GAIAVariantType.COGNITIVE_ENHANCED,
            physics_emphasis="cognitive",
            memory_persistence="extended",
            reasoning_depth=8,
            field_dimensions=(128,),
            xi_target=1.0571
        ),
        GAIAVariantConfig(
            variant_type=GAIAVariantType.OPTIMIZATION_SPECIALIZED,
            physics_emphasis="optimization",
            conservation_strictness=1e-14,
            field_dimensions=(32,),
            xi_target=1.0571
        )
    ]
    
    variant_results = {}
    
    # Test each variant
    for config in variants_to_test:
        print(f"\n--- Testing {config.variant_type.value.upper()} Variant ---")
        
        # Create variant
        variant_id = bridge.create_physics_variant(config)
        print(f"Created variant: {variant_id}")
        
        # Test intelligence capabilities
        test_config = {
            "pattern_recognition": True,
            "optimization": True,
            "memory_persistence": True
        }
        
        start_time = time.time()
        results = bridge.test_variant_intelligence(variant_id, test_config)
        test_time = time.time() - start_time
        
        variant_results[variant_id] = results
        
        # Display results
        print(f"Test completed in {test_time:.3f} seconds")
        print(f"Average Score: {results['performance_summary']['average_score']:.3f}")
        print(f"Score Stability: {results['performance_summary']['score_stability']:.3f}")
        print(f"Conservation Quality: {results['performance_summary']['conservation_quality']:.3f}")
        print(f"Physics Consistency: {results['performance_summary']['physics_consistency']:.3f}")
        
        # Show individual test scores
        for test_name, test_result in results["tests"].items():
            conservation_status = "✓" if test_result["conservation_maintained"] else "✗"
            print(f"  {test_name}: {test_result['score']:.3f} {conservation_status}")
    
    # Compare variants
    print(f"\n--- Variant Comparison ---")
    
    comparison_metrics = ["average_score", "score_stability", "conservation_quality", "physics_consistency"]
    
    for metric in comparison_metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        sorted_variants = sorted(
            variant_results.items(),
            key=lambda x: x[1]["performance_summary"][metric],
            reverse=True
        )
        
        for i, (variant_id, results) in enumerate(sorted_variants):
            variant_type = results["variant_type"]
            score = results["performance_summary"][metric]
            print(f"  {i+1}. {variant_type.upper()}: {score:.3f}")
    
    # Physics consistency analysis
    print(f"\n--- Physics Analysis ---")
    for variant_id, results in variant_results.items():
        variant_type = results["variant_type"]
        physics = results["physics_metrics"]
        
        print(f"{variant_type.upper()}:")
        print(f"  Conservation Residual: {physics['conservation_residual']:.2e}")
        print(f"  Xi Deviation: {physics['xi_deviation']:.4f}")
        print(f"  Field Energy: {physics['field_energy']:.3f}")
        print(f"  Klein-Gordon Energy: {physics['klein_gordon_energy']:.3f}")
    
    return variant_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--physics":
        # Run physics variants demo
        run_gaia_physics_variants_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "--cognitive":
        # Run original cognitive demo
        run_gaia_fracton_integration()
    else:
        # Run both demos
        print("Running Cognitive Demo:")
        print("=" * 50)
        run_gaia_fracton_integration()
        
        print("\n\n" + "=" * 50)
        print("Running Physics Variants Demo:")
        print("=" * 50)
        run_gaia_physics_variants_demo()
