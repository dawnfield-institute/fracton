"""
ShadowPuppet Evolution Engine

The main evolution loop that:
- Generates component candidates
- Evaluates fitness (coherence + tests)
- Selects survivors based on threshold
- Tracks genealogy across generations

Improvements over v1:
- Crossover/recombination between high-fitness parents
- Targeted refinement loop for borderline candidates
- Better parent selection with tournament

v0.4 additions:
- Checkpoint loading for incremental evolution
- Callback system for progress monitoring
- Integration fitness evaluation
"""

import json
import time
import ast
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Type, Tuple, Callable, Protocol
from dataclasses import dataclass, field

import numpy as np

from .protocols import ProtocolSpec, GrowthGap, ComponentOrganism
from .coherence import CoherenceEvaluator
from .genealogy import GenealogyTree
from .generators import CodeGenerator, MockGenerator


class EvolutionCallbacks(Protocol):
    """
    Protocol for evolution progress callbacks.
    
    Implement any subset of these methods to receive events.
    All methods are optional - unimplemented ones will be skipped.
    
    Example:
        class MyCallbacks:
            def on_generation_start(self, generation: int, population: int):
                print(f"Starting gen {generation} with {population} components")
            
            def on_birth(self, component, fitness: float):
                print(f"Born: {component.id} (fitness={fitness:.3f})")
        
        evolution = SoftwareEvolution(callbacks=MyCallbacks())
    """
    
    def on_evolution_start(self, gaps: List[GrowthGap], config: 'EvolutionConfig') -> None:
        """Called when evolution begins."""
        ...
    
    def on_generation_start(self, generation: int, population: int) -> None:
        """Called at the start of each generation."""
        ...
    
    def on_generation_end(self, generation: int, stats: 'GenerationStats') -> None:
        """Called at the end of each generation with stats."""
        ...
    
    def on_birth(self, component: ComponentOrganism, fitness: float) -> None:
        """Called when a new component survives initial evaluation."""
        ...
    
    def on_death(self, component: ComponentOrganism, reason: str) -> None:
        """Called when a component fails to survive."""
        ...
    
    def on_refinement(self, component: ComponentOrganism, old_fitness: float, new_fitness: float) -> None:
        """Called when a component is refined."""
        ...
    
    def on_crossover(self, parent_a: ComponentOrganism, parent_b: ComponentOrganism, child: ComponentOrganism) -> None:
        """Called when crossover produces a child."""
        ...
    
    def on_evolution_end(self, results: Dict[str, Any]) -> None:
        """Called when evolution completes."""
        ...


@dataclass
class EvolutionConfig:
    """Configuration for software evolution."""
    coherence_threshold: float = 0.70
    reproduction_threshold: float = 0.80
    max_population: int = 50
    candidates_per_gap: int = 3
    mutation_rate: float = 0.2
    max_generations: int = 10
    save_checkpoints: bool = True
    output_dir: Optional[Path] = None
    # New options
    enable_crossover: bool = True
    crossover_rate: float = 0.3  # Probability of crossover vs mutation
    enable_refinement: bool = True
    refinement_threshold: float = 0.5  # Score below which to attempt refinement
    max_refinement_attempts: int = 2



@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    population: int
    births: int
    deaths: int
    mean_coherence: float
    max_coherence: float
    best_component_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CodeEnvironment:
    """
    Environment for evolving software components.
    
    Provides:
    - Selection pressure (coherence threshold)
    - Integration energy (resource for surviving components)
    - PAC invariant enforcement
    """
    
    def __init__(
        self,
        coherence_threshold: float = 0.70,
        max_population: int = 50
    ):
        self.coherence_threshold = coherence_threshold
        self.max_population = max_population
        self.pac_invariants: List[str] = []
    
    def harvest_integration_energy(self, component: ComponentOrganism) -> float:
        """Calculate integration energy for component."""
        base_energy = 10.0
        efficiency = 1.0 + 0.5 * component.coherence_score
        return base_energy * efficiency
    
    def check_survival(self, component: ComponentOrganism) -> bool:
        """Check if component survives selection pressure."""
        return component.coherence_score >= self.coherence_threshold


class SoftwareEvolution:
    """
    Main evolution simulation for software components.
    
    Evolves a population of code components through:
    1. BIRTH: Generate candidates for each gap
    2. SELECTION: Keep best candidate per gap
    3. EVALUATION: Score coherence (structural/semantic/energetic)
    4. DEATH: Remove components below threshold
    5. REPRODUCTION: High-coherence components become templates
    
    Example:
        evolution = SoftwareEvolution(
            generator=ClaudeGenerator(),
            config=EvolutionConfig(coherence_threshold=0.75)
        )
        
        results = evolution.grow([
            GrowthGap(protocol=api_protocol),
            GrowthGap(protocol=frontend_protocol)
        ])
        
        # Get generated code
        for component in evolution.components:
            print(component.code)
    
    With callbacks:
        class MyCallbacks:
            def on_birth(self, component, fitness):
                print(f"New component: {component.id}")
        
        evolution = SoftwareEvolution(callbacks=MyCallbacks())
    
    Resume from checkpoint:
        evolution = SoftwareEvolution(config=EvolutionConfig(output_dir=Path("out/")))
        evolution.load_checkpoint(Path("out/checkpoint_gen5.json"))
        results = evolution.grow(gaps, max_generations=5)  # Continue from gen 5
    """
    
    def __init__(
        self,
        generator: Optional[CodeGenerator] = None,
        evaluator: Optional[CoherenceEvaluator] = None,
        config: Optional[EvolutionConfig] = None,
        pac_invariants: Optional[List[str]] = None,
        callbacks: Optional[EvolutionCallbacks] = None
    ):
        """
        Initialize evolution engine.
        
        Args:
            generator: Code generator (default: MockGenerator)
            evaluator: Coherence evaluator (default: CoherenceEvaluator)
            config: Evolution configuration
            pac_invariants: Global PAC invariants to enforce
            callbacks: Optional callbacks for progress monitoring
        """
        self.config = config or EvolutionConfig()
        self.generator = generator or MockGenerator()
        self.evaluator = evaluator or CoherenceEvaluator()
        self.callbacks = callbacks
        
        # Environment
        self.env = CodeEnvironment(
            coherence_threshold=self.config.coherence_threshold,
            max_population=self.config.max_population
        )
        self.env.pac_invariants = pac_invariants or []
        
        # Population
        self.components: List[ComponentOrganism] = []
        self.genealogy = GenealogyTree()
        self.next_id = 0
        self.generation = 0
        
        # History
        self.history: List[GenerationStats] = []
        
        # Output
        if self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _emit(self, event: str, *args, **kwargs) -> None:
        """Emit a callback event if callbacks are registered."""
        if self.callbacks is None:
            return
        handler = getattr(self.callbacks, event, None)
        if handler is not None:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                print(f"  [!] Callback error in {event}: {e}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Load evolution state from a checkpoint file.
        
        Restores:
        - Population (components)
        - Generation counter
        - History
        - Next ID counter
        
        Args:
            checkpoint_path: Path to checkpoint JSON file
            
        Returns:
            True if loaded successfully, False otherwise
            
        Example:
            evolution = SoftwareEvolution(config=config)
            if evolution.load_checkpoint(Path("checkpoints/gen5.json")):
                print(f"Resumed from generation {evolution.generation}")
                results = evolution.grow(gaps, max_generations=5)
        """
        if not checkpoint_path.exists():
            print(f"[!] Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            
            # Restore generation
            self.generation = data.get('generation', 0)
            
            # Restore components
            self.components = []
            for comp_data in data.get('components', []):
                component = ComponentOrganism(
                    id=comp_data['id'],
                    protocol_name=comp_data['protocol_name'],
                    code=comp_data['code'],
                    parent_id=comp_data.get('parent_id'),
                    generation=comp_data.get('generation', 0),
                    coherence_score=comp_data.get('coherence_score', 0.0),
                    structural_score=comp_data.get('structural_score', 0.0),
                    semantic_score=comp_data.get('semantic_score', 0.0),
                    energetic_score=comp_data.get('energetic_score', 0.0),
                    derivation_path=comp_data.get('derivation_path', []),
                    generator_used=comp_data.get('generator_used', 'unknown'),
                    generation_time=comp_data.get('generation_time', 0.0),
                    integration_energy=comp_data.get('integration_energy', 0.0),
                    age=comp_data.get('age', 0),
                    reuse_count=comp_data.get('reuse_count', 0)
                )
                self.components.append(component)
                self.genealogy.add(component)
            
            # Restore history
            self.history = []
            for hist_data in data.get('history', []):
                stats = GenerationStats(
                    generation=hist_data['generation'],
                    population=hist_data['population'],
                    births=hist_data['births'],
                    deaths=hist_data['deaths'],
                    mean_coherence=hist_data['mean_coherence'],
                    max_coherence=hist_data['max_coherence'],
                    best_component_id=hist_data['best_component_id'],
                    timestamp=hist_data.get('timestamp', datetime.now().isoformat())
                )
                self.history.append(stats)
            
            # Update next_id to avoid collisions
            if self.components:
                max_id_num = 0
                for comp in self.components:
                    # Extract number from ID like "UserService_5"
                    parts = comp.id.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        max_id_num = max(max_id_num, int(parts[1]))
                self.next_id = max_id_num + 1
            
            print(f"[*] Loaded checkpoint: gen={self.generation}, population={len(self.components)}")
            return True
            
        except Exception as e:
            print(f"[!] Failed to load checkpoint: {e}")
            return False
    
    def grow(
        self,
        gaps: List[GrowthGap],
        max_generations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main evolution loop.
        
        Args:
            gaps: Initial gaps to fill
            max_generations: Override config max_generations
            
        Returns:
            Results dictionary with components, history, genealogy
        """
        max_gens = max_generations or self.config.max_generations
        
        # Sort gaps by dependency order (dependencies first)
        ordered_gaps = self._order_by_dependencies(gaps)
        
        print(f"\n[ShadowPuppet] Starting evolution")
        print(f"  Generator: {self.generator.name}")
        print(f"  Gaps: {len(ordered_gaps)}")
        print(f"  Max generations: {max_gens}")
        print(f"  Coherence threshold: {self.config.coherence_threshold}")
        
        if ordered_gaps != gaps:
            print(f"  Dependency order: {[g.protocol.name for g in ordered_gaps]}")
        
        # Emit evolution start callback
        self._emit('on_evolution_start', ordered_gaps, self.config)
        
        for gen in range(max_gens):
            self.generation = gen
            births = 0
            deaths = 0
            
            print(f"\n--- Generation {gen} ---")
            
            # Emit generation start callback
            self._emit('on_generation_start', gen, len(self.components))
            
            # BIRTH: Generate components for gaps
            new_components = []
            
            for gap in ordered_gaps:
                if len(self.components) + len(new_components) >= self.config.max_population:
                    print(f"  [!] Population limit reached")
                    break
                
                # Find best parent for template
                parent = self._select_parent(gap)
                
                # Decide: crossover or mutation?
                import random
                use_crossover = (
                    self.config.enable_crossover and 
                    len(self.components) >= 2 and
                    random.random() < self.config.crossover_rate
                )
                
                # Generate candidates
                if use_crossover:
                    print(f"  [X] Crossover for: {gap.protocol.name}")
                else:
                    print(f"  [>] Generating {self.config.candidates_per_gap} candidates for: {gap.protocol.name}")
                
                candidates = []
                for idx in range(self.config.candidates_per_gap):
                    start_time = time.time()
                    
                    if use_crossover and idx == 0:
                        # First candidate via crossover
                        parent_a, parent_b = self._select_parents_for_crossover(gap)
                        if parent_a and parent_b:
                            code = self.crossover(parent_a, parent_b, gap.protocol)
                            parent = parent_a  # Track lineage from first parent
                        else:
                            use_crossover = False
                            # Fall through to normal generation
                    
                    if not use_crossover or idx > 0:
                        # Normal mutation-based generation
                        # Vary mutation rate for diversity
                        mutation_variation = self.config.mutation_rate * (
                            1.0 + 0.3 * (idx / max(1, self.config.candidates_per_gap - 1))
                        )
                        
                        # Get resolved dependencies (components for dependencies)
                        all_so_far = self.components + new_components
                        resolved_deps = self._get_resolved_dependencies(gap, all_so_far)
                        
                        # Build generation context
                        from .generators.base import GenerationContext
                        gen_context = GenerationContext(
                            protocol=gap.protocol,
                            parent=parent,
                            siblings=[c for c in self.components if c.protocol_name != gap.protocol.name][:2],
                            mutation_rate=mutation_variation,
                            pac_invariants=self.env.pac_invariants + gap.protocol.pac_invariants,
                            domain_types=gap.domain_types,
                            resolved_dependencies=resolved_deps
                        )
                        
                        # Generate code
                        try:
                            code = self.generator.generate(gen_context)
                        except Exception as e:
                            print(f"      [!] Generation failed: {e}")
                            continue
                    
                    generation_time = time.time() - start_time
                    
                    # Create component organism
                    component = ComponentOrganism(
                        id=f"{gap.protocol.name}_{self.next_id}",
                        protocol_name=gap.protocol.name,
                        code=code,
                        parent_id=parent.id if parent else None,
                        generation=gen,
                        derivation_path=(parent.derivation_path + [gap.protocol.name]) if parent else [gap.protocol.name],
                        generator_used=self.generator.name + ("+crossover" if use_crossover and idx == 0 else ""),
                        generation_time=generation_time
                    )
                    self.next_id += 1
                    
                    # Evaluate fitness
                    eval_context = {
                        'protocol': gap.protocol,
                        'pac_invariants': self.env.pac_invariants + gap.protocol.pac_invariants,
                        'parents': [parent] if parent else [],
                        'generation': gen,
                        'test_suite': gap.get_test_dict(),
                        'all_components': self.components
                    }
                    
                    fitness = self.evaluator.evaluate(component, eval_context)
                    
                    # REFINEMENT: If score is borderline, try to fix
                    if (self.config.enable_refinement and 
                        fitness < self.config.refinement_threshold and
                        fitness > 0.2):  # Don't refine hopeless cases
                        
                        violations = eval_context.get('invariant_violations', {}).get(component.id, [])
                        if not violations:
                            violations = [f"Low fitness: {fitness:.3f}"]
                        
                        for attempt in range(self.config.max_refinement_attempts):
                            refined_code = self.refine(component, eval_context, violations)
                            if refined_code:
                                component.code = refined_code
                                new_fitness = self.evaluator.evaluate(component, eval_context)
                                if new_fitness > fitness:
                                    print(f"      [R] Refined: {fitness:.3f} -> {new_fitness:.3f}")
                                    self._emit('on_refinement', component, fitness, new_fitness)
                                    fitness = new_fitness
                                    break
                    
                    candidates.append(component)
                    
                    print(f"      Candidate {idx+1}: fitness={fitness:.3f} "
                          f"(S:{component.structural_score:.2f} "
                          f"Se:{component.semantic_score:.2f} "
                          f"E:{component.energetic_score:.2f})")
                
                if not candidates:
                    print(f"      [!] No valid candidates generated")
                    continue
                
                # Select best candidate
                best = max(candidates, key=lambda c: c.coherence_score)
                print(f"      [*] Winner: {best.id} (fitness={best.coherence_score:.3f})")
                
                new_components.append(best)
                births += 1
                self._emit('on_birth', best, best.coherence_score)
            
            # Add to population and genealogy
            for comp in new_components:
                self.components.append(comp)
                self.genealogy.add(comp)
            
            # DEATH: Remove low-coherence components
            survivors = []
            for comp in self.components:
                if self.env.check_survival(comp):
                    energy = self.env.harvest_integration_energy(comp)
                    comp.integration_energy += energy
                    comp.age += 1
                    survivors.append(comp)
                else:
                    print(f"      [-] Died: {comp.id} (coherence: {comp.coherence_score:.3f})")
                    self._emit('on_death', comp, f"coherence {comp.coherence_score:.3f} < threshold {self.env.coherence_threshold}")
                    deaths += 1
            
            self.components = survivors
            
            # Record stats
            stats = self._record_stats(births, deaths)
            self.history.append(stats)
            
            # Emit generation end callback
            self._emit('on_generation_end', gen, stats)
            
            # Save checkpoint
            if self.config.save_checkpoints and self.config.output_dir:
                self._save_checkpoint(gen)
            
            # Check convergence
            if self.components and all(
                c.coherence_score >= self.config.coherence_threshold
                for c in self.components
            ):
                print(f"\n  [*] Convergence! All components above threshold.")
                break
            
            if not self.components:
                print(f"\n  [!] Population extinct!")
                break
        
        results = self._compile_results()
        self._emit('on_evolution_end', results)
        return results
    
    def _select_parent(self, gap: GrowthGap) -> Optional[ComponentOrganism]:
        """Select best component as template using tournament selection."""
        if not self.components:
            return None
        
        # Prefer same protocol family
        same_protocol = [c for c in self.components if c.protocol_name == gap.protocol.name]
        if same_protocol:
            parent = self._tournament_select(same_protocol)
            parent.reuse_count += 1
            return parent
        
        # Otherwise tournament from all
        parent = self._tournament_select(self.components)
        parent.reuse_count += 1
        return parent
    
    def _tournament_select(
        self, 
        candidates: List[ComponentOrganism], 
        tournament_size: int = 3
    ) -> ComponentOrganism:
        """Tournament selection - pick best from random subset."""
        import random
        if len(candidates) <= tournament_size:
            return max(candidates, key=lambda c: c.coherence_score)
        
        tournament = random.sample(candidates, tournament_size)
        return max(tournament, key=lambda c: c.coherence_score)
    
    def _select_parents_for_crossover(
        self, 
        gap: GrowthGap
    ) -> Tuple[Optional[ComponentOrganism], Optional[ComponentOrganism]]:
        """Select two parents for crossover."""
        candidates = [c for c in self.components if c.protocol_name == gap.protocol.name]
        if len(candidates) < 2:
            # Try any high-fitness components
            candidates = sorted(self.components, key=lambda c: c.coherence_score, reverse=True)[:5]
        
        if len(candidates) < 2:
            return None, None
        
        parent_a = self._tournament_select(candidates)
        remaining = [c for c in candidates if c.id != parent_a.id]
        if not remaining:
            return parent_a, None
        
        parent_b = self._tournament_select(remaining)
        return parent_a, parent_b
    
    def crossover(
        self, 
        parent_a: ComponentOrganism, 
        parent_b: ComponentOrganism,
        protocol: ProtocolSpec
    ) -> str:
        """
        Genetic crossover between two parent implementations.
        
        Combines methods from both parents to create offspring.
        Uses AST-based splicing for structural correctness.
        """
        try:
            tree_a = ast.parse(parent_a.code)
            tree_b = ast.parse(parent_b.code)
        except SyntaxError:
            # Fallback: just use parent_a
            return parent_a.code
        
        # Find class definitions
        class_a = None
        class_b = None
        for node in ast.walk(tree_a):
            if isinstance(node, ast.ClassDef):
                class_a = node
                break
        for node in ast.walk(tree_b):
            if isinstance(node, ast.ClassDef):
                class_b = node
                break
        
        if not class_a or not class_b:
            return parent_a.code
        
        # Extract methods from both
        methods_a = {n.name: n for n in class_a.body if isinstance(n, ast.FunctionDef)}
        methods_b = {n.name: n for n in class_b.body if isinstance(n, ast.FunctionDef)}
        
        # Crossover: randomly pick methods from each parent
        import random
        new_methods = []
        all_method_names = set(methods_a.keys()) | set(methods_b.keys())
        
        for method_name in all_method_names:
            if method_name in methods_a and method_name in methods_b:
                # Both have it - pick randomly (favor higher fitness parent)
                if parent_a.coherence_score > parent_b.coherence_score:
                    choice = methods_a[method_name] if random.random() > 0.3 else methods_b[method_name]
                else:
                    choice = methods_b[method_name] if random.random() > 0.3 else methods_a[method_name]
                new_methods.append(choice)
            elif method_name in methods_a:
                new_methods.append(methods_a[method_name])
            else:
                new_methods.append(methods_b[method_name])
        
        # Reconstruct class with crossover methods
        class_a.body = [n for n in class_a.body if not isinstance(n, ast.FunctionDef)] + new_methods
        
        # Add imports from both
        imports_a = [n for n in tree_a.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        imports_b = [n for n in tree_b.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        
        # Simple dedup by unparsing
        seen_imports = set()
        unique_imports = []
        for imp in imports_a + imports_b:
            try:
                imp_str = ast.unparse(imp)
                if imp_str not in seen_imports:
                    seen_imports.add(imp_str)
                    unique_imports.append(imp)
            except:
                unique_imports.append(imp)
        
        # Build new module
        new_tree = ast.Module(body=unique_imports + [class_a], type_ignores=[])
        ast.fix_missing_locations(new_tree)
        
        try:
            return ast.unparse(new_tree)
        except:
            return parent_a.code
    
    def refine(
        self, 
        component: ComponentOrganism, 
        context: Dict[str, Any],
        violations: List[str]
    ) -> Optional[str]:
        """
        Targeted refinement for components with specific issues.
        
        Instead of regenerating from scratch, asks the generator
        to fix specific problems.
        """
        protocol = context.get('protocol')
        if not protocol:
            return None
        
        # Build refinement prompt
        from .generators.base import GenerationContext
        
        issues = "\n".join(f"- {v}" for v in violations[:5])  # Limit issues
        
        refine_instructions = f"""The following code has issues that need fixing:

ISSUES TO FIX:
{issues}

Current implementation:
```python
{component.code}
```

Fix ONLY the listed issues. Keep everything else the same.
Return the complete corrected implementation."""
        
        gen_context = GenerationContext(
            protocol=protocol,
            parent=component,
            extra_instructions=refine_instructions,
            pac_invariants=context.get('pac_invariants', [])
        )
        
        try:
            refined_code = self.generator.generate(gen_context)
            return refined_code
        except Exception:
            return None
    
    def _record_stats(self, births: int, deaths: int) -> GenerationStats:
        """Record generation statistics."""
        if not self.components:
            return GenerationStats(
                generation=self.generation,
                population=0,
                births=births,
                deaths=deaths,
                mean_coherence=0.0,
                max_coherence=0.0,
                best_component_id=""
            )
        
        coherences = [c.coherence_score for c in self.components]
        best = max(self.components, key=lambda c: c.coherence_score)
        
        return GenerationStats(
            generation=self.generation,
            population=len(self.components),
            births=births,
            deaths=deaths,
            mean_coherence=float(np.mean(coherences)),
            max_coherence=max(coherences),
            best_component_id=best.id
        )
    
    def _save_checkpoint(self, generation: int):
        """Save evolution checkpoint."""
        checkpoint = {
            'generation': generation,
            'components': [c.to_dict() for c in self.components],
            'history': [
                {
                    'generation': s.generation,
                    'population': s.population,
                    'births': s.births,
                    'deaths': s.deaths,
                    'mean_coherence': s.mean_coherence,
                    'max_coherence': s.max_coherence,
                    'best_component_id': s.best_component_id,
                    'timestamp': s.timestamp
                }
                for s in self.history
            ]
        }
        
        path = self.config.output_dir / f"checkpoint_gen{generation}.json"
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile evolution results."""
        return {
            'success': len(self.components) > 0,
            'generations': self.generation + 1,
            'final_population': len(self.components),
            'components': [c.to_dict() for c in self.components],
            'genealogy': self.genealogy.to_dict(),
            'history': [
                {
                    'generation': s.generation,
                    'population': s.population,
                    'mean_coherence': s.mean_coherence,
                    'max_coherence': s.max_coherence
                }
                for s in self.history
            ],
            'best_component': max(
                self.components,
                key=lambda c: c.coherence_score
            ).to_dict() if self.components else None
        }
    
    def get_code(self, protocol_name: Optional[str] = None) -> Dict[str, str]:
        """
        Get generated code for components.
        
        Args:
            protocol_name: Filter by protocol (optional)
            
        Returns:
            Dict mapping component ID to code
        """
        components = self.components
        if protocol_name:
            components = [c for c in components if c.protocol_name == protocol_name]
        
        return {c.id: c.code for c in components}
    
    def save_code(self, output_dir: Path):
        """Save all generated code to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for component in self.components:
            filename = f"{component.protocol_name.lower()}.py"
            path = output_dir / filename
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(component.code)
            
            print(f"  Saved: {path}")
    
    def _order_by_dependencies(self, gaps: List[GrowthGap]) -> List[GrowthGap]:
        """
        Topologically sort gaps so dependencies are generated first.
        
        Uses Kahn's algorithm for topological sort.
        
        Args:
            gaps: List of gaps to sort
            
        Returns:
            Sorted list with dependencies before dependents
        """
        # Build dependency graph
        gap_by_name = {g.protocol.name: g for g in gaps}
        all_names = set(gap_by_name.keys())
        
        # Count incoming edges (dependencies within our gap set)
        in_degree = {name: 0 for name in all_names}
        for gap in gaps:
            for dep in gap.protocol.dependencies:
                if dep in all_names:
                    in_degree[gap.protocol.name] += 1
        
        # Start with gaps that have no dependencies (in our set)
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            name = queue.pop(0)
            result.append(gap_by_name[name])
            
            # Reduce in-degree for dependents
            for gap in gaps:
                if name in gap.protocol.dependencies:
                    in_degree[gap.protocol.name] -= 1
                    if in_degree[gap.protocol.name] == 0:
                        queue.append(gap.protocol.name)
        
        # If we couldn't order all gaps (cycle), return original order
        if len(result) != len(gaps):
            print(f"  [!] Dependency cycle detected, using original order")
            return gaps
        
        return result
    
    def _get_resolved_dependencies(
        self,
        gap: GrowthGap,
        all_components: List[ComponentOrganism]
    ) -> Dict[str, ComponentOrganism]:
        """
        Get already-generated components that this gap depends on.
        
        Args:
            gap: The gap being filled
            all_components: All generated components so far
            
        Returns:
            Dict mapping dependency name to best component
        """
        resolved = {}
        for dep_name in gap.protocol.dependencies:
            # Find best component for this dependency
            dep_components = [
                c for c in all_components 
                if c.protocol_name == dep_name
            ]
            if dep_components:
                best = max(dep_components, key=lambda c: c.coherence_score)
                resolved[dep_name] = best
        return resolved
