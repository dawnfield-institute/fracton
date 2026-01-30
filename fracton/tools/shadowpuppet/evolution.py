"""
ShadowPuppet Evolution Engine

The main evolution loop that:
- Generates component candidates
- Evaluates fitness (coherence + tests)
- Selects survivors based on threshold
- Tracks genealogy across generations
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field

import numpy as np

from .protocols import ProtocolSpec, GrowthGap, ComponentOrganism
from .coherence import CoherenceEvaluator
from .genealogy import GenealogyTree
from .generators import CodeGenerator, MockGenerator


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
    """
    
    def __init__(
        self,
        generator: Optional[CodeGenerator] = None,
        evaluator: Optional[CoherenceEvaluator] = None,
        config: Optional[EvolutionConfig] = None,
        pac_invariants: Optional[List[str]] = None
    ):
        """
        Initialize evolution engine.
        
        Args:
            generator: Code generator (default: MockGenerator)
            evaluator: Coherence evaluator (default: CoherenceEvaluator)
            config: Evolution configuration
            pac_invariants: Global PAC invariants to enforce
        """
        self.config = config or EvolutionConfig()
        self.generator = generator or MockGenerator()
        self.evaluator = evaluator or CoherenceEvaluator()
        
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
        
        print(f"\n[ShadowPuppet] Starting evolution")
        print(f"  Generator: {self.generator.name}")
        print(f"  Gaps: {len(gaps)}")
        print(f"  Max generations: {max_gens}")
        print(f"  Coherence threshold: {self.config.coherence_threshold}")
        
        for gen in range(max_gens):
            self.generation = gen
            births = 0
            deaths = 0
            
            print(f"\n--- Generation {gen} ---")
            
            # BIRTH: Generate components for gaps
            new_components = []
            
            for gap in gaps:
                if len(self.components) + len(new_components) >= self.config.max_population:
                    print(f"  [!] Population limit reached")
                    break
                
                # Find best parent for template
                parent = self._select_parent(gap)
                
                # Generate candidates
                print(f"  [>] Generating {self.config.candidates_per_gap} candidates for: {gap.protocol.name}")
                
                candidates = []
                for idx in range(self.config.candidates_per_gap):
                    start_time = time.time()
                    
                    # Vary mutation rate for diversity
                    mutation_variation = self.config.mutation_rate * (
                        1.0 + 0.3 * (idx / max(1, self.config.candidates_per_gap - 1))
                    )
                    
                    # Build generation context
                    from .generators.base import GenerationContext
                    gen_context = GenerationContext(
                        protocol=gap.protocol,
                        parent=parent,
                        siblings=[c for c in self.components if c.protocol_name != gap.protocol.name][:2],
                        mutation_rate=mutation_variation,
                        pac_invariants=self.env.pac_invariants + gap.protocol.pac_invariants
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
                        generator_used=self.generator.name,
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
                    deaths += 1
            
            self.components = survivors
            
            # Record stats
            stats = self._record_stats(births, deaths)
            self.history.append(stats)
            
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
        
        return self._compile_results()
    
    def _select_parent(self, gap: GrowthGap) -> Optional[ComponentOrganism]:
        """Select best component as template."""
        if not self.components:
            return None
        
        # Prefer same protocol family
        same_protocol = [c for c in self.components if c.protocol_name == gap.protocol.name]
        if same_protocol:
            parent = max(same_protocol, key=lambda c: c.coherence_score)
            parent.reuse_count += 1
            return parent
        
        # Otherwise highest coherence
        parent = max(self.components, key=lambda c: c.coherence_score)
        parent.reuse_count += 1
        return parent
    
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
            
            with open(path, 'w') as f:
                f.write(component.code)
            
            print(f"  Saved: {path}")
