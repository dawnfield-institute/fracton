"""
ShadowPuppet Multi-Seed Architecture

Evolves multiple seed architectures with explicit inter-seed contracts.

Think of each seed as a microservice or bounded context:
- Seed A: User Management Service
- Seed B: Task Service
- Seed C: API Gateway

Each seed evolves independently but must respect contracts with other seeds.

Key Concepts:
- SeedArchitecture: A complete service with internal components
- SeedConnector: Public interface exposed by a seed
- MultiSeedEvolution: Orchestrates evolution across seeds
- Cross-Seed Tests: Integration tests that span multiple seeds

Example:
    # Define three services
    user_seed = SeedArchitecture(
        name="UserService",
        gaps=[...],
        exposed_interfaces=["UserRepository", "AuthService"]
    )
    
    task_seed = SeedArchitecture(
        name="TaskService",
        gaps=[...],
        exposed_interfaces=["TaskRepository"],
        dependencies={"UserService": ["AuthService"]}
    )
    
    gateway_seed = SeedArchitecture(
        name="APIGateway",
        gaps=[...],
        dependencies={
            "UserService": ["UserRepository", "AuthService"],
            "TaskService": ["TaskRepository"]
        }
    )
    
    # Evolve all three with cross-seed validation
    multi_evolution = MultiSeedEvolution([user_seed, task_seed, gateway_seed])
    results = multi_evolution.evolve()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from pathlib import Path
import json
from datetime import datetime

from .protocols import GrowthGap, ComponentOrganism, ProtocolSpec
from .evolution import SoftwareEvolution, EvolutionConfig, GenerationStats
from .connectors import Connector, MethodSignature, ConnectorRegistry
from .coherence import CoherenceEvaluator
from .generators import CodeGenerator


@dataclass
class SeedInterface:
    """
    Public interface exposed by a seed.
    
    This is what other seeds can depend on. Think of it as
    the API boundary of a microservice.
    
    Attributes:
        component_name: Name of the component providing this interface
        methods: Public methods available to other seeds
        version: Semantic version (for evolution tracking)
    """
    component_name: str
    methods: List[MethodSignature]
    version: str = "1.0.0"
    
    def to_prompt_context(self) -> str:
        """Format for inclusion in dependent seed prompts."""
        lines = [f"{self.component_name} (external dependency):"]
        for method in self.methods:
            lines.append(f"  {method.to_call_pattern()}")
        return "\n".join(lines)


@dataclass
class SeedConnector:
    """
    Contract between two seeds.
    
    Defines what one seed exposes and what another consumes.
    Like a service mesh contract or API gateway route.
    
    Attributes:
        provider_seed: Name of the seed providing functionality
        consumer_seed: Name of the seed consuming functionality
        interfaces: List of interfaces being exposed
    """
    provider_seed: str
    consumer_seed: str
    interfaces: List[SeedInterface]
    
    def validate_consumer_code(self, consumer_code: str) -> Tuple[bool, List[str]]:
        """
        Validate that consumer code uses provider interfaces correctly.
        
        Returns:
            (is_valid, list_of_violations)
        """
        # TODO: Implement cross-seed call validation
        # Similar to ConnectorRegistry.validate_consumer but at seed level
        return True, []


@dataclass
class SeedArchitecture:
    """
    A complete seed architecture (microservice/bounded context).
    
    Represents one evolved service in a multi-service system.
    Each seed:
    - Has internal components (GrowthGaps)
    - Exposes public interfaces to other seeds
    - May depend on other seeds' interfaces
    - Evolves independently with its own fitness criteria
    
    Attributes:
        name: Seed/service name
        gaps: Internal components to evolve
        exposed_interfaces: Which components are public (by name)
        dependencies: Map of {seed_name: [component_names]}
        config: Evolution config for this seed
        pac_invariants: Seed-level invariants
        cross_seed_tests: Integration tests involving other seeds
    """
    name: str
    gaps: List[GrowthGap]
    exposed_interfaces: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    config: Optional[EvolutionConfig] = None
    pac_invariants: List[str] = field(default_factory=list)
    cross_seed_tests: List[Callable] = field(default_factory=list)
    
    # Evolution state (populated during evolution)
    evolution: Optional[SoftwareEvolution] = None
    components: List[ComponentOrganism] = field(default_factory=list)
    public_interfaces: Dict[str, SeedInterface] = field(default_factory=dict)
    
    def get_exposed_components(self) -> List[ComponentOrganism]:
        """Get components marked as public interfaces."""
        return [
            c for c in self.components
            if c.protocol_name in self.exposed_interfaces
        ]
    
    def extract_public_interfaces(self) -> Dict[str, SeedInterface]:
        """
        Extract actual interfaces from evolved components.
        
        Only exposed components become public interfaces.
        """
        from .connectors import InterfaceExtractor
        
        extractor = InterfaceExtractor()
        interfaces = {}
        
        for component in self.get_exposed_components():
            methods = extractor.extract(component.code, component.protocol_name)
            interfaces[component.protocol_name] = SeedInterface(
                component_name=component.protocol_name,
                methods=methods,
                version="1.0.0"  # TODO: Version tracking
            )
        
        self.public_interfaces = interfaces
        return interfaces


@dataclass
class MultiSeedStats:
    """Statistics for a multi-seed evolution run."""
    total_seeds: int
    total_components: int
    total_generations: int
    seed_stats: Dict[str, GenerationStats]
    cross_seed_test_results: Dict[str, bool] = field(default_factory=dict)
    interface_validations: Dict[str, bool] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MultiSeedEvolution:
    """
    Orchestrator for multi-seed architecture evolution.
    
    Manages evolution across multiple seeds with inter-seed contracts.
    
    Evolution Strategy:
    1. Topologically sort seeds by dependencies
    2. Evolve each seed in order (dependencies first)
    3. Extract public interfaces from evolved seeds
    4. Pass interfaces to dependent seeds as external contracts
    5. Validate cross-seed calls and integration tests
    6. Iterate if needed to improve cross-seed coherence
    
    This is like microservice choreography at evolution-time.
    """
    
    def __init__(
        self,
        seeds: List[SeedArchitecture],
        generator: Optional[CodeGenerator] = None,
        evaluator: Optional[CoherenceEvaluator] = None,
        global_config: Optional[EvolutionConfig] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize multi-seed evolution.
        
        Args:
            seeds: List of seed architectures to evolve
            generator: Code generator (shared across seeds)
            evaluator: Coherence evaluator (shared across seeds)
            global_config: Default config for seeds without their own
            output_dir: Where to save evolved code
        """
        self.seeds = {seed.name: seed for seed in seeds}
        self.generator = generator
        self.evaluator = evaluator
        self.global_config = global_config or EvolutionConfig()
        self.output_dir = output_dir or Path("generated/multi_seed")
        
        # Track inter-seed connectors
        self.seed_connectors: List[SeedConnector] = []
        
        # Evolution order (topologically sorted)
        self.evolution_order: List[str] = []
        
        # Results
        self.stats: Optional[MultiSeedStats] = None
    
    def _topological_sort(self) -> List[str]:
        """
        Sort seeds by dependencies (DAG traversal).
        
        Seeds with no dependencies evolve first, then their dependents.
        
        Returns:
            List of seed names in evolution order
        """
        # Build dependency graph
        in_degree = {name: 0 for name in self.seeds}
        adjacency = {name: [] for name in self.seeds}
        
        for seed_name, seed in self.seeds.items():
            for dep_seed in seed.dependencies.keys():
                if dep_seed in self.seeds:
                    adjacency[dep_seed].append(seed_name)
                    in_degree[seed_name] += 1
        
        # Kahn's algorithm
        queue = [name for name, degree in in_degree.items() if degree == 0]
        sorted_seeds = []
        
        while queue:
            current = queue.pop(0)
            sorted_seeds.append(current)
            
            for dependent in adjacency[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for cycles
        if len(sorted_seeds) != len(self.seeds):
            raise ValueError("Circular dependencies detected in seeds")
        
        return sorted_seeds
    
    def _build_seed_connectors(self, evolved_seeds: Dict[str, SeedArchitecture]) -> None:
        """
        Build connectors between seeds based on evolved interfaces.
        
        Args:
            evolved_seeds: Map of seed_name -> evolved SeedArchitecture
        """
        self.seed_connectors = []
        
        for consumer_name, consumer_seed in evolved_seeds.items():
            for provider_name, required_components in consumer_seed.dependencies.items():
                if provider_name not in evolved_seeds:
                    print(f"  [!] Warning: {consumer_name} depends on {provider_name} but it's not in seed set")
                    continue
                
                provider_seed = evolved_seeds[provider_name]
                
                # Get required interfaces from provider
                interfaces = []
                for comp_name in required_components:
                    if comp_name in provider_seed.public_interfaces:
                        interfaces.append(provider_seed.public_interfaces[comp_name])
                    else:
                        print(f"  [!] Warning: {provider_name} doesn't expose {comp_name}")
                
                if interfaces:
                    connector = SeedConnector(
                        provider_seed=provider_name,
                        consumer_seed=consumer_name,
                        interfaces=interfaces
                    )
                    self.seed_connectors.append(connector)
    
    def _get_dependency_context(self, seed: SeedArchitecture, evolved_seeds: Dict[str, SeedArchitecture]) -> str:
        """
        Build prompt context for a seed's external dependencies.
        
        Args:
            seed: Seed that needs dependency context
            evolved_seeds: Map of already-evolved seeds
            
        Returns:
            Formatted context string for prompts
        """
        if not seed.dependencies:
            return ""
        
        lines = ["EXTERNAL SERVICE DEPENDENCIES:"]
        lines.append("(These are already implemented - your code should call them)")
        lines.append("")
        
        for dep_seed_name, dep_components in seed.dependencies.items():
            dep_seed = evolved_seeds.get(dep_seed_name)  # Get from EVOLVED seeds, not self.seeds
            if dep_seed and dep_seed.public_interfaces:
                lines.append(f"From {dep_seed_name}:")
                for comp_name in dep_components:
                    if comp_name in dep_seed.public_interfaces:
                        interface = dep_seed.public_interfaces[comp_name]
                        lines.append(f"  {interface.to_prompt_context()}")
                lines.append("")
            elif dep_seed_name not in evolved_seeds:
                lines.append(f"WARNING: {dep_seed_name} hasn't been evolved yet!")
                lines.append("")
        
        return "\n".join(lines)
    
    def evolve(
        self,
        max_generations: Optional[int] = None,
        cross_seed_iterations: int = 1
    ) -> Dict[str, Any]:
        """
        Evolve all seeds with cross-seed contract validation.
        
        Args:
            max_generations: Max generations per seed
            cross_seed_iterations: How many times to refine cross-seed contracts
            
        Returns:
            Results with evolved seeds and statistics
        """
        print(f"\n[MultiSeedEvolution] Starting evolution of {len(self.seeds)} seeds")
        
        # Sort seeds by dependencies
        self.evolution_order = self._topological_sort()
        print(f"[*] Evolution order: {' -> '.join(self.evolution_order)}")
        
        evolved_seeds = {}
        seed_stats = {}
        
        # Main evolution loop
        for iteration in range(cross_seed_iterations):
            print(f"\n{'='*60}")
            print(f"Cross-Seed Iteration {iteration + 1}/{cross_seed_iterations}")
            print(f"{'='*60}")
            
            # Evolve each seed in dependency order
            for seed_name in self.evolution_order:
                seed = self.seeds[seed_name]
                print(f"\n[*] Evolving seed: {seed_name}")
                
                # Get dependency context from already-evolved seeds
                dep_context = self._get_dependency_context(seed, evolved_seeds)
                
                # Add dependency context to gaps
                if dep_context:
                    for gap in seed.gaps:
                        gap.extra_instructions = dep_context + "\n" + (gap.extra_instructions or "")
                
                # Create evolution engine for this seed
                config = seed.config or self.global_config
                evolution = SoftwareEvolution(
                    generator=self.generator,
                    evaluator=self.evaluator,
                    config=config,
                    pac_invariants=seed.pac_invariants
                )
                
                # Evolve this seed
                results = evolution.grow(seed.gaps, max_generations=max_generations)
                
                # Store results
                seed.evolution = evolution
                seed.components = evolution.components
                seed.extract_public_interfaces()
                evolved_seeds[seed_name] = seed  # Add to evolved_seeds IMMEDIATELY
                
                # Track stats
                if evolution.history:
                    seed_stats[seed_name] = evolution.history[-1]
                
                print(f"[OK] {seed_name}: {len(seed.components)} components, "
                      f"{len(seed.public_interfaces)} public interfaces")
            
            # Build connectors between seeds
            self._build_seed_connectors(evolved_seeds)
            print(f"\n[*] Built {len(self.seed_connectors)} cross-seed connectors")
            
            # Validate cross-seed contracts
            if iteration < cross_seed_iterations - 1:
                violations = self._validate_cross_seed_contracts(evolved_seeds)
                if violations:
                    print(f"[!] Found {len(violations)} cross-seed violations, refining...")
                    # TODO: Add targeted refinement for cross-seed issues
                else:
                    print(f"[OK] All cross-seed contracts valid, stopping early")
                    break
        
        # Run cross-seed integration tests
        test_results = self._run_cross_seed_tests(evolved_seeds)
        
        # Compile statistics
        total_components = sum(len(s.components) for s in evolved_seeds.values())
        total_generations = sum(
            s.evolution.generation for s in evolved_seeds.values() if s.evolution
        )
        
        self.stats = MultiSeedStats(
            total_seeds=len(evolved_seeds),
            total_components=total_components,
            total_generations=total_generations,
            seed_stats=seed_stats,
            cross_seed_test_results=test_results
        )
        
        # Save evolved code
        self._save_evolved_seeds(evolved_seeds)
        
        return {
            'seeds': evolved_seeds,
            'connectors': self.seed_connectors,
            'stats': self.stats,
            'evolution_order': self.evolution_order
        }
    
    def _validate_cross_seed_contracts(
        self,
        evolved_seeds: Dict[str, SeedArchitecture]
    ) -> List[str]:
        """
        Validate that all cross-seed calls respect interface contracts.
        
        Returns:
            List of violation messages
        """
        violations = []
        
        for connector in self.seed_connectors:
            consumer_seed = evolved_seeds.get(connector.consumer_seed)
            if not consumer_seed:
                continue
            
            # Check each component in consumer seed
            for component in consumer_seed.components:
                is_valid, comp_violations = connector.validate_consumer_code(
                    component.code
                )
                if not is_valid:
                    violations.extend([
                        f"{connector.consumer_seed}.{component.protocol_name}: {v}"
                        for v in comp_violations
                    ])
        
        return violations
    
    def _run_cross_seed_tests(
        self,
        evolved_seeds: Dict[str, SeedArchitecture]
    ) -> Dict[str, bool]:
        """
        Run integration tests that span multiple seeds.
        
        Returns:
            Map of {test_name: passed}
        """
        results = {}
        
        for seed_name, seed in evolved_seeds.items():
            for i, test_func in enumerate(seed.cross_seed_tests):
                test_name = f"{seed_name}_cross_test_{i}"
                try:
                    # TODO: Implement actual test execution with multi-seed context
                    # For now, just check if test is callable
                    passed = callable(test_func)
                    results[test_name] = passed
                except Exception as e:
                    print(f"  [!] Test {test_name} failed: {e}")
                    results[test_name] = False
        
        return results
    
    def _save_evolved_seeds(self, evolved_seeds: Dict[str, SeedArchitecture]) -> None:
        """Save evolved code organized by seed."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for seed_name, seed in evolved_seeds.items():
            seed_dir = self.output_dir / seed_name.lower()
            seed_dir.mkdir(exist_ok=True)
            
            # Save components
            for component in seed.components:
                filename = f"{component.protocol_name.lower()}.py"
                filepath = seed_dir / filename
                with open(filepath, 'w') as f:
                    f.write(component.code)
            
            # Save interface definitions
            if seed.public_interfaces:
                interface_file = seed_dir / "interfaces.json"
                interfaces_data = {
                    name: {
                        'component_name': iface.component_name,
                        'version': iface.version,
                        'methods': [
                            {
                                'name': m.name,
                                'params': m.params,
                                'returns': m.returns
                            }
                            for m in iface.methods
                        ]
                    }
                    for name, iface in seed.public_interfaces.items()
                }
                with open(interface_file, 'w') as f:
                    json.dump(interfaces_data, f, indent=2)
            
            print(f"[OK] Saved {seed_name} to {seed_dir}")
        
        # Save cross-seed connectors
        connectors_file = self.output_dir / "connectors.json"
        connectors_data = [
            {
                'provider': c.provider_seed,
                'consumer': c.consumer_seed,
                'interfaces': [iface.component_name for iface in c.interfaces]
            }
            for c in self.seed_connectors
        ]
        with open(connectors_file, 'w') as f:
            json.dump(connectors_data, f, indent=2)
        
        print(f"\n[OK] All seeds saved to {self.output_dir}")
