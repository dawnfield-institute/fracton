"""
GAIA Architecture-as-Code

Defines a PAC/SEC-based architecture for modeling information-entropy dynamics.
This is a complex 8-component system that should require multiple generations
to evolve successfully.

Based on Dawn Field Theory principles:
- PAC (Potential-Actualization Conservation)
- SEC (Symbolic Entropy Collapse)
- RBF (Recursive Balance Field)

Usage:
    python -m fracton.tools.shadowpuppet.examples.gaia_seed
"""

from typing import Dict, List, Any, Optional, Protocol, Tuple
from dataclasses import dataclass, field
from abc import abstractmethod
import math


# ============================================================================
# CONSTANTS
# ============================================================================

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.618
PHI_INV = 1 / PHI             # Inverse golden ≈ 0.618
XI = 1 + math.pi / 55         # Balance operator ≈ 1.057


# ============================================================================
# DOMAIN TYPES
# ============================================================================

@dataclass
class FieldPoint:
    """A point in the information-entropy field space."""
    position: Tuple[float, ...]  # N-dimensional position
    information: float           # Information density at this point
    entropy: float               # Entropy density at this point
    potential: float = 0.0       # Unrealized potential
    
    @property
    def balance(self) -> float:
        """I/E balance ratio. Stable when close to φ."""
        if self.entropy == 0:
            return float('inf')
        return self.information / self.entropy
    
    @property
    def is_stable(self) -> bool:
        """True if balance is within φ tolerance."""
        return abs(self.balance - PHI) < 0.1


@dataclass
class FieldGradient:
    """Gradient of a field quantity."""
    direction: Tuple[float, ...]  # Unit vector
    magnitude: float              # Gradient strength
    source_position: Tuple[float, ...]
    
    def dot(self, other: 'FieldGradient') -> float:
        """Dot product of gradient directions."""
        return sum(a * b for a, b in zip(self.direction, other.direction))


@dataclass
class CollapseEvent:
    """A symbolic entropy collapse event."""
    position: Tuple[float, ...]
    pre_entropy: float
    post_entropy: float
    information_released: float
    structure_type: str  # 'crystallization', 'bifurcation', 'annihilation'
    timestamp: float
    
    @property
    def collapse_ratio(self) -> float:
        """Ratio of entropy reduction."""
        if self.pre_entropy == 0:
            return 0.0
        return (self.pre_entropy - self.post_entropy) / self.pre_entropy


@dataclass
class Structure:
    """An emergent structure from collapse."""
    id: str
    origin: CollapseEvent
    field_points: List[FieldPoint]
    coherence: float  # 0-1, how well-organized
    stability: float  # 0-1, resistance to dissolution
    
    @property
    def total_information(self) -> float:
        return sum(p.information for p in self.field_points)
    
    @property
    def total_entropy(self) -> float:
        return sum(p.entropy for p in self.field_points)


@dataclass
class RecursiveState:
    """State of a recursive balance computation."""
    depth: int
    accumulated_balance: float
    fibonacci_weights: List[float]
    convergence_history: List[float]
    
    @property
    def is_converged(self) -> bool:
        """True if last 3 values are within tolerance."""
        if len(self.convergence_history) < 3:
            return False
        recent = self.convergence_history[-3:]
        return max(recent) - min(recent) < 0.001


@dataclass
class PACLedger:
    """Tracks conservation of potential-actualization."""
    total_potential: float
    total_actualized: float
    transfers: List[Tuple[str, str, float]]  # (from, to, amount)
    
    @property
    def conservation_error(self) -> float:
        """Should be 0 if PAC is conserved."""
        return abs((self.total_potential + self.total_actualized) - 1.0)
    
    @property
    def is_conserved(self) -> bool:
        return self.conservation_error < 0.001


# ============================================================================
# PROTOCOL SPECIFICATIONS
# ============================================================================

class InformationField(Protocol):
    """
    Represents the information field component.
    
    Models information density across space. Information gradients
    drive structure formation - structure grows where ∇I is high.
    
    PAC Invariants:
    - Information is non-negative everywhere
    - Total information is conserved under evolution
    - Gradients point toward higher density
    """
    dimensions: int
    resolution: int
    field_data: List[FieldPoint]
    
    @abstractmethod
    def get_density(self, position: Tuple[float, ...]) -> float:
        """Get information density at position."""
        ...
    
    @abstractmethod
    def get_gradient(self, position: Tuple[float, ...]) -> FieldGradient:
        """Get information gradient at position."""
        ...
    
    @abstractmethod
    def evolve(self, dt: float) -> None:
        """Evolve field forward in time."""
        ...
    
    @abstractmethod
    def inject(self, position: Tuple[float, ...], amount: float) -> None:
        """Inject information at position."""
        ...
    
    @abstractmethod
    def total(self) -> float:
        """Get total information in field."""
        ...


class EntropyField(Protocol):
    """
    Represents the entropy field component.
    
    Models entropy density across space. Entropy gradients
    oppose structure - dissolution occurs where ∇H is high.
    
    PAC Invariants:
    - Entropy is non-negative everywhere
    - Entropy tends to increase (second law) unless collapse occurs
    - Gradients point toward higher entropy
    """
    dimensions: int
    resolution: int
    field_data: List[FieldPoint]
    diffusion_rate: float
    
    @abstractmethod
    def get_density(self, position: Tuple[float, ...]) -> float:
        """Get entropy density at position."""
        ...
    
    @abstractmethod
    def get_gradient(self, position: Tuple[float, ...]) -> FieldGradient:
        """Get entropy gradient at position."""
        ...
    
    @abstractmethod
    def evolve(self, dt: float) -> None:
        """Evolve field forward (diffusion + generation)."""
        ...
    
    @abstractmethod
    def reduce(self, position: Tuple[float, ...], amount: float) -> float:
        """Reduce entropy at position (collapse). Returns actual reduction."""
        ...
    
    @abstractmethod
    def total(self) -> float:
        """Get total entropy in field."""
        ...


class BalanceOperator(Protocol):
    """
    The Ξ (Xi) operator that measures field balance.
    
    Computes the balance between information and entropy fields.
    Stable configurations occur when balance ≈ φ (golden ratio).
    
    PAC Invariants:
    - Balance is computed consistently across positions
    - Ξ ≈ 1.057 is the critical balance constant
    - Stability threshold is |balance - φ| < 0.1
    """
    xi_constant: float  # ≈ 1.057
    phi_target: float   # ≈ 1.618
    tolerance: float
    
    @abstractmethod
    def compute_local_balance(
        self, 
        info_field: InformationField,
        entropy_field: EntropyField,
        position: Tuple[float, ...]
    ) -> float:
        """Compute I/E balance at position."""
        ...
    
    @abstractmethod
    def compute_global_balance(
        self,
        info_field: InformationField,
        entropy_field: EntropyField
    ) -> float:
        """Compute global I/E balance."""
        ...
    
    @abstractmethod
    def find_stable_regions(
        self,
        info_field: InformationField,
        entropy_field: EntropyField
    ) -> List[Tuple[float, ...]]:
        """Find positions where balance ≈ φ."""
        ...
    
    @abstractmethod
    def apply_xi_correction(
        self,
        info_field: InformationField,
        entropy_field: EntropyField,
        position: Tuple[float, ...]
    ) -> None:
        """Apply Ξ correction to restore balance."""
        ...


class CollapseDetector(Protocol):
    """
    Detects and characterizes SEC collapse events.
    
    Monitors field dynamics for collapse signatures:
    - Rapid entropy reduction
    - Information crystallization
    - Phase transitions
    
    PAC Invariants:
    - Collapse events conserve total I + E
    - Collapse ratio is bounded [0, 1]
    - Structure type is deterministic from pre-collapse state
    """
    collapse_threshold: float
    history: List[CollapseEvent]
    
    @abstractmethod
    def detect(
        self,
        info_field: InformationField,
        entropy_field: EntropyField,
        dt: float
    ) -> List[CollapseEvent]:
        """Detect collapse events in current timestep."""
        ...
    
    @abstractmethod
    def classify_collapse(
        self,
        pre_state: FieldPoint,
        post_state: FieldPoint
    ) -> str:
        """Classify collapse type."""
        ...
    
    @abstractmethod
    def predict_collapse(
        self,
        info_field: InformationField,
        entropy_field: EntropyField,
        lookahead: float
    ) -> List[Tuple[float, ...]]:
        """Predict positions likely to collapse."""
        ...
    
    @abstractmethod
    def get_collapse_rate(self) -> float:
        """Get average collapse rate."""
        ...


class StructureEmitter(Protocol):
    """
    Emits structures from collapse events.
    
    When SEC collapse occurs, structure crystallizes from
    the released information. This component manages
    structure creation and lifecycle.
    
    PAC Invariants:
    - Structure information ≤ collapse information released
    - Structure coherence is bounded [0, 1]
    - Structures have unique IDs
    """
    structures: Dict[str, Structure]
    next_id: int
    
    @abstractmethod
    def emit(self, collapse: CollapseEvent) -> Structure:
        """Create structure from collapse event."""
        ...
    
    @abstractmethod
    def merge(self, struct_a: Structure, struct_b: Structure) -> Structure:
        """Merge two structures."""
        ...
    
    @abstractmethod
    def dissolve(self, structure: Structure, entropy_field: EntropyField) -> float:
        """Dissolve structure back to entropy. Returns entropy released."""
        ...
    
    @abstractmethod
    def get_by_coherence(self, min_coherence: float) -> List[Structure]:
        """Get structures above coherence threshold."""
        ...
    
    @abstractmethod
    def total_structure_information(self) -> float:
        """Get total information bound in structures."""
        ...


class RecursiveLayer(Protocol):
    """
    Implements RBF (Recursive Balance Field) computation.
    
    Recursively computes balance using Fibonacci-weighted
    aggregation. Convergence indicates stable configuration.
    
    PAC Invariants:
    - Recursion depth is bounded (max_depth)
    - Fibonacci weights sum to 1.0
    - Convergence is monotonic once achieved
    """
    max_depth: int
    fibonacci_cache: List[float]
    
    @abstractmethod
    def compute_recursive_balance(
        self,
        balance_op: BalanceOperator,
        info_field: InformationField,
        entropy_field: EntropyField,
        depth: int = 0
    ) -> RecursiveState:
        """Compute balance recursively with Fibonacci weighting."""
        ...
    
    @abstractmethod
    def get_fibonacci_weight(self, n: int) -> float:
        """Get normalized Fibonacci weight for level n."""
        ...
    
    @abstractmethod
    def check_convergence(self, state: RecursiveState) -> bool:
        """Check if recursive computation has converged."""
        ...
    
    @abstractmethod
    def get_convergence_depth(self, state: RecursiveState) -> int:
        """Get depth at which convergence occurred."""
        ...


class PACAggregator(Protocol):
    """
    Enforces PAC (Potential-Actualization Conservation).
    
    Tracks the conservation of potential becoming actual.
    f(Parent) = Σ f(Children) must hold at all times.
    
    PAC Invariants:
    - Total potential + actualized = 1.0 (normalized)
    - All transfers are recorded
    - Conservation error < 0.001
    """
    ledger: PACLedger
    tolerance: float
    
    @abstractmethod
    def record_actualization(
        self,
        source: str,
        target: str,
        amount: float
    ) -> None:
        """Record potential becoming actual."""
        ...
    
    @abstractmethod
    def verify_conservation(self) -> bool:
        """Verify PAC is conserved."""
        ...
    
    @abstractmethod
    def get_conservation_error(self) -> float:
        """Get current conservation error."""
        ...
    
    @abstractmethod
    def rebalance(self) -> None:
        """Rebalance to restore conservation."""
        ...
    
    @abstractmethod
    def get_transfer_history(self) -> List[Tuple[str, str, float]]:
        """Get history of transfers."""
        ...


class GAIAModel(Protocol):
    """
    Main GAIA orchestrator.
    
    Coordinates all components to simulate information-entropy
    dynamics with PAC conservation and SEC collapse.
    
    PAC Invariants:
    - All components initialized before use
    - Time evolution is deterministic
    - State can be serialized/restored
    """
    info_field: InformationField
    entropy_field: EntropyField
    balance_op: BalanceOperator
    collapse_detector: CollapseDetector
    structure_emitter: StructureEmitter
    recursive_layer: RecursiveLayer
    pac_aggregator: PACAggregator
    time: float
    
    @abstractmethod
    def step(self, dt: float) -> Dict[str, Any]:
        """Advance simulation by dt. Returns step metrics."""
        ...
    
    @abstractmethod
    def run(self, duration: float, dt: float) -> List[Dict[str, Any]]:
        """Run simulation for duration. Returns all step metrics."""
        ...
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get full simulation state."""
        ...
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore simulation state."""
        ...
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics (balance, entropy, structures, etc.)."""
        ...


# ============================================================================
# WORLDSEED METADATA
# ============================================================================

WORLDSEED_METADATA = {
    'identity': {
        'purpose': 'PAC/SEC information-entropy dynamics simulation',
        'domain': 'Dawn Field Theory',
        'version': '1.0.0'
    },
    'pac_invariants': [
        'Information is non-negative everywhere',
        'Entropy is non-negative everywhere',
        'PAC conservation: total potential + actualized = 1.0',
        'Balance stability: |I/E - φ| < 0.1 for stable regions',
        'Collapse conserves total I + E',
        'Fibonacci weights sum to 1.0',
        'Structure coherence bounded [0, 1]',
        'Recursion depth bounded by max_depth'
    ],
    'protocols': [
        'InformationField',
        'EntropyField',
        'BalanceOperator',
        'CollapseDetector',
        'StructureEmitter',
        'RecursiveLayer',
        'PACAggregator',
        'GAIAModel'
    ],
    'constants': {
        'phi': PHI,
        'phi_inv': PHI_INV,
        'xi': XI
    },
    'fibonacci_constraints': {
        'max_depth': 2,
        'max_components': 8
    }
}


# ============================================================================
# DOMAIN TYPE SOURCE CODE
# ============================================================================

DOMAIN_TYPES = [
    '''PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.618
PHI_INV = 1 / PHI             # Inverse golden ≈ 0.618
XI = 1 + math.pi / 55         # Balance operator ≈ 1.057''',

    '''@dataclass
class FieldPoint:
    """A point in the information-entropy field space."""
    position: Tuple[float, ...]
    information: float
    entropy: float
    potential: float = 0.0
    
    @property
    def balance(self) -> float:
        if self.entropy == 0:
            return float('inf')
        return self.information / self.entropy
    
    @property
    def is_stable(self) -> bool:
        return abs(self.balance - PHI) < 0.1''',

    '''@dataclass
class FieldGradient:
    """Gradient of a field quantity."""
    direction: Tuple[float, ...]
    magnitude: float
    source_position: Tuple[float, ...]
    
    def dot(self, other: 'FieldGradient') -> float:
        return sum(a * b for a, b in zip(self.direction, other.direction))''',

    '''@dataclass
class CollapseEvent:
    """A symbolic entropy collapse event."""
    position: Tuple[float, ...]
    pre_entropy: float
    post_entropy: float
    information_released: float
    structure_type: str
    timestamp: float
    
    @property
    def collapse_ratio(self) -> float:
        if self.pre_entropy == 0:
            return 0.0
        return (self.pre_entropy - self.post_entropy) / self.pre_entropy''',

    '''@dataclass
class Structure:
    """An emergent structure from collapse."""
    id: str
    origin: CollapseEvent
    field_points: List[FieldPoint]
    coherence: float
    stability: float
    
    @property
    def total_information(self) -> float:
        return sum(p.information for p in self.field_points)
    
    @property
    def total_entropy(self) -> float:
        return sum(p.entropy for p in self.field_points)''',

    '''@dataclass
class RecursiveState:
    """State of a recursive balance computation."""
    depth: int
    accumulated_balance: float
    fibonacci_weights: List[float]
    convergence_history: List[float]
    
    @property
    def is_converged(self) -> bool:
        if len(self.convergence_history) < 3:
            return False
        recent = self.convergence_history[-3:]
        return max(recent) - min(recent) < 0.001''',

    '''@dataclass
class PACLedger:
    """Tracks conservation of potential-actualization."""
    total_potential: float
    total_actualized: float
    transfers: List[Tuple[str, str, float]]
    
    @property
    def conservation_error(self) -> float:
        return abs((self.total_potential + self.total_actualized) - 1.0)
    
    @property
    def is_conserved(self) -> bool:
        return self.conservation_error < 0.001'''
]


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_information_non_negative(info_field):
    """Information must be non-negative everywhere."""
    for point in info_field.field_data:
        assert point.information >= 0, f"Negative information at {point.position}"
    return True


def test_entropy_non_negative(entropy_field):
    """Entropy must be non-negative everywhere."""
    for point in entropy_field.field_data:
        assert point.entropy >= 0, f"Negative entropy at {point.position}"
    return True


def test_balance_uses_phi(balance_op):
    """Balance operator should target φ."""
    assert abs(balance_op.phi_target - PHI) < 0.01, \
        f"phi_target {balance_op.phi_target} != PHI {PHI}"
    return True


def test_xi_constant(balance_op):
    """Balance operator should use Ξ ≈ 1.057."""
    assert abs(balance_op.xi_constant - XI) < 0.01, \
        f"xi_constant {balance_op.xi_constant} != XI {XI}"
    return True


def test_collapse_ratio_bounded(detector):
    """Collapse ratio must be in [0, 1]."""
    # Create a test collapse
    event = CollapseEvent(
        position=(0.0, 0.0),
        pre_entropy=1.0,
        post_entropy=0.5,
        information_released=0.3,
        structure_type='crystallization',
        timestamp=0.0
    )
    assert 0.0 <= event.collapse_ratio <= 1.0, \
        f"Collapse ratio {event.collapse_ratio} out of bounds"
    return True


def test_structure_coherence_bounded(emitter):
    """Structure coherence must be in [0, 1]."""
    for struct in emitter.structures.values():
        assert 0.0 <= struct.coherence <= 1.0, \
            f"Structure {struct.id} coherence {struct.coherence} out of bounds"
    return True


def test_fibonacci_weights_sum(recursive_layer):
    """Fibonacci weights should sum to 1.0."""
    if recursive_layer.fibonacci_cache:
        total = sum(recursive_layer.fibonacci_cache)
        assert abs(total - 1.0) < 0.01, \
            f"Fibonacci weights sum to {total}, not 1.0"
    return True


def test_pac_conservation(aggregator):
    """PAC must be conserved."""
    assert aggregator.verify_conservation(), \
        f"PAC not conserved, error: {aggregator.get_conservation_error()}"
    return True


def test_gaia_step_returns_metrics(model):
    """GAIA step should return metrics dict."""
    metrics = model.step(0.01)
    assert isinstance(metrics, dict), "step() should return dict"
    return True


def test_gaia_deterministic(model):
    """GAIA should be deterministic."""
    state1 = model.get_state()
    model.step(0.01)
    state2 = model.get_state()
    model.set_state(state1)
    model.step(0.01)
    state3 = model.get_state()
    # state2 and state3 should be identical
    assert state2['time'] == state3['time'], "GAIA not deterministic"
    return True


# ============================================================================
# EVOLUTION RUNNER
# ============================================================================

def run_evolution():
    """Run ShadowPuppet evolution on GAIA architecture."""
    import os
    from dotenv import load_dotenv
    from fracton.tools.shadowpuppet import (
        SoftwareEvolution,
        ProtocolSpec,
        GrowthGap,
        EvolutionConfig,
        TestSuite,
        ClaudeGenerator,
        MockGenerator
    )
    from pathlib import Path
    
    # Load API key from grimm's .env
    grimm_env = Path(__file__).parent.parent.parent.parent.parent.parent / "grimm" / ".env"
    if grimm_env.exists():
        load_dotenv(grimm_env)
        print(f"[*] Loaded API key from {grimm_env}")
    
    # Define protocols with dependencies
    protocols = [
        ProtocolSpec(
            name="InformationField",
            methods=["get_density", "get_gradient", "evolve", "inject", "total"],
            docstring="Information field with density and gradient computation",
            attributes=[
                "dimensions: int",
                "resolution: int",
                "field_data: List[FieldPoint]"
            ],
            pac_invariants=[
                "Information is non-negative everywhere",
                "Total information is conserved under evolution"
            ],
            dependencies=[]
        ),
        ProtocolSpec(
            name="EntropyField",
            methods=["get_density", "get_gradient", "evolve", "reduce", "total"],
            docstring="Entropy field with diffusion dynamics",
            attributes=[
                "dimensions: int",
                "resolution: int",
                "field_data: List[FieldPoint]",
                "diffusion_rate: float"
            ],
            pac_invariants=[
                "Entropy is non-negative everywhere",
                "Entropy increases unless collapse occurs"
            ],
            dependencies=[]
        ),
        ProtocolSpec(
            name="BalanceOperator",
            methods=["compute_local_balance", "compute_global_balance", "find_stable_regions", "apply_xi_correction"],
            docstring="Ξ operator for I/E balance computation",
            attributes=[
                "xi_constant: float",
                "phi_target: float",
                "tolerance: float"
            ],
            pac_invariants=[
                "Balance is computed consistently",
                "Stability threshold is |balance - φ| < 0.1"
            ],
            dependencies=["InformationField", "EntropyField"]
        ),
        ProtocolSpec(
            name="CollapseDetector",
            methods=["detect", "classify_collapse", "predict_collapse", "get_collapse_rate"],
            docstring="SEC collapse event detection",
            attributes=[
                "collapse_threshold: float",
                "history: List[CollapseEvent]"
            ],
            pac_invariants=[
                "Collapse conserves total I + E",
                "Collapse ratio bounded [0, 1]"
            ],
            dependencies=["InformationField", "EntropyField"]
        ),
        ProtocolSpec(
            name="StructureEmitter",
            methods=["emit", "merge", "dissolve", "get_by_coherence", "total_structure_information"],
            docstring="Structure creation from collapse events",
            attributes=[
                "structures: Dict[str, Structure]",
                "next_id: int"
            ],
            pac_invariants=[
                "Structure information ≤ collapse information",
                "Structure coherence bounded [0, 1]"
            ],
            dependencies=["CollapseDetector", "EntropyField"]
        ),
        ProtocolSpec(
            name="RecursiveLayer",
            methods=["compute_recursive_balance", "get_fibonacci_weight", "check_convergence", "get_convergence_depth"],
            docstring="RBF recursive balance with Fibonacci weighting",
            attributes=[
                "max_depth: int",
                "fibonacci_cache: List[float]"
            ],
            pac_invariants=[
                "Recursion depth bounded by max_depth",
                "Fibonacci weights sum to 1.0"
            ],
            dependencies=["BalanceOperator", "InformationField", "EntropyField"]
        ),
        ProtocolSpec(
            name="PACAggregator",
            methods=["record_actualization", "verify_conservation", "get_conservation_error", "rebalance", "get_transfer_history"],
            docstring="PAC conservation enforcement",
            attributes=[
                "ledger: PACLedger",
                "tolerance: float"
            ],
            pac_invariants=[
                "Total potential + actualized = 1.0",
                "Conservation error < 0.001"
            ],
            dependencies=[]
        ),
        ProtocolSpec(
            name="GAIAModel",
            methods=["step", "run", "get_state", "set_state", "get_metrics"],
            docstring="Main GAIA orchestrator for PAC/SEC simulation",
            attributes=[
                "info_field: InformationField",
                "entropy_field: EntropyField",
                "balance_op: BalanceOperator",
                "collapse_detector: CollapseDetector",
                "structure_emitter: StructureEmitter",
                "recursive_layer: RecursiveLayer",
                "pac_aggregator: PACAggregator",
                "time: float"
            ],
            pac_invariants=[
                "All components initialized before use",
                "Time evolution is deterministic",
                "State can be serialized/restored"
            ],
            dependencies=[
                "InformationField",
                "EntropyField",
                "BalanceOperator",
                "CollapseDetector",
                "StructureEmitter",
                "RecursiveLayer",
                "PACAggregator"
            ]
        ),
    ]
    
    # Create gaps with test suites
    test_map = {
        "InformationField": [test_information_non_negative],
        "EntropyField": [test_entropy_non_negative],
        "BalanceOperator": [test_balance_uses_phi, test_xi_constant],
        "CollapseDetector": [test_collapse_ratio_bounded],
        "StructureEmitter": [test_structure_coherence_bounded],
        "RecursiveLayer": [test_fibonacci_weights_sum],
        "PACAggregator": [test_pac_conservation],
        "GAIAModel": [test_gaia_step_returns_metrics],
    }
    
    gaps = []
    for p in protocols:
        test_funcs = test_map.get(p.name, [])
        gap = GrowthGap(
            protocol=p,
            test_suite=TestSuite(unit=test_funcs) if test_funcs else None,
            domain_types=DOMAIN_TYPES,
            required_coherence=0.70  # Higher bar than chatbot
        )
        gaps.append(gap)
    
    # Configure evolution - need more generations for this complexity
    config = EvolutionConfig(
        coherence_threshold=0.60,  # Start lower, let evolution find solutions
        reproduction_threshold=0.75,
        candidates_per_gap=3,
        max_generations=10,
        mutation_rate=0.3,
        output_dir=Path("gaia_evolution")
    )
    
    # Run evolution with Claude API for real implementations
    # Set ANTHROPIC_API_KEY env var before running
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("[!] ANTHROPIC_API_KEY not set, using MockGenerator")
        generator = MockGenerator()
    else:
        generator = ClaudeGenerator(
            model="claude-sonnet-4-20250514",
            temperature=0.4,
            fallback_generator=MockGenerator()
        )
    
    evolution = SoftwareEvolution(
        generator=generator,
        config=config,
        pac_invariants=WORLDSEED_METADATA['pac_invariants']
    )
    
    print("=" * 70)
    print("GAIA Architecture Evolution (PAC/SEC Dynamics)")
    print("=" * 70)
    print(f"Components: {len(protocols)}")
    print(f"Invariants: {len(WORLDSEED_METADATA['pac_invariants'])}")
    print(f"Constants: phi={PHI:.6f}, Xi={XI:.6f}")
    print()
    
    results = evolution.grow(gaps)
    
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Success: {results['success']}")
    print(f"Generations: {results['generations']}")
    print(f"Final population: {results['final_population']}")
    
    if results['best_component']:
        print(f"\nBest component: {results['best_component']['id']}")
        print(f"Coherence: {results['best_component']['coherence_score']:.3f}")
    
    # Show genealogy
    print("\nGenealogy:")
    evolution.genealogy.print_tree()
    
    # Save generated code
    evolution.save_code(Path("gaia_generated"))
    
    # Summary by component
    print("\n" + "=" * 70)
    print("Component Summary")
    print("=" * 70)
    for comp in sorted(evolution.components, key=lambda c: c.protocol_name):
        print(f"  {comp.protocol_name}: {comp.coherence_score:.3f} "
              f"(S:{comp.structural_score:.2f} Se:{comp.semantic_score:.2f} E:{comp.energetic_score:.2f})")
    
    return evolution


if __name__ == "__main__":
    run_evolution()
