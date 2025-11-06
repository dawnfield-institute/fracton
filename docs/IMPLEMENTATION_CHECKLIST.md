# Reality Engine Integration - Implementation Checklist

**Phase**: 0 - Reality Engine Integration into Fracton  
**Start Date**: November 2025  
**Estimated Duration**: 4 weeks  
**Status**: � Week 1 In Progress - Days 1-5 Complete (71% of Week 1)

---

## Overview

This checklist tracks the implementation of Reality Engine v2 as the `fracton/mobius/` module. Each item includes acceptance criteria, dependencies, and validation steps.

**Progress Summary**:
- ✅ Week 1, Days 1-5: **COMPLETE** (52/52 tests passing, 100%)
- ⏳ Week 1, Days 6-7: Pending (SEC, Confluence, Time)
- 📋 Weeks 2-4: Planned

**Test Coverage**: 52/52 unit tests passing (100%)
- Substrate: 19/19 tests ✅
- Thermodynamics: 33/33 tests ✅

---

## Week 1: Core Infrastructure

### Day 1-2: Module Setup ✅ **COMPLETE**

- [x] **Create directory structure**
  ```bash
  mkdir -p fracton/mobius
  touch fracton/mobius/__init__.py
  ```
  - **Acceptance**: Directory exists with `__init__.py` ✅
  - **Validation**: `import fracton.mobius` works ✅
  - **Completed**: November 4, 2025

- [x] **Update setup.py**
  ```python
  packages=find_packages(include=['fracton', 'fracton.*', 'fracton.mobius'])
  ```
  - **Acceptance**: Package includes mobius module ✅
  - **Validation**: `pip install -e .` includes mobius ✅

- [x] **Create module exports**
  ```python
  # fracton/mobius/__init__.py
  from .substrate import MobiusManifold
  from .thermodynamics import ThermodynamicField
  from .constants import *
  # SEC, Confluence, Time, RealityEngine - phased implementation
  ```
  - **Acceptance**: Core imports available ✅
  - **Validation**: Import test passes ✅

- [x] **Add to Fracton main __init__.py**
  ```python
  # Optional import
  try:
      from . import mobius
  except ImportError:
      pass
  ```
  - **Acceptance**: Fracton imports without errors ✅
  - **Validation**: Works with/without torch installed ✅

### Day 3-4: Constants & Substrate ✅ **COMPLETE**

- [x] **Create constants.py**
  - **File**: `fracton/mobius/constants.py` (221 lines) ✅
  - **Content**:
    - Universal constants (Ξ=1.0571, λ=0.020, DEPTH_MAX=2) ✅
    - Physical parameters (K_BOLTZMANN, K_T) ✅
    - Validated values from legacy experiments ✅
    - Validation framework with validate_constant() ✅
  - **Acceptance**: All constants defined ✅
  - **Validation**: Import and print all constants ✅
  - **Completed**: November 4, 2025

- [x] **Document constant origins**
  ```python
  # Validated from cosmo.py, brain.py, vcpu.py
  XI = 1.0571  # ±0.001, universal constant
  LAMBDA = 0.020  # ±0.001 Hz, fundamental frequency
  DEPTH_MAX = 2  # Exact, structure depth limit
  ```
  - **Acceptance**: Each constant has source comment ✅
  - **Validation**: Comments reference specific experiments ✅

- [x] **Add validation ranges**
  ```python
  validate_constant(name, measured_value, target, tolerance)
  print_validation_report()
  ```
  - **Acceptance**: Validation framework implemented ✅
  - **Validation**: validate_constant() function working ✅

- [x] **Create substrate.py**
  - **File**: `fracton/mobius/substrate.py` (419 lines) ✅
  - **Class**: `MobiusManifold` ✅
  - **Dependencies**: torch, MemoryField (adapted API) ✅
  - **Acceptance**: File exists with class definition ✅
  - **Validation**: Can instantiate MobiusManifold ✅
  - **Completed**: November 4, 2025

- [x] **Implement __init__**
  ```python
  def __init__(self, size=(128, 32), device='cuda'):
      self.size = size
      self.device = device
      # Direct tensor storage (MemoryField API adapted)
      self._P, self._A, self._M = None, None, None
      # Möbius coordinates
      self.u_coords, self.v_coords = ...
      self.U, self.V = torch.meshgrid(...)
  ```
  - **Acceptance**: Initializes without errors ✅
  - **Validation**: Creates coordinate grids correctly ✅

- [x] **Implement anti-periodic boundaries**
  - **Methods**:
    - `_create_antiperiodic_field()` ✅
    - `_enforce_antiperiodicity(field)` ✅
    - `validate_antiperiodicity(field)` ✅
  - **Constraint**: f(u+π, v) = -f(u, 1-v) ✅
  - **Acceptance**: Enforcement reduces error to ~0 ✅
  - **Validation**: validate_antiperiodicity() returns < 1e-6 ✅

- [x] **Implement initialization modes**
  - **Methods**: `initialize_fields(mode)` ✅
  - **Modes**:
    - `'big_bang'`: High P, zero A (primordial state) ✅
    - `'random'`: Random fields ✅
    - `'structured'`: Localized patterns ✅
  - **Acceptance**: All modes create valid fields ✅
  - **Validation**: Test each mode ✅

- [x] **Add topology metrics**
  - **Method**: `get_topology_metrics()` ✅
  - **Metrics**:
    - Euler characteristic (0 for Möbius) ✅
    - Orientability (False for Möbius) ✅
    - Boundary components (1 for Möbius) ✅
    - Anti-periodic error ✅
    - Field energies ✅
  - **Acceptance**: Returns complete dict ✅
  - **Validation**: Metrics match theory ✅

- [x] **Create unit tests**
  - **File**: `tests/test_mobius_substrate.py` ✅
  - **Coverage**: 19/19 tests passing (100%) ✅
  - **Tests**:
    - Initialization & coordinates ✅
    - All three field modes ✅
    - Anti-periodic enforcement ✅
    - Validation functions ✅
    - Topology metrics ✅
    - GPU/CPU compatibility ✅
  - **Acceptance**: All tests pass ✅
  - **Validation**: `pytest tests/test_mobius_substrate.py -v` ✅

### Day 5: Thermodynamics ✅ **COMPLETE**

- [x] **Create thermodynamics.py**
  - **File**: `fracton/mobius/thermodynamics.py` (413 lines) ✅
  - **Class**: `ThermodynamicField` ✅
  - **Dependencies**: torch, numpy, constants ✅
  - **Acceptance**: File exists with class definition ✅
  - **Validation**: Can instantiate ThermodynamicField ✅
  - **Completed**: November 4, 2025

- [x] **Implement temperature field**
  - **Method**: `initialize_temperature(A, mode)` ✅
  - **Modes**:
    - `'uniform'`: Constant T = K_T ✅
    - `'from_field'`: T ∝ |A|² (local energy) ✅
    - `'hot_big_bang'`: T = 100 * K_T (Planck scale) ✅
  - **Acceptance**: All modes work ✅
  - **Validation**: Temperature fields valid ✅

- [x] **Implement entropy computation**
  - **Method**: `compute_entropy(field, bins)` ✅
  - **Algorithm**: Shannon entropy H = -Σ p(x) log p(x) ✅
  - **Acceptance**: Returns non-negative entropy ✅
  - **Validation**: Uniform < structured < random ✅

- [x] **Implement free energy**
  - **Method**: `compute_free_energy(E, S)` ✅
  - **Formula**: F = E - T*S ✅
  - **Acceptance**: Correct thermodynamic formula ✅
  - **Validation**: F decreases with T ✅

- [x] **Implement Landauer principle**
  - **Method**: `landauer_erasure_cost(bits_erased)` ✅
  - **Formula**: E = kT ln(2) per bit (dimensionless units) ✅
  - **Acceptance**: Scales linearly with bits and T ✅
  - **Validation**: Cost > 0 for bits > 0 ✅

- [x] **Implement heat diffusion**
  - **Method**: `apply_heat_diffusion(dt, alpha)` ✅
  - **Equation**: ∂T/∂t = α∇²T (Fourier's law) ✅
  - **Laplacian**: `_compute_laplacian_2d(field)` ✅
  - **Acceptance**: Temperature smooths over time ✅
  - **Validation**: Conserves mean temperature ✅

- [x] **Implement thermal fluctuations**
  - **Method**: `inject_thermal_fluctuations(field, dt)` ✅
  - **Algorithm**: Langevin noise σ = √(2kTdt) ✅
  - **Acceptance**: Adds stochastic perturbations ✅
  - **Validation**: Amplitude scales as √T and √dt ✅

- [x] **Implement entropy tracking**
  - **Method**: `track_entropy_production(S_before, S_after)` ✅
  - **Monitoring**: 2nd law violations (ΔS < 0) ✅
  - **Acceptance**: Tracks cumulative entropy change ✅
  - **Validation**: Detects violations correctly ✅

- [x] **Create unit tests**
  - **File**: `tests/test_mobius_thermodynamics.py` ✅
  - **Coverage**: 33/33 tests passing (100%) ✅
  - **Tests**:
    - Temperature initialization (4 tests) ✅
    - Entropy calculations (4 tests) ✅
    - Free energy (2 tests) ✅
    - Landauer principle (4 tests) ✅
    - Heat diffusion (3 tests) ✅
    - Thermal fluctuations (3 tests) ✅
    - Entropy production (4 tests) ✅
    - Laplacian computation (3 tests) ✅
    - Device compatibility (4 tests) ✅
    - State reporting (2 tests) ✅
  - **Acceptance**: All tests pass ✅
  - **Validation**: `pytest tests/test_mobius_thermodynamics.py -v` ✅

### Day 6-7: SEC, Confluence, Time 🔄 **IN PROGRESS**

- [x] **Create sec_operator.py**
  - **File**: `fracton/mobius/sec_operator.py` (335 lines) ✅
  - **Class**: `SymbolicEntropyCollapse` ✅
  - **Energy Functional**: E(A|P,T) = α||A-P||² + β||∇A||² + γ∫T·|A|² ✅
  - **Acceptance**: File exists with SEC class ✅
  - **Validation**: Can instantiate SymbolicEntropyCollapse ✅
  - **Completed**: November 4, 2025

- [x] **Implement SEC initialization**
  ```python
  def __init__(self, alpha=0.1, beta=0.05, gamma=0.01, device='cpu'):
      self.alpha = alpha  # Potential-Actual coupling
      self.beta = beta    # Spatial smoothing (MED)
      self.gamma = gamma  # Thermodynamic coupling
      # Track thermodynamics
      self.total_heat_generated = 0.0
      self.total_entropy_reduced = 0.0
      self.collapse_events = []
  ```
  - **Acceptance**: Parameters stored correctly ✅
  - **Validation**: Default and custom initialization work ✅

- [x] **Implement energy computation**
  - **Method**: `compute_energy(A, P, T)` ✅
  - **Components**:
    - Coupling: α||A-P||² ✅
    - Smoothness: β||∇A||² ✅
    - Thermal: γ∫T·|A|² ✅
  - **Returns**: Dict with all energy components ✅
  - **Acceptance**: Returns correct energies ✅
  - **Validation**: Total = sum of components ✅

- [x] **Implement field evolution**
  - **Method**: `evolve(A, P, T, dt, add_thermal_noise)` ✅
  - **Dynamics**: ∂A/∂t = -α(A-P) - β∇²A + thermal_noise ✅
  - **Integration**: Forward Euler ✅
  - **Noise**: Langevin σ = √(2kTdt) ✅
  - **Acceptance**: Field evolves toward Potential ✅
  - **Validation**: Distance to P decreases ✅

- [x] **Implement heat generation**
  - **Method**: `_compute_heat_generation(A_before, A_after, dt)` ✅
  - **Sources**:
    - Kinetic: (1/2)|dA/dt|² ✅
    - Landauer: kT ln(2) per bit erased ✅
  - **Tracking**: Cumulative heat ✅
  - **Acceptance**: Returns positive heat ✅
  - **Validation**: Heat tracked correctly ✅

- [x] **Implement entropy tracking**
  - **Method**: `_compute_field_entropy(field, bins)` ✅
  - **Formula**: H = -Σ p(x) log p(x) ✅
  - **Tracking**: Cumulative entropy reduction ✅
  - **Collapse detection**: Rapid entropy reduction events ✅
  - **Acceptance**: Entropy computed correctly ✅
  - **Validation**: Tracks reduction during collapse ✅

- [x] **Implement collapse detection**
  - **Method**: `detect_collapse_regions(A, threshold)` ✅
  - **Metric**: Gradient magnitude ||∇A|| ✅
  - **Output**: Binary mask of high-gradient regions ✅
  - **Acceptance**: Detects sharp transitions ✅
  - **Validation**: Threshold adjusts sensitivity ✅

- [x] **Implement Laplacian**
  - **Method**: `_compute_laplacian_2d(field)` ✅
  - **Scheme**: Finite differences ∇²f ≈ (f[i+1,j] + f[i-1,j] + f[i,j+1] + f[i,j-1] - 4f[i,j]) ✅
  - **Boundaries**: Zero padding ✅
  - **Acceptance**: Laplacian computed correctly ✅
  - **Validation**: Constant field → zero Laplacian (interior) ✅

- [x] **Implement state reporting**
  - **Method**: `get_sec_state()` ✅
  - **Returns**: Dict with heat, entropy, collapse count, parameters ✅
  - **Repr**: Human-readable string ✅
  - **Acceptance**: State retrievable ✅
  - **Validation**: All fields present ✅

- [x] **Add convenience constructor**
  ```python
  def create_sec_operator(coupling_strength, smoothness_strength, 
                          thermal_strength, device):
      return SymbolicEntropyCollapse(alpha=..., beta=..., gamma=...)
  ```
  - **Acceptance**: Constructor works ✅
  - **Validation**: Parameters passed correctly ✅

- [x] **Create unit tests**
  - **File**: `tests/test_mobius_sec.py` ✅
  - **Coverage**: 25/25 tests passing (100%) ✅
  - **Tests**:
    - Initialization (3 tests) ✅
    - Energy functional (4 tests) ✅
    - Field evolution (4 tests) ✅
    - Heat generation (2 tests) ✅
    - Entropy tracking (2 tests) ✅
    - Collapse detection (3 tests) ✅
    - Laplacian computation (3 tests) ✅
    - State reporting (2 tests) ✅
    - Device compatibility (2 tests) ✅
  - **Acceptance**: All tests pass ✅
  - **Validation**: `pytest tests/test_mobius_sec.py -v` ✅
  - **Completed**: November 4, 2025

- [ ] **Create confluence.py**
  - **Validation**: `substrate = MobiusManifold()` works

- [ ] **Implement initialize_fields**
  - **Modes**: 'big_bang', 'random', 'structured'
  - **Returns**: (P, A, M) tensors
  - **Acceptance**: All three modes work
  - **Validation**: 
    ```python
    P, A, M = substrate.initialize_fields('big_bang')
    assert P.shape == (128, 32)
    assert A.shape == (128, 32)
    assert M.shape == (128, 32)
    ```

- [ ] **Implement _create_antiperiodic_field**
  ```python
  def _create_antiperiodic_field(self):
      field = torch.sin(self.U) * (2 * self.V - 1)
      # Add harmonics...
      return field
  ```
  - **Acceptance**: Returns field satisfying anti-periodicity
  - **Validation**: `validate_antiperiodicity(field) < 0.1`

- [ ] **Implement validate_antiperiodicity**
  ```python
  def validate_antiperiodicity(self, field):
      # Check: f(u+π, 1-v) ≈ -f(u, v)
      error = ...
      return error.item()
  ```
  - **Acceptance**: Returns float error metric
  - **Validation**: Error < 0.1 for antiperiodic fields

- [ ] **Implement get_topology_metrics**
  ```python
  def get_topology_metrics(self):
      return {
          'euler_characteristic': -1,
          'orientable': False,
          'antiperiodic_error': ...,
          'field_energy': ...,
          'field_entropy': ...
      }
  ```
  - **Acceptance**: Returns dict with all metrics
  - **Validation**: Check metric values make sense

- [ ] **Write unit tests**
  - **File**: `tests/test_mobius_substrate.py`
  - **Tests**:
    - Initialization
    - Field modes
    - Anti-periodicity validation
    - Topology metrics
    - GPU/CPU compatibility
  - **Acceptance**: All tests pass
  - **Validation**: `pytest tests/test_mobius_substrate.py`

---

## Week 2: Thermodynamics and Dynamics

### Day 1-2: Thermodynamic Fields

- [ ] **Create thermodynamics.py**
  - **File**: `fracton/mobius/thermodynamics.py`
  - **Class**: `ThermodynamicField`
  - **Acceptance**: File exists with class
  - **Validation**: Can instantiate ThermodynamicField

- [ ] **Implement __init__**
  ```python
  def __init__(self, size, device='cuda'):
      self.size = size
      self.device = device
      self.k_B = 1.380649e-23
      self.k_T = 1.0
      self.T = torch.ones(size, device=device) * self.k_T
  ```
  - **Acceptance**: Initializes temperature field
  - **Validation**: `thermo.T.shape == size`

- [ ] **Implement initialize_temperature**
  - **Modes**: 'uniform', 'from_field', 'hot_big_bang'
  - **Acceptance**: All modes work
  - **Validation**: Temperature field created correctly

- [ ] **Implement thermodynamic methods**
  - `compute_entropy(field)` - Shannon entropy
  - `compute_free_energy(E, S)` - F = E - TS
  - `landauer_erasure_cost(bits)` - kT ln(2) per bit
  - **Acceptance**: All methods implemented
  - **Validation**: Unit tests for each method

- [ ] **Implement heat diffusion**
  ```python
  def apply_heat_diffusion(self, dt=0.001, alpha=0.01):
      laplacian_T = self._compute_laplacian_2d(self.T)
      self.T += dt * alpha * laplacian_T
      self.T = torch.clamp(self.T, min=0.01)
  ```
  - **Acceptance**: Temperature diffuses
  - **Validation**: Heat spreads from hot spots

- [ ] **Implement thermal noise**
  ```python
  def inject_thermal_fluctuations(self, field, dt=0.001):
      noise_amplitude = torch.sqrt(2 * self.T * dt)
      noise = noise_amplitude * torch.randn_like(field)
      return field + noise
  ```
  - **Acceptance**: Adds Langevin noise
  - **Validation**: Noise amplitude proportional to √T

- [ ] **Implement 2nd law tracking**
  ```python
  def track_entropy_production(self, S_before, S_after):
      dS = S_after - S_before
      self.total_entropy_change += dS
      if dS < 0:
          print(f"⚠️  2nd Law Violation: ΔS = {dS}")
  ```
  - **Acceptance**: Tracks entropy changes
  - **Validation**: Warns on violations

- [ ] **Write unit tests**
  - **File**: `tests/test_thermodynamics.py`
  - **Tests**:
    - Temperature initialization
    - Entropy calculation
    - Free energy
    - Landauer costs
    - Heat diffusion
    - Thermal noise
    - 2nd law tracking
  - **Acceptance**: All tests pass
  - **Validation**: `pytest tests/test_thermodynamics.py`

### Day 3-4: SEC Operator

- [ ] **Port GeometricSEC from PACEngine**
  - **Source**: `PACEngine/modules/geometric_sec.py`
  - **Destination**: `fracton/mobius/sec_operator.py`
  - **Acceptance**: Code ported and adapted
  - **Validation**: Imports without errors

- [ ] **Create SymbolicEntropyCollapse class**
  ```python
  class SymbolicEntropyCollapse:
      def __init__(self, alpha=0.1, beta=0.05, gamma=0.01):
          # Use PACEngine SEC as foundation
          self.pac_sec = PACEngineSEC(alpha, beta)
          self.gamma = gamma
  ```
  - **Acceptance**: Class wraps PACEngine SEC
  - **Validation**: Can instantiate

- [ ] **Implement evolve method**
  ```python
  def evolve(self, A, P, T, dt=0.001):
      # Use PACEngine's energy functional
      A_evolved = self.pac_sec.step(A, P, dt)
      
      # Add thermal noise
      noise = torch.sqrt(2 * T * dt * self.gamma) * torch.randn_like(A)
      A_evolved += noise
      
      # Calculate heat generation
      heat_generated = torch.abs(A_evolved - A) * self.alpha
      
      return A_evolved, heat_generated
  ```
  - **Acceptance**: Evolves field with thermodynamics
  - **Validation**: 
    - Energy decreases
    - Heat generated > 0
    - Shape preserved

- [ ] **Implement compute_energy**
  ```python
  def compute_energy(self, A, P):
      coupling_energy = self.alpha * ((A - P)**2).sum()
      gradient_energy = self.beta * (grad_A**2).sum()
      return (coupling_energy + gradient_energy).item()
  ```
  - **Acceptance**: Returns scalar energy
  - **Validation**: Energy positive, decreases over time

- [ ] **Write unit tests**
  - **File**: `tests/test_sec_operator.py`
  - **Tests**:
    - Initialization
    - Evolution (single step)
    - Energy computation
    - Heat generation
    - Thermodynamic coupling
    - Comparison with PACEngine
  - **Acceptance**: All tests pass
  - **Validation**: `pytest tests/test_sec_operator.py`

### Day 5: Confluence and Time Emergence

- [ ] **Create confluence.py**
  - **File**: `fracton/mobius/confluence.py`
  - **Class**: `MobiusConfluence`
  - **Acceptance**: File exists
  - **Validation**: Can import

- [ ] **Implement step method**
  ```python
  def step(self, A, substrate):
      # Shift by π in u
      u_shift = A.shape[0] // 2
      A_shifted = torch.roll(A, shifts=u_shift, dims=0)
      
      # Flip in v
      A_flipped = torch.flip(A_shifted, dims=[1])
      
      # Apply sign change
      P_next = -A_flipped
      
      return P_next
  ```
  - **Acceptance**: Returns next potential field
  - **Validation**: Anti-periodicity enforced

- [ ] **Create time_emergence.py**
  - **File**: `fracton/mobius/time_emergence.py`
  - **Class**: `DisequilibriumTime`
  - **Acceptance**: File exists
  - **Validation**: Can import

- [ ] **Implement disequilibrium methods**
  ```python
  def compute_disequilibrium(self, P, A):
      return torch.abs(P - A)
  
  def compute_pressure(self, P, A):
      return self.compute_disequilibrium(P, A).mean().item()
  ```
  - **Acceptance**: Returns scalar pressure
  - **Validation**: Pressure >= 0

- [ ] **Implement time rate computation**
  ```python
  def compute_time_rate(self, pressure, T):
      T_mean = T.mean().item()
      time_rate = pressure * (1.0 + 0.1 * T_mean)
      return time_rate
  ```
  - **Acceptance**: Returns time rate
  - **Validation**: Rate > 0 when pressure > 0

- [ ] **Implement time dilation**
  ```python
  def compute_interaction_density(self, A):
      energy_density = A**2
      return energy_density / energy_density.mean()
  
  def compute_time_dilation(self, interaction_density):
      return 1.0 / (1.0 + interaction_density)
  ```
  - **Acceptance**: Returns dilation factors
  - **Validation**: Dense regions have dilation < 1

- [ ] **Write unit tests**
  - **File**: `tests/test_confluence_time.py`
  - **Tests**:
    - Confluence step
    - Anti-periodicity after confluence
    - Disequilibrium computation
    - Time rate calculation
    - Interaction density
    - Time dilation
  - **Acceptance**: All tests pass
  - **Validation**: `pytest tests/test_confluence_time.py`

### Day 6-7: Unified Reality Engine

- [ ] **Create reality_engine.py**
  - **File**: `fracton/mobius/reality_engine.py`
  - **Class**: `RealityEngine(RecursiveEngine)`
  - **Acceptance**: Subclasses Fracton's RecursiveEngine
  - **Validation**: Can instantiate

- [ ] **Implement __init__**
  ```python
  def __init__(self, size=(128, 32), device='cuda'):
      super().__init__(device=device)
      
      # Möbius substrate
      self.substrate = MobiusManifold(size, device)
      
      # Thermodynamics
      self.thermo = ThermodynamicField(size, device)
      
      # Dynamics
      self.sec = SymbolicEntropyCollapse()
      self.confluence = MobiusConfluence()
      self.time = DisequilibriumTime()
      
      # Fracton components
      self.pac = PACRegulation(tolerance=1e-12)
      self.rbf = RBFEngine()
      self.qbe = QBERegulator()
  ```
  - **Acceptance**: Initializes all components
  - **Validation**: No errors on instantiation

- [ ] **Implement initialize method**
  ```python
  def initialize(self, mode='big_bang'):
      P, A, M = self.substrate.initialize_fields(mode)
      T = self.thermo.initialize_temperature(A, ...)
      return P, A, M, T
  ```
  - **Acceptance**: All modes work
  - **Validation**: Fields created correctly

- [ ] **Implement evolve method**
  ```python
  def evolve(self, steps=1000):
      P, A, M = self.substrate.get_fields()
      T = self.thermo.T
      
      for step in self.iterate(steps):
          # Time emergence
          pressure = self.time.compute_pressure(P, A)
          time_rate = self.time.compute_time_rate(pressure, T)
          
          # SEC collapse
          A, heat = self.sec.evolve(A, P, T, time_rate)
          self.thermo.add_heat(heat)
          
          # Thermodynamics
          S_before = self.thermo.compute_entropy(A)
          self.thermo.apply_heat_diffusion(time_rate)
          T = self.thermo.T
          S_after = self.thermo.compute_entropy(A)
          self.thermo.track_entropy_production(S_before, S_after)
          
          # PAC enforcement
          P, A, M = self.pac.enforce(P, A, M)
          
          # RBF balance
          B = self.rbf.compute_balance(P, A, M)
          P, A = self.rbf.apply_balance(P, A, B)
          
          # QBE regulation
          P, A = self.qbe.regulate(P, A, self.time.current_time)
          
          # Confluence (time step!)
          P = self.confluence.step(A, self.substrate)
          
          # Update time
          self.time.update_time(time_rate)
          
          # Store fields
          self.substrate.memory_field.write(torch.stack([P, A, M]))
          
          yield {...}
  ```
  - **Acceptance**: Full evolution loop works
  - **Validation**: 
    - Runs for 1000+ steps
    - No crashes
    - PAC error < 1e-10
    - Entropy increases

- [ ] **Write integration tests**
  - **File**: `tests/test_reality_engine.py`
  - **Tests**:
    - Initialization
    - Single evolution step
    - Multiple steps (100+)
    - PAC conservation
    - Entropy increase
    - Energy decrease
    - Time emergence
    - State tracking
  - **Acceptance**: All tests pass
  - **Validation**: `pytest tests/test_reality_engine.py`

---

## Week 3: Law Discovery and Validation

### Day 1-3: Law Discovery Framework

- [ ] **Create law_discovery.py**
  - **File**: `fracton/mobius/law_discovery.py`
  - **Acceptance**: File exists
  - **Validation**: Can import

- [ ] **Implement conservation law detection**
  ```python
  def detect_conservation_laws(states):
      # Check for conserved quantities
      # P - A - M = const
      # Total energy = const
      # etc.
      pass
  ```
  - **Acceptance**: Identifies conserved quantities
  - **Validation**: Finds PAC conservation

- [ ] **Implement force law extraction**
  ```python
  def extract_force_laws(states):
      # Analyze field interactions
      # Look for inverse-square patterns
      # Identify attraction/repulsion
      pass
  ```
  - **Acceptance**: Extracts force relationships
  - **Validation**: Manual check on test data

- [ ] **Implement symmetry detection**
  ```python
  def detect_symmetries(states):
      # Check for rotational symmetry
      # Check for translational invariance
      # Check for scale invariance
      pass
  ```
  - **Acceptance**: Identifies symmetries
  - **Validation**: Finds expected symmetries

- [ ] **Implement constant measurement**
  ```python
  def measure_constants(states):
      # Extract Ξ
      # Extract λ (frequency)
      # Extract max depth
      pass
  ```
  - **Acceptance**: Measures constants from evolution
  - **Validation**: Compare to targets

- [ ] **Integrate into RealityEngine**
  ```python
  class RealityEngine:
      def discover_laws(self, states):
          laws = {}
          laws['conservation'] = detect_conservation_laws(states)
          laws['forces'] = extract_force_laws(states)
          laws['symmetries'] = detect_symmetries(states)
          laws['constants'] = measure_constants(states)
          return laws
  ```
  - **Acceptance**: Method exists
  - **Validation**: Returns dict of laws

### Day 4-7: Legacy Experiment Validation

- [ ] **Reproduce cosmo.py results**
  - **Run**: Long evolution (50k+ steps)
  - **Measure**: Ξ, λ, depth
  - **Compare**: To known values
  - **Acceptance**: Within tolerance
  - **Validation**:
    ```python
    assert abs(Xi - 1.0571) < 0.001
    assert abs(freq - 0.020) < 0.001
    assert max_depth <= 2
    ```

- [ ] **Reproduce brain.py results**
  - **Run**: Cognitive emergence pattern
  - **Measure**: Similar metrics
  - **Compare**: To legacy
  - **Acceptance**: Qualitative match
  - **Validation**: Manual inspection

- [ ] **Reproduce vcpu.py results**
  - **Run**: Logic formation pattern
  - **Measure**: Structure emergence
  - **Compare**: To legacy
  - **Acceptance**: Qualitative match
  - **Validation**: Manual inspection

- [ ] **Create validation script**
  - **File**: `tests/test_legacy_validation.py`
  - **Content**: Run all legacy validations
  - **Acceptance**: Script completes
  - **Validation**: `pytest tests/test_legacy_validation.py`

- [ ] **Document validation results**
  - **File**: `docs/VALIDATION_RESULTS.md`
  - **Content**: 
    - Measured vs target values
    - Plots and visualizations
    - Analysis and discussion
  - **Acceptance**: Document created
  - **Validation**: Review completeness

---

## Week 4: Polish and Documentation

### Day 1-2: Performance Optimization

- [ ] **Profile critical paths**
  - Use PyTorch profiler
  - Identify bottlenecks
  - **Acceptance**: Profile data collected
  - **Validation**: Review hotspots

- [ ] **Optimize GPU operations**
  - Kernel fusion where possible
  - Minimize CPU-GPU transfers
  - Batch operations
  - **Acceptance**: Measurable speedup
  - **Validation**: Benchmark before/after

- [ ] **Memory optimization**
  - Reduce intermediate allocations
  - Reuse tensors
  - In-place operations
  - **Acceptance**: Lower memory usage
  - **Validation**: Monitor GPU memory

- [ ] **Create benchmark suite**
  - **File**: `tests/benchmark_reality_engine.py`
  - **Metrics**: Steps/second, memory usage
  - **Acceptance**: Benchmarks run
  - **Validation**: Compare CPU vs GPU

### Day 3-5: Documentation

- [ ] **Complete API reference**
  - Docstrings for all public methods
  - Type hints
  - Example usage
  - **Acceptance**: All classes documented
  - **Validation**: Generate API docs

- [ ] **Write theory guide**
  - **File**: `docs/MOBIUS_THEORY.md`
  - **Content**:
    - Möbius topology
    - Thermodynamic-information duality
    - Time emergence
    - SEC dynamics
  - **Acceptance**: Guide complete
  - **Validation**: Peer review

- [ ] **Create tutorial notebooks**
  - **File**: `examples/notebooks/reality_simulation_tutorial.ipynb`
  - **Content**:
    - Basic usage
    - Visualization
    - Law discovery
    - Advanced features
  - **Acceptance**: Notebooks run without errors
  - **Validation**: Test in clean environment

- [ ] **Update main README**
  - Add Möbius module section
  - Link to integration docs
  - Update examples
  - **Acceptance**: README updated
  - **Validation**: Links work

### Day 6-7: Final Integration

- [ ] **Run full test suite**
  ```bash
  pytest tests/ -v
  ```
  - **Acceptance**: All tests pass
  - **Validation**: No failures or warnings

- [ ] **Test installation**
  ```bash
  pip install -e .
  python -c "from fracton.mobius import RealityEngine"
  ```
  - **Acceptance**: Installs and imports cleanly
  - **Validation**: In fresh virtualenv

- [ ] **Test examples**
  - Run all example scripts
  - Run all notebooks
  - **Acceptance**: All complete without errors
  - **Validation**: Output looks correct

- [ ] **Update CHANGELOG**
  - **File**: `CHANGELOG.md`
  - **Content**: Document Möbius module addition
  - **Acceptance**: Entry added
  - **Validation**: Follows format

- [ ] **Tag release**
  ```bash
  git tag -a v0.2.0-mobius -m "Möbius module integration"
  ```
  - **Acceptance**: Tag created
  - **Validation**: Tag pushed to remote

---

## Success Criteria

### Technical Requirements

- [x] **PAC Error**: <1e-12 for synthetic tests, <1e-10 for evolution
- [ ] **Anti-periodic Error**: <0.1 after confluence
- [ ] **2nd Law**: Entropy never decreases (within numerical tolerance)
- [ ] **Ξ Emergence**: 1.0571 ± 0.001 from long runs
- [ ] **Frequency**: 0.020 ± 0.001 Hz from analysis
- [ ] **Structure Depth**: ≤ 2 from legacy validation
- [ ] **GPU Speedup**: >10x vs CPU for size (256, 64)

### Functional Requirements

- [ ] **Initialization**: All three modes work
- [ ] **Evolution**: Stable for 100k+ steps
- [ ] **Law Discovery**: Identifies basic patterns
- [ ] **Time Dilation**: Observable in dense regions
- [ ] **Temperature**: Cooling matches theory
- [ ] **Integration**: Works with all Fracton components

### Quality Requirements

- [ ] **Test Coverage**: >80% for mobius module
- [ ] **Documentation**: All public APIs documented
- [ ] **Examples**: At least 3 working examples
- [ ] **Validation**: Legacy experiments reproduced

---

## Dependencies

### External
- PyTorch >= 1.9.0
- NumPy
- Matplotlib (for visualization)

### Internal (Fracton)
- `fracton.core.recursive_engine`
- `fracton.core.gpu_accelerated_memory_field`
- `fracton.core.pac_regulation`
- `fracton.field.rbf_engine`
- `fracton.field.qbe_regulator`

### External (PACEngine)
- `PACEngine/modules/geometric_sec.py` (for SEC operator)
- `PACEngine/core/pac_kernel.py` (reference for PAC)

---

## Notes

### Development Environment

```bash
# Set up development environment
cd fracton
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
pip install torch torchvision  # GPU version
```

### Testing

```bash
# Run specific test file
pytest tests/test_mobius_substrate.py -v

# Run all mobius tests
pytest tests/ -k mobius -v

# Run with coverage
pytest tests/ --cov=fracton.mobius --cov-report=html
```

### Profiling

```python
# Profile evolution
import torch
from fracton.mobius import RealityEngine

reality = RealityEngine(size=(128, 32))
reality.initialize('big_bang')

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for i, state in enumerate(reality.evolve(steps=100)):
        if i >= 100:
            break

print(prof.key_averages().table())
```

---

## Progress Tracking

Use this section to track overall progress:

```
Week 1: [░░░░░░░░░░] 0% - Not started
Week 2: [░░░░░░░░░░] 0% - Not started
Week 3: [░░░░░░░░░░] 0% - Not started
Week 4: [░░░░░░░░░░] 0% - Not started
```

Update as tasks complete!

---

**Last Updated**: November 4, 2025  
**Status**: 📋 Ready for Implementation  
**Next Action**: Begin Week 1, Day 1 - Module Setup
