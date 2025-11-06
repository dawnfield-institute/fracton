"""
Demonstration of Möbius thermodynamics module.
Shows temperature initialization, entropy tracking, Landauer costs, heat diffusion.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from fracton.mobius.substrate import MobiusManifold
from fracton.mobius.thermodynamics import ThermodynamicField
from fracton.mobius.constants import K_BOLTZMANN


def demo_temperature_initialization():
    """Demo different temperature initialization modes."""
    print("=" * 60)
    print("TEMPERATURE INITIALIZATION MODES")
    print("=" * 60)
    
    substrate = MobiusManifold(size=(64, 16), device='cpu')
    P, A, M = substrate.initialize_fields('big_bang')
    
    thermo = ThermodynamicField(size=(64, 16), device='cpu')
    
    # Uniform temperature
    print("\n1. UNIFORM TEMPERATURE (T=300K)")
    T_uniform = thermo.initialize_temperature(A, mode='uniform', T0=300.0)
    print(f"   T_mean: {T_uniform.mean():.2f}")
    print(f"   T_std:  {T_uniform.std():.6f}")
    
    # From field
    print("\n2. FROM FIELD (T ∝ |A|)")
    T_field = thermo.initialize_temperature(A, mode='from_field')
    print(f"   T_mean: {T_field.mean():.2f}")
    print(f"   T_std:  {T_field.std():.2f}")
    print(f"   T_min:  {T_field.min():.2f}")
    print(f"   T_max:  {T_field.max():.2f}")
    
    # Hot Big Bang
    print("\n3. HOT BIG BANG (Planck temperature)")
    T_hot = thermo.initialize_temperature(A, mode='hot_big_bang')
    print(f"   T_mean: {T_hot.mean():.2f}")
    print(f"   Entropy: High initial entropy state")
    print(f"   Purpose: Primordial high-temperature epoch")


def demo_entropy_calculations():
    """Demo entropy computation."""
    print("\n" + "=" * 60)
    print("ENTROPY CALCULATIONS")
    print("=" * 60)
    
    thermo = ThermodynamicField(size=(64, 16), device='cpu')
    
    # Low entropy: uniform field
    uniform = torch.ones(64, 16) * 5.0
    S_uniform = thermo.compute_entropy(uniform)
    print(f"\n1. UNIFORM FIELD")
    print(f"   Field: constant = 5.0")
    print(f"   Entropy: {S_uniform:.6f} (LOW)")
    
    # Medium entropy: structured field
    x = torch.linspace(0, 2*np.pi, 64).unsqueeze(1).expand(64, 16)
    y = torch.linspace(0, 2*np.pi, 16).unsqueeze(0).expand(64, 16)
    structured = torch.sin(x) * torch.cos(y)
    S_structured = thermo.compute_entropy(structured)
    print(f"\n2. STRUCTURED FIELD")
    print(f"   Field: sin(x) * cos(y)")
    print(f"   Entropy: {S_structured:.6f} (MEDIUM)")
    
    # High entropy: random field
    random = torch.randn(64, 16)
    S_random = thermo.compute_entropy(random)
    print(f"\n3. RANDOM FIELD")
    print(f"   Field: Gaussian noise")
    print(f"   Entropy: {S_random:.6f} (HIGH)")
    
    print(f"\n   Entropy ordering: uniform < structured < random")
    print(f"   {S_uniform:.4f} < {S_structured:.4f} < {S_random:.4f}")


def demo_free_energy():
    """Demo free energy calculations."""
    print("\n" + "=" * 60)
    print("FREE ENERGY: F = E - TS")
    print("=" * 60)
    
    substrate = MobiusManifold(size=(64, 16), device='cpu')
    P, A, M = substrate.initialize_fields('big_bang')
    
    thermo = ThermodynamicField(size=(64, 16), device='cpu')
    
    E = 10000.0  # Internal energy
    S = 50.0     # Entropy
    
    # Cold system
    print("\n1. COLD SYSTEM (T=100K)")
    thermo.initialize_temperature(A, mode='uniform', T0=100.0)
    F_cold = thermo.compute_free_energy(E, S)
    print(f"   E = {E:.2f}")
    print(f"   S = {S:.2f}")
    print(f"   T = 100.0")
    print(f"   F = E - TS = {F_cold:.2f}")
    
    # Hot system
    print("\n2. HOT SYSTEM (T=300K)")
    thermo.initialize_temperature(A, mode='uniform', T0=300.0)
    F_hot = thermo.compute_free_energy(E, S)
    print(f"   E = {E:.2f}")
    print(f"   S = {S:.2f}")
    print(f"   T = 300.0")
    print(f"   F = E - TS = {F_hot:.2f}")
    
    print(f"\n   Heating decreases free energy: {F_cold:.2f} → {F_hot:.2f}")
    print(f"   ΔF = {F_hot - F_cold:.2f}")


def demo_landauer_principle():
    """Demo Landauer erasure costs."""
    print("\n" + "=" * 60)
    print("LANDAUER PRINCIPLE: E = kT ln(2) per bit")
    print("=" * 60)
    
    substrate = MobiusManifold(size=(64, 16), device='cpu')
    P, A, M = substrate.initialize_fields('big_bang')
    
    thermo = ThermodynamicField(size=(64, 16), device='cpu')
    
    bits_erased = 1000
    
    # Room temperature
    print(f"\n1. ROOM TEMPERATURE (T=300K)")
    thermo.initialize_temperature(A, mode='uniform', T0=300.0)
    cost_room = thermo.landauer_erasure_cost(bits_erased)
    cost_per_bit = K_BOLTZMANN * 300.0 * np.log(2)
    print(f"   Erasing {bits_erased} bits")
    print(f"   Cost per bit: {cost_per_bit:.6f}")
    print(f"   Total cost:   {cost_room:.2f}")
    
    # Hot system
    print(f"\n2. HOT SYSTEM (T=1000K)")
    thermo.initialize_temperature(A, mode='uniform', T0=1000.0)
    cost_hot = thermo.landauer_erasure_cost(bits_erased)
    print(f"   Erasing {bits_erased} bits")
    print(f"   Total cost:   {cost_hot:.2f}")
    
    print(f"\n   Ratio: {cost_hot / cost_room:.2f} (scales linearly with T)")
    print(f"   Information erasure is thermodynamically expensive!")


def demo_heat_diffusion():
    """Demo heat diffusion dynamics."""
    print("\n" + "=" * 60)
    print("HEAT DIFFUSION: ∂T/∂t = α∇²T")
    print("=" * 60)
    
    substrate = MobiusManifold(size=(64, 16), device='cpu')
    P, A, M = substrate.initialize_fields('big_bang')
    
    thermo = ThermodynamicField(size=(64, 16), device='cpu')
    
    # Start with non-uniform temperature
    T_initial = thermo.initialize_temperature(A, mode='from_field')
    
    print(f"\nINITIAL STATE:")
    print(f"   T_mean: {T_initial.mean():.4f}")
    print(f"   T_std:  {T_initial.std():.4f}")
    print(f"   T_min:  {T_initial.min():.4f}")
    print(f"   T_max:  {T_initial.max():.4f}")
    
    # Apply diffusion
    steps = [0, 10, 50, 100]
    T_history = [T_initial.clone()]
    
    for i in range(100):
        thermo.apply_heat_diffusion(dt=0.01, alpha=0.5)
        if i+1 in steps[1:]:
            T_history.append(thermo.T.clone())
    
    print(f"\nDIFFUSION EVOLUTION:")
    for step, T in zip(steps, T_history):
        print(f"   Step {step:3d}: T_std = {T.std():.6f}")
    
    print(f"\n   Heat diffuses from hot to cold regions")
    print(f"   Temperature field smooths over time")
    print(f"   Mean temperature conserved: {thermo.T.mean():.4f}")


def demo_thermal_fluctuations():
    """Demo thermal noise injection."""
    print("\n" + "=" * 60)
    print("THERMAL FLUCTUATIONS: σ = √(2kTdt)")
    print("=" * 60)
    
    substrate = MobiusManifold(size=(64, 16), device='cpu')
    P, A, M = substrate.initialize_fields('big_bang')
    
    thermo = ThermodynamicField(size=(64, 16), device='cpu')
    
    # Start with structured field
    A_clean = torch.sin(torch.linspace(0, 2*np.pi, 64).unsqueeze(1).expand(64, 16))
    
    print(f"\nCLEAN FIELD:")
    print(f"   A_mean: {A_clean.mean():.6f}")
    print(f"   A_std:  {A_clean.std():.6f}")
    
    # Add thermal noise at different temperatures
    temps = [100.0, 300.0, 1000.0]
    
    print(f"\nAFTER THERMAL FLUCTUATIONS:")
    for T in temps:
        thermo.initialize_temperature(A, mode='uniform', T0=T)
        A_noisy = thermo.inject_thermal_fluctuations(A_clean.clone(), dt=0.1)
        noise_amplitude = (A_noisy - A_clean).std()
        print(f"   T={T:6.1f}K: noise_std = {noise_amplitude:.6f}")
    
    print(f"\n   Noise amplitude scales as √T")
    print(f"   Thermal fluctuations disrupt structure")


def demo_entropy_production():
    """Demo entropy tracking and 2nd law monitoring."""
    print("\n" + "=" * 60)
    print("ENTROPY PRODUCTION & 2ND LAW")
    print("=" * 60)
    
    substrate = MobiusManifold(size=(64, 16), device='cpu')
    P, A, M = substrate.initialize_fields('big_bang')
    
    thermo = ThermodynamicField(size=(64, 16), device='cpu')
    thermo.initialize_temperature(A, mode='hot_big_bang')
    
    # Simulate evolution
    print(f"\nEVOLUTION TRACKING:")
    
    A_current = A.clone()
    for step in range(10):
        # Compute entropy before
        S_before = thermo.compute_entropy(A_current)
        
        # Apply dynamics (simplified: diffusion + noise)
        A_current = A_current + 0.1 * torch.randn_like(A_current)
        
        # Compute entropy after
        S_after = thermo.compute_entropy(A_current)
        
        # Track
        thermo.track_entropy_production(S_before, S_after)
        
        if step % 2 == 0:
            state = thermo.get_thermodynamic_state()
            print(f"   Step {step:2d}: ΔS = {S_after - S_before:+.6f}, "
                  f"Total ΔS = {state['entropy_change']:.6f}, "
                  f"Violations = {state['violations']}")
    
    print(f"\n   2nd Law: ΔS_total ≥ 0 (for isolated systems)")
    print(f"   Violations indicate numerical errors or external work")


def demo_complete_evolution():
    """Demo complete thermodynamic evolution."""
    print("\n" + "=" * 60)
    print("COMPLETE THERMODYNAMIC EVOLUTION")
    print("=" * 60)
    
    # Initialize
    print("\nINITIALIZATION:")
    substrate = MobiusManifold(size=(128, 32), device='cpu')
    P, A, M = substrate.initialize_fields('big_bang')
    print(f"   Substrate: Möbius manifold (128×32)")
    
    thermo = ThermodynamicField(size=(128, 32), device='cpu')
    T = thermo.initialize_temperature(A, mode='hot_big_bang')
    print(f"   Temperature: Hot Big Bang (T={T.mean():.2f})")
    
    S0 = thermo.compute_entropy(A)
    print(f"   Initial entropy: {S0:.6f}")
    
    # Evolution
    print(f"\nEVOLUTION:")
    for step in range(20):
        # Heat diffusion (cooling)
        thermo.apply_heat_diffusion(dt=0.05, alpha=0.3)
        
        # Thermal fluctuations
        A = thermo.inject_thermal_fluctuations(A, dt=0.05)
        
        # Track entropy
        S = thermo.compute_entropy(A)
        thermo.track_entropy_production(S0, S)
        S0 = S
        
        # Information erasure (simulated computation)
        if step % 5 == 0:
            bits_erased = 100
            cost = thermo.landauer_erasure_cost(bits_erased)
            thermo.heat_generated += cost
        
        if step % 5 == 0:
            state = thermo.get_thermodynamic_state()
            print(f"   Step {step:2d}: T={state['T_mean']:6.2f}, "
                  f"S={S:.6f}, ΔS={state['entropy_change']:+.6f}, "
                  f"Heat={state['heat_generated']:.1f}")
    
    # Final state
    print(f"\nFINAL STATE:")
    state = thermo.get_thermodynamic_state()
    print(f"   Temperature: {state['T_mean']:.2f} (cooled)")
    print(f"   Total entropy change: {state['entropy_change']:.6f}")
    print(f"   Total heat generated: {state['heat_generated']:.2f}")
    print(f"   2nd law violations: {state['violations']}")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "MÖBIUS THERMODYNAMICS DEMONSTRATION" + " " * 13 + "║")
    print("╚" + "=" * 58 + "╝")
    
    demo_temperature_initialization()
    demo_entropy_calculations()
    demo_free_energy()
    demo_landauer_principle()
    demo_heat_diffusion()
    demo_thermal_fluctuations()
    demo_entropy_production()
    demo_complete_evolution()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Concepts Demonstrated:")
    print("  1. Temperature field initialization (3 modes)")
    print("  2. Entropy calculation (Shannon entropy proxy)")
    print("  3. Free energy F = E - TS")
    print("  4. Landauer principle: kT ln(2) per bit erased")
    print("  5. Heat diffusion: ∂T/∂t = α∇²T")
    print("  6. Thermal fluctuations: σ = √(2kTdt)")
    print("  7. Entropy production tracking")
    print("  8. 2nd law violation monitoring")
    print("\nNext Steps:")
    print("  • Integrate SEC operator (energy functional minimization)")
    print("  • Implement Möbius confluence (time stepping)")
    print("  • Connect to time emergence (disequilibrium → time)")
    print("  • Run full Reality Engine simulation\n")


if __name__ == '__main__':
    main()
