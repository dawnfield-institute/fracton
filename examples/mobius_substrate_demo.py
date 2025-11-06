"""
Möbius Substrate Demo

Demonstrates the Möbius manifold substrate with:
- Different initialization modes
- Anti-periodicity validation
- Topology metrics
- Field visualization (text-based)
"""

import torch
import numpy as np
from fracton.mobius.substrate import MobiusManifold
from fracton.mobius.constants import (
    XI, LAMBDA, DEPTH_MAX,
    ANTIPERIODIC_TOLERANCE,
    validate_constant,
    print_validation_report
)


def print_field_summary(name: str, field: torch.Tensor):
    """Print summary statistics of a field."""
    print(f"\n{name} Field:")
    print(f"  Shape: {field.shape}")
    print(f"  Mean: {field.mean():.6f}")
    print(f"  Std:  {field.std():.6f}")
    print(f"  Min:  {field.min():.6f}")
    print(f"  Max:  {field.max():.6f}")


def visualize_field_ascii(field: torch.Tensor, width: int = 80, height: int = 20):
    """Create ASCII visualization of field."""
    # Resize to visualization size
    field_np = field.cpu().numpy()
    
    # Normalize to 0-1
    f_min, f_max = field_np.min(), field_np.max()
    if f_max > f_min:
        normalized = (field_np - f_min) / (f_max - f_min)
    else:
        normalized = np.zeros_like(field_np)
    
    # Resize
    from scipy.ndimage import zoom
    scale = (height / field_np.shape[0], width / field_np.shape[1])
    resized = zoom(normalized, scale, order=1)
    
    # Convert to ASCII
    chars = ' .:-=+*#%@'
    ascii_art = []
    for row in resized:
        line = ''
        for val in row:
            idx = int(val * (len(chars) - 1))
            line += chars[idx]
        ascii_art.append(line)
    
    return '\n'.join(ascii_art)


def main():
    """Run Möbius substrate demonstration."""
    
    print("="*80)
    print("MÖBIUS SUBSTRATE DEMONSTRATION")
    print("="*80)
    
    print("\n📐 Creating Möbius manifold (128 × 32 grid)...")
    substrate = MobiusManifold(size=(128, 32), device='cpu')
    print(substrate)
    
    # Show universal constants
    print("\n🌌 Universal Constants (targets for emergence):")
    print(f"  Ξ (Xi):         {XI:.4f} ± {0.001}")
    print(f"  λ (Lambda):     {LAMBDA:.3f} Hz ± {0.001}")
    print(f"  Max Depth:      {DEPTH_MAX}")
    
    # Test Big Bang initialization
    print("\n" + "="*80)
    print("💥 BIG BANG INITIALIZATION")
    print("="*80)
    print("(Maximum entropy, zero structure - universe at t=0)")
    
    P, A, M = substrate.initialize_fields('big_bang')
    
    print_field_summary("Potential (P)", P)
    print_field_summary("Actual (A)", A)
    print_field_summary("Memory (M)", M)
    
    # Validate anti-periodicity
    error_P = substrate.validate_antiperiodicity(P)
    error_A = substrate.validate_antiperiodicity(A)
    
    print(f"\n✓ Anti-periodic validation:")
    print(f"  P error: {error_P:.6f} (tolerance: {ANTIPERIODIC_TOLERANCE})")
    print(f"  A error: {error_A:.6f} (tolerance: {ANTIPERIODIC_TOLERANCE})")
    print(f"  A satisfied: {'✅ YES' if error_A < ANTIPERIODIC_TOLERANCE else '⚠️  NO (will be enforced during evolution)'}")
    
    # Get topology metrics
    print(f"\n📊 Topology Metrics:")
    metrics = substrate.get_topology_metrics()
    print(f"  Euler characteristic: {metrics['euler_characteristic']}")
    print(f"  Orientable: {metrics['orientable']}")
    print(f"  Field energy: {metrics['field_energy']:.6f}")
    print(f"  Field entropy: {metrics['field_entropy']:.6f}")
    
    # Test Random initialization
    print("\n" + "="*80)
    print("🎲 RANDOM INITIALIZATION")
    print("="*80)
    
    P, A, M = substrate.initialize_fields('random')
    
    print_field_summary("Potential (P)", P)
    print_field_summary("Actual (A)", A)
    
    error_P = substrate.validate_antiperiodicity(P)
    error_A = substrate.validate_antiperiodicity(A)
    
    print(f"\n✓ Anti-periodic validation:")
    print(f"  P error: {error_P:.6f} ({'✅ OK' if error_P < ANTIPERIODIC_TOLERANCE else '⚠️  HIGH'})")
    print(f"  A error: {error_A:.6f} ({'✅ OK' if error_A < ANTIPERIODIC_TOLERANCE else '⚠️  HIGH'})")
    
    # Test Structured initialization
    print("\n" + "="*80)
    print("🏗️  STRUCTURED INITIALIZATION")
    print("="*80)
    print("(Localized structures for testing)")
    
    P, A, M = substrate.initialize_fields('structured')
    
    print_field_summary("Potential (P)", P)
    
    error_P = substrate.validate_antiperiodicity(P)
    print(f"\n✓ Anti-periodic validation:")
    print(f"  P error: {error_P:.6f} ({'✅ OK' if error_P < ANTIPERIODIC_TOLERANCE else '⚠️  HIGH'})")
    
    # Try to visualize (requires scipy)
    try:
        print("\n📈 Field Visualization (Potential field):")
        print("─" * 80)
        ascii_viz = visualize_field_ascii(P, width=80, height=16)
        print(ascii_viz)
        print("─" * 80)
    except ImportError:
        print("\n(Install scipy for field visualization)")
    
    # Summary
    print("\n" + "="*80)
    print("✅ MÖBIUS SUBSTRATE READY")
    print("="*80)
    print("\nThe geometric foundation is in place!")
    print("Next steps:")
    print("  1. Add thermodynamic fields (temperature, Landauer costs)")
    print("  2. Integrate SEC operator from PACEngine")
    print("  3. Implement Möbius confluence (time stepping)")
    print("  4. Add time emergence from disequilibrium")
    print("  5. Build unified Reality Engine interface")
    print("\n🌌 Physics will emerge from geometry + conservation + thermodynamics!")


if __name__ == '__main__':
    main()
