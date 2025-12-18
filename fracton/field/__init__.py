"""
Fracton Field Primitives

Universal field computation primitives for infodynamics.
Provides initialization, evolution, encoding, resonance, and analysis tools
that work across different Dawn Field applications (Reality Engine, GAIA, etc.)

v2.0: Added encoding, evolution, and resonance modules for PAC-Lazy substrate.
"""

from .initializers import (
    FieldInitializer,
    CMBInitializer,
    GaussianHotspotInitializer,
    BraidedStrandInitializer,
    UniformInitializer,
    initialize_field
)

from .rbf_engine import (
    RBFEngine,
    compute_rbf_balance
)

from .qbe_regulator import (
    QBERegulator,
    enforce_qbe
)

from .encoding import (
    spherical_encode,
    spherical_encode_batch,
    decode_spherical,
    create_reference_fields,
    field_distance,
    interpolate_fields
)

from .evolution import (
    evolve,
    evolve_batch,
    evolve_with_source,
    compute_field_energy,
    dissipate,
    amplify
)

from .resonance import (
    compute_resonance,
    compute_resonance_batch,
    find_resonant,
    compute_resonance_matrix,
    resonance_energy,
    harmonic_resonance,
    phase_coherence,
    resonance_gradient,
    ResonanceMesh
)

__all__ = [
    # Initializers
    'FieldInitializer',
    'CMBInitializer',
    'GaussianHotspotInitializer',
    'BraidedStrandInitializer',
    'UniformInitializer',
    'initialize_field',
    # RBF Engine
    'RBFEngine',
    'compute_rbf_balance',
    # QBE Regulator
    'QBERegulator',
    'enforce_qbe',
    # Encoding (new)
    'spherical_encode',
    'spherical_encode_batch',
    'decode_spherical',
    'create_reference_fields',
    'field_distance',
    'interpolate_fields',
    # Evolution (new)
    'evolve',
    'evolve_batch',
    'evolve_with_source',
    'compute_field_energy',
    'dissipate',
    'amplify',
    # Resonance (new)
    'compute_resonance',
    'compute_resonance_batch',
    'find_resonant',
    'compute_resonance_matrix',
    'resonance_energy',
    'harmonic_resonance',
    'phase_coherence',
    'resonance_gradient',
    'ResonanceMesh'
]
