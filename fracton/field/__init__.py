"""
Fracton Field Primitives

Universal field computation primitives for infodynamics.
Provides initialization, evolution, and analysis tools that work
across different Dawn Field applications (Reality Engine, GAIA, etc.)
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
    'enforce_qbe'
]
