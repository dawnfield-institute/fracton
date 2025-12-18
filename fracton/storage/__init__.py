"""
Fracton Storage Module
======================

Kronos-based persistent storage for PAC-Lazy substrate.

Provides:
- KronosBackend: Main storage interface
- FDOSerializer: PAC delta ↔ FDO format conversion
- TemporalIndex: Time-based pattern retrieval
- EpisodeTracker: Field evolution sequences

Example:
    from fracton.storage import KronosBackend
    from fracton.core import PACSystem
    from pathlib import Path
    
    # Create persistent substrate
    backend = KronosBackend(Path("./kronos_data"), namespace="gaia")
    substrate = PACSystem(device='cuda', kronos_backend=backend)
    
    # Patterns auto-persist to disk
    node_id = substrate.inject(pattern)
    
    # Save full state as episode
    episode_id = substrate.save_state()
    
    # Restore later
    substrate.restore_state(episode_id)
"""

from .kronos_backend import KronosBackend
from .fdo_serializer import FDOSerializer
from .temporal_index import TemporalIndex
from .episode_tracker import EpisodeTracker

__all__ = [
    'KronosBackend',
    'FDOSerializer', 
    'TemporalIndex',
    'EpisodeTracker'
]
