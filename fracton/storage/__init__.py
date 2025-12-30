"""
Fracton Storage Module
======================

Kronos-based persistent storage for PAC-Lazy substrate.

Provides:
- KronosMemory: Unified semantic memory with PAC+SEC+PAS
- KronosBackend: Low-level storage interface
- FDOSerializer: PAC delta ↔ FDO format conversion
- TemporalIndex: Time-based pattern retrieval
- EpisodeTracker: Field evolution sequences

Example (Semantic Memory):
    from fracton.storage import KronosMemory, NodeType, RelationType
    from pathlib import Path

    # Create unified memory system
    memory = KronosMemory(
        storage_path=Path("./kronos_data"),
        namespace="my_project",
        device="cpu"
    )

    # Create graphs
    await memory.create_graph("research", "Research papers and ideas")
    await memory.create_graph("code", "Code and commits")

    # Store memories
    paper_id = await memory.store(
        content="Attention Is All You Need - transformer architecture",
        graph="research",
        node_type=NodeType.PAPER,
        importance=1.0
    )

    commit_id = await memory.store(
        content="Implemented attention mechanism",
        graph="code",
        node_type=NodeType.COMMIT,
        parent_id=None
    )

    # Link across graphs
    await memory.link_across_graphs(
        from_graph="code",
        from_id=commit_id,
        to_graph="research",
        to_id=paper_id,
        relation=RelationType.IMPLEMENTS
    )

    # Query with SEC resonance ranking
    results = await memory.query(
        query_text="attention mechanism implementations",
        graphs=["research", "code"],
        expand_graph=True
    )

    # Trace evolution
    trace = await memory.trace_evolution("code", commit_id, direction="both")

Example (Low-level PAC Storage):
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
from .kronos_memory import (
    KronosMemory,
    PACMemoryNode,
    NodeType,
    RelationType,
    ResonanceResult
)
from .backends import (
    GraphBackend,
    VectorBackend,
    BackendConfig,
)
from .backend_factory import BackendFactory, create_backends_from_env
from .embeddings import EmbeddingService, create_embedding_service, EMBEDDING_MODELS

__all__ = [
    # Unified Semantic Memory (PAC+SEC+PAS)
    'KronosMemory',
    'PACMemoryNode',
    'NodeType',
    'RelationType',
    'ResonanceResult',

    # Low-level Storage
    'KronosBackend',
    'FDOSerializer',
    'TemporalIndex',
    'EpisodeTracker',

    # Backend Abstraction
    'GraphBackend',
    'VectorBackend',
    'BackendConfig',
    'BackendFactory',
    'create_backends_from_env',

    # Embeddings
    'EmbeddingService',
    'create_embedding_service',
    'EMBEDDING_MODELS',
]
# Backend imports added in separate append
