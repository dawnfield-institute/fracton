"""
KRONOS Backend Abstraction Layer

Pluggable backends for graph storage and vector search.
"""

from .base import GraphBackend, VectorBackend, BackendConfig
from .sqlite_graph import SQLiteGraphBackend
from .chromadb_vectors import ChromaVectorBackend

# Production backends (optional dependencies)
try:
    from .neo4j_graph import Neo4jGraphBackend
except ImportError:
    Neo4jGraphBackend = None

try:
    from .qdrant_vectors import QdrantVectorBackend
except ImportError:
    QdrantVectorBackend = None

__all__ = [
    # Abstract interfaces
    "GraphBackend",
    "VectorBackend",
    "BackendConfig",

    # Lightweight implementations
    "SQLiteGraphBackend",
    "ChromaVectorBackend",

    # Production implementations
    "Neo4jGraphBackend",
    "QdrantVectorBackend",
]
