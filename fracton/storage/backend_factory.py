"""
Backend Factory for KRONOS

Creates and configures backends based on BackendConfig.
Handles automatic detection and graceful fallbacks.
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional
from pathlib import Path

from .backends import (
    BackendConfig,
    GraphBackend,
    VectorBackend,
    SQLiteGraphBackend,
    ChromaVectorBackend,
    Neo4jGraphBackend,
    QdrantVectorBackend,
)

logger = logging.getLogger(__name__)


class BackendFactory:
    """
    Factory for creating KRONOS backends.

    Handles:
    - Backend creation from config
    - Automatic fallback if dependencies missing
    - Connection management
    - Error handling
    """

    @staticmethod
    async def create_backends(
        config: BackendConfig,
        storage_path: Path,
        namespace: str = "default",
    ) -> Tuple[GraphBackend, VectorBackend]:
        """
        Create graph and vector backends from config.

        Args:
            config: Backend configuration
            storage_path: Base storage path
            namespace: Namespace for this instance

        Returns:
            Tuple of (graph_backend, vector_backend)

        Raises:
            ImportError: If required backend not available
            ConnectionError: If backend connection fails
        """
        logger.info(f"Creating backends: graph={config.graph_type}, vector={config.vector_type}")

        # Create backends
        graph = BackendFactory._create_graph_backend(config, storage_path, namespace)
        vector = BackendFactory._create_vector_backend(config, storage_path, namespace)

        # Connect
        try:
            await graph.connect()
            logger.info(f"Graph backend connected: {config.graph_type}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect graph backend: {e}")

        try:
            await vector.connect()
            logger.info(f"Vector backend connected: {config.vector_type}")
        except Exception as e:
            await graph.close()
            raise ConnectionError(f"Failed to connect vector backend: {e}")

        return graph, vector

    @staticmethod
    def _create_graph_backend(
        config: BackendConfig,
        storage_path: Path,
        namespace: str,
    ) -> GraphBackend:
        """Create graph backend."""
        if config.graph_type == "sqlite":
            db_path = storage_path / namespace / f"graph_{namespace}.db"
            return SQLiteGraphBackend(db_path)

        elif config.graph_type == "neo4j":
            if Neo4jGraphBackend is None:
                raise ImportError(
                    "Neo4j backend not available. Install with: pip install neo4j"
                )
            return Neo4jGraphBackend(
                uri=config.graph_uri or "bolt://localhost:7687",
                user=config.graph_user or "neo4j",
                password=config.graph_password or "password",
            )

        else:
            raise ValueError(f"Unknown graph backend type: {config.graph_type}")

    @staticmethod
    def _create_vector_backend(
        config: BackendConfig,
        storage_path: Path,
        namespace: str,
    ) -> VectorBackend:
        """Create vector backend."""
        if config.vector_type == "chromadb":
            persist_dir = storage_path / namespace / "vectors"
            return ChromaVectorBackend(
                persist_directory=persist_dir,
                collection_name=config.collection_name,
                device=config.device,
            )

        elif config.vector_type == "qdrant":
            if QdrantVectorBackend is None:
                raise ImportError(
                    "Qdrant backend not available. Install with: pip install qdrant-client"
                )
            return QdrantVectorBackend(
                host=config.vector_host or "localhost",
                port=config.vector_port or 6333,
                collection_name=config.collection_name,
                vector_size=config.embedding_dim,
                device=config.device,
            )

        else:
            raise ValueError(f"Unknown vector backend type: {config.vector_type}")

    @staticmethod
    async def create_with_fallback(
        preferred_config: BackendConfig,
        storage_path: Path,
        namespace: str = "default",
    ) -> Tuple[GraphBackend, VectorBackend, BackendConfig]:
        """
        Create backends with automatic fallback to lightweight if production fails.

        Args:
            preferred_config: Preferred backend configuration
            storage_path: Base storage path
            namespace: Namespace

        Returns:
            Tuple of (graph_backend, vector_backend, actual_config)
        """
        try:
            # Try preferred backends
            graph, vector = await BackendFactory.create_backends(
                preferred_config, storage_path, namespace
            )
            return graph, vector, preferred_config

        except (ImportError, ConnectionError) as e:
            logger.warning(
                f"Failed to create preferred backends ({preferred_config.graph_type}, "
                f"{preferred_config.vector_type}): {e}. Falling back to lightweight."
            )

            # Fall back to lightweight
            fallback_config = BackendConfig(
                graph_type="sqlite",
                vector_type="chromadb",
                embedding_dim=preferred_config.embedding_dim,
                device=preferred_config.device,
                collection_name=preferred_config.collection_name,
            )

            try:
                graph, vector = await BackendFactory.create_backends(
                    fallback_config, storage_path, namespace
                )
                logger.info("Successfully fell back to lightweight backends")
                return graph, vector, fallback_config

            except Exception as fallback_error:
                raise RuntimeError(
                    f"Failed to create even lightweight backends: {fallback_error}"
                )

    @staticmethod
    def get_recommended_config(
        use_gpu: bool = False,
        production: bool = False,
    ) -> BackendConfig:
        """
        Get recommended backend configuration.

        Args:
            use_gpu: Whether to use GPU acceleration
            production: Whether this is for production use

        Returns:
            Recommended BackendConfig
        """
        if production:
            # Production: Neo4j + Qdrant
            return BackendConfig(
                graph_type="neo4j",
                graph_uri="bolt://localhost:7687",
                graph_user="neo4j",
                graph_password="password",
                vector_type="qdrant",
                vector_host="localhost",
                vector_port=6333,
                device="cuda" if use_gpu else "cpu",
                embedding_dim=384,
            )
        else:
            # Development: SQLite + ChromaDB
            return BackendConfig(
                graph_type="sqlite",
                vector_type="chromadb",
                device="cuda" if use_gpu else "cpu",
                embedding_dim=384,
            )

    @staticmethod
    async def health_check(
        graph: GraphBackend,
        vector: VectorBackend,
    ) -> Dict[str, bool]:
        """
        Check health of both backends.

        Returns:
            Dict with health status
        """
        return {
            "graph": await graph.health_check(),
            "vector": await vector.health_check(),
        }


# Convenience function
async def create_backends_from_env(
    storage_path: Path,
    namespace: str = "default",
) -> Tuple[GraphBackend, VectorBackend, BackendConfig]:
    """
    Create backends from environment variables.

    Environment variables:
        KRONOS_BACKEND: "lightweight" or "production" (default: lightweight)
        KRONOS_GPU: "true" or "false" (default: false)
        NEO4J_URI: Neo4j URI (default: bolt://localhost:7687)
        NEO4J_USER: Neo4j user (default: neo4j)
        NEO4J_PASSWORD: Neo4j password (default: password)
        QDRANT_HOST: Qdrant host (default: localhost)
        QDRANT_PORT: Qdrant port (default: 6333)

    Returns:
        Tuple of (graph_backend, vector_backend, config)
    """
    import os

    backend_type = os.getenv("KRONOS_BACKEND", "lightweight")
    use_gpu = os.getenv("KRONOS_GPU", "false").lower() == "true"

    if backend_type == "production":
        config = BackendConfig(
            graph_type="neo4j",
            graph_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            graph_user=os.getenv("NEO4J_USER", "neo4j"),
            graph_password=os.getenv("NEO4J_PASSWORD", "password"),
            vector_type="qdrant",
            vector_host=os.getenv("QDRANT_HOST", "localhost"),
            vector_port=int(os.getenv("QDRANT_PORT", "6333")),
            device="cuda" if use_gpu else "cpu",
        )
    else:
        config = BackendConfig(
            graph_type="sqlite",
            vector_type="chromadb",
            device="cuda" if use_gpu else "cpu",
        )

    return await BackendFactory.create_with_fallback(config, storage_path, namespace)
