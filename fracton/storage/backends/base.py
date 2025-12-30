"""
Abstract Backend Interfaces for KRONOS

Defines pluggable backends for:
- Graph storage (relationships, temporal lineage)
- Vector search (embeddings, similarity)

Implementations:
- Lightweight: SQLiteGraph + ChromaDBVectors
- Production: Neo4jGraph + QdrantVectors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

import torch


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BackendConfig:
    """Configuration for backend connections."""

    # Graph backend
    graph_type: str = "sqlite"  # "sqlite" or "neo4j"
    graph_uri: Optional[str] = None
    graph_user: Optional[str] = None
    graph_password: Optional[str] = None

    # Vector backend
    vector_type: str = "chromadb"  # "chromadb" or "qdrant"
    vector_host: Optional[str] = None
    vector_port: Optional[int] = None
    collection_name: str = "kronos_memories"

    # Common
    embedding_dim: int = 384
    device: str = "cpu"


# ============================================================================
# Graph Data Structures
# ============================================================================

@dataclass
class GraphNode:
    """Node in the graph (minimal metadata)."""

    id: str
    content: str
    timestamp: datetime
    fractal_signature: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # PAC structure
    parent_id: str = "-1"
    children_ids: List[str] = field(default_factory=list)

    # SEC dynamics
    potential: float = 1.0
    entropy: float = 0.5
    coherence: float = 0.5
    phase: str = "STABLE"


@dataclass
class GraphEdge:
    """Edge in the graph (relationship)."""

    from_id: str
    to_id: str
    relation_type: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GraphNeighborhood:
    """Subgraph neighborhood result."""

    center_id: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    depth_map: Dict[str, int]  # node_id -> distance from center


@dataclass
class TemporalPath:
    """Temporal lineage path."""

    root_id: str
    path: List[str]  # ordered sequence of node IDs
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    total_potential: float
    entropy_evolution: List[float]


# ============================================================================
# Vector Data Structures
# ============================================================================

@dataclass
class VectorPoint:
    """Point in vector space."""

    id: str
    vector: torch.Tensor  # [D]
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "id": self.id,
            "vector": self.vector.cpu().numpy().tolist(),
            "payload": self.payload,
        }


@dataclass
class VectorSearchResult:
    """Vector similarity search result."""

    id: str
    score: float
    vector: Optional[torch.Tensor] = None
    payload: Optional[Dict[str, Any]] = None


# ============================================================================
# Abstract Graph Backend
# ============================================================================

class GraphBackend(ABC):
    """
    Abstract graph storage backend.

    Stores:
    - Nodes (memories with metadata)
    - Edges (typed relationships)
    - Temporal lineage (parent-child chains)

    Implementations:
    - SQLiteGraph: Lightweight, file-based
    - Neo4jGraph: Production, distributed
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to graph store."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connection to graph store."""
        pass

    # ========================================================================
    # Node Operations
    # ========================================================================

    @abstractmethod
    async def create_node(
        self,
        node: GraphNode,
        graph_name: str = "default",
    ) -> None:
        """
        Create node in graph.

        Args:
            node: Node to create
            graph_name: Which graph to store in
        """
        pass

    @abstractmethod
    async def get_node(
        self,
        node_id: str,
        graph_name: str = "default",
    ) -> Optional[GraphNode]:
        """
        Retrieve node by ID.

        Args:
            node_id: Node identifier
            graph_name: Which graph to query

        Returns:
            Node if found, None otherwise
        """
        pass

    @abstractmethod
    async def update_node(
        self,
        node_id: str,
        updates: Dict[str, Any],
        graph_name: str = "default",
    ) -> None:
        """
        Update node metadata.

        Args:
            node_id: Node to update
            updates: Fields to update
            graph_name: Which graph to update
        """
        pass

    @abstractmethod
    async def delete_node(
        self,
        node_id: str,
        graph_name: str = "default",
    ) -> None:
        """
        Delete node from graph.

        Args:
            node_id: Node to delete
            graph_name: Which graph to delete from
        """
        pass

    # ========================================================================
    # Edge Operations
    # ========================================================================

    @abstractmethod
    async def create_edge(
        self,
        edge: GraphEdge,
        graph_name: str = "default",
    ) -> None:
        """
        Create edge between nodes.

        Args:
            edge: Edge to create
            graph_name: Which graph to create in
        """
        pass

    @abstractmethod
    async def get_edges(
        self,
        node_id: str,
        direction: str = "out",
        relation_type: Optional[str] = None,
        graph_name: str = "default",
    ) -> List[GraphEdge]:
        """
        Get edges connected to node.

        Args:
            node_id: Node to query
            direction: "out", "in", or "both"
            relation_type: Filter by relationship type
            graph_name: Which graph to query

        Returns:
            List of edges
        """
        pass

    @abstractmethod
    async def delete_edge(
        self,
        from_id: str,
        to_id: str,
        relation_type: str,
        graph_name: str = "default",
    ) -> None:
        """
        Delete specific edge.

        Args:
            from_id: Source node
            to_id: Target node
            relation_type: Relationship type
            graph_name: Which graph to delete from
        """
        pass

    # ========================================================================
    # Graph Operations
    # ========================================================================

    @abstractmethod
    async def create_graph(
        self,
        graph_name: str,
        description: str = "",
    ) -> None:
        """
        Create new named graph.

        Args:
            graph_name: Graph identifier
            description: Human-readable description
        """
        pass

    @abstractmethod
    async def list_graphs(self) -> List[Dict[str, Any]]:
        """
        List all graphs.

        Returns:
            List of graph metadata dicts
        """
        pass

    @abstractmethod
    async def get_neighbors(
        self,
        node_id: str,
        max_hops: int = 1,
        relation_types: Optional[List[str]] = None,
        graph_name: str = "default",
    ) -> GraphNeighborhood:
        """
        Get neighborhood around node.

        Args:
            node_id: Center node
            max_hops: Maximum distance to traverse
            relation_types: Filter by relationship types
            graph_name: Which graph to query

        Returns:
            Subgraph neighborhood
        """
        pass

    # ========================================================================
    # Temporal Operations
    # ========================================================================

    @abstractmethod
    async def trace_lineage(
        self,
        node_id: str,
        direction: str = "both",
        max_depth: int = 100,
        graph_name: str = "default",
    ) -> TemporalPath:
        """
        Trace temporal lineage (PAC parent-child chains).

        Args:
            node_id: Starting node
            direction: "forward", "backward", or "both"
            max_depth: Maximum path length
            graph_name: Which graph to trace

        Returns:
            Temporal path with evolution
        """
        pass

    @abstractmethod
    async def find_contradictions(
        self,
        node_id: str,
        graph_name: str = "default",
    ) -> List[str]:
        """
        Find nodes that contradict given node.

        Args:
            node_id: Node to check
            graph_name: Which graph to search

        Returns:
            List of contradicting node IDs
        """
        pass

    # ========================================================================
    # Utility
    # ========================================================================

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if backend is healthy.

        Returns:
            True if operational
        """
        pass


# ============================================================================
# Abstract Vector Backend
# ============================================================================

class VectorBackend(ABC):
    """
    Abstract vector storage backend.

    Stores:
    - Embeddings (dense vectors)
    - Payloads (metadata)

    Provides:
    - Similarity search (cosine, L2)
    - Batch operations
    - GPU acceleration (optional)

    Implementations:
    - ChromaDBVectors: Lightweight, embedded
    - QdrantVectors: Production, GPU-optimized
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to vector store."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connection to vector store."""
        pass

    # ========================================================================
    # Storage Operations
    # ========================================================================

    @abstractmethod
    async def store(
        self,
        point: VectorPoint,
    ) -> None:
        """
        Store vector point.

        Args:
            point: Point to store
        """
        pass

    @abstractmethod
    async def store_batch(
        self,
        points: List[VectorPoint],
    ) -> None:
        """
        Store batch of points.

        Args:
            points: Points to store
        """
        pass

    @abstractmethod
    async def retrieve(
        self,
        point_ids: List[str],
    ) -> List[VectorPoint]:
        """
        Retrieve points by ID.

        Args:
            point_ids: Point identifiers

        Returns:
            List of points (may be partial if some IDs not found)
        """
        pass

    @abstractmethod
    async def delete(
        self,
        point_ids: List[str],
    ) -> None:
        """
        Delete points by ID.

        Args:
            point_ids: Points to delete
        """
        pass

    # ========================================================================
    # Search Operations
    # ========================================================================

    @abstractmethod
    async def search(
        self,
        query_vector: torch.Tensor,
        limit: int = 10,
        threshold: Optional[float] = None,
        filter_payload: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding [D]
            limit: Maximum results
            threshold: Minimum similarity score
            filter_payload: Filter by payload fields

        Returns:
            List of search results, sorted by score descending
        """
        pass

    @abstractmethod
    async def search_batch(
        self,
        query_vectors: torch.Tensor,
        limit: int = 10,
        threshold: Optional[float] = None,
    ) -> List[List[VectorSearchResult]]:
        """
        Batch search for similar vectors.

        Args:
            query_vectors: Query embeddings [B, D]
            limit: Maximum results per query
            threshold: Minimum similarity score

        Returns:
            List of result lists, one per query
        """
        pass

    # ========================================================================
    # Collection Operations
    # ========================================================================

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
    ) -> None:
        """
        Create vector collection.

        Args:
            collection_name: Collection identifier
            vector_size: Embedding dimension
        """
        pass

    @abstractmethod
    async def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        """
        Delete entire collection.

        Args:
            collection_name: Collection to delete
        """
        pass

    @abstractmethod
    async def collection_info(
        self,
        collection_name: str,
    ) -> Dict[str, Any]:
        """
        Get collection metadata.

        Args:
            collection_name: Collection to query

        Returns:
            Metadata dict (size, vector_size, etc.)
        """
        pass

    # ========================================================================
    # Utility
    # ========================================================================

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if backend is healthy.

        Returns:
            True if operational
        """
        pass

    @abstractmethod
    def get_device(self) -> str:
        """
        Get compute device.

        Returns:
            "cpu" or "cuda"
        """
        pass
