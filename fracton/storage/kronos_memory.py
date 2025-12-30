"""
KRONOS Memory - Backend-Integrated Implementation

Integrates pluggable backends while preserving all PAC+SEC+PAS logic.

Combines:
- PAC (Predictive Adaptive Coding): Delta-only storage
- SEC (Symbolic Entropy Collapse): Resonance-based retrieval
- PAS (Potential Actualization): Conservation laws
- Pluggable Backends: SQLite, ChromaDB, Neo4j, Qdrant
"""

from __future__ import annotations

import torch
import uuid
import hashlib
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

from enum import Enum

from .backends import BackendConfig, GraphBackend, VectorBackend
from .backends.base import GraphNode, GraphEdge, VectorPoint, TemporalPath
from .backend_factory import BackendFactory
from .embeddings import EmbeddingService, create_embedding_service
from .foundation_integration import FoundationIntegration
from ..physics.constants import XI, PHI_XI, LAMBDA_STAR
from ..physics.phase_transitions import detect_phase

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class NodeType(str, Enum):
    """Type of memory node."""
    MEMORY = "memory"
    CONCEPT = "concept"
    FACT = "fact"
    PROCEDURE = "procedure"
    EPISODE = "episode"
    PAPER = "paper"
    COMMIT = "commit"
    DOCUMENT = "document"


class RelationType(str, Enum):
    """Type of relationship between nodes."""
    RELATES_TO = "relates_to"
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    IMPLEMENTS = "implements"
    DOCUMENTS = "documents"
    CITES = "cites"


class PhaseState(str, Enum):
    """Phase state based on potential."""
    COLLAPSED = "collapsed"
    STABLE = "stable"
    EXPANDED = "expanded"


@dataclass
class PACMemoryNode:
    """Memory node with PAC encoding."""
    # Core identity
    id: str
    fractal_signature: str

    # PAC encoding
    delta_embedding: torch.Tensor  # Delta from parent (or full if root)
    delta_content: str  # Semantic delta from parent

    # Tree structure
    parent_id: str  # "-1" for root
    children_ids: List[str] = field(default_factory=list)

    # PAS properties
    potential: float = 1.0
    entropy: float = 1.0
    phase: PhaseState = PhaseState.STABLE
    coherence: float = 0.5

    # Content
    content: str = ""
    node_type: NodeType = NodeType.MEMORY
    graph: str = "default"
    path: str = ""  # Hierarchical path

    # Temporal
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # Importance and relationships
    importance: float = 0.0
    relationships: Dict[str, List[str]] = field(default_factory=dict)

    # Lineage tracking
    forward_trace: List[str] = field(default_factory=list)
    backward_trace: List[str] = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        return self.parent_id == "-1"

    @property
    def metadata(self) -> Dict[str, Any]:
        """Access user metadata."""
        return getattr(self, '_user_metadata', {})


@dataclass
class ResonanceResult:
    """Result from SEC resonance ranking."""
    node: PACMemoryNode
    score: float  # Overall resonance score
    similarity: float  # Embedding similarity
    entropy_match: float  # Entropy resonance
    recency: float  # Temporal recency
    path_strength: float  # Graph path strength


class KronosMemory:
    """
    KRONOS Memory with Pluggable Backends.

    Integrates pluggable backends (SQLite, Neo4j, ChromaDB, Qdrant) with
    PAC+SEC+PAS logic for production-grade semantic memory.

    Architecture:
        - PAC: Delta-only storage, hierarchical reconstruction
        - SEC: Entropy-based resonance ranking
        - PAS: Conservation validation
        - Backends: Pluggable graph + vector storage
    """

    def __init__(
        self,
        storage_path: Path,
        namespace: str = "default",
        backend_config: Optional[BackendConfig] = None,
        device: str = "cpu",
        embedding_dim: int = 384,
        embedding_model: str = "mini",
        enable_auto_fallback: bool = True,
    ):
        """
        Initialize KRONOS memory.

        Args:
            storage_path: Root directory for storage
            namespace: Namespace for this instance
            backend_config: Backend configuration (None = auto-detect)
            device: PyTorch device (cpu/cuda)
            embedding_dim: Embedding dimension (overridden by model if using sentence-transformers)
            embedding_model: Embedding model key ("mini", "base", "large", "code") or custom model name
            enable_auto_fallback: Fall back to lightweight if production fails
        """
        self.storage_path = Path(storage_path)
        self.namespace = namespace
        self.device = device
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model

        # Backends (will be initialized in connect())
        self.graph_backend: Optional[GraphBackend] = None
        self.vector_backend: Optional[VectorBackend] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.foundation: Optional[FoundationIntegration] = None
        self.backend_config = backend_config or BackendConfig(
            graph_type="sqlite",
            vector_type="chromadb",
            device=device,
            embedding_dim=embedding_dim,
        )
        self.enable_auto_fallback = enable_auto_fallback

        # In-memory cache for recently accessed nodes
        self._node_cache: Dict[Tuple[str, str], PACMemoryNode] = {}  # (graph, node_id) -> node
        self._cache_size = 1000

        # Statistics
        self._stats = {
            "total_nodes": 0,
            "total_graphs": 0,
            "queries": 0,
            "reconstructions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_updates": 0,
            "collapses": 0,
        }

        # Health metrics
        self._health_metrics = {
            "c_squared": [],  # Model constant c² history
            "duty_cycle": [],  # SEC duty cycle history
            "balance_operator": [],  # Ξ values
            "med_quality": [],  # MED quality scores
        }

        logger.info(
            f"KronosMemory initialized: namespace={namespace}, "
            f"device={device}, dim={embedding_dim}"
        )

    async def connect(self) -> None:
        """Connect to backends and initialize embedding service and foundations."""
        # Initialize embedding service
        cache_dir = self.storage_path / "models"
        self.embedding_service = await create_embedding_service(
            model=self.embedding_model,
            device=self.device,
            cache_dir=cache_dir,
        )

        # Update embedding_dim if model loaded
        if self.embedding_service.is_available():
            self.embedding_dim = self.embedding_service.embedding_dim
            self.backend_config.embedding_dim = self.embedding_dim
            logger.info(f"Using real embeddings: dim={self.embedding_dim}")
        else:
            logger.info(f"Using hash-based embeddings: dim={self.embedding_dim}")

        # Initialize theoretical foundations
        self.foundation = FoundationIntegration(
            embedding_dim=self.embedding_dim,
            device=self.device,
            enable_strict_med=False,  # Non-strict for storage hierarchies
        )
        logger.info("Theoretical foundations initialized (PAC/SEC/MED)")

        # Connect to backends
        if self.enable_auto_fallback:
            self.graph_backend, self.vector_backend, actual_config = (
                await BackendFactory.create_with_fallback(
                    self.backend_config,
                    self.storage_path,
                    self.namespace,
                )
            )
            self.backend_config = actual_config
        else:
            self.graph_backend, self.vector_backend = (
                await BackendFactory.create_backends(
                    self.backend_config,
                    self.storage_path,
                    self.namespace,
                )
            )

        logger.info(
            f"Connected to backends: {self.backend_config.graph_type} + "
            f"{self.backend_config.vector_type}"
        )

    async def close(self) -> None:
        """Close backend connections."""
        if self.graph_backend:
            await self.graph_backend.close()
        if self.vector_backend:
            await self.vector_backend.close()

        logger.info("Backends closed")

    # ========================================================================
    # Graph Management
    # ========================================================================

    async def create_graph(
        self,
        graph_name: str,
        description: str = "",
    ) -> None:
        """Create a new memory graph."""
        await self.graph_backend.create_graph(graph_name, description)
        self._stats["total_graphs"] += 1
        logger.info(f"Created graph: {graph_name}")

    async def list_graphs(self) -> List[Dict[str, Any]]:
        """List all graphs."""
        return await self.graph_backend.list_graphs()

    # ========================================================================
    # Memory Storage (PAC Delta Encoding)
    # ========================================================================

    async def store(
        self,
        content: str,
        graph: str,
        node_type: NodeType,
        parent_id: Optional[str] = None,
        embedding: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.0,
    ) -> str:
        """
        Store memory with PAC delta encoding.

        Args:
            content: Memory content (text)
            graph: Which graph to store in
            node_type: Type of memory node
            parent_id: Parent node ID (None for root)
            embedding: Pre-computed embedding (None = compute)
            metadata: Additional metadata
            importance: Importance score

        Returns:
            Node ID
        """
        # Generate or validate embedding
        if embedding is None:
            embedding = await self._compute_embedding(content)

        # Generate node ID
        node_id = str(uuid.uuid4())

        # Compute PAC delta
        if parent_id is None or parent_id == "-1":
            # ROOT NODE: delta == full embedding
            delta_emb = embedding
            delta_content = content
            potential = 1.0
            parent_id = "-1"
            children_ids = []
        else:
            # CHILD NODE: compute delta from parent
            parent = await self._get_node(graph, parent_id)
            if parent is None:
                raise ValueError(f"Parent {parent_id} not found in graph {graph}")

            # Reconstruct parent embedding (PAC)
            parent_emb = await self._reconstruct_embedding(graph, parent_id)

            # Delta encoding
            delta_emb = embedding - parent_emb
            delta_content = self._compute_semantic_diff(content, parent.content)

            # Inherit potential with decay
            potential = parent.potential * LAMBDA_STAR

            # Update parent's children
            children_ids = []
            parent.children_ids.append(node_id)
            await self._update_node_children(graph, parent_id, parent.children_ids)

        # Compute SEC entropy
        entropy = self._compute_symbolic_entropy(content)

        # Detect phase
        phase = detect_phase(potential)

        # Generate fractal signature
        fractal_sig = self._generate_fractal_signature(embedding, {"entropy": entropy})

        # Create PAC memory node
        pac_node = PACMemoryNode(
            id=node_id,
            fractal_signature=fractal_sig,
            delta_embedding=delta_emb,
            delta_content=delta_content,
            parent_id=parent_id,
            children_ids=children_ids,
            potential=potential,
            entropy=entropy,
            phase=phase,
            coherence=0.5,
            content=content,
            node_type=node_type,
            graph=graph,
            importance=importance,
        )

        # Store user metadata for later retrieval
        pac_node._user_metadata = metadata or {}

        # Validate conservation and check for collapse (if has parent)
        if parent_id != "-1" and self.foundation is not None:
            # Get parent's siblings to check conservation
            parent_node = await self._get_node(graph, parent_id)
            if parent_node and parent_node.children_ids:
                # Reconstruct embeddings for validation
                parent_full = await self._reconstruct_embedding(graph, parent_id)

                # Get all children (including the one we just created)
                children_full = []
                for child_id in parent_node.children_ids:
                    if child_id == node_id:
                        # This is the new node
                        children_full.append(embedding)
                    else:
                        # Existing children
                        child_full = await self._reconstruct_embedding(graph, child_id)
                        children_full.append(child_full)

                # Validate using distance validator (E=mc²)
                if len(children_full) > 0:
                    distance_metrics = self.foundation.distance_validator.validate_energy_conservation(
                        parent_full,
                        children_full,
                        embedding_type="real" if self.embedding_service.is_available() else "synthetic",
                    )

                    # Track c² health metric
                    self._health_metrics["c_squared"].append(distance_metrics.c_squared)
                    if len(self._health_metrics["c_squared"]) > 1000:
                        self._health_metrics["c_squared"].pop(0)

                    # Create PAC nodes for balance operator calculation
                    parent_pac = self.foundation.create_pac_node_from_embedding(
                        embedding=parent_full,
                        content=parent_node.content,
                        depth=parent_node.path.count("/") if parent_node.path else 0,
                        parent_embedding=None,
                    )

                    children_pac = []
                    for i, child_full in enumerate(children_full):
                        child_pac = self.foundation.create_pac_node_from_embedding(
                            embedding=child_full,
                            content=f"child_{i}",  # Content not needed for balance calc
                            depth=parent_pac.depth + 1 + i,
                            parent_embedding=parent_full,
                        )
                        children_pac.append(child_pac)

                    # Compute balance operator Ξ
                    xi_local = self.foundation.pac_engine.compute_balance_operator(
                        parent_pac, children_pac
                    )
                    collapse_status = self.foundation.pac_engine.check_collapse_trigger(xi_local)

                    # Track balance operator
                    self._health_metrics["balance_operator"].append(xi_local)
                    if len(self._health_metrics["balance_operator"]) > 1000:
                        self._health_metrics["balance_operator"].pop(0)

                    # Check for collapse trigger
                    if collapse_status == "COLLAPSE":
                        self._stats["collapses"] += 1
                        logger.warning(
                            f"⚠️  Collapse trigger detected in {graph}: "
                            f"Ξ={xi_local:.4f} > {self.foundation.constants.XI:.4f} "
                            f"(parent={parent_id}, children={len(children_full)})"
                        )
                    elif collapse_status == "DECAY":
                        logger.warning(
                            f"⚠️  Field decay detected in {graph}: "
                            f"Ξ={xi_local:.4f} < 0.5 (duty cycle={self.foundation.constants.DUTY_CYCLE:.4f})"
                        )

                    # Compute SEC duty cycle from phase history
                    duty_cycle = self.foundation.sec_operators.compute_duty_cycle(
                        self.foundation.phase_history
                    )
                    self._health_metrics["duty_cycle"].append(duty_cycle)
                    if len(self._health_metrics["duty_cycle"]) > 1000:
                        self._health_metrics["duty_cycle"].pop(0)

                    logger.debug(
                        f"Foundation metrics: c²={distance_metrics.c_squared:.2f}, "
                        f"Ξ={xi_local:.4f} ({collapse_status}), "
                        f"duty={duty_cycle:.3f}, "
                        f"binding={'amplification' if distance_metrics.binding_energy < 0 else 'binding'}="
                        f"{abs(distance_metrics.binding_energy):.2f}"
                    )

        # Store in graph backend
        graph_node = self._pac_node_to_graph_node(pac_node)
        await self.graph_backend.create_node(graph_node, graph)

        # Store vector (delta embedding)
        vector_point = VectorPoint(
            id=node_id,
            vector=delta_emb,
            payload={
                "graph": graph,
                "node_type": node_type.value,
                "content": content[:500],  # Truncate for payload
                "fractal_signature": fractal_sig,
                "entropy": entropy,
                "potential": potential,
            },
        )
        await self.vector_backend.store(vector_point)

        # Cache
        self._node_cache[(graph, node_id)] = pac_node
        self._stats["total_nodes"] += 1

        logger.debug(f"Stored node {node_id} in {graph} (potential={potential:.3f})")

        return node_id

    # ========================================================================
    # Memory Retrieval (SEC Resonance Ranking)
    # ========================================================================

    async def query(
        self,
        query_text: str,
        graphs: Optional[List[str]] = None,
        limit: int = 10,
        threshold: float = 0.0,
        expand_graph: bool = False,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[ResonanceResult]:
        """
        Query memory with SEC resonance ranking.

        Args:
            query_text: Query text
            graphs: Which graphs to search (None = all)
            limit: Max results
            threshold: Min resonance score
            expand_graph: Expand graph context
            weights: Custom SEC weights

        Returns:
            List of resonance results, ranked by SEC score
        """
        self._stats["queries"] += 1

        # Compute query embedding
        query_emb = await self._compute_embedding(query_text)

        # Default SEC weights
        if weights is None:
            weights = {
                "similarity": 0.4,
                "entropy": 0.2,
                "recency": 0.2,
                "coherence": 0.2,
            }

        # Vector search
        search_results = await self.vector_backend.search(
            query_emb,
            limit=limit * 2,  # Get more for filtering
            threshold=threshold,
        )

        # Filter by graph if specified
        if graphs:
            search_results = [
                r for r in search_results if r.payload.get("graph") in graphs
            ]

        # Convert to PAC nodes and compute SEC resonance
        resonance_results = []
        for result in search_results[:limit]:
            graph = result.payload["graph"]
            node_id = result.id

            # Get full node
            node = await self._get_node(graph, node_id)
            if node is None:
                continue

            # Reconstruct full embedding for proper similarity
            full_emb = await self._reconstruct_embedding(graph, node_id)
            similarity = torch.cosine_similarity(query_emb, full_emb, dim=0).item()

            # SEC resonance components
            entropy_match = 1.0 - abs(node.entropy - self._compute_symbolic_entropy(query_text))
            recency = self._compute_recency(node.created_at)
            coherence_score = node.coherence

            # Foundation resonance ranking (if available)
            resonance_score = 0.0
            if self.foundation is not None:
                # Create PAC node for resonance calculation
                pac_node = self.foundation.create_pac_node_from_embedding(
                    embedding=full_emb,
                    content=node.content,
                    depth=node.path.count("/") if node.path else 0,
                )
                # Compute resonance: R(k) = φ^(1 + (k_eq - k)/2)
                query_depth = 0  # Query is at root level
                resonance_score = self.foundation.compute_resonance_score(pac_node, query_depth)
                # Normalize to [0, 1] range
                resonance_score = min(1.0, resonance_score / 10.0)

            # Overall SEC resonance with foundation resonance
            sec_score = (
                weights["similarity"] * similarity +
                weights["entropy"] * entropy_match +
                weights["recency"] * recency +
                weights["coherence"] * (coherence_score + resonance_score) / 2  # Blend coherence with resonance
            )

            # π-harmonic modulation (SEC principle)
            harmonic_boost = 1.0 + 0.1 * math.sin(sec_score * math.pi)
            final_score = sec_score * harmonic_boost

            resonance_results.append(ResonanceResult(
                node=node,
                score=final_score,
                similarity=similarity,
                entropy_match=entropy_match,
                recency=recency,
                path_strength=1.0,  # TODO: Compute from graph
            ))

        # Sort by SEC resonance
        resonance_results.sort(key=lambda r: r.score, reverse=True)

        return resonance_results[:limit]

    # ========================================================================
    # Temporal Tracing (Bifractal)
    # ========================================================================

    async def trace_evolution(
        self,
        graph: str,
        node_id: str,
        direction: str = "both",
        max_depth: int = 100,
    ) -> Dict[str, Any]:
        """
        Trace temporal evolution of an idea (bifractal).

        Args:
            graph: Graph name
            node_id: Starting node
            direction: "forward", "backward", or "both"
            max_depth: Maximum depth to trace

        Returns:
            Dict with temporal trace information
        """
        path = await self.graph_backend.trace_lineage(
            node_id, direction, max_depth, graph
        )

        # Extract evolution metadata
        return {
            "root_id": path.root_id,
            "path": path.path,
            "depth": len(path.path),
            "total_potential": path.total_potential,
            "entropy_evolution": path.entropy_evolution,
            "nodes": [self._graph_node_to_pac_node(n) for n in path.nodes],
        }

    # ========================================================================
    # Cross-Graph Linking
    # ========================================================================

    async def link_across_graphs(
        self,
        from_graph: str,
        from_id: str,
        to_graph: str,
        to_id: str,
        relation: RelationType,
        weight: float = 1.0,
    ) -> None:
        """Link nodes across different graphs."""
        edge = GraphEdge(
            from_id=from_id,
            to_id=to_id,
            relation_type=relation.value,
            weight=weight,
            metadata={"from_graph": from_graph, "to_graph": to_graph},
        )

        # Store edge in both graphs
        await self.graph_backend.create_edge(edge, from_graph)
        # Note: Cross-graph edges stored in source graph with metadata

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _compute_embedding(self, text: str) -> torch.Tensor:
        """Compute embedding for text using sentence-transformers or hash-based fallback."""
        if self.embedding_service and self.embedding_service.is_available():
            # Use real embeddings
            return await self.embedding_service.embed(text)
        else:
            # Fallback to hash-based embedding
            hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
            torch.manual_seed(hash_val % (2**32))
            return torch.randn(self.embedding_dim, device=self.device)

    def _compute_semantic_diff(self, content: str, parent_content: str) -> str:
        """Compute semantic diff between content and parent."""
        # Simple diff for now
        return content  # TODO: Implement proper diff

    def _compute_symbolic_entropy(self, text: str) -> float:
        """Compute symbolic entropy (SEC)."""
        if not text:
            return 1.0

        # Character frequency entropy
        from collections import Counter
        freq = Counter(text)
        total = len(text)
        entropy = -sum((c/total) * math.log2(c/total) for c in freq.values() if c > 0)

        # Normalize to [0, 1]
        max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1.0
        return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0

    def _generate_fractal_signature(
        self,
        embedding: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> str:
        """Generate fractal signature (neural fingerprint)."""
        # Hash embedding + metadata
        emb_hash = hashlib.sha256(embedding.cpu().numpy().tobytes()).hexdigest()[:16]
        meta_str = f"{metadata.get('entropy', 0):.3f}_{metadata.get('potential', 1):.3f}"
        return f"{emb_hash}_{meta_str}"

    def _compute_recency(self, created_at: datetime) -> float:
        """Compute recency score [0, 1]."""
        age_seconds = (datetime.now() - created_at).total_seconds()
        # Exponential decay: half-life of 1 day
        half_life = 86400
        return math.exp(-age_seconds / half_life)

    async def _reconstruct_embedding(self, graph: str, node_id: str) -> torch.Tensor:
        """Reconstruct full embedding from PAC deltas."""
        self._stats["reconstructions"] += 1

        node = await self._get_node(graph, node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in {graph}")

        # If root, delta IS the embedding
        if node.is_root:
            return node.delta_embedding

        # Reconstruct by traversing to root
        deltas = [node.delta_embedding]
        current_id = node.parent_id

        while current_id != "-1":
            parent = await self._get_node(graph, current_id)
            if parent is None:
                break
            deltas.append(parent.delta_embedding)
            current_id = parent.parent_id

        # Sum all deltas (PAC reconstruction)
        return sum(deltas)

    async def _get_node(self, graph: str, node_id: str) -> Optional[PACMemoryNode]:
        """Get node from cache or backend."""
        cache_key = (graph, node_id)

        # Check cache
        if cache_key in self._node_cache:
            self._stats["cache_hits"] += 1
            return self._node_cache[cache_key]

        self._stats["cache_misses"] += 1

        # Fetch from backend
        graph_node = await self.graph_backend.get_node(node_id, graph)
        if graph_node is None:
            return None

        # Fetch vector for delta embedding
        vectors = await self.vector_backend.retrieve([node_id])
        delta_emb = vectors[0].vector if vectors else None

        # Convert to PAC node
        pac_node = self._graph_node_to_pac_node(graph_node, delta_emb)

        # Cache
        self._node_cache[cache_key] = pac_node
        if len(self._node_cache) > self._cache_size:
            # Simple FIFO eviction
            self._node_cache.pop(next(iter(self._node_cache)))

        return pac_node

    async def _update_node_children(
        self,
        graph: str,
        node_id: str,
        children_ids: List[str],
    ) -> None:
        """Update node's children list."""
        await self.graph_backend.update_node(
            node_id,
            {"children_ids": children_ids},
            graph,
        )

        # Invalidate cache
        cache_key = (graph, node_id)
        if cache_key in self._node_cache:
            del self._node_cache[cache_key]

    def _pac_node_to_graph_node(self, pac_node: PACMemoryNode) -> GraphNode:
        """Convert PACMemoryNode to GraphNode for storage."""
        # Merge system metadata with user metadata
        metadata = {
            "node_type": pac_node.node_type.value,
            "delta_content": pac_node.delta_content,
            "importance": pac_node.importance,
            "access_count": pac_node.access_count,
        }
        # Add user metadata if present
        if hasattr(pac_node, '_user_metadata'):
            metadata.update(pac_node._user_metadata)

        return GraphNode(
            id=pac_node.id,
            content=pac_node.content,
            timestamp=pac_node.created_at,
            fractal_signature=pac_node.fractal_signature or "",
            metadata=metadata,
            parent_id=pac_node.parent_id,
            children_ids=pac_node.children_ids,
            potential=pac_node.potential,
            entropy=pac_node.entropy,
            coherence=pac_node.coherence,
            phase=pac_node.phase.value,
        )

    def _graph_node_to_pac_node(
        self,
        graph_node: GraphNode,
        delta_emb: Optional[torch.Tensor] = None,
    ) -> PACMemoryNode:
        """Convert GraphNode to PACMemoryNode."""
        # Extract system metadata
        system_keys = {"node_type", "delta_content", "importance", "access_count", "graph"}

        # Separate user metadata from system metadata
        user_metadata = {k: v for k, v in graph_node.metadata.items() if k not in system_keys}

        pac_node = PACMemoryNode(
            id=graph_node.id,
            fractal_signature=graph_node.fractal_signature,
            delta_embedding=delta_emb,
            delta_content=graph_node.metadata.get("delta_content", ""),
            parent_id=graph_node.parent_id,
            children_ids=graph_node.children_ids,
            potential=graph_node.potential,
            entropy=graph_node.entropy,
            phase=PhaseState(graph_node.phase),
            coherence=graph_node.coherence,
            content=graph_node.content,
            node_type=NodeType(graph_node.metadata.get("node_type", "memory")),
            graph=graph_node.metadata.get("graph", "default"),
            created_at=graph_node.timestamp,
            importance=graph_node.metadata.get("importance", 0.0),
            access_count=graph_node.metadata.get("access_count", 0),
        )

        # Restore user metadata
        pac_node._user_metadata = user_metadata

        return pac_node

    async def retrieve(self, graph: str, node_id: str) -> Optional[PACMemoryNode]:
        """
        Retrieve a specific node by ID.

        Args:
            graph: Graph name
            node_id: Node ID

        Returns:
            PACMemoryNode or None if not found
        """
        return await self._get_node(graph, node_id)

    async def update_node(
        self,
        graph: str,
        node_id: str,
        updates: Dict[str, Any],
    ) -> None:
        """
        Update node fields.

        Args:
            graph: Graph name
            node_id: Node ID
            updates: Fields to update
        """
        # Invalidate cache
        cache_key = (graph, node_id)
        if cache_key in self._node_cache:
            del self._node_cache[cache_key]

        # Update in backend
        await self.graph_backend.update_node(node_id, updates, graph)

        # Update stats
        self._stats["total_updates"] += 1

    async def get_neighborhood(
        self,
        graph: str,
        node_id: str,
        max_hops: int = 1,
    ) -> TemporalPath:
        """
        Get neighborhood around a node.

        Args:
            graph: Graph name
            node_id: Center node ID
            max_hops: Maximum hops to expand

        Returns:
            TemporalPath with nodes and edges
        """
        return await self.graph_backend.get_neighbors(
            node_id,
            max_hops=max_hops,
            graph_name=graph,
        )

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all backends.

        Returns:
            Dict with backend health status
        """
        return await BackendFactory.health_check(
            self.graph_backend,
            self.vector_backend,
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        health = await self.health_check()

        return {
            **self._stats,
            "backend_health": health,
            "backend_config": {
                "graph": self.backend_config.graph_type,
                "vector": self.backend_config.vector_type,
                "device": self.device,
            },
            "foundation_health": self.get_foundation_health(),
        }

    def get_foundation_health(self) -> Dict[str, Any]:
        """
        Get theoretical foundation health metrics.

        Returns:
            Dict with PAC/SEC/MED health metrics:
            - c_squared: Model constant c² statistics
            - duty_cycle: SEC duty cycle statistics
            - balance_operator: Balance operator Ξ statistics
            - med_quality: MED quality scores
        """
        import numpy as np

        def stats(values):
            if not values:
                return {"count": 0}
            return {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "latest": float(values[-1]) if values else None,
            }

        return {
            "c_squared": stats(self._health_metrics["c_squared"]),
            "duty_cycle": stats(self._health_metrics["duty_cycle"]),
            "balance_operator": stats(self._health_metrics["balance_operator"]),
            "med_quality": stats(self._health_metrics["med_quality"]),
            "constants": {
                "phi": self.foundation.constants.PHI if self.foundation else None,
                "xi": self.foundation.constants.XI if self.foundation else None,
                "lambda_star": LAMBDA_STAR,
                "duty_cycle": self.foundation.constants.DUTY_CYCLE if self.foundation else None,
            },
        }
