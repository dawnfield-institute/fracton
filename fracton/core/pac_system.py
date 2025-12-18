"""
PACSystem - PAC-Lazy Tree Management

The central system for managing PAC-Lazy tree structures.
Handles node storage, retrieval, reconstruction, and garbage collection.

Key Features:
- Delta-only storage with efficient reconstruction
- Tiered caching (hot/warm/cold) for access patterns
- Automatic phase transitions based on SEC thresholds
- Conservation-preserving garbage collection
"""

import torch
import time
from typing import Dict, List, Optional, Tuple, Set, Iterator
from dataclasses import dataclass, field
from collections import OrderedDict

from .pac_node import PACNode, PACNodeFactory
from ..physics.constants import XI, PHI_XI, LAMBDA_STAR
from ..physics.conservation import validate_pac, compute_residual
from ..physics.phase_transitions import PhaseState, detect_phase, should_collapse


@dataclass
class TieredCache:
    """
    Three-tier cache for PAC nodes based on access patterns.
    
    - Hot: Frequently accessed, kept materialized
    - Warm: Recently accessed, delta stored
    - Cold: Rarely accessed, compressed/disk
    """
    hot_size: int = 10000
    warm_size: int = 100000
    
    # Internal storage (OrderedDict for LRU behavior)
    _hot: OrderedDict = field(default_factory=OrderedDict)
    _warm: OrderedDict = field(default_factory=OrderedDict)
    _cold: Dict[int, bytes] = field(default_factory=dict)  # Compressed storage
    
    def get(self, node_id: int) -> Optional[PACNode]:
        """Get node from cache, promoting to hot tier."""
        # Check hot first
        if node_id in self._hot:
            self._hot.move_to_end(node_id)
            return self._hot[node_id]
        
        # Check warm
        if node_id in self._warm:
            node = self._warm.pop(node_id)
            self._promote_to_hot(node_id, node)
            return node
        
        # Check cold (decompress)
        if node_id in self._cold:
            node = self._decompress(self._cold[node_id])
            self._promote_to_warm(node_id, node)
            return node
        
        return None
    
    def put(self, node: PACNode, tier: str = "hot") -> None:
        """Add node to specified tier."""
        if tier == "hot":
            self._promote_to_hot(node.id, node)
        elif tier == "warm":
            self._promote_to_warm(node.id, node)
        else:
            self._demote_to_cold(node.id, node)
    
    def _promote_to_hot(self, node_id: int, node: PACNode) -> None:
        """Promote node to hot tier."""
        # Evict if necessary
        while len(self._hot) >= self.hot_size:
            evicted_id, evicted_node = self._hot.popitem(last=False)
            self._promote_to_warm(evicted_id, evicted_node)
        
        self._hot[node_id] = node
        self._hot.move_to_end(node_id)
        
        # Remove from other tiers
        self._warm.pop(node_id, None)
        self._cold.pop(node_id, None)
    
    def _promote_to_warm(self, node_id: int, node: PACNode) -> None:
        """Promote node to warm tier."""
        while len(self._warm) >= self.warm_size:
            evicted_id, evicted_node = self._warm.popitem(last=False)
            self._demote_to_cold(evicted_id, evicted_node)
        
        self._warm[node_id] = node
        self._warm.move_to_end(node_id)
        self._cold.pop(node_id, None)
    
    def _demote_to_cold(self, node_id: int, node: PACNode) -> None:
        """Demote node to cold (compressed) storage."""
        self._cold[node_id] = self._compress(node)
    
    def _compress(self, node: PACNode) -> bytes:
        """Compress node for cold storage."""
        import pickle
        import zlib
        data = node.to_dict()
        return zlib.compress(pickle.dumps(data))
    
    def _decompress(self, data: bytes) -> PACNode:
        """Decompress node from cold storage."""
        import pickle
        import zlib
        node_dict = pickle.loads(zlib.decompress(data))
        return PACNode.from_dict(node_dict)
    
    def remove(self, node_id: int) -> Optional[PACNode]:
        """Remove node from all tiers."""
        node = self._hot.pop(node_id, None)
        if node is None:
            node = self._warm.pop(node_id, None)
        if node is None and node_id in self._cold:
            node = self._decompress(self._cold.pop(node_id))
        return node
    
    def __len__(self) -> int:
        return len(self._hot) + len(self._warm) + len(self._cold)
    
    def stats(self) -> Dict[str, int]:
        return {
            "hot": len(self._hot),
            "warm": len(self._warm),
            "cold": len(self._cold),
            "total": len(self)
        }


class PACSystem:
    """
    Central management system for PAC-Lazy trees.
    
    Manages multiple trees with shared caching and conservation enforcement.
    Supports optional Kronos backend for persistent storage.
    
    Usage:
        system = PACSystem(device='cuda')
        
        # Inject a pattern
        node_id = system.inject(field_tensor)
        
        # Find resonant patterns
        similar = system.find_resonant(query_tensor, top_k=5)
        
        # Reconstruct full value
        full_value = system.reconstruct(node_id)
        
        # With persistence:
        from fracton.storage import KronosBackend
        backend = KronosBackend(Path("./data"), "myapp")
        system = PACSystem(device='cuda', kronos_backend=backend)
        
        # Save/restore state
        episode_id = system.save_state()
        system.restore_state(episode_id)
    """
    
    def __init__(self, 
                 device: str = 'cpu',
                 hot_cache_size: int = 10000,
                 warm_cache_size: int = 100000,
                 kronos_backend = None,
                 auto_persist: bool = True,
                 persist_threshold: float = 0.5):
        """
        Initialize PACSystem.
        
        Args:
            device: Torch device for tensors
            hot_cache_size: Size of hot tier cache
            warm_cache_size: Size of warm tier cache
            kronos_backend: Optional KronosBackend for persistence
            auto_persist: Auto-save important patterns (if backend provided)
            persist_threshold: Importance threshold for auto-persistence
        """
        self.device = device
        self.factory = PACNodeFactory(device=device)
        self.cache = TieredCache(hot_size=hot_cache_size, warm_size=warm_cache_size)
        
        # Kronos persistence
        self._backend = kronos_backend
        self._auto_persist = auto_persist and (kronos_backend is not None)
        self._persist_threshold = persist_threshold
        
        # Root nodes (entry points to trees)
        self._roots: Dict[int, PACNode] = {}
        
        # Quick lookup for node existence
        self._node_ids: Set[int] = set()
        
        # Statistics
        self._inject_count = 0
        self._reconstruct_count = 0
        self._gc_count = 0
        self._persist_count = 0
    
    def inject(self, 
               value: torch.Tensor,
               parent_id: Optional[int] = None,
               label: str = "",
               importance: float = 0.0,
               persist: bool = False) -> int:
        """
        Inject a pattern into the system.
        
        If parent_id is provided, stores as delta from parent.
        Otherwise creates a new root node.
        
        Args:
            value: Pattern tensor to inject
            parent_id: Optional parent node ID
            label: Optional semantic label
            importance: Importance score (used for crystallization/persistence)
            persist: Force persist to Kronos (if backend available)
            
        Returns:
            ID of the created node
        """
        self._inject_count += 1
        
        if parent_id is None:
            # Create root node (delta == value)
            node = self.factory.create_root(value.to(self.device), label=label)
            self._roots[node.id] = node
        else:
            # Create child with delta from parent
            parent = self.cache.get(parent_id)
            if parent is None:
                raise ValueError(f"Parent node {parent_id} not found")
            
            # Reconstruct parent value to compute delta
            parent_value = self.reconstruct(parent_id)
            delta = value.to(self.device) - parent_value
            
            node = self.factory.create_child(parent, delta, label=label)
        
        # Add to cache (hot tier for new nodes)
        self.cache.put(node, tier="hot")
        self._node_ids.add(node.id)
        
        # Auto-persist if backend available and importance threshold met
        should_persist = persist or (self._auto_persist and importance >= self._persist_threshold)
        if should_persist and self._backend is not None:
            crystallized = importance >= self._persist_threshold
            self._backend.save_node(
                node,
                crystallized=crystallized,
                importance=importance
            )
            self._persist_count += 1
        
        return node.id
    
    def reconstruct(self, node_id: int) -> torch.Tensor:
        """
        Reconstruct full value by traversing to root.
        
        This is the key operation - we never store absolutes,
        so reconstruction requires summing deltas to root.
        
        Args:
            node_id: Node to reconstruct
            
        Returns:
            Full reconstructed tensor
        """
        self._reconstruct_count += 1
        
        node = self.cache.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")
        
        # Check cache
        if node._materialized_valid and node._materialized is not None:
            return node._materialized
        
        # Traverse to root, collecting deltas
        deltas = []
        current = node
        while current is not None:
            deltas.append(current.delta)
            if current.parent_id == -1:
                break
            current = self.cache.get(current.parent_id)
        
        # Sum all deltas (root first)
        result = torch.zeros_like(deltas[0])
        for delta in reversed(deltas):
            result = result + delta
        
        # Cache if node is expanded
        if node.phase == PhaseState.EXPANDED:
            node._materialized = result
            node._materialized_valid = True
        
        return result
    
    def find_resonant(self, 
                      query: torch.Tensor,
                      top_k: int = 5,
                      threshold: float = 0.5) -> List[Tuple[int, float]]:
        """
        Find nodes most resonant with query pattern.
        
        Uses efficient approximation - compares with hot cache first,
        then expands search if needed.
        
        Args:
            query: Query tensor
            top_k: Number of results to return
            threshold: Minimum resonance score
            
        Returns:
            List of (node_id, resonance_score) tuples
        """
        query = query.to(self.device)
        results = []
        
        # Search hot cache (already materialized or cheap to reconstruct)
        # Convert to list to avoid mutation during iteration
        hot_items = list(self.cache._hot.items())
        for node_id, node in hot_items:
            try:
                value = self.reconstruct(node_id)
                score = self._compute_resonance(query, value)
                if score >= threshold:
                    results.append((node_id, score))
            except:
                continue
        
        # Sort by resonance score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _compute_resonance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute resonance (cosine similarity) between two tensors."""
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        
        dot = torch.dot(a_flat, b_flat)
        norm_a = torch.norm(a_flat)
        norm_b = torch.norm(b_flat)
        
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        
        return (dot / (norm_a * norm_b)).item()
    
    def get_node(self, node_id: int) -> Optional[PACNode]:
        """Get a node by ID."""
        return self.cache.get(node_id)
    
    def update_potential(self, node_id: int, new_potential: float) -> PhaseState:
        """
        Update node potential and handle phase transitions.
        
        Args:
            node_id: Node to update
            new_potential: New potential value
            
        Returns:
            New phase state
        """
        node = self.cache.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")
        
        old_phase = node.phase
        node.potential = new_potential
        new_phase = node.update_phase()
        
        # Handle phase transitions
        if old_phase != new_phase:
            if new_phase == PhaseState.COLLAPSED and node.parent_id != -1:
                # Merge to parent
                self._merge_to_parent(node)
            elif new_phase == PhaseState.EXPANDED:
                # Materialize
                self.reconstruct(node_id)  # This caches the result
        
        return new_phase
    
    def _merge_to_parent(self, node: PACNode) -> None:
        """Merge collapsed node into parent."""
        if node.parent_id == -1:
            return  # Can't merge root
        
        parent = self.cache.get(node.parent_id)
        if parent is None:
            return
        
        # Add node's delta to parent
        parent.delta = parent.delta + node.delta
        parent.invalidate_cache()
        
        # Reparent children
        for child_id in node.children_ids:
            child = self.cache.get(child_id)
            if child:
                child.parent_id = parent.id
                parent.add_child(child_id)
        
        # Remove node
        parent.remove_child(node.id)
        self.cache.remove(node.id)
        self._node_ids.discard(node.id)
    
    def garbage_collect(self, potential_threshold: float = XI / 2) -> int:
        """
        Remove nodes with potential below threshold.
        
        Conservation is maintained by merging to parents.
        
        Args:
            potential_threshold: Minimum potential to keep
            
        Returns:
            Number of nodes collected
        """
        self._gc_count += 1
        collected = 0
        
        # Collect nodes to remove (can't modify during iteration)
        to_remove = []
        for node_id in list(self._node_ids):
            node = self.cache.get(node_id)
            if node and node.potential < potential_threshold and not node.is_root:
                to_remove.append(node_id)
        
        # Merge each to parent
        for node_id in to_remove:
            node = self.cache.get(node_id)
            if node:
                self._merge_to_parent(node)
                collected += 1
        
        return collected
    
    def validate_conservation(self, root_id: int) -> Tuple[bool, float]:
        """
        Validate PAC conservation for a tree.
        
        Args:
            root_id: Root node ID
            
        Returns:
            Tuple of (is_valid, total_residual)
        """
        root = self.cache.get(root_id)
        if root is None:
            raise ValueError(f"Root {root_id} not found")
        
        total_residual = 0.0
        
        def validate_subtree(node: PACNode) -> float:
            if not node.children_ids:
                return 0.0
            
            # Get children
            children = [self.cache.get(cid) for cid in node.children_ids]
            children = [c for c in children if c is not None]
            
            if not children:
                return 0.0
            
            # Compute residual
            children_delta_sum = sum(c.delta for c in children)
            residual = torch.abs(children_delta_sum).max().item()
            
            # Recurse
            child_residuals = sum(validate_subtree(c) for c in children)
            
            return residual + child_residuals
        
        total_residual = validate_subtree(root)
        is_valid = total_residual < 1e-6
        
        return is_valid, total_residual
    
    def stats(self) -> Dict[str, any]:
        """Get system statistics."""
        stats = {
            "node_count": len(self._node_ids),
            "root_count": len(self._roots),
            "inject_count": self._inject_count,
            "reconstruct_count": self._reconstruct_count,
            "gc_count": self._gc_count,
            "persist_count": self._persist_count,
            "cache": self.cache.stats()
        }
        
        if self._backend is not None:
            stats["kronos"] = self._backend.get_stats()
        
        return stats
    
    # === Kronos Persistence Methods ===
    
    def save_state(self, name: str = None, metadata: dict = None) -> str:
        """
        Save complete substrate state as Kronos episode.
        
        Args:
            name: Optional episode name
            metadata: Optional metadata to store
            
        Returns:
            episode_id
            
        Raises:
            RuntimeError: If no Kronos backend configured
        """
        if self._backend is None:
            raise RuntimeError("No Kronos backend configured. Initialize with kronos_backend parameter.")
        
        # Collect all nodes from cache
        all_nodes = []
        
        # Hot tier
        for node in self.cache._hot.values():
            all_nodes.append(node)
        
        # Warm tier
        for node in self.cache._warm.values():
            all_nodes.append(node)
        
        # Cold tier (decompress)
        for node_id, compressed in self.cache._cold.items():
            node = self.cache._decompress(compressed)
            all_nodes.append(node)
        
        # Save as episode
        episode_id = self._backend.save_episode(
            nodes=all_nodes,
            name=name,
            metadata={
                "node_count": len(all_nodes),
                "root_count": len(self._roots),
                "device": self.device,
                **(metadata or {})
            }
        )
        
        return episode_id
    
    def restore_state(self, episode_id: str):
        """
        Restore substrate state from Kronos episode.
        
        Clears current state and loads from episode.
        
        Args:
            episode_id: Episode to restore from
            
        Raises:
            RuntimeError: If no Kronos backend configured
        """
        if self._backend is None:
            raise RuntimeError("No Kronos backend configured. Initialize with kronos_backend parameter.")
        
        # Load episode
        nodes, metadata = self._backend.load_episode(episode_id, device=self.device)
        
        # Clear current state
        self.cache._hot.clear()
        self.cache._warm.clear()
        self.cache._cold.clear()
        self._roots.clear()
        self._node_ids.clear()
        
        # Restore nodes
        for node in nodes:
            # Move tensor to correct device
            if node.delta.device.type != self.device:
                node.delta = node.delta.to(self.device)
            
            # Add to cache
            self.cache.put(node, tier="hot")
            self._node_ids.add(node.id)
            
            # Check if root
            if node.parent_id == -1:
                self._roots[node.id] = node
    
    def persist_node(self, node_id: int, importance: float = 1.0, crystallized: bool = True) -> str:
        """
        Manually persist a specific node to Kronos.
        
        Args:
            node_id: Node to persist
            importance: Importance score
            crystallized: Mark as crystallized
            
        Returns:
            doc_id
        """
        if self._backend is None:
            raise RuntimeError("No Kronos backend configured.")
        
        node = self.cache.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")
        
        doc_id = self._backend.save_node(
            node,
            crystallized=crystallized,
            importance=importance
        )
        self._persist_count += 1
        
        return doc_id
    
    def load_from_kronos(self, node_id: int) -> Optional[int]:
        """
        Load a specific node from Kronos into cache.
        
        Args:
            node_id: Node ID to load
            
        Returns:
            node_id if loaded, None if not found
        """
        if self._backend is None:
            return None
        
        node = self._backend.load_node(node_id)
        if node is None:
            return None
        
        # Move to device
        if node.delta.device.type != self.device:
            node.delta = node.delta.to(self.device)
        
        # Add to cache
        self.cache.put(node, tier="hot")
        self._node_ids.add(node.id)
        
        if node.parent_id == -1:
            self._roots[node.id] = node
        
        return node.id
    
    def __len__(self) -> int:
        return len(self._node_ids)
    
    def __repr__(self) -> str:
        stats = self.stats()
        backend_info = " +kronos" if self._backend else ""
        return (f"PACSystem(nodes={stats['node_count']}, "
                f"roots={stats['root_count']}, device={self.device}{backend_info})")
