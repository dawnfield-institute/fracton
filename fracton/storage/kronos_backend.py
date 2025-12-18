"""
Kronos Backend - Persistent PAC Storage
========================================

Main storage interface for Fracton's PAC-Lazy substrate.
Provides persistent storage with temporal indexing and episode tracking.

Directory Structure:
    {base_path}/
    └── {namespace}/
        ├── indices/
        │   ├── temporal_index.json
        │   └── crystallized_index.json
        ├── snapshots/
        │   ├── {doc_id}.fdo.yaml
        │   └── {doc_id}_delta.npy
        └── episodes/
            └── {episode_id}/
                ├── episode.yaml
                └── nodes/

Example:
    backend = KronosBackend(Path("./kronos"), namespace="gaia")
    
    # Save/load individual nodes
    doc_id = backend.save_node(node)
    node = backend.load_node(node_id)
    
    # Save/load complete state as episode
    episode_id = backend.save_episode(nodes, metadata)
    nodes, meta = backend.load_episode(episode_id)
    
    # Query by time
    node_ids = backend.query_temporal(start, end)
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from .fdo_serializer import FDOSerializer, save_fdo, load_fdo, list_fdo_files
from .temporal_index import TemporalIndex
from .episode_tracker import EpisodeTracker


class KronosBackend:
    """
    Persistent PAC storage using Kronos architecture.
    
    Provides:
    - Node persistence (save/load individual PACNodes)
    - Episode management (save/restore full state)
    - Temporal queries (find nodes by time range)
    - Crystallized pattern index (important patterns)
    """
    
    def __init__(
        self, 
        base_path: Path, 
        namespace: str = "default",
        device: str = 'cpu'
    ):
        """
        Initialize Kronos backend.
        
        Args:
            base_path: Root directory for Kronos storage
            namespace: Namespace for this substrate (allows multiple)
            device: Default device for loading tensors
        """
        self.base_path = Path(base_path)
        self.namespace = namespace
        self.device = device
        
        # Directory structure
        self.namespace_dir = self.base_path / namespace
        self.snapshots_dir = self.namespace_dir / "snapshots"
        self.indices_dir = self.namespace_dir / "indices"
        self.episodes_dir = self.namespace_dir / "episodes"
        
        # Create directories
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.serializer = FDOSerializer(self.snapshots_dir, device)
        self.temporal_index = TemporalIndex(self.indices_dir / "temporal_index.json")
        self.episode_tracker = EpisodeTracker(self.episodes_dir)
        
        # Crystallized patterns index (in-memory with disk persistence)
        self._crystallized_index_path = self.indices_dir / "crystallized_index.json"
        self._crystallized: Dict[str, float] = {}  # doc_id -> importance
        self._load_crystallized_index()
        
        # Node ID to doc_id mapping
        self._node_to_doc: Dict[int, str] = {}
        self._doc_to_node: Dict[str, int] = {}
        self._rebuild_node_mapping()
    
    # === Node Operations ===
    
    def save_node(
        self, 
        node,  # PACNode
        metadata: Optional[Dict[str, Any]] = None,
        crystallized: bool = False,
        importance: float = 0.0
    ) -> str:
        """
        Save a PACNode to persistent storage.
        
        Args:
            node: PACNode to save
            metadata: Additional metadata
            crystallized: Mark as crystallized pattern
            importance: Importance score for crystallized patterns
        
        Returns:
            doc_id
        """
        # Save to disk
        meta = metadata or {}
        meta["crystallized"] = crystallized
        meta["importance"] = importance
        
        doc_id = self.serializer.save(node, **meta)
        
        # Update indices
        self.temporal_index.add(
            doc_id=doc_id,
            node_id=node.id,
            metadata={"crystallized": crystallized, "importance": importance}
        )
        
        # Update node mapping
        self._node_to_doc[node.id] = doc_id
        self._doc_to_node[doc_id] = node.id
        
        # Update crystallized index
        if crystallized:
            self._crystallized[doc_id] = importance
            self._save_crystallized_index()
        
        return doc_id
    
    def load_node(self, node_id: int):  # -> Optional[PACNode]
        """Load a PACNode by its ID."""
        doc_id = self._node_to_doc.get(node_id)
        if not doc_id:
            return None
        
        return self.serializer.load(doc_id)
    
    def load_node_by_doc(self, doc_id: str):  # -> PACNode
        """Load a PACNode by its doc_id."""
        return self.serializer.load(doc_id)
    
    def delete_node(self, node_id: int) -> bool:
        """Delete a node from storage."""
        doc_id = self._node_to_doc.get(node_id)
        if not doc_id:
            return False
        
        # Remove from serializer
        self.serializer.delete(doc_id)
        
        # Remove from temporal index
        self.temporal_index.remove(doc_id)
        
        # Remove from crystallized index
        if doc_id in self._crystallized:
            del self._crystallized[doc_id]
            self._save_crystallized_index()
        
        # Remove from mappings
        del self._node_to_doc[node_id]
        del self._doc_to_node[doc_id]
        
        return True
    
    def node_exists(self, node_id: int) -> bool:
        """Check if node exists in storage."""
        return node_id in self._node_to_doc
    
    # === Batch Operations ===
    
    def save_nodes(
        self, 
        nodes: list,  # List[PACNode]
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Save multiple nodes efficiently."""
        doc_ids = []
        for node in nodes:
            doc_id = self.save_node(node, metadata)
            doc_ids.append(doc_id)
        return doc_ids
    
    def load_nodes(self, node_ids: List[int]) -> list:  # List[PACNode]
        """Load multiple nodes by ID."""
        nodes = []
        for node_id in node_ids:
            node = self.load_node(node_id)
            if node:
                nodes.append(node)
        return nodes
    
    # === Query Operations ===
    
    def query_temporal(
        self, 
        start: datetime, 
        end: datetime
    ) -> List[int]:
        """Find nodes created in time range."""
        results = self.temporal_index.query_range(start, end)
        return [node_id for _, node_id in results]
    
    def query_recent(self, count: int = 10) -> List[int]:
        """Get N most recently saved nodes."""
        results = self.temporal_index.query_recent(count)
        return [node_id for _, node_id in results]
    
    def query_crystallized(
        self, 
        min_importance: float = 0.0
    ) -> List[int]:
        """Find crystallized patterns by importance threshold."""
        node_ids = []
        for doc_id, importance in self._crystallized.items():
            if importance >= min_importance:
                node_id = self._doc_to_node.get(doc_id)
                if node_id is not None:
                    node_ids.append(node_id)
        return node_ids
    
    def list_all_nodes(self) -> List[int]:
        """List all stored node IDs."""
        return list(self._node_to_doc.keys())
    
    # === Episode Operations ===
    
    def save_episode(
        self,
        nodes: list,  # List[PACNode]
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> str:
        """
        Save a sequence of nodes as an episode.
        
        Use for:
        - Checkpointing learning sessions
        - Saving complete substrate state
        - Recording important moments
        
        Returns:
            episode_id
        """
        # Create episode
        episode_id = self.episode_tracker.create_episode(name, metadata)
        
        # Add all nodes
        self.episode_tracker.add_nodes_batch(episode_id, nodes, metadata)
        
        # Finalize with summary
        self.episode_tracker.finalize_episode(episode_id, {
            "node_count": len(nodes),
            "total_energy": sum(getattr(n, 'energy', 0) for n in nodes),
            "crystallized_count": sum(
                1 for n in nodes 
                if self._node_to_doc.get(n.id, "") in self._crystallized
            )
        })
        
        return episode_id
    
    def load_episode(
        self, 
        episode_id: str,
        device: Optional[str] = None
    ) -> Tuple[list, Dict[str, Any]]:  # Tuple[List[PACNode], Dict]
        """
        Load complete episode with nodes and metadata.
        
        Returns:
            (nodes, metadata)
        """
        device = device or self.device
        return self.episode_tracker.load_episode(episode_id, device)
    
    def list_episodes(self) -> List[str]:
        """List all episode IDs."""
        return self.episode_tracker.list_episodes()
    
    def list_episodes_detailed(self) -> List[Dict[str, Any]]:
        """List all episodes with metadata."""
        return self.episode_tracker.list_episodes_detailed()
    
    def delete_episode(self, episode_id: str) -> bool:
        """Delete an episode."""
        return self.episode_tracker.delete_episode(episode_id)
    
    # === Maintenance ===
    
    def rebuild_indices(self):
        """Rebuild all indices from disk snapshots."""
        # Clear current indices
        self.temporal_index.clear()
        self._crystallized.clear()
        self._node_to_doc.clear()
        self._doc_to_node.clear()
        
        # Scan all FDO files
        import yaml
        for yaml_path in self.snapshots_dir.glob("*.fdo.yaml"):
            try:
                with open(yaml_path, 'r') as f:
                    fdo = yaml.safe_load(f)
                
                doc_id = fdo.get("doc_id")
                node_id = fdo.get("node", {}).get("id")
                metadata = fdo.get("metadata", {})
                
                if doc_id and node_id is not None:
                    # Update mappings
                    self._node_to_doc[node_id] = doc_id
                    self._doc_to_node[doc_id] = node_id
                    
                    # Update temporal index
                    created = metadata.get("created")
                    if created:
                        self.temporal_index.add(
                            doc_id=doc_id,
                            node_id=node_id,
                            timestamp=datetime.fromisoformat(created),
                            metadata=metadata
                        )
                    
                    # Update crystallized index
                    if metadata.get("crystallized"):
                        self._crystallized[doc_id] = metadata.get("importance", 0.0)
                        
            except Exception as e:
                print(f"Warning: Failed to index {yaml_path}: {e}")
        
        self._save_crystallized_index()
    
    def compact(
        self, 
        keep_crystallized: bool = True,
        min_importance: float = 0.0
    ) -> int:
        """
        Remove old non-important patterns to save space.
        
        Args:
            keep_crystallized: Keep all crystallized patterns
            min_importance: Keep patterns with importance >= this
        
        Returns:
            Number of patterns removed
        """
        removed = 0
        
        for doc_id in list(self.serializer.list_all()):
            # Check if crystallized
            if keep_crystallized and doc_id in self._crystallized:
                if self._crystallized[doc_id] >= min_importance:
                    continue
            
            # Remove
            node_id = self._doc_to_node.get(doc_id)
            if node_id is not None:
                self.delete_node(node_id)
                removed += 1
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "namespace": self.namespace,
            "total_nodes": len(self._node_to_doc),
            "crystallized_count": len(self._crystallized),
            "episode_count": len(self.list_episodes()),
            "temporal_entries": len(self.temporal_index),
            "storage_path": str(self.namespace_dir),
        }
    
    # === Internal ===
    
    def _load_crystallized_index(self):
        """Load crystallized index from disk."""
        if self._crystallized_index_path.exists():
            try:
                with open(self._crystallized_index_path, 'r') as f:
                    self._crystallized = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._crystallized = {}
    
    def _save_crystallized_index(self):
        """Save crystallized index to disk."""
        with open(self._crystallized_index_path, 'w') as f:
            json.dump(self._crystallized, f, indent=2)
    
    def _rebuild_node_mapping(self):
        """Rebuild node ID to doc_id mapping from disk."""
        import yaml
        
        for yaml_path in self.snapshots_dir.glob("*.fdo.yaml"):
            try:
                with open(yaml_path, 'r') as f:
                    fdo = yaml.safe_load(f)
                
                doc_id = fdo.get("doc_id")
                node_id = fdo.get("node", {}).get("id")
                
                if doc_id and node_id is not None:
                    self._node_to_doc[node_id] = doc_id
                    self._doc_to_node[doc_id] = node_id
                    
            except Exception:
                pass
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"KronosBackend(namespace='{self.namespace}', "
            f"nodes={stats['total_nodes']}, "
            f"crystallized={stats['crystallized_count']}, "
            f"episodes={stats['episode_count']})"
        )
