"""
Episode Tracker - Field Evolution Sequences
============================================

Tracks episodes (sequences of PAC node snapshots) for:
- Recording learning sessions
- Time-travel debugging
- Analyzing pattern evolution

Episode Structure:
    {episode_id}/
    ├── episode.yaml        # Episode metadata
    └── nodes/              # Node snapshots
        ├── {doc_id}.fdo.yaml
        └── {doc_id}_delta.npy
"""

import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import uuid


class EpisodeTracker:
    """
    Manages episodes (sequences of PAC snapshots).
    
    An episode captures a coherent sequence of field evolution,
    useful for:
    - Saving/restoring learning sessions
    - Debugging consciousness evolution
    - Analyzing pattern formation over time
    """
    
    def __init__(self, episodes_dir: Path):
        self.episodes_dir = Path(episodes_dir)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
    
    def create_episode(
        self,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new episode.
        
        Returns:
            episode_id
        """
        # Generate episode ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        episode_id = f"episode_{timestamp}_{short_uuid}"
        
        # Create directory structure
        episode_dir = self.episodes_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "nodes").mkdir(exist_ok=True)
        
        # Create episode metadata
        episode_meta = {
            "episode_id": episode_id,
            "name": name or episode_id,
            "created": datetime.now().isoformat(),
            "status": "active",
            "node_count": 0,
            "metadata": metadata or {}
        }
        
        with open(episode_dir / "episode.yaml", 'w') as f:
            yaml.dump(episode_meta, f, default_flow_style=False)
        
        return episode_id
    
    def add_node_to_episode(
        self,
        episode_id: str,
        node,  # PACNode
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a node snapshot to an episode.
        
        Returns:
            doc_id
        """
        from .fdo_serializer import save_fdo
        
        episode_dir = self.episodes_dir / episode_id
        nodes_dir = episode_dir / "nodes"
        
        if not nodes_dir.exists():
            raise ValueError(f"Episode {episode_id} not found")
        
        # Save node to episode
        doc_id = save_fdo(node, nodes_dir, metadata)
        
        # Update episode metadata
        self._increment_node_count(episode_id)
        
        return doc_id
    
    def add_nodes_batch(
        self,
        episode_id: str,
        nodes: list,  # List[PACNode]
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Add multiple nodes to episode efficiently."""
        from .fdo_serializer import save_fdo
        
        episode_dir = self.episodes_dir / episode_id
        nodes_dir = episode_dir / "nodes"
        
        if not nodes_dir.exists():
            raise ValueError(f"Episode {episode_id} not found")
        
        doc_ids = []
        for node in nodes:
            doc_id = save_fdo(node, nodes_dir, metadata)
            doc_ids.append(doc_id)
        
        # Update count
        self._update_node_count(episode_id, len(nodes))
        
        return doc_ids
    
    def finalize_episode(
        self,
        episode_id: str,
        summary: Optional[Dict[str, Any]] = None
    ):
        """
        Mark episode as complete with summary.
        """
        episode_dir = self.episodes_dir / episode_id
        meta_path = episode_dir / "episode.yaml"
        
        if not meta_path.exists():
            raise ValueError(f"Episode {episode_id} not found")
        
        with open(meta_path, 'r') as f:
            meta = yaml.safe_load(f)
        
        meta["status"] = "complete"
        meta["completed"] = datetime.now().isoformat()
        if summary:
            meta["summary"] = summary
        
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f, default_flow_style=False)
    
    def load_episode(
        self,
        episode_id: str,
        device: str = 'cpu'
    ) -> Tuple[List, Dict[str, Any]]:
        """
        Load all nodes from an episode.
        
        Returns:
            (nodes, metadata)
        """
        from .fdo_serializer import load_fdo, list_fdo_files
        
        episode_dir = self.episodes_dir / episode_id
        nodes_dir = episode_dir / "nodes"
        meta_path = episode_dir / "episode.yaml"
        
        if not meta_path.exists():
            raise ValueError(f"Episode {episode_id} not found")
        
        # Load metadata
        with open(meta_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        # Load all nodes
        doc_ids = list_fdo_files(nodes_dir)
        nodes = [load_fdo(doc_id, nodes_dir, device) for doc_id in doc_ids]
        
        return nodes, metadata
    
    def get_episode_metadata(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Get episode metadata without loading nodes."""
        meta_path = self.episodes_dir / episode_id / "episode.yaml"
        
        if not meta_path.exists():
            return None
        
        with open(meta_path, 'r') as f:
            return yaml.safe_load(f)
    
    def list_episodes(self) -> List[str]:
        """List all episode IDs."""
        episodes = []
        for path in self.episodes_dir.iterdir():
            if path.is_dir() and (path / "episode.yaml").exists():
                episodes.append(path.name)
        return sorted(episodes)
    
    def list_episodes_detailed(self) -> List[Dict[str, Any]]:
        """List all episodes with metadata."""
        episodes = []
        for episode_id in self.list_episodes():
            meta = self.get_episode_metadata(episode_id)
            if meta:
                episodes.append(meta)
        return sorted(episodes, key=lambda e: e.get("created", ""), reverse=True)
    
    def delete_episode(self, episode_id: str) -> bool:
        """Delete an episode and all its nodes."""
        episode_dir = self.episodes_dir / episode_id
        
        if not episode_dir.exists():
            return False
        
        shutil.rmtree(episode_dir)
        return True
    
    def _increment_node_count(self, episode_id: str):
        """Increment node count in episode metadata."""
        self._update_node_count(episode_id, 1)
    
    def _update_node_count(self, episode_id: str, delta: int):
        """Update node count in episode metadata."""
        meta_path = self.episodes_dir / episode_id / "episode.yaml"
        
        with open(meta_path, 'r') as f:
            meta = yaml.safe_load(f)
        
        meta["node_count"] = meta.get("node_count", 0) + delta
        meta["last_updated"] = datetime.now().isoformat()
        
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f, default_flow_style=False)
