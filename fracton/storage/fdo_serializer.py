"""
FDO Serializer - PAC Delta ↔ FDO Format Conversion
===================================================

Converts between PACNode objects and FDO (Field Differential Object) format
for persistent storage.

FDO v2.0 Schema:
- YAML metadata file: {node_id}.fdo.yaml
- NumPy delta file: {node_id}_delta.npy (separate for efficiency)
"""

import yaml
import numpy as np
import torch
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import asdict

# Import PACNode (lazy to avoid circular imports)
def _get_pac_node_class():
    from fracton.core.pac_node import PACNode
    return PACNode


FDO_SCHEMA_VERSION = "fdo-v2.0"


def serialize_node(
    node,  # PACNode
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Serialize a PACNode to FDO format.
    
    Returns:
        (yaml_dict, delta_array) - Metadata dict and numpy array
    """
    # Generate doc ID from node ID
    doc_id = f"{node.id:016x}"
    
    # Get phase as string (PhaseState enum)
    phase_value = node.phase.value if hasattr(node.phase, 'value') else str(node.phase)
    
    # Build YAML structure
    fdo = {
        "schema_version": FDO_SCHEMA_VERSION,
        "doc_type": "pac_node",
        "doc_id": doc_id,
        
        # PAC structure
        "node": {
            "id": node.id,
            "parent_id": node.parent_id,
            "children_ids": list(node.children_ids) if node.children_ids else [],
            "label": node.label,
        },
        
        # Physics state
        "physics": {
            "potential": float(node.potential),
            "phase": phase_value,
        },
        
        # Metadata
        "metadata": {
            "created": datetime.fromtimestamp(node.created_at).isoformat() if node.created_at else datetime.now().isoformat(),
            **(metadata or {})
        },
        
        # Reference to delta file
        "delta_file": f"{doc_id}_delta.npy",
    }
    
    # Convert delta tensor to numpy
    if node.delta is not None:
        if isinstance(node.delta, torch.Tensor):
            delta_array = node.delta.cpu().numpy()
        else:
            delta_array = np.array(node.delta)
    else:
        delta_array = np.array([])
    
    return fdo, delta_array


def deserialize_node(
    fdo: Dict[str, Any],
    delta_array: np.ndarray,
    device: str = 'cpu'
):  # -> PACNode
    """
    Deserialize FDO format back to PACNode.
    
    Args:
        fdo: YAML dict from .fdo.yaml file
        delta_array: NumPy array from _delta.npy file
        device: Target device for tensor
    
    Returns:
        PACNode instance
    """
    PACNode = _get_pac_node_class()
    from fracton.physics.phase_transitions import PhaseState
    
    # Convert numpy to torch tensor
    if len(delta_array) > 0:
        delta = torch.from_numpy(delta_array).to(device)
    else:
        delta = torch.tensor([], device=device)
    
    # Build node
    node_data = fdo.get("node", {})
    physics_data = fdo.get("physics", {})
    metadata = fdo.get("metadata", {})
    
    # Parse phase
    phase_str = physics_data.get("phase", "STABLE")
    try:
        phase = PhaseState[phase_str] if isinstance(phase_str, str) else PhaseState.STABLE
    except KeyError:
        phase = PhaseState.STABLE
    
    # Parse created timestamp
    created_str = metadata.get("created")
    if created_str:
        try:
            created_at = datetime.fromisoformat(created_str).timestamp()
        except:
            created_at = time.time()
    else:
        created_at = time.time()
    
    node = PACNode(
        id=node_data.get("id", 0),
        delta=delta,
        potential=physics_data.get("potential", 1.0),
        parent_id=node_data.get("parent_id", -1),
        children_ids=list(node_data.get("children_ids", [])),
        label=node_data.get("label", ""),
        created_at=created_at,
        phase=phase,
    )
    
    return node


def save_fdo(
    node,  # PACNode
    directory: Path,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save PACNode to FDO files in directory.
    
    Creates:
        - {doc_id}.fdo.yaml
        - {doc_id}_delta.npy
    
    Returns:
        doc_id
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    fdo, delta_array = serialize_node(node, metadata)
    doc_id = fdo["doc_id"]
    
    # Save YAML
    yaml_path = directory / f"{doc_id}.fdo.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(fdo, f, default_flow_style=False, sort_keys=False)
    
    # Save delta array
    npy_path = directory / f"{doc_id}_delta.npy"
    np.save(npy_path, delta_array)
    
    return doc_id


def load_fdo(
    doc_id: str,
    directory: Path,
    device: str = 'cpu'
):  # -> PACNode
    """
    Load PACNode from FDO files.
    
    Args:
        doc_id: Document ID (hex string)
        directory: Directory containing FDO files
        device: Target device for tensor
    
    Returns:
        PACNode instance
    """
    directory = Path(directory)
    
    # Load YAML
    yaml_path = directory / f"{doc_id}.fdo.yaml"
    with open(yaml_path, 'r') as f:
        fdo = yaml.safe_load(f)
    
    # Load delta array
    npy_path = directory / fdo.get("delta_file", f"{doc_id}_delta.npy")
    delta_array = np.load(npy_path)
    
    return deserialize_node(fdo, delta_array, device)


def list_fdo_files(directory: Path) -> list:
    """List all doc_ids in a directory."""
    directory = Path(directory)
    if not directory.exists():
        return []
    
    doc_ids = []
    for path in directory.glob("*.fdo.yaml"):
        # Extract doc_id from filename
        doc_id = path.stem.replace(".fdo", "")
        doc_ids.append(doc_id)
    
    return doc_ids


class FDOSerializer:
    """
    High-level FDO serialization interface.
    
    Example:
        serializer = FDOSerializer(Path("./data"), device='cuda')
        
        # Save node
        doc_id = serializer.save(node, importance=0.9)
        
        # Load node
        node = serializer.load(doc_id)
        
        # List all
        doc_ids = serializer.list_all()
    """
    
    def __init__(self, directory: Path, device: str = 'cpu'):
        self.directory = Path(directory)
        self.device = device
        self.directory.mkdir(parents=True, exist_ok=True)
    
    def save(
        self, 
        node,  # PACNode
        **metadata
    ) -> str:
        """Save node with optional metadata."""
        return save_fdo(node, self.directory, metadata)
    
    def load(self, doc_id: str):  # -> PACNode
        """Load node by doc_id."""
        return load_fdo(doc_id, self.directory, self.device)
    
    def load_all(self) -> list:
        """Load all nodes in directory."""
        return [self.load(doc_id) for doc_id in self.list_all()]
    
    def list_all(self) -> list:
        """List all doc_ids."""
        return list_fdo_files(self.directory)
    
    def delete(self, doc_id: str) -> bool:
        """Delete FDO files for doc_id."""
        yaml_path = self.directory / f"{doc_id}.fdo.yaml"
        npy_path = self.directory / f"{doc_id}_delta.npy"
        
        deleted = False
        if yaml_path.exists():
            yaml_path.unlink()
            deleted = True
        if npy_path.exists():
            npy_path.unlink()
            deleted = True
        
        return deleted
