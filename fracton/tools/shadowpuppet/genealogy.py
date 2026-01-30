"""
ShadowPuppet Genealogy Tracking

Tracks the full provenance tree of generated components.
Every component knows its lineage - who generated it,
from what template, and how it evolved.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .protocols import ComponentOrganism


@dataclass
class GenealogyNode:
    """A node in the genealogy tree."""
    component_id: str
    protocol_name: str
    parent_id: Optional[str]
    generation: int
    coherence_score: float
    generator_used: str
    children: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class GenealogyTree:
    """
    Genealogy tree for tracking component evolution.
    
    Like Kronos provenance tracking, but for code:
    - Every component has recorded ancestry
    - Can trace derivation paths
    - Understand why code evolved the way it did
    
    Example:
        tree = GenealogyTree()
        tree.add(component)
        
        # Get lineage
        path = tree.get_derivation_path(component.id)
        print(f"Lineage: {' -> '.join(path)}")
        
        # Get children
        children = tree.get_children(parent.id)
    """
    
    def __init__(self):
        self.nodes: Dict[str, GenealogyNode] = {}
        self.roots: List[str] = []  # Components with no parent
    
    def add(self, component: ComponentOrganism):
        """Add component to genealogy tree."""
        node = GenealogyNode(
            component_id=component.id,
            protocol_name=component.protocol_name,
            parent_id=component.parent_id,
            generation=component.generation,
            coherence_score=component.coherence_score,
            generator_used=component.generator_used
        )
        
        self.nodes[component.id] = node
        
        # Track parent-child relationship
        if component.parent_id:
            if component.parent_id in self.nodes:
                self.nodes[component.parent_id].children.append(component.id)
        else:
            self.roots.append(component.id)
    
    def get_derivation_path(self, component_id: str) -> List[str]:
        """Get full derivation path from root to component."""
        path = []
        current_id = component_id
        
        while current_id:
            path.insert(0, current_id)
            node = self.nodes.get(current_id)
            if node:
                current_id = node.parent_id
            else:
                break
        
        return path
    
    def get_children(self, component_id: str) -> List[str]:
        """Get direct children of a component."""
        node = self.nodes.get(component_id)
        if node:
            return node.children
        return []
    
    def get_descendants(self, component_id: str) -> List[str]:
        """Get all descendants (recursive)."""
        descendants = []
        
        def collect(cid):
            for child_id in self.get_children(cid):
                descendants.append(child_id)
                collect(child_id)
        
        collect(component_id)
        return descendants
    
    def get_siblings(self, component_id: str) -> List[str]:
        """Get siblings (same parent)."""
        node = self.nodes.get(component_id)
        if not node or not node.parent_id:
            return []
        
        parent_node = self.nodes.get(node.parent_id)
        if not parent_node:
            return []
        
        return [c for c in parent_node.children if c != component_id]
    
    def get_generation(self, gen: int) -> List[str]:
        """Get all components from a specific generation."""
        return [
            cid for cid, node in self.nodes.items()
            if node.generation == gen
        ]
    
    def get_by_protocol(self, protocol_name: str) -> List[str]:
        """Get all components implementing a protocol."""
        return [
            cid for cid, node in self.nodes.items()
            if node.protocol_name == protocol_name
        ]
    
    def summary(self) -> Dict[str, Any]:
        """Get tree summary statistics."""
        if not self.nodes:
            return {'total': 0}
        
        generations = {}
        protocols = {}
        generators = {}
        
        for node in self.nodes.values():
            generations[node.generation] = generations.get(node.generation, 0) + 1
            protocols[node.protocol_name] = protocols.get(node.protocol_name, 0) + 1
            generators[node.generator_used] = generators.get(node.generator_used, 0) + 1
        
        # Calculate tree depth
        max_depth = 0
        for cid in self.nodes.keys():
            path = self.get_derivation_path(cid)
            max_depth = max(max_depth, len(path))
        
        return {
            'total': len(self.nodes),
            'roots': len(self.roots),
            'max_depth': max_depth,
            'generations': generations,
            'protocols': protocols,
            'generators': generators
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tree to dictionary."""
        return {
            'nodes': {
                cid: {
                    'protocol_name': node.protocol_name,
                    'parent_id': node.parent_id,
                    'generation': node.generation,
                    'coherence_score': node.coherence_score,
                    'generator_used': node.generator_used,
                    'children': node.children,
                    'timestamp': node.timestamp
                }
                for cid, node in self.nodes.items()
            },
            'roots': self.roots,
            'summary': self.summary()
        }
    
    def print_tree(self, component_id: Optional[str] = None, indent: int = 0):
        """Print tree visualization."""
        if component_id is None:
            # Print from roots
            for root_id in self.roots:
                self.print_tree(root_id, indent)
            return
        
        node = self.nodes.get(component_id)
        if not node:
            return
        
        prefix = "  " * indent
        score_str = f"{node.coherence_score:.3f}" if node.coherence_score else "-.---"
        print(f"{prefix}├─ {node.protocol_name} ({component_id}) [{score_str}]")
        
        for child_id in node.children:
            self.print_tree(child_id, indent + 1)
