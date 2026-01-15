"""
KronosNode - Conceptual identity in genealogy tree.

A node represents a crystallized concept where identity IS the confluence
pattern from parent potentials. Not just a label with links, but the
actual pattern of how this concept emerged.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DocumentReference:
    """Reference to supporting evidence."""

    doc_id: str
    title: str
    authors: List[str]
    year: int
    doi: Optional[str] = None
    uri: str = ""
    excerpt: str = ""  # Relevant portion that supports this concept

    def __post_init__(self):
        """Validate document reference."""
        if not self.title:
            raise ValueError("Document must have a title")
        if self.year < 1600 or self.year > datetime.now().year + 1:
            raise ValueError(f"Invalid year: {self.year}")


@dataclass
class CrystallizationEvent:
    """Record of when/how concept crystallized."""

    timestamp: datetime
    document: DocumentReference
    context: str  # What prompted this crystallization
    confidence: float  # How clear was the crystallization (0-1)

    def __post_init__(self):
        """Validate crystallization event."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")


@dataclass
class KronosNode:
    """
    A crystallized concept in the knowledge graph.

    Identity IS the confluence pattern - the weighted combination of
    parent potentials that this concept crystallized from.

    PAC Conservation:
    - Conceptual: this_meaning = Σ (parent_meaning * confluence_weight)
    - Storage: full_embedding = parent.embedding + self.delta_embedding
    """

    # === Identity (what this concept IS) ===
    id: str  # Stable identifier (e.g., "quantum_entanglement")
    name: str  # Human-readable (e.g., "Quantum Entanglement")
    definition: str  # Full text definition

    # === Confluence Pattern (how this crystallized) ===
    confluence_pattern: Dict[str, float] = field(default_factory=dict)
    # parent_id → weight (must sum to ~1.0)
    # Example: {"superposition": 0.4, "nonlocality": 0.35, "measurement": 0.25}

    parent_potentials: List[str] = field(default_factory=list)
    # Concepts this emerged from (sorted by confluence weight, descending)

    child_actualizations: List[str] = field(default_factory=list)
    # Concepts that emerged from this

    sibling_nodes: List[str] = field(default_factory=list)
    # Alternative actualizations from same parent potentials

    # === Lineage (full derivation history) ===
    derivation_path: List[str] = field(default_factory=list)
    # Full path from root to this node
    # Example: ["physics", "quantum_mechanics", "quantum_foundations", "EPR_paradox"]

    actualization_depth: int = 0
    # How many levels from root potential (0 = root)

    # === Evidence (grounding in sources) ===
    supported_by: List[DocumentReference] = field(default_factory=list)
    # Papers, sources that document this concept

    first_crystallization: Optional[datetime] = None
    # When concept first emerged historically

    crystallization_events: List[CrystallizationEvent] = field(default_factory=list)
    # Historical trail of how concept developed

    # === Embeddings (dual representation, delta-only) ===
    delta_embedding: Optional[np.ndarray] = None  # Δ from parent (semantic)
    delta_structural: Optional[np.ndarray] = None  # Δ from parent (structural)

    # Full embeddings reconstructed on demand:
    # semantic_embedding = parent.semantic + self.delta_embedding
    # structural_embedding = parent.structural + self.delta_structural

    # === Metadata ===
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def __post_init__(self):
        """Validate node construction."""
        # Validate ID format
        if not self.id or not self.id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid node ID: {self.id}")

        # Validate confluence pattern
        if self.confluence_pattern:
            total_weight = sum(self.confluence_pattern.values())
            if not (0.95 <= total_weight <= 1.05):  # Allow small floating point error
                raise ValueError(
                    f"Confluence pattern must sum to ~1.0, got {total_weight}"
                )

            # Ensure all parents in pattern are in parent_potentials list
            for parent_id in self.confluence_pattern.keys():
                if parent_id not in self.parent_potentials:
                    self.parent_potentials.append(parent_id)

        # Sort parent_potentials by confluence weight (highest first)
        if self.confluence_pattern and self.parent_potentials:
            self.parent_potentials.sort(
                key=lambda pid: self.confluence_pattern.get(pid, 0.0),
                reverse=True,
            )

        # Validate depth matches derivation path
        if self.derivation_path and self.actualization_depth != len(
            self.derivation_path
        ):
            self.actualization_depth = len(self.derivation_path)

    @property
    def is_root(self) -> bool:
        """Is this a root node (no parents)?"""
        return not self.parent_potentials

    @property
    def is_leaf(self) -> bool:
        """Is this a leaf node (no children)?"""
        return not self.child_actualizations

    @property
    def primary_parent(self) -> Optional[str]:
        """Get primary parent (highest confluence weight)."""
        if not self.confluence_pattern:
            return self.parent_potentials[0] if self.parent_potentials else None

        return max(self.confluence_pattern.items(), key=lambda x: x[1])[0]

    @property
    def documentation_depth(self) -> int:
        """Number of supporting documents."""
        return len(self.supported_by)

    def get_full_path_str(self) -> str:
        """Get derivation path as string.

        Example: "physics → quantum_mechanics → quantum_foundations → EPR_paradox"
        """
        return " → ".join(self.derivation_path) if self.derivation_path else self.id

    def add_child(self, child_id: str) -> None:
        """Add child actualization."""
        if child_id not in self.child_actualizations:
            self.child_actualizations.append(child_id)

    def add_sibling(self, sibling_id: str) -> None:
        """Add sibling node."""
        if sibling_id not in self.sibling_nodes and sibling_id != self.id:
            self.sibling_nodes.append(sibling_id)

    def add_supporting_document(self, doc: DocumentReference) -> None:
        """Add supporting evidence."""
        if doc not in self.supported_by:
            self.supported_by.append(doc)

    def record_crystallization(self, event: CrystallizationEvent) -> None:
        """Record when/how concept crystallized."""
        self.crystallization_events.append(event)

        # Update first_crystallization if this is earlier
        if (
            not self.first_crystallization
            or event.timestamp < self.first_crystallization
        ):
            self.first_crystallization = event.timestamp

    def record_access(self) -> None:
        """Record access for usage tracking."""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "definition": self.definition,
            "confluence_pattern": self.confluence_pattern,
            "parent_potentials": self.parent_potentials,
            "child_actualizations": self.child_actualizations,
            "sibling_nodes": self.sibling_nodes,
            "derivation_path": self.derivation_path,
            "actualization_depth": self.actualization_depth,
            "supported_by": [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "authors": doc.authors,
                    "year": doc.year,
                    "doi": doc.doi,
                    "uri": doc.uri,
                    "excerpt": doc.excerpt,
                }
                for doc in self.supported_by
            ],
            "first_crystallization": (
                self.first_crystallization.isoformat()
                if self.first_crystallization
                else None
            ),
            "crystallization_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "document": {
                        "doc_id": event.document.doc_id,
                        "title": event.document.title,
                        "authors": event.document.authors,
                        "year": event.document.year,
                        "doi": event.document.doi,
                        "uri": event.document.uri,
                        "excerpt": event.document.excerpt,
                    },
                    "context": event.context,
                    "confidence": event.confidence,
                }
                for event in self.crystallization_events
            ],
            "delta_embedding": (
                self.delta_embedding.tolist() if self.delta_embedding is not None else None
            ),
            "delta_structural": (
                self.delta_structural.tolist()
                if self.delta_structural is not None
                else None
            ),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KronosNode":
        """Create node from dictionary."""
        # Reconstruct supported_by
        supported_by = [
            DocumentReference(
                doc_id=doc["doc_id"],
                title=doc["title"],
                authors=doc["authors"],
                year=doc["year"],
                doi=doc.get("doi"),
                uri=doc.get("uri", ""),
                excerpt=doc.get("excerpt", ""),
            )
            for doc in data.get("supported_by", [])
        ]

        # Reconstruct crystallization_events
        crystallization_events = []
        for event in data.get("crystallization_events", []):
            doc_data = event["document"]
            crystallization_events.append(
                CrystallizationEvent(
                    timestamp=datetime.fromisoformat(event["timestamp"]),
                    document=DocumentReference(
                        doc_id=doc_data["doc_id"],
                        title=doc_data["title"],
                        authors=doc_data["authors"],
                        year=doc_data["year"],
                        doi=doc_data.get("doi"),
                        uri=doc_data.get("uri", ""),
                        excerpt=doc_data.get("excerpt", ""),
                    ),
                    context=event["context"],
                    confidence=event["confidence"],
                )
            )

        # Reconstruct embeddings
        delta_embedding = (
            np.array(data["delta_embedding"])
            if data.get("delta_embedding") is not None
            else None
        )
        delta_structural = (
            np.array(data["delta_structural"])
            if data.get("delta_structural") is not None
            else None
        )

        return cls(
            id=data["id"],
            name=data["name"],
            definition=data["definition"],
            confluence_pattern=data.get("confluence_pattern", {}),
            parent_potentials=data.get("parent_potentials", []),
            child_actualizations=data.get("child_actualizations", []),
            sibling_nodes=data.get("sibling_nodes", []),
            derivation_path=data.get("derivation_path", []),
            actualization_depth=data.get("actualization_depth", 0),
            supported_by=supported_by,
            first_crystallization=(
                datetime.fromisoformat(data["first_crystallization"])
                if data.get("first_crystallization")
                else None
            ),
            crystallization_events=crystallization_events,
            delta_embedding=delta_embedding,
            delta_structural=delta_structural,
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"KronosNode(id='{self.id}', name='{self.name}', "
            f"depth={self.actualization_depth}, "
            f"parents={len(self.parent_potentials)}, "
            f"children={len(self.child_actualizations)})"
        )
