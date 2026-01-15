"""
KronosEdge - Directional relationships in genealogy tree.

Edges encode temporal/causal/hierarchical relationships with inherent
directionality. Not arbitrary similarity scores, but meaningful flow.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List


class RelationType(Enum):
    """Directional relationship types with semantic meaning."""

    # Temporal/Causal - What came before/after
    PRECEDES = "precedes"  # A came before B temporally
    ADVANCES_FROM = "advances_from"  # B is an advancement of A

    # Hierarchical - General/specific
    GENERALIZES = "generalizes"  # A is more general than B
    SPECIALIZES = "specializes"  # B is more specific than A

    # Functional - Enablement
    ENABLES = "enables"  # A makes B possible
    IS_ENABLED_BY = "is_enabled_by"  # B is made possible by A

    # Epistemic - Knowledge relationships
    SUPPORTS = "supports"  # A provides evidence for B
    CONTRADICTS = "contradicts"  # A contradicts B
    EXTENDS = "extends"  # B extends/builds on A
    CONFLUENCE = "confluence"  # A and B merge into C (parent potentials)

    # Structural - Tree relationships
    PARENT_OF = "parent_of"  # A is parent of B
    CHILD_OF = "child_of"  # B is child of A
    SIBLING_OF = "sibling_of"  # A and B share parents

    def inverse(self) -> "RelationType":
        """Get inverse relationship type."""
        inverses = {
            RelationType.PRECEDES: RelationType.ADVANCES_FROM,
            RelationType.ADVANCES_FROM: RelationType.PRECEDES,
            RelationType.GENERALIZES: RelationType.SPECIALIZES,
            RelationType.SPECIALIZES: RelationType.GENERALIZES,
            RelationType.ENABLES: RelationType.IS_ENABLED_BY,
            RelationType.IS_ENABLED_BY: RelationType.ENABLES,
            RelationType.PARENT_OF: RelationType.CHILD_OF,
            RelationType.CHILD_OF: RelationType.PARENT_OF,
            # Self-inverse types
            RelationType.SUPPORTS: RelationType.SUPPORTS,
            RelationType.CONTRADICTS: RelationType.CONTRADICTS,
            RelationType.EXTENDS: RelationType.EXTENDS,
            RelationType.CONFLUENCE: RelationType.CONFLUENCE,
            RelationType.SIBLING_OF: RelationType.SIBLING_OF,
        }
        return inverses.get(self, self)

    @property
    def is_hierarchical(self) -> bool:
        """Is this a hierarchical relationship?"""
        return self in (
            RelationType.GENERALIZES,
            RelationType.SPECIALIZES,
            RelationType.PARENT_OF,
            RelationType.CHILD_OF,
        )

    @property
    def is_temporal(self) -> bool:
        """Is this a temporal relationship?"""
        return self in (RelationType.PRECEDES, RelationType.ADVANCES_FROM)

    @property
    def is_functional(self) -> bool:
        """Is this a functional relationship?"""
        return self in (RelationType.ENABLES, RelationType.IS_ENABLED_BY)

    @property
    def is_epistemic(self) -> bool:
        """Is this an epistemic relationship?"""
        return self in (
            RelationType.SUPPORTS,
            RelationType.CONTRADICTS,
            RelationType.EXTENDS,
        )


@dataclass
class KronosEdge:
    """
    Directional relationship between concepts.

    Unlike similarity scores, these edges encode meaningful flow:
    temporal, causal, hierarchical, or epistemic relationships.
    """

    source_id: str  # Concept this relationship originates from
    target_id: str  # Concept this relationship points to

    # Relationship type with inherent directionality
    relationship_type: RelationType

    # Strength and evidence
    strength: float = 1.0  # 0.0 to 1.0 (how strong is this relationship?)
    evidence_count: int = 0  # Number of supporting documents
    supporting_documents: List[str] = field(default_factory=list)  # doc_ids

    # Temporal markers
    established_at: datetime = field(default_factory=datetime.now)
    last_validated: datetime = field(default_factory=datetime.now)

    # Metadata
    notes: str = ""  # Optional context about this relationship

    def __post_init__(self):
        """Validate edge construction."""
        # Validate IDs
        if not self.source_id or not self.target_id:
            raise ValueError("Source and target IDs are required")

        if self.source_id == self.target_id:
            raise ValueError("Self-loops not allowed (source == target)")

        # Validate strength
        if not 0 <= self.strength <= 1:
            raise ValueError(f"Strength must be 0-1, got {self.strength}")

        # Validate evidence count matches documents
        if self.supporting_documents:
            self.evidence_count = len(self.supporting_documents)

    def inverse_edge(self) -> "KronosEdge":
        """Create inverse edge (swap source/target, inverse relationship)."""
        return KronosEdge(
            source_id=self.target_id,
            target_id=self.source_id,
            relationship_type=self.relationship_type.inverse(),
            strength=self.strength,
            evidence_count=self.evidence_count,
            supporting_documents=self.supporting_documents.copy(),
            established_at=self.established_at,
            last_validated=self.last_validated,
            notes=f"Inverse of: {self.notes}" if self.notes else "",
        )

    def add_supporting_document(self, doc_id: str) -> None:
        """Add supporting document."""
        if doc_id not in self.supporting_documents:
            self.supporting_documents.append(doc_id)
            self.evidence_count = len(self.supporting_documents)

    def validate(self) -> None:
        """Mark as validated (update timestamp)."""
        self.last_validated = datetime.now()

    def strengthen(self, amount: float = 0.1) -> None:
        """Increase strength by amount."""
        self.strength = min(1.0, self.strength + amount)
        self.validate()

    def weaken(self, amount: float = 0.1) -> None:
        """Decrease strength by amount."""
        self.strength = max(0.0, self.strength - amount)
        self.validate()

    @property
    def is_strong(self) -> bool:
        """Is this a strong relationship (strength > 0.7)?"""
        return self.strength > 0.7

    @property
    def is_weak(self) -> bool:
        """Is this a weak relationship (strength < 0.3)?"""
        return self.strength < 0.3

    @property
    def is_well_supported(self) -> bool:
        """Is this well-supported by evidence (3+ documents)?"""
        return self.evidence_count >= 3

    @property
    def age_days(self) -> float:
        """How many days since this edge was established?"""
        return (datetime.now() - self.established_at).total_seconds() / 86400

    @property
    def days_since_validation(self) -> float:
        """How many days since last validation?"""
        return (datetime.now() - self.last_validated).total_seconds() / 86400

    def needs_revalidation(self, threshold_days: int = 90) -> bool:
        """Does this edge need revalidation?"""
        return self.days_since_validation > threshold_days

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type.value,
            "strength": self.strength,
            "evidence_count": self.evidence_count,
            "supporting_documents": self.supporting_documents.copy(),
            "established_at": self.established_at.isoformat(),
            "last_validated": self.last_validated.isoformat(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KronosEdge":
        """Create edge from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relationship_type=RelationType(data["relationship_type"]),
            strength=data.get("strength", 1.0),
            evidence_count=data.get("evidence_count", 0),
            supporting_documents=data.get("supporting_documents", []),
            established_at=datetime.fromisoformat(data["established_at"]),
            last_validated=datetime.fromisoformat(data["last_validated"]),
            notes=data.get("notes", ""),
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"KronosEdge({self.source_id} --[{self.relationship_type.value}]"
            f"({self.strength:.2f})--> {self.target_id})"
        )

    def __eq__(self, other) -> bool:
        """Edges are equal if they connect same nodes with same relationship."""
        if not isinstance(other, KronosEdge):
            return False
        return (
            self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.relationship_type == other.relationship_type
        )

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash((self.source_id, self.target_id, self.relationship_type))
