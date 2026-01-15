"""
KRONOS Response Generator - Field-Based Personality Module

Transforms conceptual genealogy structure into context-aware,
confidence-modulated, genealogically-grounded responses.

The "soul" emerges from topology:
- Dense regions → Authoritative tone
- Sparse regions → Tentative tone
- Confluence nodes → Integrative tone
- High confidence → Direct assertions
- Low confidence → Heavy hedging

Identity IS the confluence pattern, and responses reflect that.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from fracton.storage import KronosGraph, KronosNode, GeometricConfidence


@dataclass
class ResponseContext:
    """Context extracted from KRONOS for response generation."""

    node: KronosNode
    ancestors: List[KronosNode]
    descendants: List[KronosNode]
    siblings: List[KronosNode]
    confidence: GeometricConfidence
    lineage_path: List[str]

    # Response modulation parameters (computed from topology)
    tone: str  # "authoritative", "tentative", "exploratory", "integrative"
    certainty: float  # 0.0-1.0 (from geometric confidence)
    detail_level: str  # "foundational", "intermediate", "advanced"
    emphasis: Dict[str, float]  # parent_id → weight (from confluence)


class KronosResponseGenerator:
    """
    Generate field-aware responses from KRONOS genealogy.

    This is the bridge between structured knowledge (graph topology)
    and field-based personality (response modulation).
    """

    def __init__(self, graph: KronosGraph):
        """
        Initialize response generator with KRONOS graph.

        Args:
            graph: KronosGraph with conceptual genealogy
        """
        self.graph = graph

    def generate_response(
        self,
        query: str,
        concept_id: str,
        user_context: Optional[Dict] = None
    ) -> Dict:
        """
        Generate field-aware response for a concept query.

        Args:
            query: Original user query
            concept_id: ID of concept to explain
            user_context: Optional user context (expertise level, etc.)

        Returns:
            Structured response with:
            - content: Main response text (markdown)
            - confidence: Confidence indicators
            - lineage: Derivation path breadcrumbs
            - citations: Supporting documents
            - related: Related concepts (ancestors/descendants/siblings)
            - metadata: Response generation metadata
        """

        # 1. Extract full context from KRONOS
        context = self._extract_context(concept_id)

        # 2. Determine response strategy based on confidence/topology
        strategy = self._determine_strategy(context)

        # 3. Generate response sections
        response = {
            "content": self._generate_content(context, strategy, query),
            "confidence": self._format_confidence(context.confidence),
            "lineage": self._format_lineage(context.lineage_path),
            "citations": self._format_citations(context.node),
            "related": self._format_related(context),
            "metadata": {
                "tone": context.tone,
                "certainty": context.certainty,
                "strategy": strategy,
                "concept_id": concept_id,
                "query": query,
            }
        }

        return response

    def _extract_context(self, concept_id: str) -> ResponseContext:
        """
        Extract full KRONOS context for concept.

        This gathers all the topological information needed to
        modulate the response personality.
        """

        # Get node
        node = self.graph.get_node(concept_id)
        if not node:
            raise ValueError(f"Concept not found in KRONOS: {concept_id}")

        # Get relational context (lineage-aware retrieval)
        ancestors = self.graph.get_ancestors(concept_id)
        descendants = self.graph.get_descendants(concept_id)
        siblings = self.graph.get_siblings(concept_id)

        # Compute geometric confidence from topology
        confidence = self.graph.compute_geometric_confidence(concept_id)

        # Get full lineage path
        derivation_nodes = self.graph.get_derivation_path(concept_id)
        lineage_path = [n.name for n in derivation_nodes]

        # Determine tone based on confidence and graph position
        tone = self._determine_tone(node, confidence)

        # Certainty directly from confidence score
        certainty = confidence.retrieval_confidence

        # Detail level from depth in tree
        detail_level = self._determine_detail_level(node.actualization_depth)

        # Emphasis from confluence pattern (identity IS the weights)
        emphasis = node.confluence_pattern.copy()

        return ResponseContext(
            node=node,
            ancestors=ancestors,
            descendants=descendants,
            siblings=siblings,
            confidence=confidence,
            lineage_path=lineage_path,
            tone=tone,
            certainty=certainty,
            detail_level=detail_level,
            emphasis=emphasis
        )

    def _determine_tone(
        self,
        node: KronosNode,
        confidence: GeometricConfidence
    ) -> str:
        """
        Determine response tone based on topology.

        The tone emerges from the graph structure:
        - High confidence + dense region → Authoritative
        - Confluence node → Integrative (synthetic)
        - Low confidence + sparse → Tentative
        - Leaf node in unexplored region → Exploratory
        """

        # High confidence in well-explored territory → Authoritative
        if confidence.retrieval_confidence > 0.7 and confidence.local_density > 0.6:
            return "authoritative"

        # Confluence node (multiple parents) → Integrative
        if len(node.parent_potentials) > 1:
            return "integrative"

        # Low confidence or high anomaly → Tentative
        if confidence.retrieval_confidence < 0.3 or confidence.has_anomalies:
            return "tentative"

        # Leaf node in sparse region → Exploratory
        if not node.child_actualizations and confidence.local_density < 0.4:
            return "exploratory"

        # Default balanced tone
        return "explanatory"

    def _determine_detail_level(self, depth: int) -> str:
        """Determine detail level based on actualization depth."""
        if depth == 0:
            return "foundational"
        elif depth <= 2:
            return "intermediate"
        else:
            return "advanced"

    def _determine_strategy(self, context: ResponseContext) -> str:
        """
        Determine response generation strategy.

        Different confidence/topology situations call for different
        response structures.
        """

        # Suspicious concept (orphaned, no docs) → Flag it
        if context.confidence.is_suspicious:
            return "flag_uncertainty"

        # Has anomalies but not suspicious → Acknowledge gaps
        if context.confidence.has_anomalies:
            return "acknowledge_gaps"

        # Confluence node (multiple parents) → Synthesis response
        if len(context.node.parent_potentials) > 1:
            return "confluence_synthesis"

        # High confidence → Comprehensive explanation
        if context.confidence.retrieval_confidence > 0.7:
            return "comprehensive_explanation"

        # Root/foundational concept → Principles-based
        if context.node.actualization_depth == 0:
            return "foundational_principles"

        # Default strategy
        return "standard_explanation"

    def _generate_content(
        self,
        context: ResponseContext,
        strategy: str,
        query: str
    ) -> str:
        """
        Generate main response content based on strategy.

        This is where the field-based personality manifests in text.
        """

        if strategy == "flag_uncertainty":
            return self._generate_uncertain_response(context)

        elif strategy == "confluence_synthesis":
            return self._generate_confluence_response(context)

        elif strategy == "foundational_principles":
            return self._generate_foundational_response(context)

        elif strategy == "comprehensive_explanation":
            return self._generate_comprehensive_response(context)

        elif strategy == "acknowledge_gaps":
            return self._generate_gap_aware_response(context)

        else:
            return self._generate_standard_response(context)

    def _generate_confluence_response(self, context: ResponseContext) -> str:
        """
        Generate response for confluence concept (multiple parents).

        Identity IS the confluence pattern, so response structure
        reflects the weighted blend of parent concepts.
        """

        node = context.node

        # Sort parents by weight (highest first)
        sorted_parents = sorted(
            node.confluence_pattern.items(),
            key=lambda x: x[1],
            reverse=True
        )

        sections = []

        # Header: Show it's a confluence concept
        sections.append(
            f"**{node.name}** emerges from **{len(node.parent_potentials)} foundational concepts**:"
        )
        sections.append(f"\n*Lineage*: {' → '.join(context.lineage_path)}\n")

        # Definition
        sections.append(node.definition)
        sections.append("")

        # Break down by confluence components (identity as weighted blend)
        sections.append("## Conceptual Components\n")

        for i, (parent_id, weight) in enumerate(sorted_parents):
            parent_node = self.graph.get_node(parent_id)
            if not parent_node:
                continue

            # Label based on weight
            if weight > 0.5:
                label = "Primary component"
            elif weight > 0.3:
                label = "Secondary component"
            else:
                label = "Contributing component"

            sections.append(f"### {label} ({weight*100:.0f}%): {parent_node.name}")
            sections.append(parent_node.definition)
            sections.append("")

        # Historical crystallization (how concept emerged)
        if node.crystallization_events:
            sections.append("## Historical Crystallization\n")
            for event in node.crystallization_events:
                year = event.timestamp.year
                sections.append(f"- **{year}**: {event.context}")
                sections.append(f"  *Source*: {event.document.title}")

        # Applications (what this enables)
        if context.descendants:
            sections.append(f"\n## Enables ({len(context.descendants)} applications)\n")
            for desc in context.descendants[:5]:  # Top 5
                sections.append(f"- **{desc.name}**: {desc.definition}")

        return "\n".join(sections)

    def _generate_uncertain_response(self, context: ResponseContext) -> str:
        """
        Generate response for low-confidence/suspicious concepts.

        Heavy hedging, explicit uncertainty flags, epistemic issues highlighted.
        """

        node = context.node
        confidence = context.confidence

        sections = []

        # Warning header
        sections.append(f"⚠️ **{node.name}** (Low Confidence)\n")
        sections.append(f"*Lineage*: {' → '.join(context.lineage_path)}\n")

        # Definition with epistemic hedging
        sections.append(f"Based on **limited information**, {node.name.lower()} appears to involve:\n")
        sections.append(node.definition)
        sections.append("")

        # Epistemic issues (why low confidence)
        sections.append("## Epistemic Issues\n")
        anomalies = confidence.get_anomaly_report()
        if anomalies:
            for anomaly in anomalies:
                sections.append(f"- {anomaly}")
        else:
            sections.append("- Low documentation depth")
            sections.append("- Sparse neighborhood in knowledge graph")

        # Confidence metrics
        sections.append(f"\n**Confidence Score**: {confidence.retrieval_confidence:.2f} (low)")
        sections.append(f"**Hallucination Risk**: {confidence.hallucination_risk:.2f}")
        sections.append(f"**Recommendation**: {confidence.action_recommendation}")

        # What's needed to improve confidence
        sections.append("\n## Required for Validation\n")
        if confidence.documentation_depth == 0:
            sections.append("- Supporting documentation from credible sources")
        if confidence.orphan_score > 0.5:
            sections.append("- Clear connection to established parent concepts")
        if confidence.missing_expected_children:
            sections.append(
                f"- Development of {len(confidence.missing_expected_children)} "
                f"expected subconcepts"
            )
        if confidence.local_density < 0.3:
            sections.append("- Additional related concepts to increase local density")

        # Final warning
        sections.append(
            "\n⚠️ This concept may represent speculative theorizing not yet "
            "validated by the field, or may be a hallucination."
        )

        return "\n".join(sections)

    def _generate_foundational_response(self, context: ResponseContext) -> str:
        """
        Generate response for root/foundational concepts.

        Authoritative, principle-based, shows core branches.
        """

        node = context.node

        sections = []

        # Title
        sections.append(f"# {node.name}\n")
        sections.append(f"*Foundational Concept (Root Level)*\n")

        # Definition
        sections.append(node.definition)
        sections.append("")

        # Core branches (immediate children)
        if context.descendants:
            sections.append("## Core Branches\n")

            # Get immediate children only
            immediate_children = [
                d for d in context.descendants
                if d.actualization_depth == node.actualization_depth + 1
            ]

            for child in immediate_children:
                sections.append(f"### {child.name}")
                sections.append(child.definition)
                sections.append("")

        # Historical development
        if node.crystallization_events:
            sections.append("## Historical Development\n")
            for event in node.crystallization_events:
                sections.append(
                    f"- **{event.timestamp.year}**: {event.context}"
                )

        # Key references
        if node.supported_by:
            sections.append(f"\n## Key References ({len(node.supported_by)})\n")
            for doc in node.supported_by[:3]:  # Top 3
                authors = ", ".join(doc.authors[:2])
                if len(doc.authors) > 2:
                    authors += " et al."
                sections.append(f"- {doc.title} ({authors}, {doc.year})")

        return "\n".join(sections)

    def _generate_comprehensive_response(self, context: ResponseContext) -> str:
        """
        Generate comprehensive response for well-established concepts.

        Confident, detailed, well-cited, shows full context.
        """

        node = context.node

        sections = []

        # Title with lineage
        sections.append(f"# {node.name}\n")
        sections.append(f"*{' → '.join(context.lineage_path)}*\n")

        # Definition
        sections.append(node.definition)
        sections.append("")

        # Context from parents (what it builds on)
        if node.parent_potentials:
            sections.append("## Builds On\n")
            for parent_id in node.parent_potentials[:3]:
                parent = self.graph.get_node(parent_id)
                if parent:
                    sections.append(f"- **{parent.name}**: {parent.definition}")
            sections.append("")

        # Key insights from documentation
        if node.supported_by:
            sections.append("## Key Insights\n")
            for doc in node.supported_by[:3]:
                if doc.excerpt:
                    sections.append(f"- {doc.excerpt}")
                    sections.append(f"  *[{doc.title}, {doc.year}]*")
                else:
                    authors = ", ".join(doc.authors[:2])
                    sections.append(f"- {doc.title} ({authors}, {doc.year})")
            sections.append("")

        # Applications (descendants)
        if context.descendants:
            sections.append("## Applications\n")
            for desc in context.descendants[:5]:
                sections.append(f"- **{desc.name}**: {desc.definition}")

        # Related concepts (siblings)
        if context.siblings:
            sibling_names = [s.name for s in context.siblings[:3]]
            sections.append(f"\n**Related Concepts**: {', '.join(sibling_names)}")

        return "\n".join(sections)

    def _generate_gap_aware_response(self, context: ResponseContext) -> str:
        """
        Generate response that acknowledges gaps but isn't suspicious.

        Moderate confidence, notes knowledge gaps, but provides useful info.
        """

        node = context.node
        confidence = context.confidence

        sections = []

        # Header
        sections.append(f"# {node.name}\n")
        sections.append(f"*{' → '.join(context.lineage_path)}*\n")

        # Definition
        sections.append(node.definition)
        sections.append("")

        # Main content (standard explanation)
        if node.supported_by:
            sections.append("## Overview\n")
            for doc in node.supported_by[:2]:
                if doc.excerpt:
                    sections.append(f"{doc.excerpt}\n")

        # Knowledge gaps section
        sections.append("## Knowledge Gaps\n")
        anomalies = confidence.get_anomaly_report()
        for anomaly in anomalies:
            sections.append(f"- {anomaly}")

        # What we do know
        if context.ancestors:
            sections.append("\n## Known Context\n")
            sections.append("This concept builds on:")
            for ancestor in context.ancestors[:3]:
                sections.append(f"- {ancestor.name}")

        return "\n".join(sections)

    def _generate_standard_response(self, context: ResponseContext) -> str:
        """Generate standard explanation for typical queries."""

        node = context.node

        sections = []

        # Header
        sections.append(f"# {node.name}\n")
        sections.append(f"*{' → '.join(context.lineage_path)}*\n")

        # Definition
        sections.append(node.definition)
        sections.append("")

        # Context from parents
        if node.parent_potentials:
            parent_names = []
            for pid in node.parent_potentials[:2]:
                p = self.graph.get_node(pid)
                if p:
                    parent_names.append(p.name)
            if parent_names:
                sections.append(f"Builds on: {', '.join(parent_names)}\n")

        # Documentation
        if node.supported_by:
            sections.append(f"**Sources** ({len(node.supported_by)}):")
            for doc in node.supported_by[:2]:
                authors = ", ".join(doc.authors[:2])
                sections.append(f"- {doc.title} ({authors}, {doc.year})")

        return "\n".join(sections)

    def _format_confidence(self, confidence: GeometricConfidence) -> Dict:
        """Format confidence metrics for display."""
        return {
            "score": round(confidence.retrieval_confidence, 3),
            "level": self._confidence_level(confidence.retrieval_confidence),
            "emoji": self._confidence_emoji(confidence.retrieval_confidence),
            "interpretation": confidence.interpretation,
            "recommendation": confidence.action_recommendation,
            "has_anomalies": confidence.has_anomalies,
            "metrics": {
                "local_density": round(confidence.local_density, 2),
                "branch_symmetry": round(confidence.branch_symmetry, 2),
                "traversal_distance": round(confidence.traversal_distance, 1),
                "documentation_depth": confidence.documentation_depth,
                "orphan_score": round(confidence.orphan_score, 2),
            }
        }

    def _confidence_level(self, score: float) -> str:
        """Convert confidence score to level."""
        if score > 0.7:
            return "high"
        elif score > 0.4:
            return "medium"
        else:
            return "low"

    def _confidence_emoji(self, score: float) -> str:
        """Get emoji for confidence level."""
        if score > 0.7:
            return "🟢"
        elif score > 0.4:
            return "🟡"
        else:
            return "🔴"

    def _format_lineage(self, path: List[str]) -> Dict:
        """Format lineage path for display."""
        return {
            "path": path,
            "depth": len(path),
            "formatted": " → ".join(path),
            "root": path[0] if path else None,
            "leaf": path[-1] if path else None,
        }

    def _format_citations(self, node: KronosNode) -> List[Dict]:
        """Format citations for display."""
        citations = []
        for doc in node.supported_by:
            citations.append({
                "id": doc.doc_id,
                "title": doc.title,
                "authors": doc.authors,
                "year": doc.year,
                "doi": doc.doi,
                "uri": doc.uri,
                "excerpt": doc.excerpt,
            })
        return citations

    def _format_related(self, context: ResponseContext) -> Dict:
        """Format related concepts for display."""
        return {
            "ancestors": [
                {"id": n.id, "name": n.name}
                for n in context.ancestors[:5]
            ],
            "descendants": [
                {"id": n.id, "name": n.name}
                for n in context.descendants[:5]
            ],
            "siblings": [
                {"id": n.id, "name": n.name}
                for n in context.siblings[:5]
            ],
        }
