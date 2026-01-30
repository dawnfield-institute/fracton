"""
Chatbot Architecture-as-Code

Defines a simple conversational chatbot using Python protocols.
ShadowPuppet will generate implementations from these specifications.

Usage:
    python -m fracton.tools.shadowpuppet.examples.chatbot_seed
"""

from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass, field
from abc import abstractmethod
from datetime import datetime


# ============================================================================
# DOMAIN TYPES
# ============================================================================

@dataclass
class Message:
    """A chat message."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """A conversation session."""
    id: str
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_message(self, role: str, content: str) -> Message:
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        return msg
    
    def get_history(self, limit: int = 10) -> List[Message]:
        return self.messages[-limit:]


@dataclass
class Intent:
    """Detected user intent."""
    name: str
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# PROTOCOL SPECIFICATIONS (Architecture-as-Code)
# ============================================================================

class IntentClassifier(Protocol):
    """
    Classifies user messages into intents.
    
    Analyzes message content to determine user's intention.
    Should support multiple intents with confidence scores.
    
    PAC Invariants:
    - Confidence scores sum to 1.0 for all candidates
    - Unknown messages return 'unknown' intent
    - Entity extraction is consistent across similar messages
    """
    intents: List[str]
    threshold: float
    
    @abstractmethod
    def classify(self, message: str) -> Intent:
        """Classify message into intent."""
        ...
    
    @abstractmethod
    def extract_entities(self, message: str, intent: Intent) -> Dict[str, Any]:
        """Extract entities from message given intent."""
        ...
    
    @abstractmethod
    def train(self, examples: List[Dict[str, Any]]) -> None:
        """Train classifier on examples."""
        ...


class ResponseGenerator(Protocol):
    """
    Generates responses based on intent and context.
    
    Produces natural language responses appropriate to the intent.
    Should maintain conversation context for coherent replies.
    
    PAC Invariants:
    - Responses are always non-empty strings
    - Context is preserved across turns
    - Fallback response for unknown intents
    """
    templates: Dict[str, List[str]]
    context_window: int
    
    @abstractmethod
    def generate(self, intent: Intent, conversation: Conversation) -> str:
        """Generate response for intent."""
        ...
    
    @abstractmethod
    def set_template(self, intent_name: str, templates: List[str]) -> None:
        """Set response templates for intent."""
        ...
    
    @abstractmethod
    def format_response(self, template: str, entities: Dict[str, Any]) -> str:
        """Format template with entity values."""
        ...


class ConversationManager(Protocol):
    """
    Manages conversation state and history.
    
    Handles conversation lifecycle, storage, and retrieval.
    Should support multiple concurrent conversations.
    
    PAC Invariants:
    - Conversation IDs are unique
    - Messages are stored in order
    - Context is isolated between conversations
    """
    conversations: Dict[str, Conversation]
    max_history: int
    
    @abstractmethod
    def create_conversation(self, user_id: Optional[str] = None) -> Conversation:
        """Create new conversation."""
        ...
    
    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Retrieve conversation by ID."""
        ...
    
    @abstractmethod
    def add_message(self, conversation_id: str, role: str, content: str) -> Message:
        """Add message to conversation."""
        ...
    
    @abstractmethod
    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation context."""
        ...
    
    @abstractmethod
    def update_context(self, conversation_id: str, updates: Dict[str, Any]) -> None:
        """Update conversation context."""
        ...


class ChatBot(Protocol):
    """
    Main chatbot orchestrator.
    
    Coordinates intent classification, response generation,
    and conversation management for a complete chat experience.
    
    PAC Invariants:
    - All components are initialized before use
    - Each message gets exactly one response
    - Errors return helpful error messages
    """
    classifier: IntentClassifier
    generator: ResponseGenerator
    conversation_manager: ConversationManager
    name: str
    
    @abstractmethod
    def chat(self, message: str, conversation_id: Optional[str] = None) -> str:
        """Process message and return response."""
        ...
    
    @abstractmethod
    def start_conversation(self) -> str:
        """Start new conversation, return ID."""
        ...
    
    @abstractmethod
    def end_conversation(self, conversation_id: str) -> None:
        """End and cleanup conversation."""
        ...
    
    @abstractmethod
    def get_history(self, conversation_id: str) -> List[Message]:
        """Get conversation history."""
        ...


# ============================================================================
# WORLDSEED METADATA
# ============================================================================

WORLDSEED_METADATA = {
    'identity': {
        'purpose': 'Conversational chatbot with intent classification',
        'domain': 'Natural language processing',
        'version': '1.0.0'
    },
    'pac_invariants': [
        'Confidence scores are valid probabilities',
        'Conversation state is consistent',
        'Responses are always generated',
        'Context is preserved across turns'
    ],
    'protocols': [
        'IntentClassifier',
        'ResponseGenerator',
        'ConversationManager',
        'ChatBot'
    ],
    'fibonacci_constraints': {
        'max_depth': 2,
        'max_components': 4
    }
}


# ============================================================================
# EVOLUTION RUNNER
# ============================================================================

def run_evolution():
    """Run ShadowPuppet evolution on this architecture."""
    from fracton.tools.shadowpuppet import (
        SoftwareEvolution,
        ProtocolSpec,
        GrowthGap,
        EvolutionConfig,
        MockGenerator
    )
    from pathlib import Path
    
    # Define protocols
    protocols = [
        ProtocolSpec(
            name="IntentClassifier",
            methods=["classify", "extract_entities", "train"],
            docstring="Classifies user messages into intents",
            attributes=["intents", "threshold"],
            pac_invariants=[
                "Confidence scores sum to 1.0",
                "Unknown messages return 'unknown' intent"
            ]
        ),
        ProtocolSpec(
            name="ResponseGenerator",
            methods=["generate", "set_template", "format_response"],
            docstring="Generates responses based on intent and context",
            attributes=["templates", "context_window"],
            pac_invariants=[
                "Responses are never empty",
                "Fallback for unknown intents"
            ]
        ),
        ProtocolSpec(
            name="ConversationManager",
            methods=["create_conversation", "get_conversation", "add_message", "get_context", "update_context"],
            docstring="Manages conversation state and history",
            attributes=["conversations", "max_history"],
            pac_invariants=[
                "Conversation IDs are unique",
                "Messages stored in order"
            ]
        ),
        ProtocolSpec(
            name="ChatBot",
            methods=["chat", "start_conversation", "end_conversation", "get_history"],
            docstring="Main chatbot orchestrator",
            attributes=["classifier", "generator", "conversation_manager", "name"],
            pac_invariants=[
                "Each message gets one response",
                "Errors return helpful messages"
            ]
        ),
    ]
    
    # Create gaps
    gaps = [GrowthGap(protocol=p) for p in protocols]
    
    # Configure evolution
    config = EvolutionConfig(
        coherence_threshold=0.65,
        candidates_per_gap=2,
        max_generations=5,
        output_dir=Path("chatbot_evolution")
    )
    
    # Run evolution
    evolution = SoftwareEvolution(
        generator=MockGenerator(),
        config=config,
        pac_invariants=WORLDSEED_METADATA['pac_invariants']
    )
    
    print("=" * 60)
    print("ChatBot Architecture Evolution")
    print("=" * 60)
    
    results = evolution.grow(gaps)
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Success: {results['success']}")
    print(f"Generations: {results['generations']}")
    print(f"Final population: {results['final_population']}")
    
    if results['best_component']:
        print(f"\nBest component: {results['best_component']['id']}")
        print(f"Coherence: {results['best_component']['coherence_score']:.3f}")
    
    # Show genealogy
    print("\nGenealogy:")
    evolution.genealogy.print_tree()
    
    # Save generated code
    evolution.save_code(Path("chatbot_generated"))
    
    return evolution


if __name__ == "__main__":
    run_evolution()
