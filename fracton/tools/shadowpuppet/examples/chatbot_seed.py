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
# DOMAIN TYPE SOURCE CODE (for generators)
# ============================================================================

DOMAIN_TYPES = [
    '''@dataclass
class Message:
    """A chat message."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)''',
    
    '''@dataclass
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
        return self.messages[-limit:]''',
    
    '''@dataclass
class Intent:
    """Detected user intent."""
    name: str
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)'''
]


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_intent_confidence_valid(classifier):
    """Test that classifier returns valid confidence scores."""
    intent = classifier.classify("hello there")
    assert 0.0 <= intent.confidence <= 1.0, f"Confidence {intent.confidence} out of range [0, 1]"
    assert isinstance(intent.name, str), "Intent name must be string"
    return True


def test_intent_unknown_fallback(classifier):
    """Test that unknown messages return 'unknown' intent."""
    intent = classifier.classify("asdfghjkl zxcvbnm qwertyuiop")  # Gibberish
    # Either low confidence or 'unknown' intent
    assert intent.confidence < 0.5 or intent.name == 'unknown', \
        "Gibberish should have low confidence or 'unknown' intent"
    return True


def test_response_not_empty(generator):
    """Test that generator always returns non-empty response."""
    from dataclasses import dataclass, field
    
    # Create minimal test data
    intent = Intent(name='greeting', confidence=0.9)
    conversation = Conversation(id='test-123')
    
    response = generator.generate(intent, conversation)
    assert response, "Response should not be empty"
    assert isinstance(response, str), "Response must be string"
    assert len(response.strip()) > 0, "Response should have content"
    return True


def test_conversation_message_order(manager):
    """Test that messages are stored in order."""
    conv = manager.create_conversation()
    
    manager.add_message(conv.id, 'user', 'First message')
    manager.add_message(conv.id, 'assistant', 'Second message')
    manager.add_message(conv.id, 'user', 'Third message')
    
    retrieved = manager.get_conversation(conv.id)
    assert len(retrieved.messages) == 3, f"Expected 3 messages, got {len(retrieved.messages)}"
    assert retrieved.messages[0].content == 'First message', "Messages not in order"
    assert retrieved.messages[1].content == 'Second message', "Messages not in order"
    assert retrieved.messages[2].content == 'Third message', "Messages not in order"
    return True


def test_conversation_context_isolation(manager):
    """Test that context is isolated between conversations."""
    conv1 = manager.create_conversation()
    conv2 = manager.create_conversation()
    
    manager.update_context(conv1.id, {'key': 'value1'})
    manager.update_context(conv2.id, {'key': 'value2'})
    
    ctx1 = manager.get_context(conv1.id)
    ctx2 = manager.get_context(conv2.id)
    
    assert ctx1.get('key') == 'value1', "Context 1 incorrect"
    assert ctx2.get('key') == 'value2', "Context 2 incorrect"
    return True


def test_chatbot_returns_response(chatbot):
    """Test that chatbot.chat returns a response."""
    conv_id = chatbot.start_conversation()
    response = chatbot.chat("hello", conv_id)
    
    assert response, "ChatBot should return a response"
    assert isinstance(response, str), "Response must be string"
    
    # Check history
    history = chatbot.get_history(conv_id)
    assert len(history) >= 2, "History should have user message and response"
    
    chatbot.end_conversation(conv_id)
    return True


# ============================================================================
# EVOLUTION RUNNER
# ============================================================================

def run_evolution():
    """Run ShadowPuppet evolution on this architecture."""
    import os
    from fracton.tools.shadowpuppet import (
        SoftwareEvolution,
        ProtocolSpec,
        GrowthGap,
        EvolutionConfig,
        TestSuite
    )
    from fracton.tools.shadowpuppet.generators import (
        ClaudeGenerator,
        MockGenerator
    )
    from pathlib import Path
    
    # Load API key from grimm/.env if not set
    if not os.environ.get('ANTHROPIC_API_KEY'):
        env_path = Path(__file__).parents[5] / 'grimm' / '.env'
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith('ANTHROPIC_API_KEY='):
                    os.environ['ANTHROPIC_API_KEY'] = line.split('=', 1)[1].strip()
                    print(f"[*] Loaded API key from {env_path}")
                    break
    
    # Define protocols with explicit dependencies
    protocols = [
        ProtocolSpec(
            name="IntentClassifier",
            methods=["classify", "extract_entities", "train"],
            docstring="Classifies user messages into intents",
            attributes=["intents: List[str]", "threshold: float"],
            pac_invariants=[
                "Confidence scores sum to 1.0",
                "Unknown messages return 'unknown' intent"
            ],
            dependencies=[]  # No dependencies
        ),
        ProtocolSpec(
            name="ResponseGenerator",
            methods=["generate", "set_template", "format_response"],
            docstring="Generates responses based on intent and context",
            attributes=["templates: Dict[str, List[str]]", "context_window: int"],
            pac_invariants=[
                "Responses are never empty",
                "Fallback for unknown intents"
            ],
            dependencies=[]  # No dependencies
        ),
        ProtocolSpec(
            name="ConversationManager",
            methods=["create_conversation", "get_conversation", "add_message", "get_context", "update_context"],
            docstring="Manages conversation state and history",
            attributes=["conversations: Dict[str, Conversation]", "max_history: int"],
            pac_invariants=[
                "Conversation IDs are unique",
                "Messages stored in order"
            ],
            dependencies=[]  # No dependencies
        ),
        ProtocolSpec(
            name="ChatBot",
            methods=["chat", "start_conversation", "end_conversation", "get_history"],
            docstring="Main chatbot orchestrator",
            attributes=["classifier: IntentClassifier", "generator: ResponseGenerator", "conversation_manager: ConversationManager", "name: str"],
            pac_invariants=[
                "Each message gets one response",
                "Errors return helpful messages"
            ],
            dependencies=["IntentClassifier", "ResponseGenerator", "ConversationManager"]
        ),
    ]
    
    # Create gaps with test suites and domain types
    gaps = []
    for p in protocols:
        # Select tests for this protocol
        test_funcs = []
        if p.name == "IntentClassifier":
            test_funcs = [test_intent_confidence_valid, test_intent_unknown_fallback]
        elif p.name == "ResponseGenerator":
            test_funcs = [test_response_not_empty]
        elif p.name == "ConversationManager":
            test_funcs = [test_conversation_message_order, test_conversation_context_isolation]
        elif p.name == "ChatBot":
            test_funcs = [test_chatbot_returns_response]
        
        gap = GrowthGap(
            protocol=p,
            test_suite=TestSuite(unit=test_funcs) if test_funcs else None,
            domain_types=DOMAIN_TYPES
        )
        gaps.append(gap)
    
    # Configure evolution
    config = EvolutionConfig(
        coherence_threshold=0.65,
        candidates_per_gap=2,
        max_generations=5,
        output_dir=Path("chatbot_evolution")
    )
    
    # Select generator based on API key availability
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("[!] ANTHROPIC_API_KEY not set, using MockGenerator")
        generator = MockGenerator()
    else:
        generator = ClaudeGenerator(
            model="claude-sonnet-4-20250514",
            temperature=0.3,
            fallback_generator=MockGenerator()
        )
    
    # Run evolution
    evolution = SoftwareEvolution(
        generator=generator,
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
