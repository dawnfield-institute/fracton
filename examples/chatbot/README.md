# Fracton Agentic Chatbot

A conversational AI agent powered by KronosMemory with full PAC/SEC/MED theoretical foundations.

## Features

- **Long-term Memory**: Stores conversation with hierarchical parent-child relationships
- **SEC Resonance Ranking**: Retrieves context using golden ratio resonance
- **Conservation Validation**: Every message validated with PAC three-dimensional conservation
- **Real-time Monitoring**: Balance operator Ξ, duty cycle, c² tracked in real-time
- **Collapse Detection**: Automatic warnings when Ξ exceeds threshold
- **Multi-modal**: CLI and Web interfaces

## Architecture

```
User Message
    ↓
Store with PAC Conservation (parent-child relationship)
    ↓
Retrieve Context via SEC Resonance Ranking
    ↓
Generate Response (LLM or mock)
    ↓
Store Response with Conservation Validation
    ↓
Monitor Foundation Health (c², Ξ, duty cycle)
```

## Quick Start

### Option 1: CLI Interface

```bash
# Using Docker Compose
docker-compose up chatbot

# Or run locally
python chatbot.py
```

### Option 2: Web Interface

```bash
# Using Docker Compose
docker-compose up chatbot

# Or run locally
python web_chatbot.py
```

Then open http://localhost:8080 in your browser.

## Configuration

### Environment Variables

```bash
# Device (CPU or CUDA)
DEVICE=cpu

# LLM Provider (optional, uses mock if not set)
LLM_PROVIDER=openai  # or 'anthropic'
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...

# Data directories
FRACTON_DATA_DIR=/app/data
FRACTON_CACHE_DIR=/app/cache
FRACTON_LOG_DIR=/app/logs

# Backend (default: ChromaDB)
FRACTON_BACKEND=chromadb
CHROMADB_HOST=chromadb
CHROMADB_PORT=8000
```

### Docker Compose

```yaml
# In docker-compose.yml
chatbot:
  build:
    context: .
    dockerfile: examples/chatbot/Dockerfile
  ports:
    - "8080:8080"
  environment:
    - DEVICE=cpu
    - LLM_PROVIDER=openai
    - OPENAI_API_KEY=${OPENAI_API_KEY}
```

## Usage

### CLI Mode

```
$ python chatbot.py

╔══════════════════════════════════════════════════════════════╗
║              Fracton Agentic Chatbot Demo                   ║
║  Powered by KronosMemory with PAC/SEC/MED Foundations       ║
╚══════════════════════════════════════════════════════════════╝

🚀 Initializing Agentic Chatbot...
✅ Chatbot initialized
   Device: cpu
   Embedding model: mini
   Storage: ./data/chatbot

📊 Foundation Health Metrics
────────────────────────────────────────────────────────────
   c² (model constant):    No data yet
   Balance operator Ξ:     No data yet
   Duty cycle:             No data yet
   Total nodes:            1
   Collapse triggers:      0
────────────────────────────────────────────────────────────

💡 Type your messages (or 'quit' to exit, 'health' for metrics)

👤 You: Hello!

============================================================
Turn 1
============================================================

💬 User: Hello!

🔍 Retrieved 0 relevant context items:

🤖 Bot: Hello! I'm an agentic chatbot powered by Fracton's KronosMemory...

👤 You: What can you remember?

============================================================
Turn 2
============================================================

💬 User: What can you remember?

🔍 Retrieved 1 relevant context items:
   1. [0.823] Hello!...

🤖 Bot: I remember we discussed: Hello!... My memory uses PAC conservation...
```

### Web Mode

Open http://localhost:8080

Features:
- Real-time chat interface
- Live health metrics in header (c², Ξ, duty cycle, collapses)
- Beautiful gradient UI
- Typing indicators
- Message timestamps

## API Endpoints

### POST /api/chat

Send a message and get response.

**Request:**
```json
{
  "message": "Hello!"
}
```

**Response:**
```json
{
  "response": "Hello! I'm an agentic chatbot...",
  "context": [
    {
      "content": "Previous message...",
      "score": 0.823,
      "similarity": 0.891,
      "role": "user",
      "timestamp": "2025-12-29T22:00:00"
    }
  ],
  "health": {
    "c_squared": { "latest": 1.45, "mean": 1.42, ... },
    "balance_operator": { "latest": 1.0234, ... },
    "duty_cycle": { "latest": 0.618, ... },
    "constants": { "phi": 1.618, "xi": 1.0571, ... },
    "collapses": 0
  }
}
```

### GET /api/health

Get foundation health metrics.

**Response:**
```json
{
  "c_squared": { "count": 10, "mean": 1.45, "latest": 1.53, ... },
  "balance_operator": { "count": 10, "mean": 1.0234, ... },
  "duty_cycle": { "count": 10, "mean": 0.621, ... },
  "constants": { "phi": 1.618033988749895, ... },
  "total_nodes": 25,
  "collapses": 0
}
```

### GET /api/stats

Get full memory statistics.

## How It Works

### 1. Message Storage

Every message is stored with full PAC conservation validation:

```python
msg_id = await memory.store(
    content=message,
    graph="conversations",
    node_type=NodeType.FACT,  # User messages
    parent_id=last_message_id,  # Creates hierarchy
    metadata={"role": "user", "timestamp": "...", "turn": 1}
)
```

**Conservation Validated**:
- Value: f(parent) = Σ f(children)
- Complexity: ||C(parent)||² = ||Σ C(children)||²
- Effect: Effect(parent) = Σ Effect(children)

### 2. Context Retrieval

Uses SEC resonance ranking for semantic relevance:

```python
results = await memory.query(
    query_text=user_message,
    graphs=["conversations"],
    limit=5,
)
```

**Resonance Formula**: R(k) = φ^(1 + (k_eq - k)/2)

Where:
- φ = 1.618... (golden ratio)
- k = hierarchy depth
- k_eq = equilibrium depth

### 3. Response Generation

Three modes:
1. **OpenAI GPT-4**: If `OPENAI_API_KEY` set
2. **Anthropic Claude**: If `ANTHROPIC_API_KEY` set
3. **Mock**: Rule-based for testing (default)

### 4. Health Monitoring

Real-time metrics tracked:

- **c² (model constant)**: E_children / E_parent
  - Synthetic: ≈ 1.0 (perfect conservation)
  - Real embeddings: ≈ 100-1000 (semantic amplification)

- **Balance operator Ξ**: 1 + π/F₁₀ ≈ 1.0571
  - COLLAPSE: Ξ > 1.0571 → Warning logged
  - STABLE: 0.9514 ≤ Ξ ≤ 1.0571
  - DECAY: Ξ < 0.9514 → Warning logged

- **Duty cycle**: φ/(φ+1) ≈ 0.618
  - Target equilibrium for SEC dynamics
  - Tracks attraction/repulsion balance

## Memory Hierarchy

```
Conversation Root
├── Turn 1: User Message
│   └── Turn 1: Bot Response
│       ├── Turn 2: User Message
│       │   └── Turn 2: Bot Response
│       │       └── ...
```

Each edge validated with PAC conservation!

## Testing

The chatbot serves as a comprehensive test for:

1. **PAC Conservation**: Every message creates parent-child relationship
2. **SEC Resonance**: Context retrieval uses golden ratio ranking
3. **MED Bounds**: Emergent structures monitored (non-strict mode)
4. **Distance Validation**: c² measured for each relationship
5. **Collapse Detection**: Ξ monitored in real-time
6. **Backend Integration**: Works with ChromaDB, SQLite, Neo4j, Qdrant

## Example Conversation with Health Metrics

```
Turn 1: "Hello!"
  → c²: 1.02, Ξ: 1.0123, Duty: 0.615
  → Status: STABLE ✅

Turn 5: "Tell me about memory"
  → c²: 1.45, Ξ: 1.0234, Duty: 0.621
  → Status: STABLE ✅
  → Collapses: 0

Turn 10: (complex conversation)
  → c²: 1.89, Ξ: 1.0645, Duty: 0.618
  → Status: ⚠️ COLLAPSE DETECTED
  → Collapses: 1
```

## Development

### Add New Response Modes

```python
async def _generate_custom(self, user_message, context):
    """Custom response generation."""
    # Your logic here
    return response
```

### Custom Memory Backends

```python
chatbot = AgenticChatbot(
    backend="neo4j",  # or 'qdrant', 'sqlite'
    backend_config={...}
)
```

### Extended Features

Ideas for extension:
- Multi-user conversations with separate graphs
- Personality modes (different response styles)
- Tool integration (web search, calculator, etc.)
- Memory summarization (compress old conversations)
- Export conversation history
- Sentiment analysis with PAC correlation

## Troubleshooting

**"Chatbot not initialized"**:
- Wait for startup (model download on first run ~18s)
- Check ChromaDB is running: `docker-compose ps`

**"OpenAI API error"**:
- Check API key is set: `echo $OPENAI_API_KEY`
- Falls back to mock mode automatically

**Memory not persisting**:
- Check volume mounts: `./data:/app/data`
- Verify permissions on data directory

**High collapse count**:
- This is normal for complex conversations
- Indicates semantic richness (children > parent energy)
- Monitor Ξ values in health metrics

## Performance

**Typical Metrics** (CPU, mini embeddings):
- Message storage: ~50ms
- Context retrieval: ~30ms
- Conservation validation: ~3ms
- Total latency: ~100ms + LLM time

**With GPU** (CUDA):
- Message storage: ~20ms
- Context retrieval: ~10ms
- Conservation validation: ~1ms
- Total latency: ~40ms + LLM time

## License

Apache 2.0 - Part of the Fracton SDK
