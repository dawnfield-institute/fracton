# Docker Deployment Guide

This guide covers running Fracton and the agentic chatbot example using Docker.

## Quick Start

### 1. Build Images

```bash
# Build CPU runtime
docker-compose build fracton-cpu

# Or build GPU runtime (requires NVIDIA Docker)
docker-compose build fracton-gpu

# Build chatbot
docker-compose build chatbot
```

### 2. Start Services

```bash
# Start chatbot with ChromaDB backend
docker-compose up chatbot

# Start with Neo4j backend
docker-compose --profile neo4j up chatbot neo4j

# Start with Qdrant backend
docker-compose --profile qdrant up chatbot qdrant

# Start development environment
docker-compose --profile dev up dev
```

### 3. Access Chatbot

**CLI Mode** (default):
```bash
docker-compose up chatbot
# Interact in terminal
```

**Web Mode**:
```bash
# Modify docker-compose.yml to run web_chatbot.py
# Or use environment variable
docker-compose run -e MODE=web chatbot

# Then open: http://localhost:8080
```

## Docker Images

### Fracton Base Images

**fracton:cpu** - CPU runtime
- Python 3.11 slim
- PyTorch CPU
- All Fracton dependencies
- Ready for production

**fracton:gpu** - GPU runtime
- NVIDIA CUDA 12.1
- PyTorch with CUDA support
- GPU-accelerated embeddings
- Requires NVIDIA Docker

**fracton:dev** - Development environment
- Based on CPU runtime
- Includes pytest, black, ruff, mypy
- Jupyter notebook support
- Source code mounted as volume

### Application Images

**fracton-chatbot** - Agentic chatbot example
- Built on fracton:cpu
- FastAPI web server
- CLI and web interfaces
- OpenAI/Anthropic integration

## Docker Compose Services

### Core Services

**chromadb** - Vector database (default backend)
- Port: 8000
- Persistent storage
- Health checks enabled

**neo4j** - Graph database (optional)
- Ports: 7474 (HTTP), 7687 (Bolt)
- Profile: `neo4j`
- Default auth: neo4j/fracton123

**qdrant** - Vector database (optional)
- Ports: 6333 (HTTP), 6334 (gRPC)
- Profile: `qdrant`
- Persistent storage

### Application Services

**chatbot** - Agentic chatbot
- Port: 8080 (web mode)
- Depends on: chromadb
- Volumes: ./data, ./cache, ./logs

**fracton-cpu** - CPU runtime container
- Base service for custom apps
- ChromaDB integration

**fracton-gpu** - GPU runtime container
- Requires NVIDIA Docker
- Profile: `gpu`
- GPU reservation

**dev** - Development environment
- Interactive shell
- Source mounted as volume
- Profile: `dev`

## Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Device
DEVICE=cpu  # or 'cuda'

# LLM Provider (optional)
LLM_PROVIDER=openai  # or 'anthropic'
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Backend Configuration
FRACTON_BACKEND=chromadb  # or 'neo4j', 'qdrant', 'sqlite'
CHROMADB_HOST=chromadb
CHROMADB_PORT=8000

# Neo4j (if using)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=fracton123

# Qdrant (if using)
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Directories
FRACTON_DATA_DIR=/app/data
FRACTON_CACHE_DIR=/app/cache
FRACTON_LOG_DIR=/app/logs
```

### Volume Mounts

Data persists in local directories:

```yaml
volumes:
  - ./data:/app/data        # KronosMemory storage
  - ./cache:/app/cache      # Model cache
  - ./logs:/app/logs        # Application logs
```

Create directories:
```bash
mkdir -p data cache logs
```

## Usage Examples

### Run Chatbot (CLI)

```bash
docker-compose up chatbot
```

Interact in terminal:
```
👤 You: Hello!
🤖 Bot: Hello! I'm an agentic chatbot...
```

### Run Chatbot (Web)

Modify `docker-compose.yml`:
```yaml
chatbot:
  command: ["python", "/app/chatbot/web_chatbot.py"]
```

Then:
```bash
docker-compose up chatbot
# Open http://localhost:8080
```

### Run Tests

```bash
# Start dev container
docker-compose --profile dev run dev

# Inside container
pytest tests/ -v

# Run specific tests
pytest tests/test_foundations.py -v

# Run with coverage
pytest tests/ --cov=fracton.storage --cov-report=html
```

### Development Mode

```bash
# Start dev environment with shell
docker-compose --profile dev run dev

# Inside container you have:
# - Source code mounted (editable)
# - All dev tools (pytest, black, ruff, mypy)
# - Jupyter notebook
# - Access to ChromaDB backend
```

### GPU Mode

Requires NVIDIA Docker runtime.

```bash
# Build GPU image
docker-compose build fracton-gpu

# Start with GPU
docker-compose --profile gpu up fracton-gpu

# Verify GPU available
docker-compose --profile gpu run fracton-gpu python -c "import torch; print(torch.cuda.is_available())"
```

## Backend Configurations

### ChromaDB (Default)

```bash
# Standalone
docker-compose up chromadb

# With chatbot
docker-compose up chatbot
```

### Neo4j

```bash
# Start Neo4j + chatbot
docker-compose --profile neo4j up neo4j chatbot

# Access Neo4j Browser: http://localhost:7474
# Username: neo4j
# Password: fracton123
```

Configure chatbot for Neo4j:
```yaml
environment:
  - FRACTON_BACKEND=neo4j
  - NEO4J_URI=bolt://neo4j:7687
```

### Qdrant

```bash
# Start Qdrant + chatbot
docker-compose --profile qdrant up qdrant chatbot

# Access Qdrant API: http://localhost:6333
```

Configure chatbot for Qdrant:
```yaml
environment:
  - FRACTON_BACKEND=qdrant
  - QDRANT_HOST=qdrant
  - QDRANT_PORT=6333
```

### SQLite (No backend service needed)

```yaml
environment:
  - FRACTON_BACKEND=sqlite
```

## Production Deployment

### Build Production Images

```bash
# Build optimized images
docker-compose build --no-cache fracton-cpu chatbot

# Tag for registry
docker tag fracton:cpu your-registry/fracton:1.0.0-cpu
docker tag fracton-chatbot:latest your-registry/fracton-chatbot:1.0.0

# Push to registry
docker push your-registry/fracton:1.0.0-cpu
docker push your-registry/fracton-chatbot:1.0.0
```

### Production Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  chatbot:
    image: your-registry/fracton-chatbot:1.0.0
    restart: always
    environment:
      - DEVICE=cpu
      - FRACTON_BACKEND=chromadb
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - fracton-data:/app/data
      - fracton-logs:/app/logs
    depends_on:
      - chromadb
    networks:
      - fracton-network

  chromadb:
    image: ghcr.io/chroma-core/chroma:latest
    restart: always
    volumes:
      - chroma-data:/chroma/chroma
    networks:
      - fracton-network

volumes:
  fracton-data:
  fracton-logs:
  chroma-data:

networks:
  fracton-network:
```

Deploy:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Health Checks

All services have health checks:

```bash
# Check service health
docker-compose ps

# View health check logs
docker inspect --format='{{json .State.Health}}' fracton-chatbot | jq

# Manual health check
docker-compose exec chatbot python -c "import fracton; print('OK')"
```

## Troubleshooting

### "ChromaDB connection refused"

```bash
# Check ChromaDB is running
docker-compose ps chromadb

# View ChromaDB logs
docker-compose logs chromadb

# Restart ChromaDB
docker-compose restart chromadb
```

### "CUDA not available" (GPU mode)

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Verify GPU device in compose
docker-compose --profile gpu config | grep -A5 "devices:"

# Check NVIDIA driver
nvidia-smi
```

### "Permission denied" on volumes

```bash
# Fix permissions
sudo chown -R $USER:$USER data cache logs

# Or run as root (not recommended)
docker-compose run --user root chatbot
```

### Model download slow

First run downloads sentence-transformers models (~90MB).

```bash
# Pre-download models
docker-compose run chatbot python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Models cached in ./cache directory
ls -lh cache/
```

### Container logs

```bash
# View all logs
docker-compose logs

# Follow chatbot logs
docker-compose logs -f chatbot

# Last 100 lines
docker-compose logs --tail=100 chatbot

# Specific service
docker-compose logs chromadb
```

## Advanced Usage

### Custom Fracton App

Create custom `Dockerfile`:

```dockerfile
FROM fracton:cpu

WORKDIR /app

# Copy your app
COPY my_app.py .

# Install additional dependencies
RUN pip install --no-cache-dir your-deps

CMD ["python", "my_app.py"]
```

Build and run:
```bash
docker build -t my-fracton-app .
docker run --network fracton-network my-fracton-app
```

### Multi-Container Setup

Run multiple chatbot instances:

```yaml
services:
  chatbot-1:
    extends: chatbot
    container_name: fracton-chatbot-1
    ports:
      - "8081:8080"

  chatbot-2:
    extends: chatbot
    container_name: fracton-chatbot-2
    ports:
      - "8082:8080"
```

### Monitoring

Add monitoring with Prometheus/Grafana:

```yaml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

## Performance Tuning

### CPU Optimization

```yaml
chatbot:
  environment:
    - OMP_NUM_THREADS=4
    - MKL_NUM_THREADS=4
  deploy:
    resources:
      limits:
        cpus: '4.0'
        memory: 8G
```

### GPU Optimization

```yaml
fracton-gpu:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['0']  # Specific GPU
            capabilities: [gpu]
```

### Memory Limits

```yaml
chatbot:
  deploy:
    resources:
      limits:
        memory: 4G
      reservations:
        memory: 2G
```

## Security

### Non-root User

Add to Dockerfile:

```dockerfile
RUN useradd -m -u 1000 fracton
USER fracton
```

### Secrets Management

Use Docker secrets:

```yaml
secrets:
  openai_key:
    external: true

services:
  chatbot:
    secrets:
      - openai_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_key
```

### Network Isolation

```yaml
networks:
  fracton-internal:
    driver: bridge
    internal: true
  fracton-external:
    driver: bridge
```

## Cleanup

```bash
# Stop all services
docker-compose down

# Remove volumes (deletes data!)
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Full cleanup
docker system prune -a --volumes
```

## Next Steps

- See [examples/chatbot/README.md](examples/chatbot/README.md) for chatbot details
- See [tests/TESTING_REPORT.md](tests/TESTING_REPORT.md) for test coverage
- See [README.md](README.md) for Fracton SDK overview
