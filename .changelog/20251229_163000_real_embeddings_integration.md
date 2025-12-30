# Real Embeddings Integration

**Date**: 2025-12-29 16:30
**Type**: engineering

## Summary

Integrated sentence-transformers for real semantic embeddings with automatic fallback to hash-based embeddings when the library is unavailable. System now supports GPU acceleration and multiple embedding models.

## Changes

### Added
- `fracton/storage/embeddings.py` (230 lines) - Embedding service with sentence-transformers
- `EmbeddingService` class with GPU support and batch processing
- Model registry with 4 preset models (mini, base, large, code)
- Automatic fallback to hash-based embeddings
- Model caching to disk

### Changed
- Updated `KronosMemory.__init__()` to accept `embedding_model` parameter
- Modified `KronosMemory.connect()` to initialize embedding service
- Replaced `_compute_embedding()` to use real embeddings when available
- Updated embedding dimension based on loaded model

### Exports
- Added `EmbeddingService`, `create_embedding_service`, `EMBEDDING_MODELS` to `fracton.storage`

## Details

### Embedding Models

**Model Registry**:
```python
EMBEDDING_MODELS = {
    "mini": {
        "name": "all-MiniLM-L6-v2",
        "dim": 384,
        "speed": "fast",
        "quality": "good",
        "size_mb": 80,
    },
    "base": {
        "name": "all-mpnet-base-v2",
        "dim": 768,
        "speed": "medium",
        "quality": "excellent",
        "size_mb": 420,
    },
    "large": {
        "name": "sentence-transformers/all-distilroberta-v1",
        "dim": 768,
        "speed": "medium",
        "quality": "excellent",
        "size_mb": 290,
    },
    "code": {
        "name": "microsoft/codebert-base",
        "dim": 768,
        "speed": "medium",
        "quality": "excellent-code",
        "size_mb": 500,
    },
}
```

### Usage

**With sentence-transformers** (real embeddings):
```python
# Install: pip install sentence-transformers
from fracton.storage import KronosMemory

memory = KronosMemory(
    storage_path=Path("./data"),
    embedding_model="mini",  # or "base", "large", "code", or custom model name
    device="cuda",  # GPU acceleration
)
await memory.connect()
# Uses real semantic embeddings from all-MiniLM-L6-v2
```

**Without sentence-transformers** (hash-based fallback):
```python
memory = KronosMemory(
    storage_path=Path("./data"),
    device="cpu",
)
await memory.connect()
# Falls back to deterministic hash-based embeddings
# Warning logged: "sentence-transformers not installed, falling back to hash-based embeddings"
```

### Features

**Automatic Fallback**:
- If sentence-transformers not installed → hash-based embeddings
- No code changes required
- Graceful degradation

**GPU Acceleration**:
- Set `device="cuda"` for GPU inference
- Automatic tensor device management
- Works with both embeddings and backends

**Batch Processing**:
- `EmbeddingService.embed_batch()` for efficient batch inference
- Converts lists of texts to tensors in one call

**Model Caching**:
- Models cached to `{storage_path}/models/`
- Avoid re-downloading on subsequent runs

**Embedding Dimension**:
- Automatically updated based on loaded model
- Overrides `embedding_dim` parameter if model loaded
- Ensures consistency between embeddings and vector backends

### Implementation

**EmbeddingService**:
```python
class EmbeddingService:
    async def initialize(self) -> None:
        """Load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=str(self.cache_dir),
            )
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            # Fallback to hash-based
            self.model = None

    async def embed(self, text: str) -> torch.Tensor:
        """Embed single text."""
        if self.model is None:
            return self._hash_embeddings([text])[0]
        return self.model.encode(
            [text],
            convert_to_tensor=True,
            device=self.device,
        )[0]
```

**KronosMemory Integration**:
```python
async def connect(self) -> None:
    # Initialize embedding service
    cache_dir = self.storage_path / "models"
    self.embedding_service = await create_embedding_service(
        model=self.embedding_model,
        device=self.device,
        cache_dir=cache_dir,
    )

    # Update embedding_dim if model loaded
    if self.embedding_service.is_available():
        self.embedding_dim = self.embedding_service.embedding_dim
        logger.info(f"Using real embeddings: dim={self.embedding_dim}")
    else:
        logger.info(f"Using hash-based embeddings: dim={self.embedding_dim}")

async def _compute_embedding(self, text: str) -> torch.Tensor:
    """Compute embedding using sentence-transformers or hash fallback."""
    if self.embedding_service and self.embedding_service.is_available():
        return await self.embedding_service.embed(text)
    else:
        # Deterministic hash-based fallback
        hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        torch.manual_seed(hash_val % (2**32))
        return torch.randn(self.embedding_dim, device=self.device)
```

## Test Results

**All tests passing with both real and fallback embeddings**:
- 19/19 KronosMemory tests passing
- Model downloads automatically on first run
- Fallback works when sentence-transformers unavailable

## Related

- Backend integration: Previous session
- Next: Docker setup for production deployment
