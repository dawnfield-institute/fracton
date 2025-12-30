"""
Embedding Service for KRONOS Memory

Provides sentence-transformers integration with GPU support and caching.
"""

import torch
import logging
from typing import List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service using sentence-transformers.

    Supports:
    - Multiple model backends
    - GPU acceleration
    - Batch processing
    - Local caching
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize embedding service.

        Args:
            model_name: Sentence-transformers model name
            device: "cpu" or "cuda"
            cache_dir: Optional cache directory for models
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self._embedding_dim = None

        logger.info(f"EmbeddingService initialized: model={model_name}, device={device}")

    async def initialize(self) -> None:
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            # Load model
            if self.cache_dir:
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=str(self.cache_dir),
                )
            else:
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )

            # Get embedding dimension
            self._embedding_dim = self.model.get_sentence_embedding_dimension()

            logger.info(
                f"Loaded model {self.model_name}: dim={self._embedding_dim}, device={self.device}"
            )

        except ImportError:
            logger.warning(
                "sentence-transformers not installed, falling back to hash-based embeddings"
            )
            self.model = None

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        # Default for all-MiniLM-L6-v2
        return 384

    def is_available(self) -> bool:
        """Check if sentence-transformers is available."""
        return self.model is not None

    async def embed(self, text: str) -> torch.Tensor:
        """
        Compute embedding for single text.

        Args:
            text: Input text

        Returns:
            Embedding tensor
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Compute embeddings for batch of texts.

        Args:
            texts: List of input texts

        Returns:
            Tensor of shape (len(texts), embedding_dim)
        """
        if not texts:
            return torch.zeros((0, self.embedding_dim), device=self.device)

        if self.model is None:
            # Fallback to hash-based embeddings
            return self._hash_embeddings(texts)

        # Encode with sentence-transformers
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
        )

        return embeddings

    def _hash_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Fallback hash-based embeddings when sentence-transformers unavailable.

        Args:
            texts: List of input texts

        Returns:
            Tensor of shape (len(texts), embedding_dim)
        """
        import hashlib

        embeddings = []

        for text in texts:
            # Deterministic random embedding based on hash
            hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
            torch.manual_seed(hash_val % (2**32))
            embedding = torch.randn(self.embedding_dim, device=self.device)
            embeddings.append(embedding)

        return torch.stack(embeddings)


# ============================================================================
# Model Registry
# ============================================================================

EMBEDDING_MODELS = {
    # Fast, lightweight models (good for development)
    "mini": {
        "name": "all-MiniLM-L6-v2",
        "dim": 384,
        "speed": "fast",
        "quality": "good",
        "size_mb": 80,
    },
    # Balanced models
    "base": {
        "name": "all-mpnet-base-v2",
        "dim": 768,
        "speed": "medium",
        "quality": "excellent",
        "size_mb": 420,
    },
    # High-quality models
    "large": {
        "name": "sentence-transformers/all-distilroberta-v1",
        "dim": 768,
        "speed": "medium",
        "quality": "excellent",
        "size_mb": 290,
    },
    # Specialized models
    "code": {
        "name": "microsoft/codebert-base",
        "dim": 768,
        "speed": "medium",
        "quality": "excellent-code",
        "size_mb": 500,
    },
}


def get_model_name(model_key: str = "mini") -> str:
    """
    Get model name from registry.

    Args:
        model_key: Key from EMBEDDING_MODELS

    Returns:
        Model name for sentence-transformers
    """
    if model_key in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[model_key]["name"]
    # Return key as-is if not in registry (allows custom models)
    return model_key


async def create_embedding_service(
    model: str = "mini",
    device: str = "cpu",
    cache_dir: Optional[Path] = None,
) -> EmbeddingService:
    """
    Create and initialize embedding service.

    Args:
        model: Model key from registry or custom model name
        device: "cpu" or "cuda"
        cache_dir: Optional cache directory

    Returns:
        Initialized EmbeddingService
    """
    model_name = get_model_name(model)

    service = EmbeddingService(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
    )

    await service.initialize()

    return service
