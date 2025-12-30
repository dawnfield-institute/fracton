"""
Distance Validator (E=mc² Framework)

Validates PAC conservation through Euclidean distance geometry.

Key Discovery (Experiment 6):
    E = mc²

Where:
    E = semantic energy (||embedding||²)
    m = semantic "mass" (information density)
    c² = model-specific constant

For synthetic embeddings: c² ≈ 1.0 (perfect conservation)
For real LLMs: c² ≈ 416 (llama3.2), model-specific

This provides geometric validation of PAC conservation.

References:
- foundational/arithmetic/euclidean_distance_validation/RESULTS.md
- foundational/arithmetic/euclidean_distance_validation/experiments/
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class DistanceMetrics:
    """Metrics from distance validation."""

    parent_energy: float  # E_parent = ||parent||²
    children_energy: float  # E_children = Σ ||child||²
    c_squared: float  # c² = E_children / E_parent
    binding_energy: float  # Binding = E_parent - E_children (negative = amplification)
    conservation_residual: float  # |E_parent - E_children|
    is_conserved: bool  # Within tolerance
    embedding_type: str  # 'synthetic' or 'real'


@dataclass
class FractalMetrics:
    """Fractal dimension metrics."""

    fractal_dimension: float  # D = log(N) / log(1/λ)
    branching_factor: int  # N
    scaling_factor: float  # λ
    depth: int  # Tree depth


class DistanceValidator:
    """
    Distance validator using E=mc² framework.

    Validates PAC conservation through geometric properties:
    1. Energy conservation (E=mc²)
    2. Distance conservation
    3. Fractal scaling
    4. Context-relative invariance
    """

    def __init__(
        self,
        device: str = "cpu",
        synthetic_tolerance: float = 0.2,  # ±20% for synthetic
        real_c_squared_range: Tuple[float, float] = (100.0, 1000.0),
    ):
        """
        Initialize distance validator.

        Args:
            device: Computation device
            synthetic_tolerance: Tolerance for synthetic c² around 1.0
            real_c_squared_range: Valid range for real LLM c²
        """
        self.device = device
        self.synthetic_tolerance = synthetic_tolerance
        self.real_c_squared_range = real_c_squared_range

    def validate_energy_conservation(
        self,
        parent_embedding: torch.Tensor,
        children_embeddings: List[torch.Tensor],
        embedding_type: str = "unknown",
    ) -> DistanceMetrics:
        """
        Validate E=mc² energy conservation.

        Args:
            parent_embedding: Parent embedding
            children_embeddings: List of children embeddings
            embedding_type: 'synthetic', 'real', or 'unknown'

        Returns:
            DistanceMetrics with validation results
        """
        # Compute energies: E = ||embedding||²
        parent_energy = self._compute_energy(parent_embedding)
        children_energy = sum(
            self._compute_energy(child) for child in children_embeddings
        )

        # Compute c²
        if parent_energy == 0:
            c_squared = float("inf")
        else:
            c_squared = children_energy / parent_energy

        # Binding energy (negative means amplification)
        binding_energy = parent_energy - children_energy

        # Conservation residual
        residual = abs(binding_energy)

        # Determine if conserved based on embedding type
        if embedding_type == "unknown":
            # Auto-detect from c²
            if (
                1.0 - self.synthetic_tolerance
                < c_squared
                < 1.0 + self.synthetic_tolerance
            ):
                embedding_type = "synthetic"
            elif (
                self.real_c_squared_range[0]
                < c_squared
                < self.real_c_squared_range[1]
            ):
                embedding_type = "real"

        # Check conservation
        if embedding_type == "synthetic":
            is_conserved = (
                1.0 - self.synthetic_tolerance
                < c_squared
                < 1.0 + self.synthetic_tolerance
            )
        elif embedding_type == "real":
            is_conserved = (
                self.real_c_squared_range[0]
                < c_squared
                < self.real_c_squared_range[1]
            )
        else:
            is_conserved = False

        return DistanceMetrics(
            parent_energy=parent_energy,
            children_energy=children_energy,
            c_squared=c_squared,
            binding_energy=binding_energy,
            conservation_residual=residual,
            is_conserved=is_conserved,
            embedding_type=embedding_type,
        )

    def validate_distance_conservation(
        self,
        parent_embedding: torch.Tensor,
        children_embeddings: List[torch.Tensor],
        tolerance: float = 0.1,
    ) -> Tuple[bool, float]:
        """
        Validate distance conservation: ||parent||² ≈ Σ ||children||²

        Args:
            parent_embedding: Parent embedding
            children_embeddings: Children embeddings
            tolerance: Relative tolerance

        Returns:
            (is_conserved, ratio) where ratio = Σ||children||² / ||parent||²
        """
        metrics = self.validate_energy_conservation(
            parent_embedding, children_embeddings
        )

        is_conserved = abs(metrics.c_squared - 1.0) < tolerance
        return is_conserved, metrics.c_squared

    def compute_fractal_dimension(
        self,
        embeddings_by_level: List[List[torch.Tensor]],
    ) -> FractalMetrics:
        """
        Compute fractal dimension from multi-level embeddings.

        D = log(N) / log(1/λ)

        Where:
            N = branching factor
            λ = scaling factor (distance ratio between levels)

        Args:
            embeddings_by_level: List of embedding lists, one per level

        Returns:
            FractalMetrics
        """
        if len(embeddings_by_level) < 2:
            return FractalMetrics(
                fractal_dimension=0.0,
                branching_factor=0,
                scaling_factor=0.0,
                depth=len(embeddings_by_level),
            )

        # Compute branching factor (average)
        branching_factors = []
        for i in range(len(embeddings_by_level) - 1):
            if len(embeddings_by_level[i]) > 0:
                bf = len(embeddings_by_level[i + 1]) / len(
                    embeddings_by_level[i]
                )
                branching_factors.append(bf)

        avg_branching = (
            np.mean(branching_factors) if branching_factors else 1.0
        )

        # Compute scaling factor (distance ratio)
        scaling_factors = []
        for i in range(len(embeddings_by_level) - 1):
            if embeddings_by_level[i] and embeddings_by_level[i + 1]:
                # Average distance at level i
                dist_i = self._average_pairwise_distance(
                    embeddings_by_level[i]
                )
                # Average distance at level i+1
                dist_i1 = self._average_pairwise_distance(
                    embeddings_by_level[i + 1]
                )

                if dist_i > 0:
                    scaling_factors.append(dist_i1 / dist_i)

        avg_scaling = np.mean(scaling_factors) if scaling_factors else 1.0

        # Compute fractal dimension
        if avg_scaling > 0 and avg_scaling != 1.0:
            fractal_dim = np.log(avg_branching) / np.log(1.0 / avg_scaling)
        else:
            fractal_dim = 0.0

        return FractalMetrics(
            fractal_dimension=fractal_dim,
            branching_factor=int(avg_branching),
            scaling_factor=avg_scaling,
            depth=len(embeddings_by_level),
        )

    def compute_context_relative_variance(
        self,
        embedding: torch.Tensor,
        context_embeddings: List[torch.Tensor],
    ) -> float:
        """
        Compute context-relative variance.

        Measures how much embedding changes relative to context.

        Args:
            embedding: Target embedding
            context_embeddings: Context embeddings

        Returns:
            Variance ratio (embedding / context)
        """
        if not context_embeddings:
            return 1.0

        # Embedding variance
        embedding_var = torch.var(embedding).item()

        # Context variance
        context_tensor = torch.stack(context_embeddings)
        context_var = torch.var(context_tensor).item()

        if context_var == 0:
            return 1.0

        return embedding_var / context_var

    def measure_amplification(
        self,
        parent_embedding: torch.Tensor,
        children_embeddings: List[torch.Tensor],
    ) -> Tuple[float, str]:
        """
        Measure semantic amplification/binding.

        For synthetic: ~91% binding (c² ≈ 0.09)
        For real LLMs: ~330% amplification (c² ≈ 4.3)

        Args:
            parent_embedding: Parent embedding
            children_embeddings: Children embeddings

        Returns:
            (amplification_factor, interpretation)
        """
        metrics = self.validate_energy_conservation(
            parent_embedding, children_embeddings
        )

        # Amplification factor: (children - parent) / parent
        amplification = (
            metrics.children_energy - metrics.parent_energy
        ) / metrics.parent_energy

        # Interpretation
        if amplification > 0:
            interpretation = f"+{amplification*100:.0f}% amplification (whole > parts)"
        elif amplification < 0:
            interpretation = (
                f"{abs(amplification)*100:.0f}% binding (whole < parts)"
            )
        else:
            interpretation = "perfect conservation"

        return amplification, interpretation

    def _compute_energy(self, embedding: torch.Tensor) -> float:
        """
        Compute semantic energy: E = ||embedding||²

        Args:
            embedding: Embedding vector

        Returns:
            Energy (scalar)
        """
        return torch.norm(embedding).item() ** 2

    def _average_pairwise_distance(
        self, embeddings: List[torch.Tensor]
    ) -> float:
        """
        Compute average pairwise Euclidean distance.

        Args:
            embeddings: List of embeddings

        Returns:
            Average distance
        """
        if len(embeddings) < 2:
            return 0.0

        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = torch.norm(embeddings[i] - embeddings[j]).item()
                distances.append(dist)

        return np.mean(distances) if distances else 0.0
