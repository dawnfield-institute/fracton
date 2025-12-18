"""
Training Interventions - Active nurturing of generalization.

Provides intervention types and a trainer that applies them dynamically
based on PAC tree health to encourage generalization over memorization.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import numpy as np


class InterventionType(Enum):
    """Types of training interventions."""
    ADD_NOISE = "add_noise"           # Add Gaussian noise to embeddings
    ADD_DROPOUT = "add_dropout"       # Temporary dropout increase
    TOKEN_MASKING = "token_masking"   # Mask tokens to force context
    POSITION_PERTURB = "position_perturb"  # Shuffle positions slightly
    BRANCH_PRUNE = "branch_prune"     # Remove overly specific branches
    BYREF_MERGE = "byref_merge"       # Connect similar branches
    LEARNING_RATE_BOOST = "lr_boost"  # Temporary LR increase
    GRADIENT_CLIP = "gradient_clip"   # Tighter gradient clipping


@dataclass
class Intervention:
    """A specific intervention to apply."""
    type: InterventionType
    target_patterns: List[str] = field(default_factory=list)
    strength: float = 0.1  # 0.0 to 1.0
    duration_steps: int = 100
    step_applied: int = 0
    
    def is_active(self, current_step: int) -> bool:
        """Check if intervention is still active."""
        return current_step < self.step_applied + self.duration_steps
    
    def remaining_steps(self, current_step: int) -> int:
        """Get remaining steps for this intervention."""
        return max(0, self.step_applied + self.duration_steps - current_step)


@dataclass
class InterventionConfig:
    """Configuration for intervention thresholds and strengths."""
    # When to trigger interventions
    entropy_collapse_threshold: float = 0.3
    specific_ratio_threshold: float = 0.4
    byref_similarity_threshold: float = 0.95
    
    # Intervention strengths
    noise_strength: float = 0.1
    dropout_strength: float = 0.2
    mask_ratio: float = 0.15
    position_perturb_range: int = 2
    
    # Durations
    default_duration: int = 100
    
    # Limits
    max_active_interventions: int = 3


class InterventionApplicator:
    """
    Applies interventions to model inputs/outputs.
    
    Each intervention type has a specific application method.
    """
    
    def __init__(self, config: Optional[InterventionConfig] = None):
        self.config = config or InterventionConfig()
    
    def apply_noise(
        self, 
        embeddings: np.ndarray, 
        strength: float
    ) -> np.ndarray:
        """Add Gaussian noise to embeddings."""
        noise = np.random.normal(0, strength, embeddings.shape)
        return embeddings + noise
    
    def apply_token_masking(
        self, 
        token_ids: np.ndarray,
        mask_token_id: int,
        ratio: float
    ) -> np.ndarray:
        """Randomly mask tokens."""
        mask = np.random.random(token_ids.shape) < ratio
        masked = token_ids.copy()
        masked[mask] = mask_token_id
        return masked
    
    def apply_position_perturbation(
        self, 
        positions: np.ndarray,
        max_shift: int
    ) -> np.ndarray:
        """Slightly shuffle position indices."""
        shifts = np.random.randint(-max_shift, max_shift + 1, positions.shape)
        perturbed = positions + shifts
        # Clip to valid range
        perturbed = np.clip(perturbed, 0, positions.max())
        return perturbed


class GeneralizationNurturingTrainer:
    """
    Trainer wrapper that actively nurtures generalization.
    
    Monitors PAC tree health during training and applies interventions
    when patterns suggest memorization over generalization.
    
    Usage:
        from fracton.monitoring import (
            LanguageGeneralizationMonitor,
            GeneralizationNurturingTrainer
        )
        
        monitor = LanguageGeneralizationMonitor(model.pac_tree)
        trainer = GeneralizationNurturingTrainer(model, monitor, base_trainer)
        
        for batch in dataloader:
            loss = trainer.training_step(batch)
    """
    
    def __init__(
        self,
        model: Any,
        monitor: 'LanguageGeneralizationMonitor',
        base_trainer: Any,
        config: Optional[InterventionConfig] = None,
        verbose: bool = False
    ):
        """
        Initialize the nurturing trainer.
        
        Args:
            model: The model being trained
            monitor: LanguageGeneralizationMonitor attached to model
            base_trainer: The underlying trainer (handles actual training)
            config: Intervention configuration
            verbose: Print intervention information
        """
        self.model = model
        self.monitor = monitor
        self.base_trainer = base_trainer
        self.config = config or InterventionConfig()
        self.verbose = verbose
        
        self.applicator = InterventionApplicator(self.config)
        self._active_interventions: List[Intervention] = []
        self._intervention_history: List[Intervention] = []
        self._step_count = 0
    
    def training_step(self, batch: Any) -> float:
        """
        Execute one training step with intervention logic.
        
        Args:
            batch: Training batch
            
        Returns:
            Training loss
        """
        self._step_count += 1
        self.monitor.step()
        
        # Update active interventions
        self._active_interventions = [
            i for i in self._active_interventions 
            if i.is_active(self._step_count)
        ]
        
        # Check if we need new interventions
        self._check_and_apply_interventions()
        
        # Modify batch if needed
        modified_batch = self._apply_active_interventions(batch)
        
        # Run actual training step
        if hasattr(self.base_trainer, 'step_with_interventions'):
            loss = self.base_trainer.step_with_interventions(
                modified_batch, 
                self._active_interventions
            )
        else:
            # Fallback: just run normal step
            loss = self.base_trainer.training_step(modified_batch)
        
        return loss
    
    def _check_and_apply_interventions(self) -> None:
        """Check tree health and apply interventions if needed."""
        if len(self._active_interventions) >= self.config.max_active_interventions:
            return
        
        health = self.monitor.get_health_summary()
        
        # Check for entropy collapse risk (too many specific patterns)
        if health['specific_ratio'] > self.config.specific_ratio_threshold:
            high_risk = self.monitor.get_high_risk_patterns()
            if high_risk:
                self._add_intervention(Intervention(
                    type=InterventionType.ADD_NOISE,
                    target_patterns=high_risk[:10],  # Top 10 most risky
                    strength=self.config.noise_strength,
                    duration_steps=self.config.default_duration,
                    step_applied=self._step_count
                ))
        
        # Check for byref optimization opportunities
        byref_candidates = self.monitor.byref_candidates
        if byref_candidates:
            top_candidate = byref_candidates[0]
            if top_candidate.similarity_score > self.config.byref_similarity_threshold:
                self._add_intervention(Intervention(
                    type=InterventionType.BYREF_MERGE,
                    target_patterns=[top_candidate.branch_a, top_candidate.branch_b],
                    strength=1.0,
                    duration_steps=1,  # Immediate, one-time
                    step_applied=self._step_count
                ))
        
        # If overfitting detected, add token masking
        if health['is_overfitting']:
            self._add_intervention(Intervention(
                type=InterventionType.TOKEN_MASKING,
                target_patterns=[],  # Global
                strength=self.config.mask_ratio,
                duration_steps=self.config.default_duration * 2,
                step_applied=self._step_count
            ))
    
    def _add_intervention(self, intervention: Intervention) -> None:
        """Add an intervention to active list."""
        # Check if similar intervention already active
        for active in self._active_interventions:
            if active.type == intervention.type:
                return  # Don't duplicate
        
        self._active_interventions.append(intervention)
        self._intervention_history.append(intervention)
        
        if self.verbose:
            print(f"[Step {self._step_count}] Applied {intervention.type.value} "
                  f"(strength={intervention.strength}, duration={intervention.duration_steps})")
    
    def _apply_active_interventions(self, batch: Any) -> Any:
        """Apply all active interventions to the batch."""
        modified = batch
        
        for intervention in self._active_interventions:
            if intervention.type == InterventionType.ADD_NOISE:
                if hasattr(modified, 'embeddings'):
                    modified.embeddings = self.applicator.apply_noise(
                        modified.embeddings,
                        intervention.strength
                    )
                    
            elif intervention.type == InterventionType.TOKEN_MASKING:
                if hasattr(modified, 'input_ids') and hasattr(modified, 'mask_token_id'):
                    modified.input_ids = self.applicator.apply_token_masking(
                        modified.input_ids,
                        modified.mask_token_id,
                        intervention.strength
                    )
                    
            elif intervention.type == InterventionType.POSITION_PERTURB:
                if hasattr(modified, 'position_ids'):
                    modified.position_ids = self.applicator.apply_position_perturbation(
                        modified.position_ids,
                        self.config.position_perturb_range
                    )
        
        return modified
    
    @property
    def active_interventions(self) -> List[Intervention]:
        """Get currently active interventions."""
        return self._active_interventions.copy()
    
    @property
    def intervention_history(self) -> List[Intervention]:
        """Get full intervention history."""
        return self._intervention_history.copy()
    
    def get_intervention_stats(self) -> Dict[str, int]:
        """Get counts of interventions by type."""
        stats = {}
        for intervention in self._intervention_history:
            key = intervention.type.value
            stats[key] = stats.get(key, 0) + 1
        return stats
