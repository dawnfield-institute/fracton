"""CascadePlugin — applies φ⁻ᵏ energy decay at each recursion depth."""

from __future__ import annotations

from typing import Any

from fracton.physics.constants import PHI_INV


class CascadePlugin:
    """Apply golden-ratio energy cascade decay at each recursion level.

    At depth k, the effective energy is scaled by φ⁻ᵏ. This enforces
    the PAC cascade principle: energy organizes hierarchically with
    golden-ratio partitioning.

    Args:
        base_decay: Decay base per recursion level (default: φ⁻¹ ≈ 0.618).
        min_energy: Minimum energy below which recursion is halted.
    """

    def __init__(
        self,
        base_decay: float = PHI_INV,
        min_energy: float = 1e-12,
    ):
        self.base_decay = base_decay
        self.min_energy = min_energy
        self._last_scale: float = 1.0

    @property
    def name(self) -> str:
        return "cascade"

    def on_recurse(self, context: Any, depth: int) -> None:
        """Scale energy by φ⁻ᵏ and store scale factor on context."""
        self._last_scale = self.base_decay ** depth
        if hasattr(context, "metadata") and isinstance(context.metadata, dict):
            context.metadata["cascade_scale"] = self._last_scale

    def on_crystallize(self, context: Any, result: Any) -> Any:
        """Scale numeric results by the cascade factor."""
        if isinstance(result, (int, float)):
            return result * self._last_scale
        return result

    def validate(self, context: Any) -> bool:
        """Halt recursion when energy drops below minimum."""
        return self._last_scale >= self.min_energy
