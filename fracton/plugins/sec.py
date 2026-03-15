"""SECPlugin — gates recursion based on entropy thresholds."""

from __future__ import annotations

from typing import Any

from fracton.physics.constants import XI_SEC


class SECPlugin:
    """Gate recursion based on the SEC entropy threshold.

    Recursion is allowed only when the context entropy is above
    the SEC collapse threshold (XI_SEC ≈ 0.0618). Below this threshold,
    the system has "crystallized" and further recursion is suppressed.

    Args:
        threshold: Entropy threshold for recursion gating (default: XI_SEC).
    """

    def __init__(self, threshold: float = XI_SEC):
        self.threshold = threshold
        self._last_entropy: float = 1.0

    @property
    def name(self) -> str:
        return "sec"

    def on_recurse(self, context: Any, depth: int) -> None:
        """Track current entropy from context."""
        if hasattr(context, "entropy"):
            self._last_entropy = context.entropy
        elif isinstance(context, dict):
            self._last_entropy = context.get("entropy", 1.0)

    def on_crystallize(self, context: Any, result: Any) -> Any:
        """No-op — SEC gates recursion, doesn't modify results."""
        return result

    def validate(self, context: Any) -> bool:
        """Allow recursion only when entropy exceeds the SEC threshold."""
        return self._last_entropy > self.threshold
