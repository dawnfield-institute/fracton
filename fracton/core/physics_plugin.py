"""
PhysicsPlugin protocol — interface for physics modules to plug into RecursiveExecutor.

Bridges the runtime half (RecursiveExecutor, MemoryField, PACRegulator) with the
physics half (cascade, corrections, SEC evolution) via lifecycle hooks called at
each recursion/crystallization point.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PhysicsPlugin(Protocol):
    """Interface for physics modules to plug into RecursiveExecutor.

    Plugins receive callbacks at two points in the recursive lifecycle:
    - on_recurse: called before each recursive descent (can modify context)
    - on_crystallize: called when a result is produced (can modify result)
    - validate: called to check whether recursion should continue
    """

    @property
    def name(self) -> str:
        """Plugin identifier."""
        ...

    def on_recurse(self, context: Any, depth: int) -> None:
        """Called before each recursive descent.

        Args:
            context: ExecutionContext (or dict) for this recursion level.
            depth: Current recursion depth.
        """
        ...

    def on_crystallize(self, context: Any, result: Any) -> Any:
        """Called when a recursion level produces a result.

        Args:
            context: ExecutionContext for this level.
            result: The computed result.

        Returns:
            Optionally modified result.
        """
        ...

    def validate(self, context: Any) -> bool:
        """Check whether recursion should continue.

        Args:
            context: ExecutionContext to validate.

        Returns:
            True if recursion may proceed, False to halt.
        """
        ...
