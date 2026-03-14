"""
Timing utilities for experiments.

Provides a context-manager timer for measuring computation time.
"""

import time


class Timer:
    """Context-manager timer for experiment sections.

    Examples:
        >>> with Timer() as t:
        ...     result = heavy_computation()
        >>> print(f"Took {t.elapsed:.3f}s")
    """

    def __init__(self):
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self.start


def timer() -> Timer:
    """Create a Timer instance (convenience factory).

    Returns:
        Timer context manager.

    Examples:
        >>> with timer() as t:
        ...     do_work()
        >>> print(f"{t.elapsed:.3f}s")
    """
    return Timer()
