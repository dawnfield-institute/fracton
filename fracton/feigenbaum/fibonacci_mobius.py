"""
Fibonacci Mobius transformation class.

M_n(z) = (F_{n+1}*z + F_n) / (F_n*z + F_{n-1})

Promoted from core/feigenbaum_mobius.py (v2.0) to feigenbaum/ (v2.1).
Rewritten to use fracton.fibonacci.fib() instead of local arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from ..constants.mathematical import PHI, PHI_INV
from ..fibonacci import fib


@dataclass
class FibonacciMobius:
    """Fibonacci Mobius transformation M_n(z) = (F_{n+1}z + F_n) / (F_n z + F_{n-1}).

    Key properties:
        - Fixed points: phi (stable) and -1/phi (unstable)
        - Eigenvalue at -1/phi: phi^(2n) for n even
        - Exact identity: F_{n+1} - F_n * phi = 1/phi^n
    """

    n: int

    def __post_init__(self):
        self.a = float(fib(self.n + 1))  # F_{n+1}
        self.b = float(fib(self.n))      # F_n
        self.c = float(fib(self.n))      # F_n
        self.d = float(fib(self.n - 1))  # F_{n-1}

    def __call__(self, z: complex) -> complex:
        """Apply M_n(z) = (az + b) / (cz + d)."""
        denom = self.c * z + self.d
        if abs(denom) < 1e-15:
            return complex("inf")
        return (self.a * z + self.b) / denom

    def derivative_at(self, z: complex) -> complex:
        """M'_n(z) = det(M) / (cz + d)^2."""
        det = self.a * self.d - self.b * self.c  # = (-1)^n
        return det / (self.c * z + self.d) ** 2

    @property
    def fixed_points(self) -> Tuple[float, float]:
        """Return (phi, -1/phi) — the universal fixed points."""
        return (PHI, -PHI_INV)

    @property
    def eigenvalue_at_unstable(self) -> float:
        """Eigenvalue at the unstable fixed point -1/phi.

        For M_10: equals phi^20 exactly.

        Proof: M'(-1/phi) = 1/(c(-1/phi) + d)^2 = 1/(F_n(-1/phi) + F_{n-1})^2
        For n=10: = 1/(55(-1/phi) + 34)^2 = 1/(89 - 55*phi)^2 = phi^20
        Because: 89 - 55*phi = 1/phi^10 (exact algebraic identity)
        """
        z = -PHI_INV
        return abs(self.derivative_at(z))

    def verify_eigenvalue_identity(self) -> Dict[str, Any]:
        """Verify the exact algebraic identity: F_{n+1} - F_n * phi = 1/phi^n.

        This is NOT numerical — it is algebraically exact.
        """
        lhs = self.a - self.b * PHI
        rhs = PHI_INV ** self.n
        error = abs(lhs - rhs)

        return {
            "n": self.n,
            "F_{n+1}": int(self.a),
            "F_n": int(self.b),
            "F_{n+1} - F_n*phi": lhs,
            "1/phi^n": rhs,
            "error": error,
            "is_exact": error < 1e-14,
        }


# Canonical M_10 instance
M10 = FibonacciMobius(10)
