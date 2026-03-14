"""
PAC energy cascade via eigenvalue-based partitioning.

Models how energy flows through a hierarchy of scales, with organization
measured by the dominant eigenvalue fraction of a coupling matrix.
The Landauer floor (kT·ln2) enforces minimum dissipation.

Canonical reference: milestone4 exp_03, milestone3 exp_03.
"""

import math
from typing import Any, Dict, List, Optional, Tuple


def coupling_matrix(
    n_modes: int,
    coupling_decay: float = 0.3,
    feedback: Optional[List[float]] = None,
) -> List[List[float]]:
    """Build a structured coupling matrix C[i,j] = exp(-|i-j| × decay).

    Args:
        n_modes: Number of modes.
        coupling_decay: Exponential decay rate for off-diagonal coupling.
        feedback: Optional nonlinear feedback vector from previous scale.

    Returns:
        n_modes × n_modes coupling matrix as nested list.
    """
    C = [[0.0] * n_modes for _ in range(n_modes)]
    for i in range(n_modes):
        for j in range(n_modes):
            C[i][j] = math.exp(-abs(i - j) * coupling_decay)

    if feedback is not None:
        for i in range(min(n_modes, len(feedback))):
            for j in range(min(n_modes, len(feedback))):
                C[i][j] += 0.1 * feedback[i] * feedback[j]

    return C


def _eigenvalues_power_method(
    matrix: List[List[float]],
    n_iter: int = 100,
) -> Tuple[float, List[float]]:
    """Compute dominant eigenvalue and eigenvector via power iteration.

    Pure Python — no numpy dependency.
    """
    n = len(matrix)
    # Start with uniform vector
    v = [1.0 / math.sqrt(n)] * n

    eigenvalue = 0.0
    for _ in range(n_iter):
        # Matrix-vector product
        w = [0.0] * n
        for i in range(n):
            for j in range(n):
                w[i] += matrix[i][j] * v[j]

        # Compute eigenvalue (Rayleigh quotient)
        eigenvalue = sum(w[i] * v[i] for i in range(n))

        # Normalize
        norm = math.sqrt(sum(x * x for x in w))
        if norm < 1e-30:
            break
        v = [x / norm for x in w]

    return eigenvalue, v


def _trace(matrix: List[List[float]]) -> float:
    """Sum of diagonal elements."""
    return sum(matrix[i][i] for i in range(len(matrix)))


def participation_ratio(eigenvalues: List[float]) -> float:
    """Compute participation ratio PR = (sum lambda_i)^2 / sum(lambda_i^2).

    Measures effective number of participating modes (1 = single mode,
    N = all modes equal).

    Args:
        eigenvalues: List of eigenvalues.

    Returns:
        Participation ratio.
    """
    s1 = sum(eigenvalues)
    s2 = sum(e * e for e in eigenvalues)
    if s2 < 1e-30:
        return 0.0
    return s1 * s1 / s2


def energy_cascade(
    injection_energy: float,
    n_scales: int,
    n_modes: int = 8,
    coupling_decay: float = 0.3,
    nonlinear_strength: float = 0.3,
    landauer_floor: float = None,
    dissipation_rate: float = 0.02,
) -> List[Dict[str, Any]]:
    """Run a PAC energy cascade through multiple scales.

    Energy flows from injection scale through n_scales, with organization
    measured by eigenvalue structure and Landauer minimum enforced.

    Args:
        injection_energy: Initial energy at injection scale.
        n_scales: Number of cascade scales.
        n_modes: Number of modes per scale.
        coupling_decay: Coupling matrix decay rate.
        nonlinear_strength: Strength of nonlinear feedback.
        landauer_floor: Minimum dissipation per step (default: ln(2)).
        dissipation_rate: Fraction of energy lost per scale (default 2%).

    Returns:
        List of per-scale dicts with keys:
        k_index, wavenumber, P_input, org_fraction, E_organized,
        E_transfer, participation_ratio, alive.
    """
    if landauer_floor is None:
        landauer_floor = math.log(2)

    results = []
    P = injection_energy
    prev_eigenvector = None

    for k in range(n_scales):
        wavenumber = 2 ** k

        # Build coupling matrix with optional nonlinear feedback
        feedback = None
        if prev_eigenvector is not None and nonlinear_strength > 0:
            feedback = [v * nonlinear_strength for v in prev_eigenvector]

        C = coupling_matrix(n_modes, coupling_decay, feedback)

        # Eigenvalue analysis
        lambda_max, eigenvector = _eigenvalues_power_method(C)
        trace_C = _trace(C)

        org_fraction = lambda_max / trace_C if trace_C > 0 else 0.0
        E_organized = P * org_fraction
        E_transfer = P * (1 - dissipation_rate)

        # Landauer floor enforcement
        alive = True
        if E_transfer < landauer_floor:
            E_transfer = landauer_floor
            alive = False

        # Approximate participation ratio from coupling structure
        # (full eigendecomposition would give exact PR)
        pr = n_modes * (1 - org_fraction) + org_fraction

        results.append({
            "k_index": k,
            "wavenumber": wavenumber,
            "P_input": P,
            "org_fraction": org_fraction,
            "E_organized": E_organized,
            "E_transfer": E_transfer,
            "participation_ratio": pr,
            "alive": alive,
        })

        prev_eigenvector = eigenvector
        P = E_transfer

    return results
