"""
Experiment result storage and serialization.

Provides structured result collection following the Part-based architecture
used across all DFT experiments (Part A, B, C... then Synthesis).
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class PartResult:
    """Result from a single experiment part (A, B, C, ...).

    Each part has a description, finding, pass/fail status, and optional
    rows of tabular data.

    Examples:
        >>> part = PartResult("A", "Cascade density profile")
        >>> part.add_row({"r": 1.5, "rho": 0.667})
        >>> part.finding = "Density matches 1/r prediction"
        >>> part.passed = True
    """

    def __init__(self, label: str, description: str):
        self.label = label
        self.description = description
        self.finding: Optional[str] = None
        self.passed: Optional[bool] = None
        self.rows: List[Dict[str, Any]] = []
        self.data: Dict[str, Any] = {}

    def add_row(self, row: Dict[str, Any]) -> None:
        """Add a data row (for tabular results)."""
        self.rows.append(row)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON output."""
        d: Dict[str, Any] = {"description": self.description}
        if self.finding:
            d["finding"] = self.finding
        if self.passed is not None:
            d["passed"] = self.passed
        if self.rows:
            d["rows"] = self.rows
        if self.data:
            d.update(self.data)
        return d


class ExperimentResult:
    """Structured experiment result following DFT conventions.

    Collects parts, tracks pass/fail, and serializes to timestamped JSON.

    Examples:
        >>> result = ExperimentResult("exp_42_cascade_test")
        >>> part_a = result.add_part("A", "Test cascade convergence")
        >>> part_a.passed = True
        >>> part_a.finding = "Converges to phi^-k"
        >>> result.synthesize("CONFIRMED", "All parts pass")
        >>> result.save("path/to/results/")
    """

    def __init__(self, name: str, timestamp: Optional[str] = None):
        self.name = name
        self.timestamp = timestamp or datetime.now().isoformat()
        self.parts: Dict[str, PartResult] = {}
        self.synthesis: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def add_part(self, label: str, description: str) -> PartResult:
        """Create and register a new experiment part."""
        part = PartResult(label, description)
        self.parts[label] = part
        return part

    def synthesize(self, status: str, verdict: str, **extra: Any) -> Dict[str, Any]:
        """Set synthesis/conclusion for the experiment.

        Args:
            status: "CONFIRMED", "PARTIAL", or "FAIL".
            verdict: Summary conclusion.
            **extra: Additional synthesis fields.

        Returns:
            The synthesis dict.
        """
        pass_fail = {}
        for label, part in self.parts.items():
            if part.passed is not None:
                key = f"{label}_{part.description[:30].replace(' ', '_').lower()}"
                pass_fail[key] = part.passed

        self.synthesis = {
            "status": status,
            "verdict": verdict,
            "pass_fail": pass_fail,
            **extra,
        }
        return self.synthesis

    @property
    def overall_passed(self) -> bool:
        """True if all parts with pass/fail tracking passed."""
        tracked = [p.passed for p in self.parts.values() if p.passed is not None]
        return all(tracked) if tracked else False

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization to dict."""
        d: Dict[str, Any] = {
            "experiment": self.name,
            "timestamp": self.timestamp,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        d["parts"] = {label: part.to_dict() for label, part in self.parts.items()}
        if self.synthesis:
            d["synthesis"] = self.synthesis
        return d

    def save(self, results_dir: Union[str, Path], filename: Optional[str] = None) -> Path:
        """Save results as timestamped JSON.

        Args:
            results_dir: Directory for output files.
            filename: Override filename (default: {name}_{date}.json).

        Returns:
            Path to the saved file.
        """
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{date_str}.json"

        filepath = results_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return filepath


def save_results(
    results: Dict[str, Any],
    results_dir: Union[str, Path],
    name: str,
) -> Path:
    """Save a raw results dict to timestamped JSON (legacy interface).

    For new code, prefer ExperimentResult.save().

    Args:
        results: Dict to serialize.
        results_dir: Output directory.
        name: Experiment name (used in filename).

    Returns:
        Path to saved file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = results_dir / f"{name}_{date_str}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return filepath
