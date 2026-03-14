"""
Experiment metadata and documentation helpers.

Provides structured metadata generation for experiment headers
and docstrings following DFT conventions.
"""

from datetime import datetime
from typing import Dict, Optional


def experiment_header(
    name: str,
    description: str,
    paper: Optional[str] = None,
    section: Optional[str] = None,
    milestone: Optional[str] = None,
) -> Dict[str, str]:
    """Print experiment header and return metadata dict.

    Args:
        name: Experiment name (e.g., "exp_42_cascade_test").
        description: One-line description.
        paper: Source paper reference (e.g., "PACSeries Paper 4").
        section: Section within the paper.
        milestone: Milestone identifier (e.g., "milestone4").

    Returns:
        Metadata dict with experiment, description, paper, section,
        timestamp, milestone fields.

    Examples:
        >>> meta = experiment_header("exp_42", "Test cascade convergence",
        ...                          paper="PACSeries Paper 6")
    """
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"  {description}")
    if paper:
        print(f"  Source: {paper}" + (f" {section}" if section else ""))
    if milestone:
        print(f"  Milestone: {milestone}")
    print(f"{'=' * 70}\n")

    return {
        "experiment": name,
        "description": description,
        "paper": paper or "",
        "section": section or "",
        "timestamp": datetime.now().isoformat(),
        "milestone": milestone or "",
    }


def experiment_docstring(
    purpose: str,
    hypothesis: list,
    design: list,
    corpus_context: list,
    output: str,
) -> str:
    """Generate a standard experiment docstring.

    All DFT experiments follow this template in their module docstring.

    Args:
        purpose: What this experiment tests.
        hypothesis: List of hypothesis strings.
        design: List of design parts (e.g., ["Part A: ...", "Part B: ..."]).
        corpus_context: List of related experiment references.
        output: Output path/description.

    Returns:
        Formatted docstring.
    """
    lines = [f"PURPOSE: {purpose}", ""]

    lines.append("HYPOTHESIS:")
    for i, h in enumerate(hypothesis, 1):
        lines.append(f"    {i}. {h}")
    lines.append("")

    lines.append("DESIGN:")
    for d in design:
        lines.append(f"    - {d}")
    lines.append("")

    lines.append("CORPUS CONTEXT:")
    for c in corpus_context:
        lines.append(f"    - {c}")
    lines.append("")

    lines.append(f"OUTPUT: {output}")
    return "\n".join(lines)
