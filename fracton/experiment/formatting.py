"""
Experiment output formatting.

Standardized print functions for headers, sections, tables, and results.
Matches the formatting conventions used across all DFT experiments.
"""

import sys
from typing import Any, Dict, List, Optional, Sequence, Union


def _ensure_utf8() -> None:
    """Ensure stdout can handle Unicode (Greek letters, etc.)."""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def print_header(title: str, subtitle: Optional[str] = None, width: int = 72) -> None:
    """Print a major section header.

    Args:
        title: Header title.
        subtitle: Optional subtitle line.
        width: Line width (default 72).

    Examples:
        >>> print_header("Part A: Cascade Density")
        ========================================================================
        Part A: Cascade Density
        ========================================================================
    """
    _ensure_utf8()
    print(f"\n{'=' * width}")
    print(title)
    if subtitle:
        print(subtitle)
    print("=" * width)


def print_section(title: str, width: int = 60) -> None:
    """Print a minor section divider.

    Args:
        title: Section title.
        width: Line width.
    """
    _ensure_utf8()
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print("-" * width)


def print_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    col_widths: Optional[Sequence[int]] = None,
    fmt: Optional[Sequence[str]] = None,
    indent: int = 2,
) -> None:
    """Print a formatted table with aligned columns.

    Args:
        headers: Column header strings.
        rows: List of row tuples/lists.
        col_widths: Optional explicit column widths.
        fmt: Optional format strings per column (e.g., [">10s", ">15.8f"]).
        indent: Left indent spaces.

    Examples:
        >>> print_table(["r/r_s", "rho", "Status"],
        ...             [(1.5, 0.667, "PASS"), (2.0, 0.500, "PASS")])
    """
    _ensure_utf8()
    n_cols = len(headers)

    if col_widths is None:
        col_widths = []
        for i in range(n_cols):
            max_w = len(str(headers[i]))
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i])))
            col_widths.append(max_w + 2)

    prefix = " " * indent

    # Header
    header_parts = []
    for i, h in enumerate(headers):
        w = col_widths[i] if i < len(col_widths) else 12
        header_parts.append(f"{h:>{w}s}")
    print(f"{prefix}{'  '.join(header_parts)}")

    # Separator
    total_w = sum(col_widths) + 2 * (n_cols - 1)
    print(f"{prefix}{'-' * total_w}")

    # Rows
    for row in rows:
        parts = []
        for i, val in enumerate(row):
            w = col_widths[i] if i < len(col_widths) else 12
            if fmt and i < len(fmt):
                parts.append(f"{val:{fmt[i]}}")
            elif isinstance(val, float):
                parts.append(f"{val:>{w}.6g}")
            else:
                parts.append(f"{str(val):>{w}s}")
        print(f"{prefix}{'  '.join(parts)}")


def print_result(
    label: str,
    predicted: float,
    measured: float,
    unit: str = "",
    error_type: str = "pct",
) -> None:
    """Print a single predicted-vs-measured comparison.

    Args:
        label: Name of the quantity.
        predicted: PAC-derived value.
        measured: Measured/CODATA value.
        unit: Unit string (e.g., "m/s", "ppm").
        error_type: "pct" for percent, "ppm" for parts-per-million.

    Examples:
        >>> print_result("alpha_EM", 0.007297311, 0.007297353, error_type="ppm")
          alpha_EM:  predicted = 0.007297311  measured = 0.007297353  error = 5.7 ppm
    """
    _ensure_utf8()
    if measured != 0:
        diff = abs(predicted - measured) / abs(measured)
        if error_type == "ppm":
            err_str = f"{diff * 1e6:.1f} ppm"
        else:
            err_str = f"{diff * 100:.4f}%"
    else:
        err_str = "N/A"

    unit_str = f" {unit}" if unit else ""
    print(f"  {label}:  predicted = {predicted:.10g}{unit_str}"
          f"  measured = {measured:.10g}{unit_str}  error = {err_str}")
