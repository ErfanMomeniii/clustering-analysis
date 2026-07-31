"""Artefact emission for the phase reports.

Every number quoted in a report must be traceable to a run, so the pipeline
never hand-copies figures into LaTeX. Instead each stage writes:

  - ``results/<phase>/*.json`` — the machine-readable record of the run;
  - ``reports/tables/*.tex`` — booktabs tables the report ``\\input``s.

That way ``dvc repro`` regenerates the report content, and a stale number is a
pipeline failure rather than a transcription error nobody notices.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
TABLES_ROOT = REPO_ROOT / "reports" / "tables"
FIGURES_ROOT = REPO_ROOT / "reports" / "figures"


def _jsonable(obj: Any):
    """Coerce numpy scalars/arrays into JSON-serialisable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=False) + "\n")
    return path


def results_path(phase: str, name: str) -> Path:
    return RESULTS_ROOT / phase / name


def figure_path(name: str) -> Path:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    return FIGURES_ROOT / name


# Non-ASCII characters that analysis code naturally emits (σ for standard
# deviations, × for factors) but that pdflatex cannot typeset. Substituted after
# backslash escaping so the replacements' own backslashes survive.
_UNICODE_TO_LATEX = {
    "σ": r"$\sigma$",
    "×": r"$\times$",
    "±": r"$\pm$",
    "≥": r"$\geq$",
    "≤": r"$\leq$",
    "≈": r"$\approx$",
    "→": r"$\rightarrow$",
    "—": "---",
    "–": "--",
}


def _escape(text: str) -> str:
    """Make arbitrary driver-generated text safe for a pdflatex table cell."""
    for a, b in (("\\", r"\textbackslash "), ("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
        text = text.replace(a, b)
    for char, latex in _UNICODE_TO_LATEX.items():
        text = text.replace(char, latex)
    return text


def safe_name(text: str) -> str:
    """Escape an identifier for interpolation into a caption.

    Captions carry intentional LaTeX (``$k=2$``, ``\\,``, ``\\sim``) so they cannot
    be escaped wholesale; but identifiers interpolated into them (``ae_kmeans``,
    ``davies_bouldin``) contain underscores that abort the build. Escape the
    values, never the surrounding caption.
    """
    return text.replace("_", r"\_")


def thousands(value: int | float) -> str:
    """Format an integer with LaTeX-safe thousands separators.

    LaTeX collapses a bare ``,`` in maths mode, so groups are wrapped as
    ``{,}``. Captions must call this on the *number* rather than running a
    blanket comma replacement over the whole string, which would corrupt thin
    spaces (``\\,``) into ``\\{,}`` and break the build.
    """
    return f"{int(value):,}".replace(",", "{,}")


def _fmt(value: Any, precision: int) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float | np.floating):
        v = float(value)
        if np.isnan(v):
            return "--"
        if v != 0 and (abs(v) >= 1e5 or abs(v) < 1e-3):
            # Inline maths rather than \num{}: siunitx is absent from BasicTeX
            # installs, and a report that will not compile is not a deliverable.
            mantissa, exponent = f"{v:.{min(precision, 3)}e}".split("e")
            return f"${mantissa}\\times 10^{{{int(exponent)}}}$"
        return f"{v:.{precision}f}"
    if isinstance(value, int | np.integer):
        return thousands(value)
    return _escape(str(value))


def write_latex_table(
    rows: Sequence[dict],
    path: Path,
    *,
    columns: Sequence[str] | None = None,
    headers: dict[str, str] | None = None,
    caption: str | None = None,
    label: str | None = None,
    precision: int = 3,
) -> Path:
    """Write rows as a standalone booktabs tabular for ``\\input``.

    Emits only the ``tabular`` (optionally wrapped in ``table``) so the report
    controls placement. Missing / NaN cells render as ``--`` rather than as the
    string "nan", which would otherwise reach the graded PDF.
    """
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("% no rows produced by this run\n")
        return path

    cols = list(columns) if columns else list(rows[0].keys())
    headers = headers or {}
    align = "l" + "r" * (len(cols) - 1)

    lines = []
    if caption or label:
        lines.append("\\begin{table}[htbp]\n\\centering")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(_escape(headers.get(c, c)) for c in cols) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(_fmt(row.get(c), precision) for c in cols) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    if caption or label:
        lines.append("\\end{table}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_matrix_table(
    names: Sequence[str],
    matrix: np.ndarray,
    path: Path,
    *,
    caption: str | None = None,
    label: str | None = None,
    precision: int = 3,
) -> Path:
    """Write a square labelled matrix (e.g. pairwise ARI) as a booktabs table."""
    rows = []
    for i, name in enumerate(names):
        row = {"": name}
        for j, other in enumerate(names):
            row[other] = float(matrix[i, j])
        rows.append(row)
    return write_latex_table(
        rows,
        path,
        columns=[""] + list(names),
        caption=caption,
        label=label,
        precision=precision,
    )
