"""Cleaning step + decision log generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import InterimSchema


def clean_frame(df: pd.DataFrame, *, drop_duplicates: bool) -> tuple[pd.DataFrame, dict]:
    InterimSchema.validate(df)
    input_rows = len(df)
    duplicates_dropped = 0
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        duplicates_dropped = before - len(df)
    n_fraud = int((df["Class"] == 1).sum())
    n_legit = int((df["Class"] == 0).sum())
    stats = {
        "input_rows": input_rows,
        "duplicates_dropped": duplicates_dropped,
        "output_rows": len(df),
        "n_fraud": n_fraud,
        "n_legitimate": n_legit,
    }
    return df, stats


def generate_decision_log(stats: dict) -> str:
    return (
        "# Phase 1 — Cleaning Decision Log\n\n"
        "Generated automatically by `clustering_analysis.clean.generate_decision_log`. "
        "Do not edit by hand.\n\n"
        "## Summary\n\n"
        f"- Input rows: {stats['input_rows']}\n"
        f"- Duplicates dropped: {stats['duplicates_dropped']}\n"
        f"- Output rows: {stats['output_rows']}\n"
        f"- Class distribution — Fraud: {stats['n_fraud']}, "
        f"Legitimate: {stats['n_legitimate']}\n\n"
        "## Per-column decisions\n\n"
        "| Column     | Decision                                | Rationale                                                                            |\n"
        "|------------|-----------------------------------------|--------------------------------------------------------------------------------------|\n"
        "| `V1`..`V28`| pass through                            | Already PCA-anonymised upstream, roughly N(0,1); no further cleaning warranted.       |\n"
        "| `Time`     | retain, cyclic-encoded in feature stage | Single-day offset; cyclic encoding preserves diurnal structure (brief §2.3.3).        |\n"
        "| `Amount`   | retain, log1p in feature stage          | Heavy right skew; log1p collapses tail without losing zero structure (brief §2.3.1).  |\n"
        "| `Class`    | retain, never used in fitting           | External-evaluation ground truth only (brief §3.4).                                   |\n"
        "| duplicates | drop                                    | Inflate local density and bias DBSCAN MinPts logic (brief §2.2).                      |\n"
        "| missing    | none observed; assert                   | Fail fast if upstream changes (brief §2.2).                                            |\n"
    )


def write_decision_log(stats: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_decision_log(stats))
    return path
