# Phase 1 — Cleaning Decision Log

Generated automatically by `clustering_analysis.clean.generate_decision_log`. Do not edit by hand.

## Summary

- Input rows: 284807
- Duplicates dropped: 1081
- Output rows: 283726
- Class distribution — Fraud: 473, Legitimate: 283253

## Per-column decisions

| Column     | Decision                                | Rationale                                                                            |
|------------|-----------------------------------------|--------------------------------------------------------------------------------------|
| `V1`..`V28`| pass through                            | Already PCA-anonymised upstream, roughly N(0,1); no further cleaning warranted.       |
| `Time`     | retain, cyclic-encoded in feature stage | Single-day offset; cyclic encoding preserves diurnal structure (brief §2.3.3).        |
| `Amount`   | retain, log1p in feature stage          | Heavy right skew; log1p collapses tail without losing zero structure (brief §2.3.1).  |
| `Class`    | retain, never used in fitting           | External-evaluation ground truth only (brief §3.4).                                   |
| duplicates | drop                                    | Inflate local density and bias DBSCAN MinPts logic (brief §2.2).                      |
| missing    | none observed; assert                   | Fail fast if upstream changes (brief §2.2).                                            |
