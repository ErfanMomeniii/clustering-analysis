"""Model (centroid) registry for the production pipeline (Phase 3 §4.5).

Serialises a fitted clustering artefact together with the metadata that makes it
auditable months later: fit timestamp, data version, hyperparameters, and the
full metric scoreboard. Without the metadata a persisted centroid file is
unusable — nobody can tell which data version or params produced it.

Entries are content-addressed by a short hash of ``(name, data_version, params)``
so re-registering the same configuration overwrites rather than accumulating
near-duplicates, while a genuine change lands as a new entry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = REPO_ROOT / "models" / "registry"
INDEX_NAME = "index.json"


@dataclass
class ModelCard:
    """Metadata describing one registered clustering artefact."""

    name: str
    algorithm: str
    k: int
    data_version: str
    n_rows: int
    n_features: int
    feature_space: str
    params: dict[str, Any]
    metrics: dict[str, float]
    seeds: dict[str, int]
    fit_date: str
    entry_id: str
    artifact_path: str | None = None
    notes: str = ""
    tags: list[str] = field(default_factory=list)


def _portable_path(path: Path) -> str:
    """Record a repo-relative path when possible, else an absolute one.

    Repo-relative keeps a committed registry index portable across machines. A
    registry rooted outside the repo (a temp dir in tests, a shared volume in
    production) simply records its absolute path rather than failing.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _entry_id(name: str, data_version: str, params: dict) -> str:
    payload = json.dumps(
        {"name": name, "data_version": data_version, "params": params},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def register_model(
    *,
    name: str,
    algorithm: str,
    k: int,
    data_version: str,
    n_rows: int,
    n_features: int,
    feature_space: str,
    params: dict,
    metrics: dict,
    seeds: dict,
    artifact: Any | None = None,
    notes: str = "",
    tags: list[str] | None = None,
    fit_date: str | None = None,
    root: Path | None = None,
) -> ModelCard:
    """Persist an artefact plus its card and update the registry index.

    ``fit_date`` is injectable so tests stay deterministic; production callers
    leave it unset and get the current UTC timestamp.
    """
    root = root or REGISTRY_ROOT
    root.mkdir(parents=True, exist_ok=True)
    entry = _entry_id(name, data_version, params)

    artifact_path: str | None = None
    if artifact is not None:
        dest = root / f"{name}-{entry}.joblib"
        joblib.dump(artifact, dest)
        artifact_path = _portable_path(dest)

    card = ModelCard(
        name=name,
        algorithm=algorithm,
        k=k,
        data_version=data_version,
        n_rows=n_rows,
        n_features=n_features,
        feature_space=feature_space,
        params=params,
        metrics={str(m): (None if v is None else float(v)) for m, v in metrics.items()},
        seeds=seeds,
        fit_date=fit_date or datetime.now(UTC).isoformat(),
        entry_id=entry,
        artifact_path=artifact_path,
        notes=notes,
        tags=tags or [],
    )
    (root / f"{name}-{entry}.json").write_text(json.dumps(asdict(card), indent=2) + "\n")
    _update_index(card, root=root)
    return card


def _update_index(card: ModelCard, *, root: Path) -> Path:
    index_path = root / INDEX_NAME
    entries = {}
    if index_path.exists():
        entries = {e["entry_id"]: e for e in json.loads(index_path.read_text())["entries"]}
    entries[card.entry_id] = {
        "entry_id": card.entry_id,
        "name": card.name,
        "algorithm": card.algorithm,
        "k": card.k,
        "data_version": card.data_version,
        "fit_date": card.fit_date,
        "metrics": card.metrics,
        "artifact_path": card.artifact_path,
        "tags": card.tags,
    }
    ordered = sorted(entries.values(), key=lambda e: (e["name"], e["fit_date"]))
    index_path.write_text(json.dumps({"entries": ordered}, indent=2) + "\n")
    return index_path


def list_models(root: Path | None = None) -> list[dict]:
    index_path = (root or REGISTRY_ROOT) / INDEX_NAME
    if not index_path.exists():
        return []
    return json.loads(index_path.read_text())["entries"]


def load_card(entry_id: str, root: Path | None = None) -> ModelCard:
    root = root or REGISTRY_ROOT
    for path in root.glob("*.json"):
        if path.name == INDEX_NAME:
            continue
        payload = json.loads(path.read_text())
        if payload.get("entry_id") == entry_id:
            return ModelCard(**payload)
    raise KeyError(f"No registry entry with id {entry_id!r} under {root}")


def load_artifact(entry_id: str, root: Path | None = None) -> Any:
    card = load_card(entry_id, root=root)
    if not card.artifact_path:
        raise ValueError(f"Registry entry {entry_id!r} has no serialised artefact")
    stored = Path(card.artifact_path)
    return joblib.load(stored if stored.is_absolute() else REPO_ROOT / stored)


def data_version_from_manifest(manifest_path: Path | None = None) -> str:
    """Derive the data version from the ingest manifest's checksum.

    Ties every registered model to the exact raw file it was fit on; falls back
    to ``"unknown"`` when the manifest is absent so registration never becomes
    the reason a pipeline run fails.
    """
    manifest_path = manifest_path or REPO_ROOT / "data" / "raw" / "manifest.json"
    if not manifest_path.exists():
        return "unknown"
    return json.loads(manifest_path.read_text()).get("sha256", "unknown")[:12]
