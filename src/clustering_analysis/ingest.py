"""Source acquisition + checksum + manifest for DS-10 raw data."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .io_utils import ensure_parents, raw_path
from .schemas import RawSchema

CHUNK = 1 << 20  # 1 MiB


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(path: Path, n_rows: int, source: str) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "row_count": n_rows,
        "sha256": sha256_of_file(path),
        "source": source,
        "downloaded_at": datetime.now(UTC).isoformat(),
    }


def validate_raw_frame(df: pd.DataFrame) -> None:
    RawSchema.validate(df)


def verify_source_expectations(
    path: Path,
    df: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    expected_columns: int | None = None,
    checksum_sha256: str | None = None,
) -> None:
    """Fail fast when the source file is not the dataset the params pin.

    The schema alone cannot catch a truncated download or a silently updated
    upstream file — both would corrupt every downstream clustering result while
    still passing column-level validation. Any expectation left as ``None`` is
    skipped, which keeps unit tests free to use small fixtures.
    """
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {path.name}, got {len(df)}")
    if expected_columns is not None and df.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} columns in {path.name}, got {df.shape[1]}")
    if checksum_sha256 is not None:
        actual = sha256_of_file(path)
        if actual != checksum_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: expected {checksum_sha256}, got {actual}"
            )


def ingest_csv(
    local_csv: Path,
    source: str = "local",
    *,
    expected_rows: int | None = None,
    expected_columns: int | None = None,
    checksum_sha256: str | None = None,
) -> tuple[Path, dict]:
    """Validate and stage a locally-present creditcard.csv into data/raw/.

    For Kaggle download, the user runs: `kaggle datasets download -d mlg-ulb/creditcardfraud`
    and unzips to data/raw/creditcard.csv before calling this function.

    The ``expected_*`` / ``checksum_sha256`` arguments come from
    ``params.yaml::ingest`` in the DVC stage, so the pinned dataset identity is
    enforced at the pipeline boundary rather than merely documented.
    """
    df = pd.read_csv(local_csv)
    validate_raw_frame(df)
    verify_source_expectations(
        local_csv,
        df,
        expected_rows=expected_rows,
        expected_columns=expected_columns,
        checksum_sha256=checksum_sha256,
    )
    dest = raw_path("creditcard.csv")
    ensure_parents(dest)
    if local_csv.resolve() != dest.resolve():
        dest.write_bytes(local_csv.read_bytes())
    parquet_dest = raw_path("creditcard.parquet")
    df.to_parquet(parquet_dest, index=False)
    manifest = build_manifest(dest, n_rows=len(df), source=source)
    manifest_path = raw_path("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return parquet_dest, manifest
