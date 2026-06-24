"""Source acquisition + checksum + manifest for DS-10 raw data."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from .io_utils import raw_path, ensure_parents
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
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }

def validate_raw_frame(df: pd.DataFrame) -> None:
    RawSchema.validate(df)

def ingest_csv(local_csv: Path, source: str = "local") -> tuple[Path, dict]:
    """Validate and stage a locally-present creditcard.csv into data/raw/.

    For Kaggle download, the user runs: `kaggle datasets download -d mlg-ulb/creditcardfraud`
    and unzips to data/raw/creditcard.csv before calling this function.
    """
    df = pd.read_csv(local_csv)
    validate_raw_frame(df)
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
