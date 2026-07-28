import hashlib

import pandas as pd
import pytest

from clustering_analysis.ingest import build_manifest, sha256_of_file, validate_raw_frame


def test_sha256_of_file_matches_python_hashlib(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert sha256_of_file(p) == expected


def test_build_manifest_records_path_size_checksum_rowcount(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    m = build_manifest(p, n_rows=2, source="file://x.csv")
    assert m["path"].endswith("x.csv")
    assert m["row_count"] == 2
    assert m["source"] == "file://x.csv"
    assert m["sha256"] == sha256_of_file(p)
    assert "downloaded_at" in m
    assert m["size_bytes"] > 0


def test_validate_raw_frame_accepts_minimum_valid():
    df = pd.DataFrame(
        [{**{f"V{i}": 0.0 for i in range(1, 29)}, "Time": 0.0, "Amount": 1.0, "Class": 0}]
    )
    validate_raw_frame(df)  # no exception


def test_validate_raw_frame_rejects_wrong_column_count():
    df = pd.DataFrame([{"V1": 0.0}])
    with pytest.raises(Exception):
        validate_raw_frame(df)
