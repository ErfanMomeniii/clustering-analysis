import hashlib

import pandas as pd
import pytest
from pandera.errors import SchemaError

from clustering_analysis.ingest import (
    build_manifest,
    sha256_of_file,
    validate_raw_frame,
    verify_source_expectations,
)


def _valid_raw_df(n=2):
    row = {**{f"V{i}": 0.0 for i in range(1, 29)}, "Time": 0.0, "Amount": 1.0, "Class": 0}
    return pd.DataFrame([row] * n)


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
    with pytest.raises(SchemaError):
        validate_raw_frame(df)


def test_verify_source_expectations_accepts_matching_source(tmp_path):
    p = tmp_path / "creditcard.csv"
    p.write_text("payload")
    df = _valid_raw_df(2)
    verify_source_expectations(
        p,
        df,
        expected_rows=2,
        expected_columns=31,
        checksum_sha256=hashlib.sha256(b"payload").hexdigest(),
    )  # no exception


def test_verify_source_expectations_rejects_truncated_row_count(tmp_path):
    p = tmp_path / "creditcard.csv"
    p.write_text("payload")
    with pytest.raises(ValueError, match="Expected 284807 rows"):
        verify_source_expectations(p, _valid_raw_df(2), expected_rows=284807)


def test_verify_source_expectations_rejects_wrong_column_count(tmp_path):
    p = tmp_path / "creditcard.csv"
    p.write_text("payload")
    with pytest.raises(ValueError, match="Expected 99 columns"):
        verify_source_expectations(p, _valid_raw_df(2), expected_columns=99)


def test_verify_source_expectations_rejects_checksum_mismatch(tmp_path):
    p = tmp_path / "creditcard.csv"
    p.write_text("payload")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_source_expectations(p, _valid_raw_df(2), checksum_sha256="deadbeef")


def test_verify_source_expectations_skips_none_expectations(tmp_path):
    p = tmp_path / "creditcard.csv"
    p.write_text("payload")
    verify_source_expectations(p, _valid_raw_df(2))  # all expectations opt-in
