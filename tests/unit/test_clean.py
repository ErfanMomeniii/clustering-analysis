import pandas as pd
import pytest

from clustering_analysis.clean import clean_frame, generate_decision_log


def _row(v1=0.0, time=0.0, amount=1.0, cls=0):
    return {
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "V1": v1,
        "Time": time,
        "Amount": amount,
        "Class": cls,
    }


def test_clean_frame_drops_exact_duplicates():
    df = pd.DataFrame([_row(), _row(), _row(v1=1.0)])
    result, stats = clean_frame(df, drop_duplicates=True)
    assert len(result) == 2
    assert stats["duplicates_dropped"] == 1


def test_clean_frame_no_dup_drop_when_disabled():
    df = pd.DataFrame([_row(), _row(), _row(v1=1.0)])
    result, stats = clean_frame(df, drop_duplicates=False)
    assert len(result) == 3
    assert stats["duplicates_dropped"] == 0


def test_clean_frame_rejects_missing_columns():
    df = pd.DataFrame([{"V1": 0.0}])
    with pytest.raises(Exception):
        clean_frame(df, drop_duplicates=True)


def test_clean_frame_records_label_distribution():
    df = pd.DataFrame([_row(cls=0), _row(cls=0, v1=1.0), _row(cls=1, v1=2.0)])
    _, stats = clean_frame(df, drop_duplicates=True)
    assert stats["n_legitimate"] == 2
    assert stats["n_fraud"] == 1


def test_generate_decision_log_includes_all_fields():
    stats = {
        "input_rows": 100,
        "duplicates_dropped": 1,
        "output_rows": 99,
        "n_fraud": 5,
        "n_legitimate": 94,
    }
    md = generate_decision_log(stats)
    assert "Input rows: 100" in md
    assert "Duplicates dropped: 1" in md
    assert "Output rows: 99" in md
    assert "Fraud: 5" in md
    assert "Legitimate: 94" in md
