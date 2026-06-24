import pandas as pd
import pytest
from clustering_analysis.schemas import RawSchema, InterimSchema, ProcessedSchema

def _valid_raw_row():
    return {
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "Time": 0.0,
        "Amount": 1.0,
        "Class": 0,
    }

def test_raw_schema_accepts_valid_row():
    df = pd.DataFrame([_valid_raw_row()])
    RawSchema.validate(df)

def test_raw_schema_rejects_invalid_class():
    df = pd.DataFrame([{**_valid_raw_row(), "Class": 2}])
    with pytest.raises(Exception):
        RawSchema.validate(df)

def test_raw_schema_rejects_negative_amount():
    df = pd.DataFrame([{**_valid_raw_row(), "Amount": -1.0}])
    with pytest.raises(Exception):
        RawSchema.validate(df)

def test_interim_schema_accepts_cleaned_row():
    df = pd.DataFrame([_valid_raw_row()])
    InterimSchema.validate(df)

def test_processed_schema_requires_engineered_features():
    row = {f"V{i}": 0.0 for i in range(1, 29)}
    row.update({"log_amount": 0.0, "time_sin": 0.0, "time_cos": 1.0})
    df = pd.DataFrame([row])
    ProcessedSchema.validate(df)

def test_processed_schema_rejects_raw_columns_present():
    row = {f"V{i}": 0.0 for i in range(1, 29)}
    row.update({"log_amount": 0.0, "time_sin": 0.0, "time_cos": 1.0, "Amount": 1.0})
    df = pd.DataFrame([row])
    with pytest.raises(Exception):
        ProcessedSchema.validate(df)
