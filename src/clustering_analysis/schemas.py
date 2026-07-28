"""Pandera schemas for raw, interim, and processed feature frames."""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

_V_COLUMNS = {f"V{i}": Column(float, nullable=False) for i in range(1, 29)}

RawSchema = DataFrameSchema(
    {
        **_V_COLUMNS,
        "Time": Column(float, Check.ge(0), nullable=False),
        "Amount": Column(float, Check.ge(0), nullable=False),
        "Class": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=True,
    coerce=True,
)

InterimSchema = DataFrameSchema(
    {
        **_V_COLUMNS,
        "Time": Column(float, Check.ge(0), nullable=False),
        "Amount": Column(float, Check.ge(0), nullable=False),
        "Class": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=True,
    coerce=True,
)

ProcessedSchema = DataFrameSchema(
    {
        **_V_COLUMNS,
        "log_amount": Column(float, nullable=False),
        "time_sin": Column(float, Check.in_range(-1.0, 1.0), nullable=False),
        "time_cos": Column(float, Check.in_range(-1.0, 1.0), nullable=False),
    },
    strict=True,
    coerce=True,
)
