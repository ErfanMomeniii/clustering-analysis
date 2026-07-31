import json

import numpy as np

from clustering_analysis.reporting import (
    write_json,
    write_latex_table,
    write_matrix_table,
)


def test_latex_table_has_booktabs_rules_and_one_row_per_record(tmp_path):
    rows = [{"algorithm": "kmeans", "silhouette": 0.42}, {"algorithm": "gmm", "silhouette": 0.31}]
    text = write_latex_table(rows, tmp_path / "t.tex").read_text()
    assert "\\toprule" in text and "\\midrule" in text and "\\bottomrule" in text
    assert text.count("\\\\") == 3  # header + two data rows


def test_latex_table_escapes_underscores_that_would_break_compilation(tmp_path):
    text = write_latex_table([{"metric": "davies_bouldin", "v": 1.0}], tmp_path / "t.tex").read_text()
    assert "davies\\_bouldin" in text
    assert "davies_bouldin" not in text.replace("davies\\_bouldin", "")


def test_latex_table_renders_nan_as_a_dash_not_the_word_nan(tmp_path):
    text = write_latex_table(
        [{"algorithm": "ward", "silhouette": float("nan")}], tmp_path / "t.tex"
    ).read_text()
    assert "nan" not in text.lower()
    assert "--" in text


def test_latex_table_renders_missing_keys_as_a_dash(tmp_path):
    rows = [{"a": 1.0, "b": 2.0}, {"a": 3.0}]
    text = write_latex_table(rows, tmp_path / "t.tex", columns=["a", "b"]).read_text()
    assert "--" in text


def test_latex_table_wraps_in_a_float_only_when_captioned(tmp_path):
    plain = write_latex_table([{"a": 1}], tmp_path / "p.tex").read_text()
    floated = write_latex_table([{"a": 1}], tmp_path / "f.tex", caption="c", label="tab:x").read_text()
    assert "\\begin{table}" not in plain
    assert "\\begin{table}" in floated and "\\label{tab:x}" in floated


def test_empty_rows_produce_a_valid_placeholder_file(tmp_path):
    text = write_latex_table([], tmp_path / "t.tex").read_text()
    assert text.startswith("%")


def test_matrix_table_is_square_with_labelled_rows(tmp_path):
    names = ["kmeans", "gmm"]
    M = np.array([[1.0, 0.5], [0.5, 1.0]])
    text = write_matrix_table(names, M, tmp_path / "m.tex").read_text()
    assert "kmeans" in text and "gmm" in text
    assert "0.500" in text


def test_write_json_coerces_numpy_types(tmp_path):
    payload = {
        "int": np.int64(3),
        "float": np.float64(1.5),
        "array": np.arange(3),
        "nested": {"v": np.float32(2.5)},
    }
    data = json.loads(write_json(payload, tmp_path / "s.json").read_text())
    assert data == {"int": 3, "float": 1.5, "array": [0, 1, 2], "nested": {"v": 2.5}}


def test_write_json_maps_nan_to_null_so_the_file_stays_valid_json(tmp_path):
    path = write_json({"metric": np.float64("nan")}, tmp_path / "s.json")
    assert json.loads(path.read_text()) == {"metric": None}


def test_write_json_creates_missing_parent_directories(tmp_path):
    path = write_json({"a": 1}, tmp_path / "deep" / "nested" / "s.json")
    assert path.exists()


def test_latex_table_converts_unicode_that_pdflatex_cannot_typeset(tmp_path):
    """Analysis code emits σ and ×; pdflatex would abort on either."""
    rows = [{"cluster": 0, "note": "V16 (-1.08σ), 3× base rate, ±0.2, ≥5, ≈7"}]
    text = write_latex_table(rows, tmp_path / "t.tex").read_text()
    for char in ("σ", "×", "±", "≥", "≈"):
        assert char not in text
    assert r"$\sigma$" in text and r"$\times$" in text


def test_large_values_use_inline_maths_not_an_undefined_macro(tmp_path):
    """siunitx is absent on BasicTeX, so \\num{} must never reach the .tex file."""
    text = write_latex_table([{"bic": 454800.0, "tiny": 1e-6}], tmp_path / "t.tex").read_text()
    assert "\\num{" not in text
    assert "\\times 10^{" in text


def test_thousands_separator_is_latex_safe():
    from clustering_analysis.reporting import thousands

    assert thousands(283726) == "283{,}726"
    assert thousands(999) == "999"
