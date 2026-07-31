import json

import numpy as np
import pytest

from clustering_analysis.registry import (
    data_version_from_manifest,
    list_models,
    load_card,
    register_model,
)


def _register(tmp_path, **overrides):
    kwargs = {
        "name": "test-model",
        "algorithm": "kmeans",
        "k": 4,
        "data_version": "abc123",
        "n_rows": 1000,
        "n_features": 31,
        "feature_space": "X_pca",
        "params": {"n_init": 20},
        "metrics": {"silhouette": 0.42},
        "seeds": {"algorithms": 46},
        "fit_date": "2026-07-31T00:00:00+00:00",
        "root": tmp_path,
    }
    kwargs.update(overrides)
    return register_model(**kwargs)


def test_register_model_writes_card_and_index(tmp_path):
    card = _register(tmp_path)
    assert (tmp_path / f"test-model-{card.entry_id}.json").exists()
    entries = list_models(root=tmp_path)
    assert len(entries) == 1
    assert entries[0]["entry_id"] == card.entry_id


def test_card_records_the_metadata_needed_for_an_audit(tmp_path):
    card = _register(tmp_path, notes="phase 2 winner", tags=["phase2"])
    payload = json.loads((tmp_path / f"test-model-{card.entry_id}.json").read_text())
    for field in ("fit_date", "data_version", "params", "metrics", "seeds", "algorithm", "k"):
        assert field in payload
    assert payload["notes"] == "phase 2 winner"
    assert payload["tags"] == ["phase2"]


def test_same_configuration_overwrites_rather_than_accumulating(tmp_path):
    a = _register(tmp_path)
    b = _register(tmp_path, metrics={"silhouette": 0.99})
    assert a.entry_id == b.entry_id
    assert len(list_models(root=tmp_path)) == 1
    assert load_card(a.entry_id, root=tmp_path).metrics["silhouette"] == 0.99


def test_changed_params_produce_a_new_entry(tmp_path):
    a = _register(tmp_path)
    b = _register(tmp_path, params={"n_init": 50})
    assert a.entry_id != b.entry_id
    assert len(list_models(root=tmp_path)) == 2


def test_changed_data_version_produces_a_new_entry(tmp_path):
    a = _register(tmp_path)
    b = _register(tmp_path, data_version="def456")
    assert a.entry_id != b.entry_id


def test_artifact_is_serialised_and_reloadable(tmp_path):
    from clustering_analysis.registry import load_artifact

    centroids = np.arange(12, dtype=float).reshape(4, 3)
    card = _register(tmp_path, artifact=centroids)
    assert card.artifact_path is not None
    np.testing.assert_array_equal(load_artifact(card.entry_id, root=tmp_path), centroids)


def test_load_artifact_raises_when_none_was_stored(tmp_path):
    from clustering_analysis.registry import load_artifact

    card = _register(tmp_path)
    with pytest.raises(ValueError, match="no serialised artefact"):
        load_artifact(card.entry_id, root=tmp_path)


def test_load_card_raises_for_unknown_entry(tmp_path):
    _register(tmp_path)
    with pytest.raises(KeyError, match="No registry entry"):
        load_card("deadbeef", root=tmp_path)


def test_data_version_falls_back_to_unknown_without_a_manifest(tmp_path):
    assert data_version_from_manifest(tmp_path / "missing.json") == "unknown"


def test_data_version_uses_the_manifest_checksum(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sha256": "0123456789abcdef"}))
    assert data_version_from_manifest(manifest) == "0123456789ab"
