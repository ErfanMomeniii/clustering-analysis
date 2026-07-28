import numpy as np

from clustering_analysis.seeds import rng_for


def test_rng_for_returns_deterministic_generator():
    a = rng_for("ingest").integers(0, 1_000_000, size=5)
    b = rng_for("ingest").integers(0, 1_000_000, size=5)
    np.testing.assert_array_equal(a, b)


def test_rng_for_different_purposes_diverge():
    a = rng_for("ingest").integers(0, 1_000_000, size=5)
    b = rng_for("scaling").integers(0, 1_000_000, size=5)
    assert not np.array_equal(a, b)


def test_rng_for_unknown_purpose_falls_back_to_global():
    g = rng_for("not_in_params").integers(0, 1_000_000, size=5)
    g2 = rng_for("not_in_params").integers(0, 1_000_000, size=5)
    np.testing.assert_array_equal(g, g2)
