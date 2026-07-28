import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from clustering_analysis.distance import AVAILABLE_METRICS, get_metric

VEC = st.lists(
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    min_size=3,
    max_size=8,
)


@settings(deadline=None, max_examples=50)
@given(a=VEC, b=VEC)
def test_euclidean_symmetric(a, b):
    if len(a) != len(b):
        return
    d = get_metric("euclidean")
    assert d(np.array(a), np.array(b)) == d(np.array(b), np.array(a))


@settings(deadline=None, max_examples=50)
@given(a=VEC)
def test_euclidean_identity_of_indiscernibles(a):
    d = get_metric("euclidean")
    assert d(np.array(a), np.array(a)) == 0


@settings(deadline=None, max_examples=30)
@given(a=VEC, b=VEC, c=VEC)
def test_euclidean_triangle_inequality(a, b, c):
    n = min(len(a), len(b), len(c))
    if n < 2:
        return
    a, b, c = np.array(a[:n]), np.array(b[:n]), np.array(c[:n])
    d = get_metric("euclidean")
    assert d(a, c) <= d(a, b) + d(b, c) + 1e-9


def test_available_metrics_contain_required_set():
    required = {"euclidean", "manhattan", "cosine", "mahalanobis"}
    assert required.issubset(set(AVAILABLE_METRICS))
