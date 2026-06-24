"""Distance metric registry (brief §2.6)."""
from __future__ import annotations
from typing import Callable
import numpy as np

def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))

def _mahalanobis_factory(VI: np.ndarray) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return a Mahalanobis metric bound to inverse covariance VI."""
    def metric(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return float(np.sqrt(diff @ VI @ diff))
    return metric

_REGISTRY = {
    "euclidean": _euclidean,
    "manhattan": _manhattan,
    "cosine": _cosine,
    "mahalanobis": _mahalanobis_factory,
}

AVAILABLE_METRICS = tuple(_REGISTRY.keys())

def get_metric(name: str, *, VI: np.ndarray | None = None):
    if name == "mahalanobis":
        if VI is None:
            raise ValueError("mahalanobis requires VI (inverse covariance matrix)")
        return _mahalanobis_factory(VI)
    return _REGISTRY[name]
