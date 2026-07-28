"""Shared interface + dispatch for clustering algorithms."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np


class Clusterer(Protocol):
    def __call__(self, X: np.ndarray, k: int, *, seed: int, **kwargs) -> np.ndarray: ...


_ALGORITHM_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def deco(fn: Callable) -> Callable:
        if name in _ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm {name!r} already registered")
        _ALGORITHM_REGISTRY[name] = fn
        return fn

    return deco


def fit_algorithm(name: str, X: np.ndarray, k: int, *, seed: int, **kwargs) -> np.ndarray:
    if name not in _ALGORITHM_REGISTRY:
        raise KeyError(f"Unknown algorithm {name!r}. Available: {sorted(_ALGORITHM_REGISTRY)}")
    return _ALGORITHM_REGISTRY[name](X, k, seed=seed, **kwargs)


ALGORITHM_REGISTRY = _ALGORITHM_REGISTRY
