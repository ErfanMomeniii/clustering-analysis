from __future__ import annotations
from pathlib import Path
import numpy as np
from .config import load_params

_PARAMS = None

def _params():
    global _PARAMS
    if _PARAMS is None:
        _PARAMS = load_params(Path(__file__).resolve().parents[2] / "params.yaml")
    return _PARAMS

def seed_for(purpose: str) -> int:
    s = _params().seeds
    return getattr(s, purpose, s.global_)

def rng_for(purpose: str) -> np.random.Generator:
    return np.random.default_rng(seed_for(purpose))
