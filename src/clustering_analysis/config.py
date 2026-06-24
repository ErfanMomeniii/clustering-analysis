from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass(frozen=True)
class Seeds:
    global_: int
    ingest: int
    scaling: int
    reduce: int
    tendency: int
    algorithms: int

@dataclass(frozen=True)
class IngestParams:
    source: str
    expected_rows: int
    expected_columns: int
    checksum_sha256: str

@dataclass(frozen=True)
class CleanParams:
    drop_duplicates: bool

@dataclass(frozen=True)
class FeatureParams:
    amount_transform: str
    time_period_seconds: int

@dataclass(frozen=True)
class ScalingParams:
    strategy: str
    v_features: list[str]
    robust_features: list[str]

@dataclass(frozen=True)
class UmapParams:
    n_neighbors: int
    min_dist: float
    n_components_viz: int
    n_components_downstream: int
    metric: str

@dataclass(frozen=True)
class ReduceParams:
    pca_variance: float
    umap: UmapParams

@dataclass(frozen=True)
class HopkinsParams:
    sample_fraction: float
    n_repeats: int
    pass_threshold: float

@dataclass(frozen=True)
class VatParams:
    n_legitimate: int
    n_fraud: int

@dataclass(frozen=True)
class TendencyParams:
    hopkins: HopkinsParams
    vat: VatParams

@dataclass(frozen=True)
class Params:
    seeds: Seeds
    ingest: IngestParams
    clean: CleanParams
    features: FeatureParams
    scaling: ScalingParams
    reduce: ReduceParams
    tendency: TendencyParams

def load_params(path: Path | str = "params.yaml") -> Params:
    with open(path) as f:
        raw = yaml.safe_load(f)
    seeds_raw = dict(raw["seeds"])
    seeds_raw["global_"] = seeds_raw.pop("global")
    return Params(
        seeds=Seeds(**seeds_raw),
        ingest=IngestParams(**raw["ingest"]),
        clean=CleanParams(**raw["clean"]),
        features=FeatureParams(**raw["features"]),
        scaling=ScalingParams(**raw["scaling"]),
        reduce=ReduceParams(
            pca_variance=raw["reduce"]["pca_variance"],
            umap=UmapParams(**raw["reduce"]["umap"]),
        ),
        tendency=TendencyParams(
            hopkins=HopkinsParams(**raw["tendency"]["hopkins"]),
            vat=VatParams(**raw["tendency"]["vat"]),
        ),
    )
