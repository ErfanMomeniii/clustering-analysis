from __future__ import annotations

from dataclasses import dataclass
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
class KmeansParams:
    n_init: int


@dataclass(frozen=True)
class WardParams:
    linkage: str


@dataclass(frozen=True)
class GmmParams:
    covariance_type: str


@dataclass(frozen=True)
class HdbscanParams:
    min_cluster_size: int
    min_samples: int
    cluster_selection_method: str


@dataclass(frozen=True)
class AlgorithmFamilies:
    kmeans: KmeansParams
    ward: WardParams
    gmm: GmmParams
    hdbscan: HdbscanParams


@dataclass(frozen=True)
class AlgorithmParams:
    k_range: list[int]
    families: AlgorithmFamilies


@dataclass(frozen=True)
class SilhouetteParams:
    sample_size: int


@dataclass(frozen=True)
class GapParams:
    n_refs: int
    seed_offset: int


@dataclass(frozen=True)
class BootstrapParams:
    n_resamples: int
    sample_fraction: float


@dataclass(frozen=True)
class KSelectionParams:
    silhouette: SilhouetteParams
    gap: GapParams
    bootstrap: BootstrapParams


@dataclass(frozen=True)
class MetricsParams:
    internal: list[str]
    external: list[str]


@dataclass(frozen=True)
class ConsensusParams:
    co_occurrence_threshold: float


@dataclass(frozen=True)
class StabilityParams:
    n_seeds: int
    seed_offset: int
    consensus: ConsensusParams


@dataclass(frozen=True)
class Params:
    seeds: Seeds
    ingest: IngestParams
    clean: CleanParams
    features: FeatureParams
    scaling: ScalingParams
    reduce: ReduceParams
    tendency: TendencyParams
    algorithms: AlgorithmParams
    k_selection: KSelectionParams
    metrics: MetricsParams
    stability: StabilityParams


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
        algorithms=AlgorithmParams(
            k_range=raw["algorithms"]["k_range"],
            families=AlgorithmFamilies(
                kmeans=KmeansParams(**raw["algorithms"]["families"]["kmeans"]),
                ward=WardParams(**raw["algorithms"]["families"]["ward"]),
                gmm=GmmParams(**raw["algorithms"]["families"]["gmm"]),
                hdbscan=HdbscanParams(**raw["algorithms"]["families"]["hdbscan"]),
            ),
        ),
        k_selection=KSelectionParams(
            silhouette=SilhouetteParams(**raw["k_selection"]["silhouette"]),
            gap=GapParams(**raw["k_selection"]["gap"]),
            bootstrap=BootstrapParams(**raw["k_selection"]["bootstrap"]),
        ),
        metrics=MetricsParams(**raw["metrics"]),
        stability=StabilityParams(
            n_seeds=raw["stability"]["n_seeds"],
            seed_offset=raw["stability"]["seed_offset"],
            consensus=ConsensusParams(**raw["stability"]["consensus"]),
        ),
    )
