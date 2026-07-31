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
class ErrorAnalysisParams:
    n_worst: int
    sample_size: int


@dataclass(frozen=True)
class Phase2Params:
    input_matrix: str
    full_algorithms: list[str]
    portfolio_n: int
    k_selection_n: int
    ablation_n: int
    consensus_n: int
    error_analysis: ErrorAnalysisParams
    primary_metric: str


@dataclass(frozen=True)
class DownstreamParams:
    test_fraction: float


@dataclass(frozen=True)
class SensitivityParams:
    alt_scaler: str


@dataclass(frozen=True)
class DriftParams:
    n_bins: int
    psi_threshold: float
    split_fraction: float


@dataclass(frozen=True)
class Phase3Params:
    input_matrix: str
    train_n: int
    eval_n: int
    downstream: DownstreamParams
    sensitivity: SensitivityParams
    drift: DriftParams


@dataclass(frozen=True)
class AutoencoderParams:
    latent_dim: int
    hidden_dims: list[int]
    epochs: int
    batch_size: int
    lr: float


@dataclass(frozen=True)
class AeKmeansParams:
    epochs: int


@dataclass(frozen=True)
class DecParams:
    init_epochs: int
    dec_epochs: int
    update_interval: int
    tol: float
    gamma: float


@dataclass(frozen=True)
class DecInfonceParams:
    contrastive_weight: float
    temperature: float
    n_negatives: int


@dataclass(frozen=True)
class DeepParams:
    autoencoder: AutoencoderParams
    ae_kmeans: AeKmeansParams
    dec: DecParams
    dec_infonce: DecInfonceParams


@dataclass(frozen=True)
class InterpretationParams:
    tree_max_depth: int
    tree_min_samples_leaf: int
    shap_background: int
    dice_features: list[str]


@dataclass(frozen=True)
class SignificanceParams:
    n_bootstrap: int
    ci_level: float
    n_permutations: int
    permutation_seed: int


@dataclass(frozen=True)
class DashboardParams:
    port: int
    sample_size: int


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
    phase2: Phase2Params
    phase3: Phase3Params
    deep: DeepParams
    interpretation: InterpretationParams
    significance: SignificanceParams
    dashboard: DashboardParams


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
        phase2=Phase2Params(
            input_matrix=raw["phase2"]["input_matrix"],
            full_algorithms=raw["phase2"]["full_algorithms"],
            portfolio_n=raw["phase2"]["portfolio_n"],
            k_selection_n=raw["phase2"]["k_selection_n"],
            ablation_n=raw["phase2"]["ablation_n"],
            consensus_n=raw["phase2"]["consensus_n"],
            error_analysis=ErrorAnalysisParams(**raw["phase2"]["error_analysis"]),
            primary_metric=raw["phase2"]["primary_metric"],
        ),
        phase3=Phase3Params(
            input_matrix=raw["phase3"]["input_matrix"],
            train_n=raw["phase3"]["train_n"],
            eval_n=raw["phase3"]["eval_n"],
            downstream=DownstreamParams(**raw["phase3"]["downstream"]),
            sensitivity=SensitivityParams(**raw["phase3"]["sensitivity"]),
            drift=DriftParams(**raw["phase3"]["drift"]),
        ),
        deep=DeepParams(
            autoencoder=AutoencoderParams(**raw["deep"]["autoencoder"]),
            ae_kmeans=AeKmeansParams(**raw["deep"]["ae_kmeans"]),
            dec=DecParams(**raw["deep"]["dec"]),
            dec_infonce=DecInfonceParams(**raw["deep"]["dec_infonce"]),
        ),
        interpretation=InterpretationParams(**raw["interpretation"]),
        significance=SignificanceParams(**raw["significance"]),
        dashboard=DashboardParams(**raw["dashboard"]),
    )
