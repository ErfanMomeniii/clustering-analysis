# %% [markdown]
# # Phase 1 — EDA + Clustering Tendency
#
# DS-10 Credit Card Fraud. This notebook produces the figures embedded in `reports/phase1_report.pdf`
# and computes Hopkins H + VAT on the engineered feature matrix.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from clustering_analysis.config import load_params
from clustering_analysis.io_utils import raw_path, interim_path, processed_path
from clustering_analysis.clean import clean_frame, write_decision_log
from clustering_analysis.features import engineer_features
from clustering_analysis.scaling import build_scaler, fit_and_describe
from clustering_analysis.reduce import fit_pca_for_variance, pca_explained_variance_curve, umap_embed
from clustering_analysis.tendency import hopkins_statistic, vat_ordering
from clustering_analysis.schemas import ProcessedSchema

# Resolve params.yaml from repo root (nbconvert runs kernel in notebooks/ dir).
_NB_CWD = Path.cwd()
_PARAMS_PATH = _NB_CWD / "params.yaml" if (_NB_CWD / "params.yaml").exists() else _NB_CWD.parent / "params.yaml"
REPO_ROOT = _PARAMS_PATH.parent
PARAMS = load_params(_PARAMS_PATH)
FIG = REPO_ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.dpi"] = 120

# %% [markdown]
# ## 1. Load raw, clean, feature-engineer

# %%
df_raw = pd.read_parquet(raw_path("creditcard.parquet"))
df_clean, stats = clean_frame(df_raw, drop_duplicates=PARAMS.clean.drop_duplicates)
write_decision_log(stats, REPO_ROOT / "reports" / "decision_log.md")
stats

# %%
df_feat = engineer_features(df_clean.drop(columns=["Class"]), period_seconds=PARAMS.features.time_period_seconds)
ProcessedSchema.validate(df_feat)  # stage-boundary contract: fail fast if shape/types drift
labels = df_clean["Class"].values
df_feat.head()

# %% [markdown]
# ## 2. Scaling comparison (Strategy A / B / C)

# %%
summary = fit_and_describe(df_feat, v_features=PARAMS.scaling.v_features, robust_features=PARAMS.scaling.robust_features)
pd.DataFrame(summary).T

# %% [markdown]
# Scaling-strategy comparison figure: V1 × V14 scatter under each strategy.

# %%
strategies = ["standard", "robust", "hybrid"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, strategy in zip(axes, strategies):
    sc = build_scaler(strategy, v_features=PARAMS.scaling.v_features, robust_features=PARAMS.scaling.robust_features)
    arr = sc.fit_transform(df_feat)
    df_arr = pd.DataFrame(arr, columns=PARAMS.scaling.v_features + PARAMS.scaling.robust_features)
    ax.scatter(df_arr["V1"], df_arr["V14"], s=1, alpha=0.1)
    ax.set_title(f"Strategy: {strategy}")
    ax.set_xlabel("V1 (scaled)")
axes[0].set_ylabel("V14 (scaled)")
fig.suptitle("Scaling strategy comparison — V1 × V14 (Phase 1 §4.5)")
fig.savefig(FIG / "scaling_comparison.pdf")
plt.show()

# %%
scaler = build_scaler(PARAMS.scaling.strategy, v_features=PARAMS.scaling.v_features, robust_features=PARAMS.scaling.robust_features)
X_scaled = scaler.fit_transform(df_feat)
print("Scaled shape:", X_scaled.shape)

# %% [markdown]
# Save the fitted scaler for downstream phases and the dashboard.

# %%
import joblib
processed_path("scaler.joblib").parent.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, processed_path("scaler.joblib").as_posix())

# %% [markdown]
# ## 3. PCA explained-variance curve

# %%
curve = pca_explained_variance_curve(X_scaled, max_components=X_scaled.shape[1])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(curve) + 1), curve, marker="o")
ax.axhline(PARAMS.reduce.pca_variance, color="r", ls="--", label=f"target = {PARAMS.reduce.pca_variance}")
ax.set_xlabel("Number of components"); ax.set_ylabel("Cumulative explained variance")
ax.set_title("PCA explained-variance curve"); ax.legend(); ax.grid(alpha=0.3)
fig.savefig(FIG / "pca_explained_variance.pdf")
plt.show()

# %%
pca, n_pc = fit_pca_for_variance(X_scaled, target_variance=PARAMS.reduce.pca_variance)
X_pca = pca.transform(X_scaled)
print(f"Retained {n_pc} components for {PARAMS.reduce.pca_variance:.0%} variance")

# %% [markdown]
# ## 4. UMAP embedding (2D for visualisation)

# %%
emb2 = umap_embed(X_scaled, n_neighbors=PARAMS.reduce.umap.n_neighbors, min_dist=PARAMS.reduce.umap.min_dist,
                  n_components=PARAMS.reduce.umap.n_components_viz, seed=PARAMS.seeds.reduce,
                  metric=PARAMS.reduce.umap.metric)
fig, ax = plt.subplots(figsize=(8, 7))
hb = ax.hexbin(emb2[:, 0], emb2[:, 1], gridsize=60, mincnt=1, cmap="viridis")
fig.colorbar(hb, label="count")
ax.set_title("UMAP 2D — density view"); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
fig.savefig(FIG / "umap_density.pdf")
plt.show()

# %%
fraud_mask = labels == 1
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(emb2[~fraud_mask, 0], emb2[~fraud_mask, 1], s=2, alpha=0.05, label="legitimate")
ax.scatter(emb2[fraud_mask, 0], emb2[fraud_mask, 1], s=10, color="red", label="fraud")
ax.set_title("UMAP 2D — fraud overlay (labels external, not used in fitting)"); ax.legend()
fig.savefig(FIG / "umap_fraud_overlay.pdf")
plt.show()

# %% [markdown]
# ## 5. Hopkins statistic — 5 repeats

# %%
H_values = []
for i in range(PARAMS.tendency.hopkins.n_repeats):
    H = hopkins_statistic(X_scaled, sample_size=int(PARAMS.tendency.hopkins.sample_fraction * len(X_scaled)), seed=PARAMS.seeds.tendency + i)
    H_values.append(H)
H_mean, H_std = float(np.mean(H_values)), float(np.std(H_values))
print(f"Hopkins H = {H_mean:.3f} ± {H_std:.3f}  (threshold {PARAMS.tendency.hopkins.pass_threshold})")
assert H_mean >= PARAMS.tendency.hopkins.pass_threshold, "Hopkins below threshold — pivot per spec §2.7"

# %% [markdown]
# ## 6. VAT heatmap on stratified subsample

# %%
rng = np.random.default_rng(PARAMS.seeds.tendency)
legit_idx = rng.choice(np.where(~fraud_mask)[0], size=PARAMS.tendency.vat.n_legitimate, replace=False)
fraud_idx = np.where(fraud_mask)[0][:PARAMS.tendency.vat.n_fraud]
sub_idx = np.concatenate([legit_idx, fraud_idx])
X_sub = X_scaled[sub_idx]

order = vat_ordering(X_sub)
from scipy.spatial.distance import squareform, pdist
D = squareform(pdist(X_sub[order]))
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(D, cmap="viridis_r", aspect="auto")
fig.colorbar(im, label="distance")
ax.set_title("VAT heatmap (reordered dissimilarity)")
fig.savefig(FIG / "vat_heatmap.pdf")
plt.show()

# %% [markdown]
# ## 7. Persist processed feature matrix for Phase 2

# %%
import numpy as np
np.save(processed_path("X_scaled.npy").as_posix(), X_scaled)
np.save(processed_path("X_pca.npy").as_posix(), X_pca)
np.save(processed_path("umap_2d.npy").as_posix(), emb2)
np.save(processed_path("labels.npy").as_posix(), labels)
print("Persisted: X_scaled, X_pca, umap_2d, labels")
