"""Deep clustering (Phase 3, Track 1).

Three models of increasing sophistication:
  - ``Autoencoder``: denoising-free MLP autoencoder for latent embedding.
  - ``ae_kmeans``: pre-train AE, then K-Means in latent space, then jointly
    fine-tune the encoder against the K-Means objective (baseline, Guo et al. 2017).
  - ``DEC``: Deep Embedded Clustering (Xie et al. 2016) — KL divergence between
    a soft assignment (Student's t) and an iteratively refined target distribution.
  - ``DECInfoNCE``: DEC + a contrastive InfoNCE auxiliary loss that pulls
    latent neighbours together and pushes non-neighbours apart (bonus +5).

torch is imported lazily inside each function so the package remains importable
without the ``[deep]`` extra installed; only calling the deep routines requires torch.
"""

from clustering_analysis.deep.ae_kmeans import fit_ae_kmeans
from clustering_analysis.deep.autoencoder import Autoencoder, encode, train_autoencoder
from clustering_analysis.deep.dec import DEC, fit_dec, target_distribution
from clustering_analysis.deep.dec_infonce import fit_dec_infonce, info_nce_loss

__all__ = [
    "Autoencoder",
    "train_autoencoder",
    "encode",
    "fit_ae_kmeans",
    "fit_dec",
    "DEC",
    "target_distribution",
    "fit_dec_infonce",
    "info_nce_loss",
]
