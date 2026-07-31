"""Deep Embedded Clustering (Xie, Girshick, Farhadi 2016).

DEC fine-tunes a pre-trained autoencoder by minimising the KL divergence
between a soft cluster assignment (Student's t-distribution on latent
distances to centroids) and an iteratively refined target distribution. The
target sharpens over iterations, yielding a self-supervised clustering that
does not use labels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .autoencoder import encode, train_autoencoder

if TYPE_CHECKING:
    import torch


def soft_assignment(z: torch.Tensor, centroids: torch.Tensor, alpha: float = 1.0):
    """Student's t-distribution soft assignment q_ij.

    q_ij ∝ (1 + ||z_i - mu_j||² / alpha)^{-1}, normalised over clusters.
    Uses squared Euclidean; alpha=1 is the DEC default.
    """
    import torch

    d = torch.cdist(z, centroids) ** 2
    q = (1.0 + d / alpha) ** (-(alpha + 1.0) / 2.0)
    return q / q.sum(dim=1, keepdim=True)


def target_distribution(q: torch.Tensor):
    """Auxiliary target p that sharpens q and normalises per-cluster.

    p_ij ∝ q_ij² / f_j, where f_j = sum_i q_ij. This down-weights uncertain
    assignments and amplifies confident ones, driving cluster purity.
    """

    f = q.sum(dim=0, keepdim=True)
    p = (q**2) / f
    return p / p.sum(dim=1, keepdim=True)


class DEC:
    """Container holding a fitted encoder + cluster centroids."""

    def __init__(self, encoder, decoder, centroids, alpha: float = 1.0):
        self.encoder = encoder
        self.decoder = decoder
        self.centroids = centroids
        self.alpha = alpha

    def predict(self, X: np.ndarray):
        import torch

        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(torch.tensor(np.asarray(X, dtype=np.float32)))
            q = soft_assignment(z, self.centroids, self.alpha)
        return q.argmax(dim=1).cpu().numpy().astype(int)

    def encode(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(torch.tensor(np.asarray(X, dtype=np.float32)))
        return z.cpu().numpy()


def fit_dec(
    X: np.ndarray,
    k: int,
    *,
    latent_dim: int = 10,
    hidden_dims: list[int] | None = None,
    init_epochs: int = 100,
    dec_epochs: int = 150,
    update_interval: int = 5,
    tol: float = 1e-3,
    gamma: float = 1.0,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
):
    """Fit DEC and return (labels, latent, dec_model, loss_history).

    Pipeline: pre-train AE → K-Means init centroids → KL fine-tune with target
    refresh every ``update_interval`` epochs. Convergence is the change in
    label assignment proportion dropping below ``tol``.
    """
    import torch
    from sklearn.cluster import KMeans

    ae = train_autoencoder(
        X,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        epochs=init_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )
    Z = encode(ae, X)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)

    encoder = ae.encoder
    decoder = ae.decoder
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    X_t = torch.tensor(np.asarray(X, dtype=np.float32))

    last_labels = km.labels_.copy()
    losses = []
    for epoch in range(dec_epochs):
        opt.zero_grad()
        z = encoder(X_t)
        q = soft_assignment(z, centroids)
        if epoch % update_interval == 0:
            p = target_distribution(q).detach()
        kl = torch.nn.functional.kl_div(q.log(), p, reduction="batchmean")
        if gamma >= 1.0:
            loss = kl
        else:
            recon = torch.nn.functional.mse_loss(decoder(z), X_t)
            loss = gamma * kl + (1 - gamma) * recon
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

        if epoch % update_interval == 0 and epoch > 0:
            with torch.no_grad():
                cur = q.argmax(dim=1).cpu().numpy()
            delta = float(np.mean(cur != last_labels))
            last_labels = cur
            if delta < tol:
                break

    with torch.no_grad():
        labels = soft_assignment(encoder(X_t), centroids).argmax(dim=1).cpu().numpy().astype(int)
    model = DEC(encoder, decoder, centroids)
    return labels, encode(ae, X), model, losses
