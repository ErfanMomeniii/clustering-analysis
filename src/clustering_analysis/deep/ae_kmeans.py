"""AE+K-Means baseline (Guo et al. 2017).

Pre-train an autoencoder, run K-Means in the latent space, then jointly
fine-tune the encoder + cluster centroids against the K-Means reconstruction-
regularised objective. This is the standard deep-clustering baseline that DEC
is benchmarked against.
"""

from __future__ import annotations

import numpy as np

from .autoencoder import encode, train_autoencoder


def fit_ae_kmeans(
    X: np.ndarray,
    k: int,
    *,
    latent_dim: int = 10,
    hidden_dims: list[int] | None = None,
    ae_epochs: int = 100,
    ft_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
):
    """Fit AE+K-Means and return (labels, latent, ae, centroids).

    Fine-tuning alternates: (1) K-Means assignment on current latent codes,
    (2) gradient step on reconstruction + cluster-distance loss. The cluster
    term pulls each point's latent code toward its assigned centroid, the
    reconstruction term prevents collapse.
    """
    import torch
    from sklearn.cluster import KMeans

    ae = train_autoencoder(
        X,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        epochs=ae_epochs,
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
    recon_weight = 1.0
    cluster_weight = 0.5

    for _ in range(ft_epochs):
        opt.zero_grad()
        z = encoder(X_t)
        recon = decoder(z)
        recon_loss = torch.nn.functional.mse_loss(recon, X_t)
        d = torch.cdist(z, centroids)
        assign = d.argmin(dim=1)
        cluster_loss = d.gather(1, assign.unsqueeze(1)).mean()
        loss = recon_weight * recon_loss + cluster_weight * cluster_loss
        loss.backward()
        opt.step()
        # update centroids as EMA of assigned points
        with torch.no_grad():
            new_c = torch.stack(
                [
                    z[assign == c].mean(dim=0) if (assign == c).any() else centroids[c]
                    for c in range(k)
                ]
            )
            centroids = 0.9 * centroids + 0.1 * new_c

    Z_final = encode(ae, X)
    km_final = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z_final)
    return km_final.labels_.astype(int), Z_final, ae, km_final.cluster_centers_
