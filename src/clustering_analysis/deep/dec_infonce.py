"""DEC + InfoNCE contrastive auxiliary loss (bonus +5).

Adds a contrastive term to DEC that pulls latent neighbours together and
pushes non-neighbours apart. The neighbour relation is recomputed periodically
in latent space (k-NN on the current codes), so the contrastive signal
co-evolves with the cluster structure — unlike a fixed-similarity pre-text,
this rewards the geometry DEC is actually learning.

Total loss = KL(q || p) + lambda * InfoNCE, where InfoNCE for anchor i with
positive p and negatives N is:
    -log( exp(sim(i,p)/tau) / sum_{n in N∪{p}} exp(sim(i,n)/tau) )

Negatives are sampled from the batch to keep the cost bounded (SimCLR style).
"""

from __future__ import annotations

import numpy as np

from .autoencoder import encode, train_autoencoder
from .dec import DEC, soft_assignment, target_distribution


def info_nce_loss(z, neighbours, *, temperature: float = 0.5, n_negatives: int = 1024):
    """InfoNCE loss over a batch of latent codes.

    ``z``: (B, D) latent codes. ``neighbours``: (B,) int index of each anchor's
    positive (must be within the batch). The candidate pool per anchor is the
    positive plus the in-batch negatives; the anchor itself is excluded, since
    sim(i, i) = 1/tau is the largest term and would otherwise dominate the
    denominator and flatten the gradient.

    When the batch offers more than ``n_negatives`` negatives, the pool is
    capped at the ``n_negatives`` *hardest* (highest-similarity) ones. Hard-
    negative mining keeps the cost bounded without introducing sampling noise,
    so the loss stays deterministic for a fixed seed.

    Returns a scalar loss.
    """
    import torch
    import torch.nn.functional as F

    B = z.shape[0]
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature  # (B, B)
    pos = sim[torch.arange(B, device=z.device), neighbours]  # (B,)

    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    pos_mask = F.one_hot(neighbours, num_classes=B).bool()
    neg_mask = ~eye & ~pos_mask  # excludes self and the positive

    neg_sim = sim.masked_fill(~neg_mask, float("-inf"))
    n_available = int(neg_mask.sum(dim=1).max().item())
    keep = min(n_negatives, n_available)
    if keep > 0:
        # hardest negatives per anchor; -inf rows/entries drop out of logsumexp
        neg_sim = neg_sim.topk(keep, dim=1).values

    # denominator = positive + retained negatives
    candidates = torch.cat([pos.unsqueeze(1), neg_sim], dim=1)
    denom = torch.logsumexp(candidates, dim=1)
    return (denom - pos).mean()


def _batch_neighbours(z):
    """Return per-row index of the nearest non-self neighbour in the batch."""
    import torch

    d = torch.cdist(z, z)
    d.fill_diagonal_(float("inf"))
    return d.argmin(dim=1)


def fit_dec_infonce(
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
    contrastive_weight: float = 0.1,
    temperature: float = 0.5,
    n_negatives: int = 1024,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
):
    """Fit DEC + InfoNCE and return (labels, latent, dec_model, loss_history).

    Mirrors ``fit_dec`` but adds the contrastive term at each step. The
    neighbour index is recomputed per batch from the current latent codes so
    the positive pair adapts as the representation evolves.
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
    n = len(X_t)
    for epoch in range(dec_epochs):
        opt.zero_grad()
        z = encoder(X_t)
        q = soft_assignment(z, centroids)
        if epoch % update_interval == 0:
            p = target_distribution(q).detach()
        kl = torch.nn.functional.kl_div(q.log(), p, reduction="batchmean")
        # InfoNCE is O(B²) in the batch, so it is evaluated on a minibatch even
        # though the KL term is full-batch: on DS-10 (283k rows) a full-batch
        # similarity matrix would need ~80 GB. Indices are drawn from a
        # per-epoch seeded generator so the run stays reproducible.
        if n > batch_size:
            bidx = torch.from_numpy(
                np.random.default_rng(seed + epoch).choice(n, size=batch_size, replace=False)
            )
            z_c = z[bidx]
        else:
            z_c = z
        neighbours = _batch_neighbours(z_c)
        ctn = info_nce_loss(z_c, neighbours, temperature=temperature, n_negatives=n_negatives)
        loss = gamma * kl + contrastive_weight * ctn
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
