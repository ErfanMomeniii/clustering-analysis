import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustering_analysis.deep import (
    Autoencoder,
    encode,
    fit_ae_kmeans,
    fit_dec,
    fit_dec_infonce,
    info_nce_loss,
    target_distribution,
)
from clustering_analysis.deep.dec import soft_assignment


@pytest.fixture
def blobs():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.3, random_state=0, n_features=8)
    return X, y


def test_autoencoder_reconstructs_input(blobs):
    X, _ = blobs
    ae = Autoencoder(input_dim=8, latent_dim=4, hidden_dims=[16], lr=1e-3, seed=0)
    ae.fit(X, epochs=30, batch_size=64)
    recon = ae.reconstruct(X)
    assert recon.shape == X.shape
    # correlation between input and reconstruction should be high (>0.7)
    corr = np.corrcoef(X.flatten(), recon.flatten())[0, 1]
    assert corr > 0.7


def test_autoencoder_transform_returns_latent(blobs):
    X, _ = blobs
    ae = Autoencoder(input_dim=8, latent_dim=4, hidden_dims=[16], seed=0).fit(
        X, epochs=20, batch_size=64
    )
    Z = encode(ae, X)
    assert Z.shape == (300, 4)
    assert np.isfinite(Z).all()


def test_autoencoder_seeded_is_deterministic(blobs):
    X, _ = blobs
    a = Autoencoder(input_dim=8, latent_dim=4, hidden_dims=[16], seed=7).fit(
        X, epochs=10, batch_size=64
    )
    b = Autoencoder(input_dim=8, latent_dim=4, hidden_dims=[16], seed=7).fit(
        X, epochs=10, batch_size=64
    )
    np.testing.assert_allclose(a.transform(X), b.transform(X), atol=1e-5)


def test_fit_ae_kmeans_recovers_blobs(blobs):
    X, y = blobs
    labels, Z, ae, centroids = fit_ae_kmeans(
        X, k=3, latent_dim=4, hidden_dims=[16], ae_epochs=20, ft_epochs=15, seed=0
    )
    assert labels.shape == (300,)
    assert len(np.unique(labels)) == 3
    assert Z.shape == (300, 4)
    assert adjusted_rand_score(y, labels) > 0.9


def test_fit_dec_recovers_blobs(blobs):
    X, y = blobs
    labels, Z, model, losses = fit_dec(
        X, k=3, latent_dim=4, hidden_dims=[16], init_epochs=20, dec_epochs=40, seed=0
    )
    assert labels.shape == (300,)
    assert len(np.unique(labels)) >= 2
    assert len(losses) > 0
    assert adjusted_rand_score(y, labels) > 0.85


def test_fit_dec_infonce_recovers_blobs(blobs):
    X, y = blobs
    labels, Z, model, losses = fit_dec_infonce(
        X,
        k=3,
        latent_dim=4,
        hidden_dims=[16],
        init_epochs=20,
        dec_epochs=40,
        contrastive_weight=0.1,
        seed=0,
    )
    assert labels.shape == (300,)
    assert len(np.unique(labels)) >= 2
    assert adjusted_rand_score(y, labels) > 0.85


def test_fit_dec_deterministic_for_fixed_seed(blobs):
    X, _ = blobs
    a, _, _, _ = fit_dec(
        X, k=3, latent_dim=4, hidden_dims=[16], init_epochs=15, dec_epochs=20, seed=11
    )
    b, _, _, _ = fit_dec(
        X, k=3, latent_dim=4, hidden_dims=[16], init_epochs=15, dec_epochs=20, seed=11
    )
    np.testing.assert_array_equal(a, b)


def test_soft_assignment_rows_sum_to_one():
    import torch

    z = torch.randn(20, 3)
    centroids = torch.randn(3, 3)
    q = soft_assignment(z, centroids)
    np.testing.assert_allclose(q.sum(dim=1).detach().numpy(), 1.0, atol=1e-5)


def test_target_distribution_rows_sum_to_one():
    import torch

    q = torch.softmax(torch.randn(20, 4), dim=1)
    p = target_distribution(q)
    np.testing.assert_allclose(p.sum(dim=1).detach().numpy(), 1.0, atol=1e-5)


def test_info_nce_loss_is_finite_and_positive():
    import torch

    z = torch.randn(32, 4)
    neighbours = torch.randint(0, 32, (32,))
    while (neighbours == torch.arange(32)).any():
        neighbours = torch.randint(0, 32, (32,))
    loss = info_nce_loss(z, neighbours, temperature=0.5, n_negatives=16)
    assert np.isfinite(loss.item())
    assert loss.item() > 0


def test_info_nce_loss_excludes_anchor_self_similarity():
    """sim(i,i) = 1/tau is the largest term; including it would swamp the denominator."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    z = torch.randn(8, 4)
    neighbours = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6])
    tau = 0.5
    got = float(info_nce_loss(z, neighbours, temperature=tau, n_negatives=1024))

    B = z.shape[0]
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / tau
    self_excluded = sim.masked_fill(torch.eye(B, dtype=torch.bool), float("-inf"))
    expected = float(
        (torch.logsumexp(self_excluded, dim=1) - sim[torch.arange(B), neighbours]).mean()
    )
    assert got == pytest.approx(expected, rel=1e-5)


def test_info_nce_loss_honours_n_negatives_cap():
    import torch

    torch.manual_seed(0)
    z = torch.randn(16, 4)
    neighbours = torch.roll(torch.arange(16), 1)
    few = float(info_nce_loss(z, neighbours, temperature=0.5, n_negatives=2))
    many = float(info_nce_loss(z, neighbours, temperature=0.5, n_negatives=14))
    # a smaller negative pool means a smaller denominator, hence a smaller loss
    assert few < many


def test_info_nce_loss_deterministic_for_same_input():
    import torch

    torch.manual_seed(0)
    z = torch.randn(16, 4)
    neighbours = torch.roll(torch.arange(16), 1)
    a = float(info_nce_loss(z, neighbours, n_negatives=8))
    b = float(info_nce_loss(z, neighbours, n_negatives=8))
    assert a == b


def test_dec_model_predict_returns_int_labels(blobs):
    X, y = blobs
    labels, _, model, _ = fit_dec(
        X, k=3, latent_dim=4, hidden_dims=[16], init_epochs=15, dec_epochs=20, seed=0
    )
    pred = model.predict(X[:50])
    assert pred.shape == (50,)
    assert pred.dtype == int
    assert len(np.unique(pred)) >= 2
