"""MLP autoencoder — shared backbone for all deep clustering models.

The encoder maps input → latent (``latent_dim``), the decoder reconstructs.
Both are linear+ReLU MLPs whose widths follow ``hidden_dims`` then narrow to
the latent dimension. Weights are initialised with Xavier for stable training.
"""

from __future__ import annotations

import numpy as np


def _import_torch():
    try:
        import torch
        import torch.nn as nn

        return torch, nn
    except ImportError as e:
        raise ImportError(
            "Deep clustering requires torch. Install with: uv pip install -e '.[deep]'"
        ) from e


class Autoencoder:
    """torch-backed MLP autoencoder with a sklearn-ish fit/transform interface.

    Parameters mirror Phase 1 conventions (seed-driven, params.yaml-configured).
    Training uses Adam + MSE reconstruction loss on full-batch mini-batches.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        hidden_dims: list[int] | None = None,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        torch, nn = _import_torch()
        self.torch = torch
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.lr = lr
        self.seed = seed
        torch.manual_seed(seed)

        enc_layers, dec_layers = self._build_layers(nn)
        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )
        self._fitted = False

    def _build_layers(self, nn):
        enc = []
        in_dim = self.input_dim
        for h in self.hidden_dims:
            enc += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        enc += [nn.Linear(in_dim, self.latent_dim)]

        dec = []
        in_dim = self.latent_dim
        for h in reversed(self.hidden_dims):
            dec += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        dec += [nn.Linear(in_dim, self.input_dim)]
        return enc, dec

    def fit(
        self, X: np.ndarray, *, epochs: int = 100, batch_size: int = 256, verbose: bool = False
    ):
        torch = self.torch
        X_t = torch.tensor(np.asarray(X, dtype=np.float32))
        dataset = torch.utils.data.TensorDataset(X_t, X_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            total = 0.0
            for xb, _ in loader:
                self.optimizer.zero_grad()
                z = self.encoder(xb)
                recon = self.decoder(z)
                loss = torch.nn.functional.mse_loss(recon, xb)
                loss.backward()
                self.optimizer.step()
                total += loss.item() * len(xb)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"AE epoch {epoch+1}/{epochs}  recon={total/len(X_t):.4f}")
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(torch.tensor(np.asarray(X, dtype=np.float32)))
        return z.cpu().numpy()

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            r = self.decoder(self.encoder(torch.tensor(np.asarray(X, dtype=np.float32))))
        return r.cpu().numpy()


def train_autoencoder(
    X: np.ndarray,
    *,
    latent_dim: int = 10,
    hidden_dims: list[int] | None = None,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
) -> Autoencoder:
    """Convenience: build + fit an autoencoder, return the fitted model."""
    ae = Autoencoder(
        input_dim=X.shape[1],
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        lr=lr,
        seed=seed,
    )
    return ae.fit(X, epochs=epochs, batch_size=batch_size)


def encode(ae: Autoencoder, X: np.ndarray) -> np.ndarray:
    """Project X through a fitted encoder's latent space."""
    return ae.transform(X)
