"""CiTFormer: Clustered inverted Transformer for multivariate forecasting."""
import numpy as np
import torch
import torch.nn as nn


def instance_norm(x):
    """Per-window (RevIN-style) normalisation without affine parameters.

    x: (B, L, C). Returns the normalised window and its mean and std, each of
    shape (B, 1, C), so that the forecast can be de-normalised.
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-5
    return (x - mean) / std, mean, std


def spectral_cluster_init(train_data, K, alpha, seed=0):
    """Initial cluster logits from the variable correlation matrix.

    Each variable is represented by its loadings on the top-K eigenvectors of
    the C x C correlation matrix (scaled by sqrt(eigenvalue), then
    row-normalised); k-means on these coordinates assigns it to one of K
    groups. The logit is `alpha` for the variable's own group and 0 elsewhere,
    so that softmax over variables starts each cluster as a weighted average
    biased towards its group.

    train_data: (T, C) array of the standardised training split.
    Returns (logits of shape (C, K), float32; group labels of shape (C,)).
    """
    from sklearn.cluster import KMeans
    corr = np.nan_to_num(np.corrcoef(np.asarray(train_data, dtype=np.float64).T))
    lam, vec = np.linalg.eigh(corr)
    lam, vec = np.clip(lam[::-1][:K], 0, None), vec[:, ::-1][:, :K]
    coords = vec * np.sqrt(lam)
    coords /= np.linalg.norm(coords, axis=1, keepdims=True) + 1e-12
    labels = KMeans(K, n_init=10, random_state=seed).fit_predict(coords)
    logits = np.zeros((corr.shape[0], K), dtype=np.float32)
    logits[np.arange(len(labels)), labels] = alpha
    return logits, labels


class CiTFormer(nn.Module):
    """Clustered inverted Transformer.

    Each variable's L-step window is embedded as one token. The C variable
    tokens are gathered into K cluster tokens with a soft cluster matrix
    W = softmax(cluster_w) (softmax over variables), a Transformer encoder
    attends over the K cluster tokens, and the result is redistributed to the
    variables with the same W. A per-variable FFN and a linear projection give
    the H-step forecast of the target, which is the last channel.

    Args:
        n_vars, L, H: number of variables, input length, forecast horizon.
        K: number of clusters (clipped to n_vars).
        d, heads, n_layers: token width, attention heads, encoder layers.
        cluster_init: (C, K) initial logits, e.g. from spectral_cluster_init;
            None gives a random initialisation.
        cluster_emb: add a learnable embedding to each cluster token.
        prenorm: pre-norm residual blocks (v3, v4); post-norm otherwise (v1, v2).
        revin: per-window normalisation of the input and de-normalisation of
            the forecast (v4).
    """

    def __init__(self, n_vars=7, L=96, H=96, K=3, d=64, heads=4, n_layers=2,
                 cluster_init=None, cluster_emb=False, prenorm=False, revin=False):
        super().__init__()
        K = min(K, n_vars)
        self.K = K
        self.prenorm = prenorm
        self.revin = revin
        self.embed = nn.Linear(L, d)
        self.embed_norm = nn.LayerNorm(d)
        self.embed_drop = nn.Dropout(0.1)
        if cluster_init is None:
            self.cluster_w = nn.Parameter(torch.randn(n_vars, K) * 0.02)
        else:
            init = torch.as_tensor(np.asarray(cluster_init), dtype=torch.float32)
            assert init.shape == (n_vars, K), f'cluster_init must be ({n_vars}, {K})'
            self.cluster_w = nn.Parameter(init.clone())
        self.cluster_emb = (nn.Parameter(torch.randn(K, d) * 0.02)
                            if cluster_emb else None)
        enc = nn.TransformerEncoderLayer(
            d, heads, d * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.cluster_attn = nn.TransformerEncoder(enc, n_layers)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d * 2, d))
        self.norm2 = nn.LayerNorm(d)
        self.proj = nn.Linear(d, H)

    def forward(self, x):
        """x: (B, L, C) -> forecast of the last channel, (B, H)."""
        if self.revin:
            x, mean, std = instance_norm(x)
        y = self._forward_prenorm(x) if self.prenorm else self._forward_postnorm(x)
        if self.revin:
            y = y * std[:, :, -1] + mean[:, :, -1]
        return y

    def _forward_postnorm(self, x):
        h = self.embed_drop(self.embed_norm(
            self.embed(x.permute(0, 2, 1))))                     # (B, C, d)
        w = torch.softmax(self.cluster_w, dim=0)                  # (C, K)
        c_rep = torch.einsum('bcd,ck->bkd', h, w)                 # (B, K, d)
        if self.cluster_emb is not None:
            c_rep = c_rep + self.cluster_emb
        c_rep = self.cluster_attn(c_rep)                          # attention over clusters
        h = self.norm1(h + torch.einsum('bkd,ck->bcd', c_rep, w))
        h = self.norm2(h + self.ffn(h))
        return self.proj(h)[:, -1, :]

    def _forward_prenorm(self, x):
        h = self.embed_drop(self.embed(x.permute(0, 2, 1)))       # (B, C, d)
        w = torch.softmax(self.cluster_w, dim=0)                  # (C, K)
        c_rep = torch.einsum('bcd,ck->bkd', self.norm1(h), w)     # (B, K, d)
        if self.cluster_emb is not None:
            c_rep = c_rep + self.cluster_emb
        c_rep = self.cluster_attn(c_rep)                          # attention over clusters
        h = h + torch.einsum('bkd,ck->bcd', c_rep, w)
        h = h + self.ffn(self.norm2(h))
        return self.proj(h)[:, -1, :]
