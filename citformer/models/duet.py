import torch
import torch.nn as nn
import torch.nn.functional as F


class DUETModel(nn.Module):
    """Re-implementation of DUET (Qiu et al., "DUET: Dual Clustering Enhanced
    Multivariate Time Series Forecasting", KDD 2025).

    - Temporal Clustering Module (TCM, Eq. 5-14): a noisy top-k router sends
      each channel's window to a weighted mix of M linear pattern extractors
      (trend/seasonal decomposition, one linear layer per component).
    - Channel Clustering Module (CCM, Eq. 15-18): a learnable Mahalanobis
      distance between the rFFT magnitudes of every pair of channels gives an
      N x N connection probability, sampled with Gumbel-Softmax into a soft mask.
    - Fusion Module (FM, Eq. 19-21): N x N masked multi-head attention over the
      TCM features followed by a feed-forward block.

    Hyper-parameters not given in the paper (`heads`, `d0`, `moving_avg_kernel`,
    `gamma`, `gumbel_tau`) use the defaults below. `revin=False` disables the
    per-window normalisation of Eq. 1; it is an ablation, not the published model.
    """

    def __init__(self, n_vars, L, H, d=64, heads=4, M=4, k=2, d0=32,
                 moving_avg_kernel=25, gamma=0.5, gumbel_tau=1.0, revin=True):
        super().__init__()
        if k > M:
            raise ValueError(f"k={k} must be <= M={M}")
        if d % heads != 0:
            raise ValueError(f"d={d} must be divisible by heads={heads}")

        self.n_vars = n_vars
        self.L = L
        self.M = M
        self.k = k
        self.gamma = gamma
        self.gumbel_tau = gumbel_tau
        self.moving_avg_kernel = moving_avg_kernel
        self.heads = heads
        self.head_dim = d // heads
        self.revin = revin

        # TCM distribution router (Eq. 5-8)
        self.encoder_mu_0 = nn.Linear(L, d0)
        self.encoder_mu_1 = nn.Linear(d0, M)
        self.encoder_sigma_0 = nn.Linear(L, d0)
        self.encoder_sigma_1 = nn.Linear(d0, M)
        self.W_H = nn.Linear(M, M, bias=False)

        # TCM linear pattern extractors (Eq. 11-13)
        self.pattern_extractors_trend = nn.ModuleList(
            [nn.Linear(L, d) for _ in range(M)])
        self.pattern_extractors_seasonal = nn.ModuleList(
            [nn.Linear(L, d) for _ in range(M)])

        # CCM: Q = A^T A, a learnable PSD metric on the rFFT magnitudes (Eq. 15)
        freq_dim = L // 2 + 1
        self.A = nn.Parameter(torch.randn(freq_dim, freq_dim) * 0.02)

        # Fusion module
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.attn_out = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.norm2 = nn.LayerNorm(d)

        self.proj = nn.Linear(d, H)

    def _instance_norm(self, x):  # x: (B, N, L)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-5
        return (x - mean) / std, mean, std

    def _moving_avg(self, x):  # (B, N, L) -> (B, N, L), replicate padding
        k = self.moving_avg_kernel
        pad_left = (k - 1) // 2
        pad_right = k - 1 - pad_left
        x_padded = F.pad(x, (pad_left, pad_right), mode='replicate')
        return F.avg_pool1d(x_padded, kernel_size=k, stride=1)

    def _tcm(self, x_norm):  # (B, N, L) -> (B, N, d)
        mu = self.encoder_mu_1(F.relu(self.encoder_mu_0(x_norm)))           # (B, N, M)
        sigma = self.encoder_sigma_1(F.relu(self.encoder_sigma_0(x_norm)))  # (B, N, M)

        # Reparameterisation (Eq. 7) during training; the mean at evaluation,
        # so that inference is deterministic.
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * F.softplus(sigma)
        else:
            z = mu
        h = self.W_H(z)                                                     # Eq. 8

        # KeepTopK then softmax (Eq. 9-10)
        topk_vals, topk_idx = torch.topk(h, self.k, dim=-1)
        masked = torch.full_like(h, float('-inf'))
        masked.scatter_(-1, topk_idx, topk_vals)
        gate = F.softmax(masked, dim=-1)                                    # (B, N, M)

        x_trend = self._moving_avg(x_norm)                                  # Eq. 11
        x_seasonal = x_norm - x_trend                                       # Eq. 12
        temp_outputs = [
            self.pattern_extractors_trend[i](x_trend)
            + self.pattern_extractors_seasonal[i](x_seasonal)
            for i in range(self.M)
        ]                                                                   # Eq. 13
        temp_stack = torch.stack(temp_outputs, dim=-2)                      # (B, N, M, d)
        x_temp = (gate.unsqueeze(-1) * temp_stack).sum(dim=-2)              # Eq. 14
        self._last_gate = gate.detach()
        return x_temp

    def _ccm(self, x_norm):  # (B, N, L) -> mask (B, N, N)
        x_chan = torch.fft.rfft(x_norm, dim=-1).abs()                       # Eq. 16

        # d_ij = (x_i - x_j)^T Q (x_i - x_j), expanded as
        # x_i^T Q x_i + x_j^T Q x_j - 2 x_i^T Q x_j to avoid a (B, N, N, F) tensor.
        Q = self.A.t() @ self.A                                             # Eq. 15
        y = torch.einsum('bnf,fg->bng', x_chan, Q)
        diag = (x_chan * y).sum(-1)
        cross = torch.einsum('bnf,bmf->bnm', x_chan, y)
        d_ij = diag.unsqueeze(-1) + diag.unsqueeze(-2) - 2 * cross
        d_ij = d_ij.clamp(min=1e-8)

        eye = torch.eye(self.n_vars, device=d_ij.device, dtype=torch.bool)
        c_ij = (1.0 / d_ij).masked_fill(eye, 0.0)

        max_c = c_ij.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        p_ij = c_ij * self.gamma / max_c                                    # Eq. 18
        p_ij = p_ij.masked_fill(eye, 1.0)
        p_ij = p_ij.clamp(1e-6, 1 - 1e-6)

        # Bernoulli(p_ij) relaxed with Gumbel-Softmax during training; the
        # probability itself at evaluation.
        if self.training:
            logits = torch.stack([torch.log(p_ij), torch.log(1 - p_ij)], dim=-1)
            gumbel = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False)
            mask = gumbel[..., 0]
        else:
            mask = p_ij
        self._last_mask = mask.detach()
        return mask

    def _fm(self, x_temp, mask):  # (B, N, d), (B, N, N) -> (B, N, d)
        B, N, d = x_temp.shape
        q = self.q_proj(x_temp).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_temp).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_temp).view(B, N, self.heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Masking as an additive log(mask) bias: same limits as Eq. 20 for a
        # 0/1 mask, and keeps gradients w.r.t. a soft mask.
        scores = scores + torch.log(mask.unsqueeze(1).clamp(min=1e-6))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                                                      # Eq. 21
        out = out.transpose(1, 2).reshape(B, N, d)
        out = self.attn_out(out)

        h = self.norm1(x_temp + out)
        h = self.norm2(h + self.ffn(h))
        return h

    def forward(self, x):
        """x: (B, L, C) -> forecast of the last channel, (B, H)."""
        x = x.permute(0, 2, 1)                                              # (B, N, L)
        if self.revin:
            x_norm, mean, std = self._instance_norm(x)                      # Eq. 1
        else:
            x_norm = x

        x_temp = self._tcm(x_norm)
        mask = self._ccm(x_norm)
        x_mix = self._fm(x_temp, mask)

        y = self.proj(x_mix)                                                # Eq. 22, (B, N, H)
        if self.revin:
            y = y * std + mean
        return y[:, -1, :]
