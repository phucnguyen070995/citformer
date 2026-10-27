import torch.nn as nn

from citformer.models.citformer import instance_norm


class iTransformerModel(nn.Module):
    """iTransformer (Liu et al., ICLR 2024).

    Each variable's L-step window is one token; a Transformer encoder applies
    full self-attention over the C variable tokens and a linear layer projects
    to H steps. Returns the forecast of the last channel, (B, H).
    `revin` enables the same per-window normalisation as CiTFormer and DUET.
    """

    def __init__(self, n_vars=7, L=96, H=96, d=64, heads=4, n_layers=2, revin=False):
        super().__init__()
        self.revin = revin
        self.embed = nn.Linear(L, d)
        enc = nn.TransformerEncoderLayer(
            d, heads, d * 2, 0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, n_layers)
        self.proj = nn.Linear(d, H)

    def forward(self, x):
        """x: (B, L, C) -> (B, H)."""
        if self.revin:
            x, mean, std = instance_norm(x)
        h = self.embed(x.permute(0, 2, 1))   # (B, C, d)
        h = self.transformer(h)              # attention over variables
        y = self.proj(h)[:, -1, :]
        if self.revin:
            y = y * std[:, :, -1] + mean[:, :, -1]
        return y
