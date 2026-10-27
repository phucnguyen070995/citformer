"""Univariate deep baselines (LSTM, N-HiTS) on the target series only.

Same split, scaling, early stopping and test windows as run_multivariate.py.

    python experiments/run_uv_deep.py --datasets Weather --out results/weather_clean/weather_uv_deep.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from citformer.data import N_TEST, metrics, prepare
from citformer.training import predict, train_model


class LSTMForecaster(nn.Module):
    """Two-layer LSTM (hidden 128) followed by an MLP head to H steps."""

    def __init__(self, H=96):
        super().__init__()
        self.lstm = nn.LSTM(1, 128, 2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, H))

    def forward(self, x):  # (B, L) -> (B, H)
        o, _ = self.lstm(x[:, :, None])
        return self.head(o[:, -1, :])


class NHiTSBlock(nn.Module):
    """N-HiTS block: input pooling, MLP, backcast and forecast heads."""

    def __init__(self, L, H, pool=4, hid=128):
        super().__init__()
        pl = max(1, L // pool)
        self.pool_size = pool
        self.fc = nn.Sequential(nn.Linear(pl, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.fh = nn.Linear(hid, H)
        self.bh = nn.Linear(hid, L)

    def forward(self, x):
        p = F.avg_pool1d(x.unsqueeze(1), self.pool_size, stride=self.pool_size).squeeze(1)
        h = self.fc(p)
        return self.bh(h), self.fh(h)


class NHiTS(nn.Module):
    """Simplified N-HiTS (Challu et al., AAAI 2023): three stacked blocks with
    pooling sizes 8, 4, 1; doubly residual backcast/forecast."""

    def __init__(self, L, H):
        super().__init__()
        self.H = H
        self.blocks = nn.ModuleList([NHiTSBlock(L, H, pool=p) for p in [8, 4, 1]])

    def forward(self, x):  # (B, L) -> (B, H)
        r = x
        f = torch.zeros(x.size(0), self.H, device=x.device)
        for b in self.blocks:
            bc, fc = b(r)
            r = r - bc
            f = f + fc
        return f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--datasets', nargs='+', default=['Weather'])
    ap.add_argument('--models', nargs='+', default=['LSTM', 'N-HiTS'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data-dir', default=None, help='directory with the data files')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    rows = []
    for name in a.datasets:
        cfg, parts, _, sc_uv = prepare(name, a.data_dir)
        # The target is the last channel; its column scaling equals the target scaler.
        (Xtr, ytr), (Xva, yva), (Xte, yte) = [(p[0][:, :, -1].copy(), p[1])
                                               for p in (parts['train'], parts['val'], parts['test'])]
        for m in a.models:
            torch.manual_seed(a.seed)
            np.random.seed(a.seed)
            model = LSTMForecaster(cfg['H']) if m == 'LSTM' else NHiTS(cfg['L'], cfg['H'])
            tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                            batch_size=cfg['batch'], shuffle=True)
            va = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva)),
                            batch_size=cfg['batch'])
            losses, best_val, elapsed, _ = train_model(model, tr, va, epochs=a.epochs, patience=20,
                                                       label=f'{name}/{m}', device=a.device)
            pred = predict(model, Xte[:N_TEST], a.device, bs=cfg['batch'])
            r = {'dataset': name, 'model': m, 'seed': a.seed,
                 **metrics(yte[:N_TEST], pred, sc_uv),
                 'epochs_run': len(losses), 'train_sec': elapsed}
            rows.append(r)
            json.dump(rows, open(a.out, 'w'), indent=1)
            print(f'{name}/{m}: MAE={r["MAE"]:.4f} RMSE={r["RMSE"]:.4f} '
                  f'epochs={r["epochs_run"]}', flush=True)


if __name__ == '__main__':
    main()
