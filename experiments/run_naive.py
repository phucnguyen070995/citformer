"""Naive and Seasonal-Naive baselines on the same N_TEST test windows, target
and metrics as the other univariate baselines.

Naive repeats the last observed value of the target for all H steps.
Seasonal-Naive repeats the last season of length m = PERIOD[dataset]; it is
skipped when m > L (Weather and Solar, m = 144 > L = 96), since the input
window does not contain a full season.

    python experiments/run_naive.py --out results/naive/naive.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from citformer.data import N_TEST, ORDER, PERIOD, metrics, prepare_target


def naive(X, H):
    """(n, L) -> (n, H): last value repeated."""
    return np.repeat(X[:, -1:], H, axis=1)


def seasonal_naive(X, H, m):
    """(n, L) -> (n, H): y[t + h] = y[t + h - m * ceil(h / m)], h = 1..H."""
    if m > X.shape[1]:
        raise ValueError(f'period {m} exceeds the input length {X.shape[1]}')
    return np.tile(X[:, -m:], (1, -(-H // m)))[:, :H]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--datasets', nargs='+', default=ORDER)
    ap.add_argument('--out', default='naive.json')
    ap.add_argument('--data-dir', default=None, help='directory with the data files')
    a = ap.parse_args()
    rows = []
    for name in a.datasets:
        cfg, parts, sc = prepare_target(name, a.data_dir)
        L, H, m = cfg['L'], cfg['H'], PERIOD[name]
        Xte, yte = (v[:N_TEST] for v in parts['test'])
        preds = {'Naive': naive(Xte, H)}
        if m <= L:
            preds['SeasonalNaive'] = seasonal_naive(Xte, H, m)
        for model, pred in preds.items():
            r = {'dataset': name, 'model': model, 'period': m if model == 'SeasonalNaive' else None,
                 'n_windows': len(Xte), **metrics(yte, pred, sc)}
            rows.append(r)
            print(f'{name:9s} {model:13s} n={len(Xte):4d} MAE={r["MAE"]:.4f} '
                  f'RMSE={r["RMSE"]:.4f}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
