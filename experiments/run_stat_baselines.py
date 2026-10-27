"""Univariate statistical and linear baselines (ARIMA, ETS, DLinear) on the
same N_TEST test windows as the deep models.

ARIMA(2,1,2) and additive-trend ETS are fitted per window on the target's
L-step history (fallback: repeat the last value if a fit fails). DLinear maps
a moving-average(25) trend and the remainder to the horizon with one joint
least-squares fit on the training windows.

    python experiments/run_stat_baselines.py --out results/stat_baselines/stat_1000.json
"""
import argparse
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from citformer.data import N_TEST, ORDER, metrics, prepare_target


def fit_one(args):
    """Fit ARIMA or ETS on one window and forecast H steps."""
    kind, x, H = args
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    warnings.filterwarnings('ignore')
    try:
        if kind == 'ARIMA':
            return np.asarray(ARIMA(x, order=(2, 1, 2)).fit().forecast(H), dtype=np.float32)
        return np.asarray(ExponentialSmoothing(x, trend='add').fit(optimized=True).forecast(H),
                          dtype=np.float32)
    except Exception:
        return np.full(H, x[-1], dtype=np.float32)


def moving_avg(x, k=25):
    """Moving average with edge replication, as in the official DLinear code."""
    pad = (k - 1) // 2
    xp = np.concatenate([np.repeat(x[:1], pad), x, np.repeat(x[-1:], pad)])
    return np.convolve(xp, np.ones(k) / k, mode='valid')


def dlinear(Xtr, ytr, Xte):
    """DLinear (Zeng et al., AAAI 2023): y = W_T trend + W_S remainder + b.

    The two linear maps are trained jointly under MSE, which has a closed-form
    solution: one least-squares fit on the concatenated [trend, remainder].
    """
    def feats(X):
        T = np.array([moving_avg(x) for x in X])
        return np.hstack([T, X - T])
    return LinearRegression().fit(feats(Xtr), ytr).predict(feats(Xte))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--datasets', nargs='+', default=ORDER)
    ap.add_argument('--models', nargs='+', default=['DLinear', 'ETS', 'ARIMA'])
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--out', default='stat_1000.json')
    ap.add_argument('--data-dir', default=None, help='directory with the data files')
    a = ap.parse_args()
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r['dataset'], r['model']) for r in rows}
    for name in a.datasets:
        cfg, parts, sc = prepare_target(name, a.data_dir)
        H = cfg['H']
        Xtr, ytr = parts['train']
        Xte, yte = (v[:N_TEST] for v in parts['test'])
        for model in a.models:
            if (name, model) in done:
                continue
            t0 = time.time()
            if model == 'DLinear':
                pred = dlinear(Xtr, ytr, Xte)
            else:
                with Pool(a.procs) as p:
                    pred = np.stack(p.map(fit_one, [(model, x, H) for x in Xte], chunksize=8))
            r = {'dataset': name, 'model': model, 'n_windows': len(Xte),
                 **metrics(yte, pred, sc), 'sec': time.time() - t0}
            rows.append(r)
            json.dump(rows, open(a.out, 'w'), indent=1)
            print(f'{name:9s} {model:8s} n={len(Xte):4d} MAE={r["MAE"]:.4f} '
                  f'RMSE={r["RMSE"]:.4f} [{r["sec"]:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
