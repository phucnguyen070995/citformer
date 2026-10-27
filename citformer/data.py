"""Dataset settings, loading, windowing and metrics shared by all scripts.

Data files are looked up in the directory given by `data_dir`, else the
CITFORMER_DATA environment variable, else `<repo>/data` (see data/README.md).
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.environ.get('CITFORMER_DATA', os.path.join(ROOT, 'data'))

# Per-dataset settings: input length L, horizon H, number of clusters K
# (K = max(3, K*)), token width d, attention heads, batch size.
DATASETS = {
    'ETTh1':    dict(file='ETTh1.csv',         L=96, H=96, K=4,  d=64,  heads=4, batch=32),
    'ETTm1':    dict(file='ETTm1.csv',         L=96, H=96, K=4,  d=64,  heads=4, batch=32),
    'Exchange': dict(file='exchange_rate.csv', L=96, H=96, K=3,  d=64,  heads=4, batch=32),
    'Weather':  dict(file='weather.csv',       L=96, H=96, K=6,  d=64,  heads=4, batch=32),
    'Covid':    dict(file='covid.csv',         L=14, H=7,  K=10, d=64,  heads=4, batch=16),
    'Solar':    dict(file='solar_AL.txt',      L=96, H=96, K=3,  d=64,  heads=4, batch=32,
                     no_header=True, kstar=1),
    'ECL':      dict(file='electricity.csv',   L=96, H=96, K=32, d=128, heads=8, batch=32),
    'Traffic':  dict(file='traffic.csv',       L=96, H=96, K=85, d=128, heads=8, batch=32),
}
ORDER = list(DATASETS)

# Seasonal period in rows: daily cycle for hourly and 10/15-minute data,
# weekly for the daily Exchange and Covid series.
PERIOD = {'ETTh1': 24, 'ETTm1': 96, 'Exchange': 7, 'Weather': 144,
          'Covid': 7, 'Solar': 144, 'ECL': 24, 'Traffic': 24}

TRAIN_FRAC, VAL_FRAC = 0.7, 0.1   # chronological split; the rest is the test split
N_TEST = 1000                     # every model is scored on the first N_TEST test windows
MISSING_CODE = -9999              # missing-value code in weather.csv


def find_file(name, data_dir=None):
    """Path of `name` (or `name.gz`) anywhere under the data directory."""
    root = data_dir or DEFAULT_DATA_DIR
    for cand in (name, name + '.gz'):
        hits = [h for h in glob.glob(os.path.join(root, '**', cand), recursive=True)
                if os.path.isfile(h)]
        if hits:
            return hits[0]
    raise FileNotFoundError(f'{name} not found under {root} (see data/README.md)')


def fill_missing_codes(df):
    """Replace MISSING_CODE cells by linear interpolation along time."""
    if (df.values == MISSING_CODE).any():
        df = df.replace(MISSING_CODE, np.nan).interpolate(limit_direction='both')
    return df


def load_df(cfg, data_dir=None):
    """All variables of a dataset as a DataFrame, target (OT or last column) last."""
    path = find_file(cfg['file'], data_dir)
    if cfg.get('no_header'):
        df = pd.read_csv(path, header=None)
        df.columns = [f'PV{i}' for i in range(df.shape[1])]
    else:
        df = pd.read_csv(path)
        df = df.drop(columns=[c for c in df.columns if c == 'date'])
    df = fill_missing_codes(df)
    target = df.columns.get_loc('OT') if 'OT' in df.columns else df.shape[1] - 1
    assert target == df.shape[1] - 1, 'target must be the last column'
    return df


def load_numeric(name, data_dir=None):
    """Numeric columns of a dataset for the data analyses (IDA, EDA, K*).

    Non-numeric columns (dates) are dropped; files without a header get the
    column names V0, V1, ...
    """
    cfg = DATASETS[name]
    header = None if cfg.get('no_header') else 0
    df = pd.read_csv(find_file(cfg['file'], data_dir), header=header)
    if header is None:
        df.columns = [f'V{i}' for i in range(df.shape[1])]
    numeric = df.drop(columns=df.select_dtypes(exclude=[np.number]).columns)
    return fill_missing_codes(numeric)


def windows(X_mv, y_uv, L, H):
    """Sliding windows: inputs (n, L, C) from X_mv ((n, L) if X_mv is 1-D) and
    targets (n, H) from y_uv."""
    n = len(X_mv) - L - H + 1
    X = np.stack([X_mv[i:i + L] for i in range(n)]).astype(np.float32)
    y = np.stack([y_uv[i + L:i + L + H] for i in range(n)]).astype(np.float32)
    return X, y


def split_sizes(n):
    """Number of rows in the train and validation splits."""
    return int(n * TRAIN_FRAC), int(n * VAL_FRAC)


def prepare(name, data_dir=None):
    """Split, scale (statistics of the train split) and window a dataset.

    Returns (settings, {'train'|'val'|'test': (X, y)}, scaled train split
    of shape (T, C), target scaler used to report metrics in original units).
    """
    cfg = DATASETS[name]
    vals = load_df(cfg, data_dir).values.astype(np.float32)
    n_train, n_val = split_sizes(len(vals))
    sc_uv = StandardScaler().fit(vals[:n_train, -1:])
    uv = sc_uv.transform(vals[:, -1:]).ravel()
    sc_mv = StandardScaler().fit(vals[:n_train])
    mv = sc_mv.transform(vals).astype(np.float32)
    L, H = cfg['L'], cfg['H']
    parts = {
        'train': windows(mv[:n_train], uv[:n_train], L, H),
        'val':   windows(mv[n_train:n_train + n_val], uv[n_train:n_train + n_val], L, H),
        'test':  windows(mv[n_train + n_val:], uv[n_train + n_val:], L, H),
    }
    return cfg, parts, mv[:n_train], sc_uv


def prepare_target(name, data_dir=None):
    """Univariate version of `prepare` for the baselines that see only the target.

    Same split, target scaler and windows as `prepare`; inputs have shape (n, L).
    Returns (settings, {'train'|'val'|'test': (X, y)}, target scaler).
    """
    cfg = DATASETS[name]
    y = load_df(cfg, data_dir).values[:, -1:].astype(np.float32)
    n_train, n_val = split_sizes(len(y))
    sc_uv = StandardScaler().fit(y[:n_train])
    z = sc_uv.transform(y).ravel()
    L, H = cfg['L'], cfg['H']
    parts = {
        'train': windows(z[:n_train], z[:n_train], L, H),
        'val':   windows(z[n_train:n_train + n_val], z[n_train:n_train + n_val], L, H),
        'test':  windows(z[n_train + n_val:], z[n_train + n_val:], L, H),
    }
    return cfg, parts, sc_uv


def metrics(y_true, y_pred, sc_uv):
    """MAE, MSE, RMSE and sMAPE (%) in the original units of the target."""
    yt = sc_uv.inverse_transform(y_true.reshape(-1, 1)).ravel()
    yp = sc_uv.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    mae = float(np.mean(np.abs(yt - yp)))
    mse = float(np.mean((yt - yp) ** 2))
    smape = float(np.mean(200 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp) + 1e-8)))
    return {'MAE': mae, 'MSE': mse, 'RMSE': mse ** 0.5, 'SMAPE': smape}
