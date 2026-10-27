"""Exploratory analysis of the target series of the eight datasets: STL
decomposition with trend/seasonal strength, and ACF/PACF.

    python analysis/eda_analysis.py
      ->  results/eda/eda_summary.csv
          results/figures/eda_<dataset>_decomp.png, eda_<dataset>_acf_pacf.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from citformer.data import ORDER, PERIOD, load_numeric


def target_series(name, data_dir=None):
    """Target column (OT, else the last column), its name and the number of variables."""
    df = load_numeric(name, data_dir)
    target_col = 'OT' if 'OT' in df.columns else df.columns[-1]
    return df[target_col].reset_index(drop=True), target_col, df.shape[1]


def seasonal_trend_strength(resid, seasonal, trend):
    """Trend and seasonal strength in [0, 1] (Hyndman & Athanasopoulos):
    F = max(0, 1 - Var(resid) / Var(component + resid))."""
    var_resid = np.var(resid)
    f_trend = max(0.0, 1 - var_resid / max(np.var(trend + resid), 1e-12))
    f_seasonal = max(0.0, 1 - var_resid / max(np.var(seasonal + resid), 1e-12))
    return f_trend, f_seasonal


def acf_decay_lag(series, max_lag, threshold=0.2):
    """Smallest lag at which |ACF| falls below `threshold` (None if it never does)."""
    vals = acf(series, nlags=max_lag, fft=True)
    below = np.where(np.abs(vals[1:]) < threshold)[0]
    return int(below[0] + 1) if len(below) else None


def run(name, fig_dir, data_dir=None):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    series, target_col, n_vars = target_series(name, data_dir)
    period = PERIOD[name]
    n = len(series)
    print(f"Target: '{target_col}' (of {n_vars} variables), {n:,} points, period={period}")

    if n < period * 2 + 1:   # STL needs at least two full periods
        print(f'SKIP: series too short ({n} rows) for period={period}')
        return {'dataset': name, 'target_col': target_col, 'n_rows': n,
                'period': period, 'trend_strength': None, 'seasonal_strength': None,
                'acf_at_period': None, 'acf_decay_lag': None}

    stl = STL(series, period=period, robust=True).fit()
    f_trend, f_seasonal = seasonal_trend_strength(
        stl.resid.values, stl.seasonal.values, stl.trend.values)
    print(f'STL strength: trend={f_trend:.3f}  seasonal={f_seasonal:.3f}')

    max_lag = min(4 * period, n - 1, 400)
    acf_vals = acf(series, nlags=max_lag, fft=True)
    acf_at_period = float(acf_vals[period]) if period <= max_lag else None
    decay_lag = acf_decay_lag(series, max_lag)
    print(f'ACF at lag {period}: {acf_at_period}   decay lag (|ACF|<0.2): {decay_lag}')

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(series.values, lw=0.6, color='#2C3E50'); axes[0].set_title(f'{name} – {target_col} (gốc)')
    axes[1].plot(stl.trend.values, lw=0.8, color='#E74C3C'); axes[1].set_title('Xu hướng (Trend)')
    axes[2].plot(stl.seasonal.values, lw=0.6, color='#2ECC71'); axes[2].set_title(f'Mùa vụ (period={period})')
    axes[3].plot(stl.resid.values, lw=0.4, color='#7F8C8D', alpha=0.7); axes[3].set_title('Phần dư (Residual)')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'eda_{name.lower()}_decomp.png'), bbox_inches='tight', dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sample = series.iloc[:min(500, n)]
    plot_acf(sample, lags=min(48, max_lag), ax=axes[0], color='#3498DB', title=f'{name} – ACF')
    plot_pacf(sample, lags=min(48, max_lag), ax=axes[1], color='#E74C3C', title=f'{name} – PACF')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'eda_{name.lower()}_acf_pacf.png'), bbox_inches='tight', dpi=120)
    plt.close(fig)

    return {'dataset': name, 'target_col': target_col, 'n_rows': n,
            'period': period, 'trend_strength': round(f_trend, 4),
            'seasonal_strength': round(f_seasonal, 4),
            'acf_at_period': round(acf_at_period, 4) if acf_at_period is not None else None,
            'acf_decay_lag': decay_lag}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-dir', default=None, help='directory with the data files')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'eda', 'eda_summary.csv'))
    ap.add_argument('--fig-dir', default=os.path.join(ROOT, 'results', 'figures'))
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    summary = pd.DataFrame([run(name, a.fig_dir, a.data_dir) for name in ORDER])
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(summary.to_string(index=False))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    summary.to_csv(a.out, index=False)
    print('Saved', a.out)


if __name__ == '__main__':
    main()
