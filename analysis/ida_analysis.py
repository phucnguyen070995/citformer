"""Initial data analysis of the eight datasets: missing values, outliers (IQR
rule), stationarity (ADF test) and the number of clusters K*.

K* is the smallest k such that the k largest eigenvalues of the C x C
correlation matrix explain at least 90% of the total variance.

    python analysis/ida_analysis.py     ->  results/ida/ida_summary.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from citformer.data import ORDER, load_numeric

THRESHOLD = 0.90


def outlier_fraction_iqr(series):
    """Fraction of values outside [Q1 - 1.5 IQR, Q3 + 1.5 IQR]."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(((series < lo) | (series > hi)).mean())


def adf_stationary_fraction(numeric_df, max_cols=None, sample_rows=5000, seed=42):
    """ADF test (alpha = 0.05) per column on the last `sample_rows` rows, on at
    most `max_cols` randomly chosen columns.

    Returns (n stationary, n tested, n failed, n columns considered).
    """
    cols = list(numeric_df.columns)
    if max_cols is not None and len(cols) > max_cols:
        rng = np.random.RandomState(seed)
        cols = list(rng.choice(cols, size=max_cols, replace=False))

    series_df = numeric_df[cols]
    if len(series_df) > sample_rows:
        series_df = series_df.iloc[-sample_rows:]   # contiguous window

    n_stationary, n_tested, n_failed = 0, 0, 0
    for col in cols:
        s = series_df[col].dropna()
        if len(s) < 30 or s.nunique() < 2:
            continue
        try:
            _, pvalue, *_ = adfuller(s.values, autolag='AIC')
            n_tested += 1
            if pvalue < 0.05:
                n_stationary += 1
        except Exception:
            n_failed += 1
    return n_stationary, n_tested, n_failed, len(cols)


def effective_rank(numeric_df, threshold=THRESHOLD):
    """K* and the eigenvalues (descending) of the correlation matrix."""
    corr = numeric_df.corr().fillna(0.0).values
    eigvals = np.clip(np.linalg.eigvalsh(corr)[::-1], 0, None)
    total = eigvals.sum()
    if total <= 0:
        return None, eigvals
    cumvar = np.cumsum(eigvals) / total
    return int(np.searchsorted(cumvar, threshold) + 1), eigvals


def run(name, data_dir=None):
    t0 = time.time()
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    numeric_df = load_numeric(name, data_dir)
    n_rows, c = numeric_df.shape
    print(f'Shape: {n_rows:,} rows x {c} numeric columns')

    missing_frac = numeric_df.isna().mean()
    print(f'Missing values: {missing_frac.sum() * 100 / max(c, 1):.4f}% of cells on average; '
          f'max column missing {missing_frac.max() * 100:.4f}%')

    cols_for_outlier = numeric_df.columns
    if c > 50:
        cols_for_outlier = np.random.RandomState(42).choice(
            numeric_df.columns, size=50, replace=False)
    outlier_fracs = [outlier_fraction_iqr(numeric_df[col].dropna()) for col in cols_for_outlier]
    print(f'Outlier fraction (IQR rule, mean over {len(cols_for_outlier)} columns): '
          f'{np.mean(outlier_fracs) * 100:.2f}% '
          f'(min {np.min(outlier_fracs) * 100:.2f}%, max {np.max(outlier_fracs) * 100:.2f}%)')

    n_stat, n_tested, n_failed, n_cols = adf_stationary_fraction(
        numeric_df, max_cols=50 if c > 50 else None)
    print(f'Stationarity (ADF, alpha=0.05), tested {n_tested}/{n_cols} columns '
          f'({n_failed} failed to fit): {n_stat}/{n_tested} = '
          f'{100 * n_stat / max(n_tested, 1):.1f}% stationary')

    k_star, _ = effective_rank(numeric_df)
    k_sqrt = round(c ** 0.5)
    print(f'K* (90% variance): {k_star}   round(sqrt(C)) = {k_sqrt}   (C={c})')
    print(f'[{time.time() - t0:.1f}s]')
    return {
        'dataset': name, 'n_rows': n_rows, 'C': c,
        'missing_pct': float(missing_frac.sum() * 100 / max(c, 1)),
        'outlier_pct_mean': float(np.mean(outlier_fracs) * 100),
        'stationary_pct': 100 * n_stat / max(n_tested, 1),
        'k_star': k_star, 'k_sqrt_heuristic': k_sqrt,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-dir', default=None, help='directory with the data files')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'ida', 'ida_summary.csv'))
    a = ap.parse_args()
    summary = pd.DataFrame([run(name, a.data_dir) for name in ORDER])
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(summary.to_string(index=False))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    summary.to_csv(a.out, index=False)
    print('Saved', a.out)


if __name__ == '__main__':
    main()
