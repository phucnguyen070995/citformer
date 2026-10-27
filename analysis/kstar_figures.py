"""Eigenvalue spectrum of the variable correlation matrix and the choice of K*.

Computes K* on the full series and on the training split only, and draws:
  fig_eigen_2d.png           eigenvalues of two ETTh1 variable pairs (strong / weak correlation)
  fig_eigen_scree.png        spectrum and cumulative variance per dataset, K* marked
  fig_eigen_corr_blocks.png  correlation matrices reordered by hierarchical clustering

    python analysis/kstar_figures.py
      ->  results/figures/fig_eigen_*.png, results/ida/kstar_train_vs_full.csv
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from citformer.data import ORDER, TRAIN_FRAC, load_numeric

K_USED = {'Solar': 3}   # K = max(3, K*)
THRESHOLD = 0.90


def spectrum(df):
    """Correlation matrix, eigenvalues (descending), cumulative share, K*."""
    corr = df.corr().fillna(0.0).values
    eig = np.clip(np.linalg.eigvalsh(corr)[::-1], 0, None)
    cum = np.cumsum(eig) / eig.sum()
    k_star = int(np.searchsorted(cum, THRESHOLD) + 1)
    return corr, eig, cum, k_star


def load(name, data_dir=None):
    """Numeric variables of a dataset without constant columns."""
    num = load_numeric(name, data_dir)
    return num.loc[:, num.std() > 0]


def fig_2d_intuition(etth1, fig_dir):
    """Two ETTh1 variable pairs with their eigenvectors scaled by sqrt(eigenvalue)."""
    pairs = [('HUFL', 'MUFL', 'Tương quan mạnh'), ('HULL', 'OT', 'Tương quan yếu')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    rng = np.random.RandomState(0)
    for ax, (a, b, title) in zip(axes, pairs):
        z = etth1[[a, b]].values
        z = (z - z.mean(0)) / z.std(0)
        idx = rng.choice(len(z), 2500, replace=False)
        ax.scatter(z[idx, 0], z[idx, 1], s=4, alpha=0.25, color='#7F8C8D')
        r = np.corrcoef(z.T)[0, 1]
        lam, vec = np.linalg.eigh(np.corrcoef(z.T))
        lam, vec = lam[::-1], vec[:, ::-1]
        for i, (col, lbl) in enumerate([('#C0392B', 'λ₁'), ('#2980B9', 'λ₂')]):
            v = vec[:, i] * 2 * np.sqrt(lam[i])
            ax.annotate('', xy=v, xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', color=col, lw=3))
            # place the label of a near-zero eigenvalue outside the point cloud
            pos = v * 1.12 if lam[i] > 0.3 else vec[:, i] * 1.6
            ax.text(pos[0], pos[1], f'{lbl} = {lam[i]:.2f}', color=col,
                    fontsize=12, fontweight='bold', ha='center', va='center')
        ax.set_title(f'{title}: {a} vs {b} (r = {r:.2f})\n'
                     f'λ₁ giải thích {100*lam[0]/2:.0f}% phương sai', fontsize=11)
        ax.set_xlabel(f'{a} (chuẩn hoá)'); ax.set_ylabel(f'{b} (chuẩn hoá)')
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')
        ax.axhline(0, color='k', lw=0.4); ax.axvline(0, color='k', lw=0.4)
    plt.suptitle('Trị riêng = độ trải của dữ liệu theo từng hướng chính (vector riêng)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_eigen_2d.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_scree(spectra, fig_dir):
    """Per dataset: share of variance per eigenvalue (bars), cumulative share
    (line), 90% threshold (dashed) and K* (star)."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for ax, name in zip(axes.flat, ORDER):
        C, eig, cum, k = spectra[name]
        n_show = min(C, max(12, int(k * 1.6)))
        x = np.arange(1, n_show + 1)
        share = eig[:n_show] / eig.sum()
        ax.bar(x, share, color=['#C0392B' if i < k else '#BDC3C7' for i in range(n_show)])
        ax2 = ax.twinx()
        ax2.plot(x, cum[:n_show], color='#2C3E50', lw=1.8, marker='o', ms=2.5)
        ax2.axhline(THRESHOLD, ls='--', color='#27AE60', lw=1.2)
        ax2.plot([k], [cum[k - 1]], marker='*', ms=16, color='#F39C12', mec='k')
        ax2.set_ylim(0, 1.05)
        ax2.set_yticks([0, 0.5, 0.9, 1.0])
        ax.set_title(f'{name}  (C={C})  →  K*={k}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Thứ tự trị riêng i')
        if name in ('ETTh1', 'Covid'):
            ax.set_ylabel('λᵢ / C (cột)')
        if name in ('Weather', 'Traffic'):
            ax2.set_ylabel('Tích luỹ (đường)')
        ax.text(0.97, 0.55, f'λ₁ = {100*eig[0]/eig.sum():.0f}%', transform=ax.transAxes,
                ha='right', fontsize=10, color='#C0392B')
    plt.suptitle('Phổ trị riêng của ma trận tương quan C×C: cột đỏ = K* trị riêng '
                 'đầu tiên đủ giải thích ≥ 90% phương sai (đường xanh lá)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_eigen_scree.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_corr_blocks(corrs, spectra, fig_dir):
    """Correlation heatmaps reordered by average-linkage clustering on 1 - |r|."""
    names = ['Exchange', 'Weather', 'Solar', 'ECL']
    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    for ax, name in zip(axes, names):
        corr = corrs[name]
        dist = 1 - np.abs(corr)
        iu = np.triu_indices_from(dist, 1)
        order = leaves_list(linkage(dist[iu], method='average'))
        im = ax.imshow(corr[np.ix_(order, order)], cmap='RdBu_r', vmin=-1, vmax=1)
        C, _, _, k = spectra[name]
        ax.set_title(f'{name}: C={C}, K*={k}', fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label='Hệ số tương quan Pearson')
    plt.suptitle('Ma trận tương quan đã sắp xếp lại theo nhóm: mỗi khối màu đậm là một '
                 'nhóm biến cùng nhịp ↔ một trị riêng lớn', fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(fig_dir, 'fig_eigen_corr_blocks.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-dir', default=None, help='directory with the data files')
    ap.add_argument('--fig-dir', default=os.path.join(ROOT, 'results', 'figures'))
    ap.add_argument('--csv', default=os.path.join(ROOT, 'results', 'ida', 'kstar_train_vs_full.csv'))
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    spectra, corrs, rows = {}, {}, []
    for name in ORDER:
        df = load(name, a.data_dir)
        _, eig_tr, _, k_tr = spectrum(df.iloc[:int(len(df) * TRAIN_FRAC)])
        corr, eig, cum, k_full = spectrum(df)
        spectra[name] = (df.shape[1], eig, cum, k_full)   # figures: full-series spectrum
        corrs[name] = corr
        rows.append({'dataset': name, 'C': df.shape[1], 'k_star_full': k_full,
                     'k_star_train': k_tr, 'k_used': K_USED.get(name, k_full),
                     'lambda1_share_train': round(eig_tr[0] / eig_tr.sum(), 4)})
        print(rows[-1])
        if name == 'ETTh1':
            fig_2d_intuition(df, a.fig_dir)
    fig_scree(spectra, a.fig_dir)
    fig_corr_blocks(corrs, spectra, a.fig_dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.csv, index=False)
    print(f'Saved {a.csv} and figures under {a.fig_dir}')


if __name__ == '__main__':
    main()
