"""Spectral initialisation of the cluster matrix on ETTh1 (K = 4): variables in
the space of their eigenvector loadings, the correlation matrix ordered by the
resulting groups, and the initial W = softmax(logits) with alpha = 2.

    python analysis/eigvec_init_figures.py   ->  results/figures/fig_eigvec_etth1.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from citformer.data import DATASETS, TRAIN_FRAC, find_file
from citformer.models import spectral_cluster_init

GROUP_COLORS = ['#C0392B', '#2980B9', '#27AE60', '#8E44AD']


def loadings(z, K):
    """Top-K eigenvalues, eigenvectors (largest entry positive) and loadings."""
    lam, vec = np.linalg.eigh(np.corrcoef(z.T))
    lam, vec = lam[::-1][:K], vec[:, ::-1][:, :K]
    vec = vec * np.sign(vec[np.abs(vec).argmax(0), range(K)])
    return lam, vec, vec * np.sqrt(lam)


def softmax_cols(logits):
    e = np.exp(logits)
    return e / e.sum(0)


def etth1(fig_dir, data_dir=None):
    df = pd.read_csv(find_file(DATASETS['ETTh1']['file'], data_dir)).drop(columns=['date'])
    n = int(len(df) * TRAIN_FRAC)
    z = ((df - df[:n].mean()) / df[:n].std()).values[:n]
    names = df.columns.tolist(); K = 4
    lam, vec, load = loadings(z, K)
    logits, lab = spectral_cluster_init(z, K, 2.0)
    # number the clusters in order of first appearance
    order = {}
    for g in lab:
        order.setdefault(g, len(order))
    lab = np.array([order[g] for g in lab]); logits = logits[:, list(order)]
    W = softmax_cols(logits)
    corr = np.corrcoef(z.T)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={'width_ratios': [1, 1, 0.9]})
    # (a) loadings on directions 2 and 3
    ax = axes[0]
    for c in range(len(names)):
        ax.scatter(load[c, 1], load[c, 2], s=280, color=GROUP_COLORS[lab[c]], edgecolor='k', zorder=3)
        ax.annotate(names[c], (load[c, 1], load[c, 2]), xytext=(9, 5), textcoords='offset points', fontsize=11)
    ax.axhline(0, color='gray', lw=0.6); ax.axvline(0, color='gray', lw=0.6)
    ax.set_xlabel(r'hệ số tải hướng 2 ($\sqrt{\lambda_2}v_2$)'); ax.set_ylabel(r'hệ số tải hướng 3 ($\sqrt{\lambda_3}v_3$)')
    ax.set_title(f'(a) 7 biến trong không gian hệ số tải\n(k-means dùng cả {K} hướng; hình chiếu lên hướng 2--3)', fontsize=11)
    # (b) correlation matrix ordered by group, one box per group
    ax = axes[1]
    idx = np.argsort(lab, kind='stable')
    ax.imshow(corr[np.ix_(idx, idx)], cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([names[i] for i in idx], rotation=45)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([names[i] for i in idx])
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f'{corr[idx[i], idx[j]]:.2f}', ha='center', va='center', fontsize=8.5,
                    color='white' if abs(corr[idx[i], idx[j]]) > 0.6 else 'black')
    start = 0
    for g in range(K):
        size = int((lab == g).sum())
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), size, size, fill=False,
                               edgecolor=GROUP_COLORS[g], lw=3))
        start += size
    ax.set_title('(b) Ma trận tương quan sắp theo nhóm tìm được\n(khung màu = 1 nhóm)', fontsize=11)
    # (c) initial W
    ax = axes[2]
    ax.imshow(W, cmap='Reds', vmin=0, vmax=0.6)
    for c in range(len(names)):
        for k in range(K):
            ax.text(k, c, f'{W[c, k]:.2f}', ha='center', va='center', fontsize=10,
                    color='white' if W[c, k] > 0.35 else 'black')
    ax.set_xticks(range(K)); ax.set_xticklabels([f'Cụm {k+1}' for k in range(K)])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for t, c in zip(ax.get_yticklabels(), lab):
        t.set_color(GROUP_COLORS[c]); t.set_fontweight('bold')
    ax.set_title(r'(c) $\mathbf{W}$ khởi tạo cho CiTFormer ($\alpha=2$)', fontsize=11)
    plt.suptitle(f'ETTh1 (C=7, K*={K}): vector riêng → nhóm biến → khởi tạo ma trận cụm',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(fig_dir, 'fig_eigvec_etth1.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('ETTh1 groups:', {g + 1: [names[c] for c in range(len(names)) if lab[c] == g] for g in range(K)})
    print('saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-dir', default=None, help='directory with the data files')
    ap.add_argument('--fig-dir', default=os.path.join(ROOT, 'results', 'figures'))
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    etth1(a.fig_dir, a.data_dir)
