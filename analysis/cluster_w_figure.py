"""Cluster matrix W = softmax(cluster_w) of CiTFormer-v1 (random init, post-norm)
on ETTh1 (K = 4) before and after training: an illustration of cluster
collapse. Single run, seed 42, at most 60 epochs.

    python analysis/cluster_w_figure.py   ->  results/figures/fig_cluster_w_etth1.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from citformer.data import DATASETS, TRAIN_FRAC, VAL_FRAC, find_file
from citformer.models import CiTFormer
from citformer.training import train_model

L, H, K = 96, 96, 4


def main(fig_dir, data_dir=None):
    torch.manual_seed(42); np.random.seed(42)
    df = pd.read_csv(find_file(DATASETS['ETTh1']['file'], data_dir), index_col='date')
    n = len(df); n_train = int(n * TRAIN_FRAC); n_val = int(n * VAL_FRAC)
    ot = df.columns.get_loc('OT')
    vals = df.values.astype(np.float32)
    z = StandardScaler().fit(vals[:n_train]).transform(vals).astype(np.float32)

    def windows(a):
        idx = range(len(a) - L - H + 1)
        X = np.stack([a[i:i + L] for i in idx]); y = np.stack([a[i + L:i + L + H, ot] for i in idx])
        return torch.tensor(X), torch.tensor(y)

    Xtr, ytr = windows(z[:n_train]); Xva, yva = windows(z[n_train:n_train + n_val])
    model = CiTFormer(df.shape[1], L, H, K=K, d=64, heads=4, n_layers=2)
    init_w = torch.softmax(model.cluster_w.detach().clone(), dim=0).numpy()
    train_model(model, DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True),
                DataLoader(TensorDataset(Xva, yva), batch_size=256), epochs=60, patience=20,
                label='CiTFormer-v1', device='cpu', log_every=10)
    w = torch.softmax(model.cluster_w.detach(), dim=0).numpy()   # (C, K), columns sum to 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, m, title in [(axes[0], init_w, 'Khởi tạo (ngẫu nhiên ≈ đều nhau)'),
                         (axes[1], w, 'Sau khi huấn luyện')]:
        im = ax.imshow(m, cmap='Reds', vmin=0, vmax=max(0.5, w.max()))
        ax.set_xticks(range(K)); ax.set_xticklabels([f'Cụm {k+1}' for k in range(K)])
        ax.set_yticks(range(len(df.columns))); ax.set_yticklabels(df.columns)
        for c in range(m.shape[0]):
            for k in range(K):
                ax.text(k, c, f'{m[c, k]:.2f}', ha='center', va='center', fontsize=9,
                        color='white' if m[c, k] > 0.35 else 'black')
        ax.set_title(title, fontweight='bold')
    fig.colorbar(im, ax=axes, shrink=0.8, label='Trọng số w(c,k) = softmax theo cột')
    plt.suptitle(f'ETTh1 (C=7, K=K*={K}): ma trận phân cụm mềm W = softmax(cluster_w) — '
                 'mỗi cột là "công thức trộn" tạo ra 1 đại diện cụm', fontweight='bold')
    out = os.path.join(fig_dir, 'fig_cluster_w_etth1.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    np.set_printoptions(precision=3, suppress=True)
    print(df.columns.tolist()); print(w); print('saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-dir', default=None, help='directory with the data files')
    ap.add_argument('--fig-dir', default=os.path.join(ROOT, 'results', 'figures'))
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    main(a.fig_dir, a.data_dir)
