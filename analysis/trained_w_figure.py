"""Cluster matrix W of CiTFormer-v4 on ETTh1 and Weather: spectral
initialisation next to the trained matrix (seed 0, same pipeline and settings
as experiments/run_multivariate.py). Boxed cells mark each variable's initial
group.

    python analysis/trained_w_figure.py   ->  results/figures/fig_trained_w_v4.png
"""
import argparse
import contextlib
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from citformer.data import load_df, prepare
from citformer.models import spectral_cluster_init
from citformer.training import train_model
from experiments.run_multivariate import build, cluster_cos_centered


def train_v4(name, seed=0, data_dir=None):
    """Train v4 on one dataset; return initial W, trained W, init groups,
    variable names and the centred cluster cosine."""
    cfg, parts, train_mv, sc_uv = prepare(name, data_dir)
    torch.manual_seed(seed); np.random.seed(seed)
    _, labels = spectral_cluster_init(train_mv, cfg['K'], 2.0, seed=seed)
    model = build('v4', cfg, train_mv, cfg['K'], seed)
    init = torch.softmax(model.cluster_w.detach().clone(), 0).numpy()
    (Xtr, ytr), (Xva, yva) = parts['train'], parts['val']
    with contextlib.redirect_stdout(io.StringIO()):
        train_model(model, DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                                      batch_size=cfg['batch'], shuffle=True),
                    DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva)), batch_size=cfg['batch']),
                    epochs=200, patience=20, label=name, device='cpu')
    trained = torch.softmax(model.cluster_w.detach(), 0).numpy()
    names = [c.split(' (')[0] for c in load_df(cfg, data_dir).columns]   # drop units
    return init, trained, labels, names, cluster_cos_centered(model)


def panel(ax, W, names, labels, title, vmax):
    order = np.argsort(labels, kind='stable')
    ax.imshow(W[order], cmap='Reds', vmin=0, vmax=vmax, aspect='auto')
    ax.set_yticks(range(len(names))); ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xticks(range(W.shape[1])); ax.set_xticklabels([f'C{k+1}' for k in range(W.shape[1])], fontsize=8)
    for y, i in enumerate(order):
        ax.add_patch(plt.Rectangle((labels[i] - 0.5, y - 0.5), 1, 1, fill=False, ec='#2C3E50', lw=0.8))
    ax.set_title(title, fontsize=10, fontweight='bold')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-dir', default=None, help='directory with the data files')
    ap.add_argument('--fig-dir', default=os.path.join(ROOT, 'results', 'figures'))
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(17, 6.2), gridspec_kw={'width_ratios': [1, 1, 1.6, 1.6]})
    for col, name in [(0, 'ETTh1'), (2, 'Weather')]:
        init, trained, labels, names, cc = train_v4(name, data_dir=a.data_dir)
        vmax = max(init.max(), trained.max())
        panel(axes[col], init, names, labels, f'{name}: khởi tạo (theo nhóm biến)', vmax)
        panel(axes[col + 1], trained, names, labels, f'{name}: sau huấn luyện (cos$_c$={cc:.2f})', vmax)
        own = np.mean([trained[labels == k, k].sum() for k in range(trained.shape[1])])
        print(f'{name}: centred cos={cc:.3f}, mean own-group weight after training={own:.2f}')
    plt.suptitle('Ma trận cụm W của CiTFormer-v4: các cụm sau huấn luyện vẫn là các nhóm biến khác nhau '
                 '(ô viền = nhóm ban đầu của biến)', fontweight='bold')
    plt.tight_layout()
    out = os.path.join(a.fig_dir, 'fig_trained_w_v4.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print('saved', out)
