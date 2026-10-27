"""Training time per epoch of iTransformer, DUET and CiTFormer-v1 (one timing
run on Kaggle, 2x T4). The values are those of the timing table of the thesis;
Covid is omitted (< 0.5 s/epoch for every model).

    python analysis/timing_figure.py   ->  results/figures/timing_per_epoch.png
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# dataset, C, seconds per epoch: iTransformer, DUET, CiTFormer-v1
ROWS = [('ETTh1', 7, 5.3, 7.6, 6.7), ('ETTm1', 7, 20.3, 29.1, 25.4), ('Exchange', 8, 1.5, 1.8, 1.6),
        ('Weather', 21, 15.1, 21.6, 18.6), ('Solar', 137, 16.8, 24.0, 20.6),
        ('ECL', 321, 12.3, 13.7, 10.2), ('Traffic', 862, 32.3, 40.8, 9.1)]


def main(fig_dir):
    x = np.arange(len(ROWS)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, (lab, col) in enumerate([('iTransformer*', '#e8457a'), ('DUET', '#73909c'), ('CiTFormer', '#c65145')]):
        ax.bar(x + (i - 1) * w, [r[2 + i] for r in ROWS], w, label=lab, color=col)
    for xi, r in zip(x, ROWS):   # DUET / CiTFormer ratio above each group
        ax.text(xi, max(r[2:]) * 1.25, f'{r[3] / r[4]:.1f}×', ha='center', color='#b03a2e',
                fontweight='bold', fontsize=10)
    ax.set_yscale('log'); ax.set_ylim(1, 80)
    ax.set_xticks(x); ax.set_xticklabels([f'{r[0]}\n(C={r[1]})' for r in ROWS])
    ax.set_ylabel('Giây / epoch (thang log)')
    ax.set_title('Thời gian huấn luyện mỗi epoch: iTransformer*, DUET và CiTFormer', fontweight='bold')
    ax.legend(loc='upper left'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(fig_dir, 'timing_per_epoch.png')
    plt.savefig(out, dpi=150)
    print('saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fig-dir', default=os.path.join(ROOT, 'results', 'figures'))
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    main(a.fig_dir)
