"""Train and evaluate the multivariate deep models (CiTFormer versions,
iTransformer, DUET) over several seeds.

Pipeline: 70/10/20 chronological split, StandardScaler fitted on the train
split, sliding windows, target = last column, metrics on the first N_TEST
test windows in original units.

Configurations (names are those stored in results/multivariate/*.json):
  v1                  CiTFormer, random cluster init, post-norm
  v2_emb              v1 + cluster embedding
  v2                  v2_emb + spectral cluster init (alpha = 2)
  v3_rand             v1 with pre-norm blocks (control for v3)
  v3                  v2 with pre-norm blocks
  v3_revin            v3 + per-window normalisation (RevIN)
  v4                  v3_revin; random init when K* = 1 (final model)
  iTransformer        iTransformer, iTransformer_revin with RevIN
  DUET                DUET, DUET_norevin without its normalisation (ablation)

`--ks sweep` runs CiTFormer with K in {K*/2, 2K*, sqrt(C)} (clipped to
[2, C], K* excluded); `--ks 8 16` runs explicit values.

Each run also records the mean pairwise cosine similarity between the columns
of the trained cluster matrix (`cluster_cos`, and `cluster_cos_centered` on
w_k - 1/C), where 1 means that all clusters are identical.

Example:
    python experiments/run_multivariate.py --datasets ETTh1 --configs v4 \\
        --seeds 0 1 2 --out results/multivariate/etth1_v4.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from citformer.data import DATASETS, N_TEST, metrics, prepare
from citformer.models import CiTFormer, DUETModel, iTransformerModel, spectral_cluster_init
from citformer.training import predict, train_model

CONFIGS = {
    'v1':       dict(alpha=None, cluster_emb=False),
    'v2_emb':   dict(alpha=None, cluster_emb=True),
    'v2':       dict(alpha=2.0,  cluster_emb=True),
    'v3_rand':  dict(alpha=None, cluster_emb=False, prenorm=True),
    'v3':       dict(alpha=2.0,  cluster_emb=True,  prenorm=True),
    'v3_revin': dict(alpha=2.0,  cluster_emb=True,  prenorm=True, revin=True),
    'v4':       dict(alpha=2.0,  cluster_emb=True,  prenorm=True, revin=True,
                     random_init_if_kstar1=True),
    'iTransformer':       dict(model='itransformer'),
    'iTransformer_revin': dict(model='itransformer', revin=True),
    'DUET':               dict(model='duet'),
    'DUET_norevin':       dict(model='duet', revin=False),
}
CITFORMER_CONFIGS = {c for c, o in CONFIGS.items() if 'model' not in o}


def sweep_ks(name, C):
    """K values of the sensitivity study: K*/2, 2K*, sqrt(C), without K*."""
    k_star = DATASETS[name]['K']
    ks = {max(2, round(k_star / 2)), min(C, 2 * k_star), max(2, round(C ** 0.5))}
    return sorted(ks - {k_star})


def _mean_offdiag_cos(w, eps):
    w = w / (w.norm(dim=0, keepdim=True) + eps)
    K = w.shape[1]
    if K < 2:
        return float('nan')
    return float((w.T @ w)[~torch.eye(K, dtype=torch.bool)].mean())


def cluster_cos(model):
    """Mean pairwise cosine between the columns of softmax(cluster_w)."""
    if not isinstance(model, CiTFormer):
        return float('nan')
    w = torch.softmax(model.cluster_w.detach().cpu(), dim=0)
    return _mean_offdiag_cos(w, 0.0)


def cluster_cos_centered(model):
    """As cluster_cos, on the deviations w_k - 1/C from the uniform average."""
    if not isinstance(model, CiTFormer):
        return float('nan')
    w = torch.softmax(model.cluster_w.detach().cpu(), dim=0)
    return _mean_offdiag_cos(w - 1.0 / w.shape[0], 1e-12)


def build(config, cfg, train_mv, K, seed):
    """Instantiate the model of a configuration for one dataset."""
    C, opts = train_mv.shape[1], CONFIGS[config]
    if opts.get('model') == 'itransformer':
        return iTransformerModel(C, cfg['L'], cfg['H'], d=cfg['d'], heads=cfg['heads'],
                                 n_layers=2, revin=opts.get('revin', False))
    if opts.get('model') == 'duet':
        return DUETModel(C, cfg['L'], cfg['H'], d=cfg['d'], heads=cfg['heads'],
                         revin=opts.get('revin', True))
    init = None
    random_init = opts.get('random_init_if_kstar1') and cfg.get('kstar', cfg['K']) == 1
    if opts['alpha'] is not None and not random_init:
        init, _ = spectral_cluster_init(train_mv, K, opts['alpha'], seed=seed)
    return CiTFormer(C, cfg['L'], cfg['H'], K=K, d=cfg['d'], heads=cfg['heads'],
                     n_layers=2, cluster_init=init, cluster_emb=opts['cluster_emb'],
                     prenorm=opts.get('prenorm', False), revin=opts.get('revin', False))


def run_one(name, cfg, parts, train_mv, sc_uv, config, seed, epochs, device, max_train, K=None):
    """Train one (dataset, configuration, seed, K) and return its result row."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    K = K or cfg['K']
    model = build(config, cfg, train_mv, K, seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = parts['train'], parts['val'], parts['test']
    if max_train:
        Xtr, ytr = Xtr[:max_train], ytr[:max_train]
    tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=cfg['batch'], shuffle=True)
    va = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva)),
                    batch_size=cfg['batch'], shuffle=False)
    losses, best_val, elapsed, mem = train_model(
        model, tr, va, epochs=epochs, patience=20, label=f'{name}/{config}/s{seed}',
        device=device)
    model = model.to(device)
    pred = predict(model, Xte[:N_TEST], device, bs=cfg['batch'])
    return {'dataset': name, 'config': config, 'seed': seed,
            'K': K if config in CITFORMER_CONFIGS else None,
            **metrics(yte[:N_TEST], pred, sc_uv), 'best_val': best_val,
            'epochs_run': len(losses), 'train_sec': elapsed,
            'sec_per_epoch': elapsed / max(len(losses), 1), 'peak_mem_mb': mem,
            'cluster_cos': cluster_cos(model), 'cluster_cos_centered': cluster_cos_centered(model)}


def summarize(rows):
    """Mean and std over seeds per (dataset, configuration, K)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.assign(K=df['K'].fillna(-1).astype(int))   # -1: model without clusters
    g = df.groupby(['dataset', 'config', 'K'])
    out = g[['MAE', 'RMSE', 'cluster_cos', 'sec_per_epoch']].agg(['mean', 'std'])
    out[('n', 'seeds')] = g.size()
    return out.round(4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--out', default='results.json',
                    help='JSON file; existing runs in it are skipped')
    ap.add_argument('--data-dir', default=None, help='directory with the data files')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--ks', nargs='+', default=None,
                    help="K values for the CiTFormer configs, or 'sweep' (default: K* only)")
    ap.add_argument('--max-train', type=int, default=0,
                    help='cap the number of training windows (smoke tests only)')
    a = ap.parse_args(argv)

    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r['dataset'], r['config'], r['seed'], r.get('K')) for r in rows}
    for name in a.datasets:
        cfg, parts, train_mv, sc_uv = prepare(name, a.data_dir)
        C = train_mv.shape[1]
        print(f'\n=== {name}: C={C} K={cfg["K"]} '
              f'train/val/test windows={[len(parts[p][0]) for p in ("train", "val", "test")]}',
              flush=True)
        if a.ks is None:
            ks = [cfg['K']]
        elif a.ks == ['sweep']:
            ks = sweep_ks(name, C)
        else:
            ks = [int(k) for k in a.ks]
        for seed in a.seeds:
            for config in a.configs:
                for K in (ks if config in CITFORMER_CONFIGS else [None]):
                    if (name, config, seed, K) in done:
                        print(f'skip {name}/{config}/K={K}/s{seed} (already in {a.out})')
                        continue
                    t0 = time.time()
                    r = run_one(name, cfg, parts, train_mv, sc_uv, config, seed,
                                a.epochs, a.device, a.max_train, K=K)
                    rows.append(r)
                    json.dump(rows, open(a.out, 'w'), indent=1)   # saved after every run
                    print(f'  -> {name}/{config}/K={K}/s{seed}: MAE={r["MAE"]:.4f} '
                          f'RMSE={r["RMSE"]:.4f} cos={r["cluster_cos"]:.3f} '
                          f'epochs={r["epochs_run"]} [{time.time() - t0:.0f}s]', flush=True)
        del parts
    print('\n' + summarize(rows).to_string())
    return rows


if __name__ == '__main__':
    main()
