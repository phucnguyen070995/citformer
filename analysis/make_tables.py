"""Generate the LaTeX table rows and the K-sweep figure of the thesis from the
stored results.

Inputs:
  results/multivariate/*.json     experiments/run_multivariate.py
  results/stat_baselines/*.json   experiments/run_stat_baselines.py
  results/weather_clean/*.json    Weather runs after filling its missing-value codes

Outputs (`--out`, default results/tables): *_rows.tex, included by the thesis
with \\inputrows. Figure (`--fig-dir`, default results/figures): fig_ksweep.png.
Cells without results are written as \\cho (placeholder macro of the thesis).

    python analysis/make_tables.py
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(ROOT, 'results')
CLEAN = os.path.join(RES, 'weather_clean')
OUT = os.path.join(RES, 'tables')
FIG = os.path.join(RES, 'figures')

# (dataset, C, K used); K = max(3, K*)
ORDER = [('ETTh1', 7, 4), ('ETTm1', 7, 4), ('Exchange', 8, 3), ('Weather', 21, 6),
         ('Covid', 55, 10), ('Solar', 137, 3), ('ECL', 321, 32), ('Traffic', 862, 85)]
KSTAR = {'Solar': 1}                     # K* where it differs from the K used
VARIANTS = ['v1', 'v2', 'v3', 'v3_rand']  # columns of the variants table
MAIN = 'v4'                              # final model
CHO = r'\cho'


def fmt(x, ref):
    """Number of decimals matched to the scale of the dataset's MAE."""
    a = abs(ref)
    if a >= 100: return f'{x:.2f}'
    if a >= 10: return f'{x:.3f}'
    if a >= 0.01: return f'{x:.4f}'
    return f'{x:.5f}'


def pm(g, col='MAE'):
    """'mean +- std' over seeds; the number of seeds is shown when below 3."""
    m = g[col].mean(); sd = g[col].std(ddof=1) if len(g) > 1 else 0.0
    txt = f'{fmt(m, m)} $\\pm$ {fmt(sd, m)}'
    return (txt + f'$^{{(n={len(g)})}}$' if len(g) < 3 else txt), m


def load_dl():
    """All deep-model runs as one DataFrame (K = -1 for models without clusters).

    Weather rows come only from results/weather_clean (the earlier Weather runs
    on the uncleaned file are ignored). Naming of stored runs:
      - runs stored as 'v4' in results/multivariate predate the final v4 and
        used a per-cluster init strength; they are relabelled 'v4_autoalpha';
      - runs stored as 'v4' in results/weather_clean are the final v4, which
        for K* > 1 is the same computation as 'v3_revin'.
    The final v4 rows are then taken from 'v3_revin' (K* > 1) or from
    'v4_autoalpha' (Solar, K* = 1: random init, so the init strength is unused),
    both identical in computation to the final v4.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, 'multivariate', '*.json'))):
        rows += [r for r in json.load(open(f)) if r['dataset'] != 'Weather']
    for f in ('weather_cit.json', 'weather_base.json', 'weather_ksweep.json'):
        path = os.path.join(CLEAN, f)
        if os.path.exists(path):
            rows += [dict(r, config='v3_revin') if r['config'] == 'v4' else r
                     for r in json.load(open(path))]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=['dataset', 'config', 'seed', 'K', 'MAE', 'RMSE', 'cluster_cos'])
    df['K'] = df['K'].fillna(-1).astype(int)
    df = df.drop_duplicates(['dataset', 'config', 'seed', 'K'], keep='last')
    df.loc[df.config == 'v4', 'config'] = 'v4_autoalpha'
    parts = []
    for name, C, K in ORDER:
        src = 'v4_autoalpha' if KSTAR.get(name, K) == 1 else 'v3_revin'
        parts.append(df[(df.dataset == name) & (df.config == src)].assign(config='v4'))
    return pd.concat([df] + parts, ignore_index=True)


def load_stat():
    """Statistical baseline runs; Weather from results/weather_clean."""
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, 'stat_baselines', '*.json'))):
        rows += [r for r in json.load(open(f)) if r['dataset'] != 'Weather']
    path = os.path.join(CLEAN, 'weather_stat.json')
    if os.path.exists(path):
        rows += json.load(open(path))
    return pd.DataFrame(rows)


def write(name, lines):
    with open(os.path.join(OUT, name), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def variants_rows(df):
    """CiTFormer v1/v2/v3/v3_rand (MAE, 3 seeds; best in bold) and the centred
    cluster cosine of v3_rand / v3."""
    lines = []
    for name, C, K in ORDER:
        d = df[(df.dataset == name) & (df.K == K)]
        stats = {c: d[d.config == c] for c in VARIANTS if len(d[d.config == c])}
        best = min(stats, key=lambda c: stats[c].MAE.mean(), default=None)
        cells = []
        for c in VARIANTS:
            if c not in stats:
                cells.append(CHO); continue
            txt, _ = pm(stats[c])
            cells.append(r'\textbf{' + txt + '}' if c == best else txt)
        cc = 'cluster_cos_centered'
        cos = (f"{stats['v3_rand'][cc].mean():.2f} / {stats['v3'][cc].mean():.2f}"
               if 'v3_rand' in stats and 'v3' in stats and cc in stats['v3'] else CHO)
        lines.append(f'{name} & {C} & {K} & ' + ' & '.join(cells) + f' & {cos} \\\\')
    write('v1v2_rows.tex', lines)
    print('\n'.join(lines))


def ksweep(df):
    """K sensitivity: MAE for each K (best in bold, K used marked with *),
    and fig_ksweep.png (MAE relative to the K used)."""
    lines, curves = [], {}
    for name, C, K in ORDER:
        # sweep of the final model if available, else of v3, else of v2
        for cfg_name in (MAIN, 'v3', 'v2'):
            d = df[(df.dataset == name) & (df.config == cfg_name)]
            if d.K.nunique() >= 2:
                break
        if d.empty or d.K.nunique() < 2:
            lines.append(f'{name} & {C} & {KSTAR.get(name, K)} & {CHO} \\\\'); continue
        g = d.groupby('K').MAE
        means = g.mean(); best = means.idxmin()
        cells = []
        for k in sorted(means.index):
            txt, _ = pm(d[d.K == k])
            lab = f'$K{{=}}{k}$' + (r'$^\ast$' if k == K else '')
            cell = f'{lab}: {txt}'
            cells.append(r'\textbf{' + cell + '}' if k == best else cell)
        lines.append(f'{name} & {C} & {KSTAR.get(name, K)} & ' + '; '.join(cells) + r' \\')
        curves[name] = (np.array(sorted(means.index)) / K,
                        np.array([means[k] for k in sorted(means.index)]) / means.get(K, np.nan))
    write('ksweep_rows.tex', lines)
    if curves:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for i, (name, (x, y)) in enumerate(curves.items()):
            ax.plot(x, y, marker='o', lw=2, color=plt.get_cmap('tab10')(i), label=name)
        ax.axvline(1, ls='--', color='gray'); ax.axhline(1, ls=':', color='gray')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('K / K*  (thang log₂; 1 = K* dùng trong luận văn)')
        ax.set_ylabel('MAE / MAE tại K*  (< 1: tốt hơn K*)')
        ax.set_title('Độ nhạy của CiTFormer theo số cụm K (trung bình 3 seed)', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_ksweep.png'), dpi=150)
        plt.close(fig)


def _row(df, name, K, configs, baselines=('iTransformer', 'iTransformer_revin', 'DUET')):
    """MAE 'mean +- std' cells of one dataset for several configs, best in bold."""
    groups = {c: df[(df.dataset == name) & (df.config == c) &
                    (df.K == (-1 if c in baselines else K))] for c in configs}
    means = {c: g.MAE.mean() for c, g in groups.items() if len(g)}
    best = min(means, key=means.get) if means else None
    cells = []
    for c, g in groups.items():
        if not len(g):
            cells.append(CHO); continue
        txt = pm(g)[0]
        cells.append(r'\textbf{' + txt + '}' if c == best else txt)
    return cells


# Datasets whose 3-seed runs were made on the same GPU setup (Kaggle, 2x T4);
# training times of the other datasets (local CPU) are not reported.
GPU_TIMED = {'Solar', 'ECL', 'Traffic'}


def seeds_rows(df):
    """Main comparison, all models with per-window normalisation: MAE over
    3 seeds (best in bold), then seconds per epoch on GPU (fastest in bold)."""
    cols = ['iTransformer_revin', 'DUET', 'v4']
    lines = []
    for name, C, K in ORDER:
        cells = _row(df, name, K, cols)
        if name in GPU_TIMED:
            spe = {c: df[(df.dataset == name) & (df.config == c) &
                         (df.K == (-1 if c != 'v4' else K))].sec_per_epoch.mean() for c in cols}
            fast = min(spe, key=spe.get)
            cells += [(r'\textbf{%.1f}' if c == fast else '%.1f') % spe[c] for c in cols]
        else:
            cells += ['--'] * 3
        lines.append(f'{name} & {C} & ' + ' & '.join(cells) + r' \\')
    write('seeds_rows.tex', lines)


# Single-seed (MAE, RMSE) of the univariate deep baselines, as printed by the
# original per-dataset runs (same architectures as experiments/run_uv_deep.py).
# Weather is replaced by results/weather_clean/weather_uv_deep.json if present.
SINGLE = {
    'LSTM': dict(ETTh1=('2.7664', '3.3903'), ETTm1=('1.2488', '1.5288'), Exchange=('0.0372', '0.0430'),
                 Weather=('15.313', '21.749'), Covid=('27.711', '31.678'), Solar=('3.9676', '6.8025'),
                 ECL=('338.74', '468.96'), Traffic=('0.0038', '0.0055')),
    'N-HiTS': dict(ETTh1=('2.2824', '2.8627'), ETTm1=('1.2765', '1.5912'), Exchange=('0.0154', '0.0193'),
                   Weather=('13.218', '18.576'), Covid=('17.369', '19.691'), Solar=('3.7840', '6.9834'),
                   ECL=('275.12', '403.73'), Traffic=('0.0034', '0.0048')),
}
_uvd = os.path.join(CLEAN, 'weather_uv_deep.json')
if os.path.exists(_uvd):
    for _r in json.load(open(_uvd)):
        SINGLE[_r['model']]['Weather'] = (f"{_r['MAE']:.3f}", f"{_r['RMSE']:.3f}")
UV = ['ARIMA', 'ETS', 'DLinear', 'LSTM', 'N-HiTS']
MV = ['iTransformer', 'DUET', 'CiTFormer-v4']     # 3-seed means, with per-window normalisation
MV_CFG = {'iTransformer': ('iTransformer_revin', -1), 'DUET': ('DUET', -1)}


def single_tables(df, st):
    """MAE / RMSE tables of all methods. Bold: best multivariate model;
    underline: best of the column. Values are compared as printed."""
    for col, idx, fname in [('MAE', 0, 'mae_table_rows.tex'), ('RMSE', 1, 'rmse_table_rows.tex')]:
        txt = {}
        for name, C, K in ORDER:
            for mdl in UV + MV:
                if mdl in SINGLE:
                    txt[mdl, name] = SINGLE[mdl][name][idx]
                    continue
                if mdl == 'CiTFormer-v4':
                    v = df[(df.dataset == name) & (df.config == 'v4') & (df.K == K)][col].mean()
                elif mdl in MV_CFG:
                    cfg, kk = MV_CFG[mdl]
                    v = df[(df.dataset == name) & (df.config == cfg) & (df.K == kk)][col].mean()
                else:
                    v = st[(st.dataset == name) & (st.model == mdl)][col].iloc[-1]
                txt[mdl, name] = fmt(v, v)
        lines = []
        for group, models in (('UV', UV), ('MV', MV)):
            for i, mdl in enumerate(models):
                cells = []
                for name, C, K in ORDER:
                    vals = {m: float(txt[m, name]) for m in UV + MV}
                    s = txt[mdl, name]
                    if mdl in MV and vals[mdl] == min(vals[m] for m in MV):
                        s = r'\textbf{' + s + '}'
                    if vals[mdl] == min(vals.values()):
                        s = r'\underline{' + s + '}'
                    cells.append(s)
                lead = (r'\multirow{%d}{*}{%s}' % (len(models), group)) if i == 0 else ''
                lines.append(f'{lead} & {mdl} & ' + ' & '.join(cells) + r' \\')
            lines.append(r'\hline')
        write(fname, lines[:-1])


# Development stages. Stages 1-2: no per-window normalisation for any model
# (iTransformer, DUET_norevin, CiTFormer v1/v2 post-norm, v3 pre-norm).
# Stage 3: normalisation switched on for all three models (revin_effect_rows).
NO_K = ('iTransformer', 'iTransformer_revin', 'DUET', 'DUET_norevin')
# STL trend strength of the target (results/eda/eda_summary.csv)
TREND = {'ETTh1': 0.97, 'ETTm1': 0.97, 'Exchange': 1.00, 'Weather': 0.54,
         'Covid': 0.95, 'Solar': 0.12, 'ECL': 0.74, 'Traffic': 0.43}


def _grp(df, name, c, K):
    return df[(df.dataset == name) & (df.config == c) & (df.K == (-1 if c in NO_K else K))]


def stage_rows(df, fname, cols, delta=None):
    """MAE 'mean +- std' per config (best in bold); with delta=(a, b), a last
    column with the relative MAE change of b with respect to a."""
    lines = []
    for name, C, K in ORDER:
        cells = ['--' if x == CHO else x for x in _row(df, name, K, cols, baselines=NO_K)]
        if delta:
            a, b = (_grp(df, name, c, K).MAE for c in delta)
            cells.append(f'${100 * (b.mean() / a.mean() - 1):+.0f}$\\%' if len(a) and len(b) else CHO)
        lines.append(f'{name} & {C} & ' + ' & '.join(cells) + r' \\')
    write(fname, lines)


def revin_effect_rows(df):
    """Relative MAE change when per-window normalisation is switched on, per
    model, datasets sorted by trend strength. A change within the larger of
    the two seed standard deviations is marked as a tie (^{\\approx})."""
    pairs = [('iTransformer', 'iTransformer_revin'), ('DUET_norevin', 'DUET'), ('v3', 'v3_revin')]
    lines = []
    for name, C, K in sorted(ORDER, key=lambda r: -TREND[r[0]]):
        cells = [f'{TREND[name]:.2f}']
        for off, on in pairs:
            a, b = _grp(df, name, off, K).MAE, _grp(df, name, on, K).MAE
            if not (len(a) and len(b)):
                cells.append('--'); continue
            d = 100 * (b.mean() / a.mean() - 1)
            tie = abs(b.mean() - a.mean()) <= max(a.std(ddof=1), b.std(ddof=1))
            cells.append(f'${d:+.0f}$\\%' + (r'$^{\approx}$' if tie else ''))
        lines.append(f'{name} & {C} & ' + ' & '.join(cells) + r' \\')
    write('revin_effect_rows.tex', lines)


def main():
    global OUT, FIG
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=OUT, help='output directory of the .tex rows')
    ap.add_argument('--fig-dir', default=FIG, help='output directory of the figure')
    a = ap.parse_args()
    OUT, FIG = a.out, a.fig_dir
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    df, st = load_dl(), load_stat()
    variants_rows(df)
    ksweep(df)
    seeds_rows(df)
    single_tables(df, st)
    stage_rows(df, 'stage1_rows.tex', ['iTransformer', 'DUET_norevin', 'v1', 'v2'])
    stage_rows(df, 'stage2_rows.tex', ['iTransformer', 'DUET_norevin', 'v3'], delta=('v2', 'v3'))
    revin_effect_rows(df)
    print('tables written to', os.path.abspath(OUT))


if __name__ == '__main__':
    main()
