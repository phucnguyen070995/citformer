import numpy as np
import pandas as pd
import pytest
import torch

from citformer import data
from citformer.data import DATASETS, fill_missing_codes, prepare, prepare_target, windows
from citformer.models import CiTFormer, DUETModel, iTransformerModel
from experiments.run_multivariate import CITFORMER_CONFIGS, CONFIGS, build, sweep_ks
from experiments.run_naive import naive, seasonal_naive


def test_missing_codes_are_interpolated():
    df = pd.DataFrame({'a': [1.0, -9999, 3.0], 'b': [-9999, 2.0, 4.0]})
    out = fill_missing_codes(df)
    assert out['a'].tolist() == [1.0, 2.0, 3.0]
    assert out['b'].tolist() == [2.0, 2.0, 4.0]


def test_windows_shapes_and_alignment():
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    Xw, yw = windows(X, X[:, -1], L=3, H=2)
    assert Xw.shape == (6, 3, 2) and yw.shape == (6, 2)
    assert np.array_equal(yw[0], X[3:5, -1])


@pytest.mark.parametrize('config', list(CONFIGS))
def test_every_config_builds_and_runs(config):
    cfg = dict(DATASETS['ETTh1'], L=16, H=4, d=16, heads=2)
    train_mv = np.random.RandomState(0).randn(200, 7).astype(np.float32)
    model = build(config, cfg, train_mv, cfg['K'], seed=0)
    expected = CiTFormer if config in CITFORMER_CONFIGS else (DUETModel, iTransformerModel)
    assert isinstance(model, expected)
    assert model(torch.randn(2, 16, 7)).shape == (2, 4)


def test_v4_uses_random_init_when_kstar_is_one():
    cfg = dict(DATASETS['Solar'], L=16, H=4, d=16, heads=2)
    train_mv = np.random.RandomState(0).randn(200, 10).astype(np.float32)
    torch.manual_seed(0)
    model = build('v4', cfg, train_mv, cfg['K'], seed=0)
    assert model.cluster_w.abs().max() < 0.2   # randn * 0.02, no spectral logits


def test_ksweep_excludes_kstar():
    assert sweep_ks('ECL', 321) == [16, 18, 64]
    assert DATASETS['ECL']['K'] not in sweep_ks('ECL', 321)


def test_target_windows_match_multivariate_pipeline(tmp_path, monkeypatch):
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randn(400, 3).cumsum(0), columns=['a', 'b', 'OT'])
    df.insert(0, 'date', range(400))
    df.to_csv(tmp_path / 'toy.csv', index=False)
    monkeypatch.setitem(data.DATASETS, 'Toy', dict(file='toy.csv', L=24, H=12, K=2, d=16, heads=2, batch=8))
    _, mv, _, sc_a = prepare('Toy', str(tmp_path))
    _, uv, sc_b = prepare_target('Toy', str(tmp_path))
    for split in ('train', 'val', 'test'):
        assert np.allclose(mv[split][0][:, :, -1], uv[split][0], atol=1e-6)
        assert np.allclose(mv[split][1], uv[split][1], atol=1e-6)
    assert np.allclose(sc_a.mean_, sc_b.mean_) and np.allclose(sc_a.scale_, sc_b.scale_)


def test_naive_and_seasonal_naive():
    X = np.arange(10, dtype=np.float32)[None, :]
    assert np.array_equal(naive(X, 3), [[9, 9, 9]])
    assert np.array_equal(seasonal_naive(X, 5, 3), [[7, 8, 9, 7, 8]])
    with pytest.raises(ValueError):
        seasonal_naive(X, 5, 11)
