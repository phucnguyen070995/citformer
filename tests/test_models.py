import numpy as np
import pytest
import torch
import torch.nn as nn

from citformer.models import CiTFormer, DUETModel, iTransformerModel, spectral_cluster_init


def _duet(n_vars=7, **kw):
    return DUETModel(n_vars=n_vars, L=96, H=96, d=32, heads=4, M=4, k=2, d0=16, **kw)


def _two_group_data(T=2000, seed=0):
    """Variables 0-3 follow one latent signal, variables 4-7 another."""
    rng = np.random.RandomState(seed)
    a, b = rng.randn(T, 1), rng.randn(T, 1)
    return np.hstack([a + 0.1 * rng.randn(T, 4), b + 0.1 * rng.randn(T, 4)])


def _record_seq_len(store):
    """Forward hook that stores the token count of the module input."""
    def hook(module, inputs, output):
        store['seq'] = inputs[0].shape[1]
    return hook


# ---------------------------------------------------------------- DUET

def test_duet_output_shape():
    assert _duet()(torch.randn(4, 96, 7)).shape == (4, 96)


def test_duet_ccm_mask_is_full_n_by_n():
    model = _duet()
    model(torch.randn(2, 96, 7))
    assert model._last_mask.shape == (2, 7, 7)


def test_duet_eval_is_deterministic_and_train_is_stochastic():
    model = _duet()
    x = torch.randn(4, 96, 7)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), model(x))
    model.train()
    with torch.no_grad():
        assert not torch.equal(model(x), model(x))


def test_duet_ccm_mask_diagonal_is_near_one():
    model = _duet()
    model(torch.randn(2, 96, 7))
    assert torch.all(torch.diagonal(model._last_mask, dim1=-2, dim2=-1) > 0.99)


def test_duet_tcm_gate_keeps_at_most_k_experts():
    model = _duet()
    model(torch.randn(2, 96, 7))
    assert torch.all((model._last_gate > 0).sum(dim=-1) <= 2)


def test_duet_tcm_extractors_are_linear():
    model = _duet()
    for extractor in list(model.pattern_extractors_trend) + list(model.pattern_extractors_seasonal):
        assert isinstance(extractor, nn.Linear)


def test_duet_invalid_arguments_raise():
    with pytest.raises(ValueError):
        DUETModel(n_vars=7, L=96, H=96, d=32, heads=4, M=2, k=3, d0=16)
    with pytest.raises(ValueError):
        DUETModel(n_vars=7, L=96, H=96, d=30, heads=4, M=4, k=2, d0=16)


def test_duet_gradients_reach_every_module():
    model = _duet()
    model(torch.randn(4, 96, 7)).sum().backward()
    for name, param in [('CCM.A', model.A),
                        ('TCM.W_H', model.W_H.weight),
                        ('TCM.extractor', model.pattern_extractors_trend[0].weight),
                        ('FM.q_proj', model.q_proj.weight)]:
        assert param.grad is not None and torch.any(param.grad != 0), name


def test_duet_revin_on_by_default_and_ablation_differs():
    torch.manual_seed(0)
    on = _duet()
    off = _duet(revin=False)
    off.load_state_dict(on.state_dict())
    on.eval(); off.eval()
    x = torch.randn(4, 96, 7) * 3 + 5
    with torch.no_grad():
        y_on, y_off = on(x), off(x)
    assert on.revin is True
    assert y_off.shape == (4, 96)
    assert not torch.allclose(y_on, y_off)


# ---------------------------------------------------------------- iTransformer

def test_itransformer_attends_over_all_variables():
    model = iTransformerModel(n_vars=9, L=96, H=24, d=32, heads=4, n_layers=1)
    captured = {}
    h = model.transformer.register_forward_hook(_record_seq_len(captured))
    out = model(torch.randn(2, 96, 9))
    h.remove()
    assert out.shape == (2, 24) and captured['seq'] == 9


# ---------------------------------------------------------------- CiTFormer

def test_citformer_output_shape_and_cluster_matrix():
    model = CiTFormer(n_vars=7, L=96, H=96, K=3, d=32, heads=4, n_layers=2)
    assert model(torch.randn(4, 96, 7)).shape == (4, 96)
    assert model.cluster_w.shape == (7, 3)


def test_citformer_attention_is_over_clusters():
    model = CiTFormer(n_vars=7, L=96, H=96, K=3, d=32, heads=4, n_layers=2)
    captured = {}
    handle = model.cluster_attn.register_forward_hook(_record_seq_len(captured))
    model(torch.randn(2, 96, 7))
    handle.remove()
    assert captured['seq'] == model.K


def test_citformer_gradients_reach_cluster_matrix():
    model = CiTFormer(n_vars=7, L=96, H=96, K=3, d=32, heads=4, n_layers=2)
    model(torch.randn(4, 96, 7)).sum().backward()
    assert model.cluster_w.grad is not None and torch.any(model.cluster_w.grad != 0)


def test_spectral_init_groups_co_moving_variables():
    logits, labels = spectral_cluster_init(_two_group_data(), K=2, alpha=2.0)
    assert logits.shape == (8, 2)
    assert len(set(labels[:4])) == 1 and len(set(labels[4:])) == 1
    assert labels[0] != labels[4]
    assert np.allclose(logits.sum(1), 2.0) and np.allclose(logits.max(1), 2.0)


def test_random_init_gives_near_identical_clusters():
    """v1 starts every cluster as an almost uniform average over variables."""
    torch.manual_seed(0)
    model = CiTFormer(n_vars=50, L=96, H=96, K=8, d=32, heads=4, n_layers=1)
    assert model.cluster_emb is None
    w = torch.softmax(model.cluster_w, dim=0)
    w = w / w.norm(dim=0, keepdim=True)
    assert (w.T @ w)[~torch.eye(8, dtype=torch.bool)].min() > 0.99


def test_spectral_init_gives_distinct_clusters_and_trains():
    logits, _ = spectral_cluster_init(_two_group_data(), K=2, alpha=3.0)
    model = CiTFormer(n_vars=8, L=96, H=24, K=2, d=32, heads=4, n_layers=1,
                      cluster_init=logits, cluster_emb=True)
    w = torch.softmax(model.cluster_w, dim=0)
    w = w / w.norm(dim=0, keepdim=True)
    assert (w[:, 0] @ w[:, 1]) < 0.5
    out = model(torch.randn(3, 96, 8))
    assert out.shape == (3, 24)
    out.sum().backward()
    assert torch.any(model.cluster_emb.grad != 0) and torch.any(model.cluster_w.grad != 0)


def test_cluster_init_shape_is_checked():
    with pytest.raises(AssertionError):
        CiTFormer(n_vars=7, L=96, H=96, K=3, d=32, heads=4, n_layers=1,
                  cluster_init=np.zeros((7, 4)))


def test_prenorm_keeps_input_scale_postnorm_removes_it():
    torch.manual_seed(0)
    x = torch.randn(4, 96, 7)
    for prenorm in (False, True):
        m = CiTFormer(n_vars=7, L=96, H=24, K=3, d=32, heads=4, n_layers=1, prenorm=prenorm).eval()
        with torch.no_grad():
            ratio = (m(10 * x).std() / m(x).std()).item()
        assert (ratio > 3) if prenorm else (ratio < 3), ratio


def test_prenorm_with_spectral_init_trains():
    logits, _ = spectral_cluster_init(_two_group_data(), K=2, alpha=2.0)
    m = CiTFormer(n_vars=8, L=96, H=24, K=2, d=32, heads=4, n_layers=1,
                  cluster_init=logits, cluster_emb=True, prenorm=True)
    out = m(torch.randn(3, 96, 8))
    assert out.shape == (3, 24)
    out.sum().backward()
    assert torch.any(m.cluster_w.grad != 0) and torch.any(m.cluster_emb.grad != 0)


def test_revin_forecast_is_equivariant_to_shift_and_scale():
    torch.manual_seed(0)
    x = torch.randn(4, 96, 7)
    for m in (CiTFormer(n_vars=7, L=96, H=24, K=3, d=32, heads=4, n_layers=1,
                        prenorm=True, revin=True).eval(),
              iTransformerModel(n_vars=7, L=96, H=24, d=32, heads=4, n_layers=1, revin=True).eval()):
        with torch.no_grad():
            y, y2 = m(x), m(3.0 * x + 5.0)
        assert torch.allclose(y2, 3.0 * y + 5.0, atol=1e-3)


def test_revin_is_off_by_default():
    assert CiTFormer(n_vars=7, K=3).revin is False
    assert iTransformerModel(n_vars=7).revin is False
