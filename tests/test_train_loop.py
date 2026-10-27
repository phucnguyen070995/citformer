import os

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from citformer.training import train_model


def _toy_loaders(n_train=64, n_val=16, L=8, H=4, batch_size=8):
    Xtr, ytr = torch.randn(n_train, L, 3), torch.randn(n_train, H)
    Xva, yva = torch.randn(n_val, L, 3), torch.randn(n_val, H)
    return (DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False))


class ToyModel(nn.Module):
    def __init__(self, L=8, C=3, H=4):
        super().__init__()
        self.lin = nn.Linear(L * C, H)

    def forward(self, x):
        return self.lin(x.reshape(x.size(0), -1))


def _frozen():
    model = ToyModel()
    for p in model.parameters():
        p.requires_grad = False
    return model


def test_returns_losses_best_val_time_and_memory():
    losses, best_val, elapsed, mem_mb = train_model(
        ToyModel(), *_toy_loaders(), epochs=5, patience=20, label='toy', device='cpu')
    assert len(losses) == 5
    assert isinstance(best_val, float)
    assert elapsed >= 0
    assert mem_mb == 0.0


def test_early_stopping():
    losses, *_ = train_model(_frozen(), *_toy_loaders(), epochs=100, patience=2,
                             label='frozen', device='cpu')
    assert len(losses) <= 5


def test_learning_rate_is_reduced_on_plateau(capsys):
    train_model(_frozen(), *_toy_loaders(), epochs=100, patience=15, label='frozen',
                device='cpu', lr_patience=3, lr=1e-3, log_every=1)
    lrs = [float(line.split('LR:')[1].split()[0])
           for line in capsys.readouterr().out.splitlines() if 'LR:' in line]
    assert len(lrs) >= 4
    assert lrs[0] == pytest.approx(1e-3)
    assert lrs[-1] < lrs[0]


def test_best_checkpoint_is_saved_and_reloadable(tmp_path):
    save_path = str(tmp_path / 'toy_best.pt')
    train_model(ToyModel(), *_toy_loaders(), epochs=5, patience=20, label='toy',
                save_path=save_path, device='cpu')
    assert os.path.exists(save_path)
    ToyModel().load_state_dict(torch.load(save_path, map_location='cpu'))


def test_final_weights_are_the_best_validation_checkpoint():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    model = ToyModel()
    _, best_val, _, _ = train_model(model, train_loader, val_loader, epochs=30, patience=30,
                                    label='toy', device='cpu')
    model.eval()
    with torch.no_grad():
        val = sum(nn.MSELoss()(model(xb), yb).item() for xb, yb in val_loader) / len(val_loader)
    assert abs(val - best_val) < 1e-5
