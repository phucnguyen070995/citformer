import copy
import time

import numpy as np
import torch
import torch.nn as nn


def train_model(model, train_loader, val_loader, epochs, patience, label,
                save_path=None, device='cpu', lr=1e-3, weight_decay=1e-5,
                grad_clip=1.0, log_every=20, lr_patience=5, lr_factor=0.5):
    """Train with Adam and MSE loss, early stopping on the validation loss.

    The learning rate is reduced on validation plateaus (ReduceLROnPlateau) and
    the best-validation weights are restored at the end (and saved to
    `save_path` if given). Uses DataParallel when several GPUs are available.

    Returns (train losses per epoch, best validation loss, training seconds,
    peak GPU memory in MB; 0 on CPU).
    """
    device = torch.device(device)
    model = model.to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=lr_factor, patience=lr_patience)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    losses = []
    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = nn.MSELoss()(pred, yb)
            opt.zero_grad()
            if loss.requires_grad:  # False only when no parameter is trainable
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += nn.MSELoss()(model(xb), yb).item()
        val_loss /= len(val_loader)
        sch.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            target = model.module if hasattr(model, 'module') else model
            best_state = copy.deepcopy(target.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (ep + 1) % log_every == 0:
            elapsed = time.time() - t0
            eta = elapsed / (ep + 1) * (epochs - ep - 1)
            cur_lr = opt.param_groups[0]['lr']
            print(f'  {label} Ep {ep+1:3d}/{epochs}  Train:{losses[-1]:.5f}  '
                  f'Val:{val_loss:.5f}  Best:{best_val:.5f}  LR:{cur_lr:.2e}  '
                  f'{elapsed/60:.1f}min  ETA:{eta/60:.1f}min')

        if epochs_no_improve >= patience:
            print(f'  {label} Early stop at epoch {ep+1} '
                  f'(no val improvement for {patience} epochs)')
            break

    elapsed = time.time() - t0
    memory_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                 if device.type == 'cuda' else 0.0)

    target = model.module if hasattr(model, 'module') else model
    if best_state is not None:
        target.load_state_dict(best_state)
    if save_path is not None and best_state is not None:
        torch.save(best_state, save_path)

    return losses, best_val, elapsed, memory_mb


def predict(model, X, device, bs=32):
    """Forecast for an array of windows X in mini-batches; returns a numpy array."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.tensor(X[i:i + bs]).to(device)).cpu().numpy())
    return np.concatenate(out)
