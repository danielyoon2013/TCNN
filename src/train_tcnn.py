"""TCNN training functions — rungs 4 (linear), 5 (1-channel), 6 (3-channel paper config).

Unified architecture supports all three variants via config flags:
  - linear=True            → single Conv1d(in_ch, 1, kernel=lookback), no activation. Rung 4.
  - num_heads=K, in_ch=1   → multi-head attention pooling, 1-channel. Rung 5.
  - num_heads=K, in_ch=3   → multi-head attention pooling, 3-channel paper config. Rung 6.

The runner calls `train_and_evaluate(cfg, panel, year, seed)` once per (year, seed)
cell. This function builds the TCNN panels on-the-fly (cached to disk per
experiment), trains the model on the rolling train+val window, and evaluates
on the test year. Returns a DataFrame of daily P&L for that test year.
"""

from __future__ import annotations
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from . import panels as panels_mod
from . import portfolio as portfolio_mod


# =============================================================================
# Determinism
# =============================================================================

def seed_everything(seed: int = 0, deterministic: bool = True):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


# =============================================================================
# Unified encoder — supports linear (rung 4), full TCNN (rungs 5, 6)
# =============================================================================

class TCNEncoder(nn.Module):
    """One class, three variants via config:
      linear=True         → Conv1d(in_ch, 1, kernel_size=lookback, bias=False), no activation
      linear=False        → 5-layer dilated TCN + multi-head additive attention pooling
    """

    def __init__(self, in_ch: int = 1, hidden: int = 48, d: int = 96,
                 kernel_size: int = 5, dilations=(1, 2, 4, 8, 16),
                 dropout: float = 0.15, num_heads: int = 4,
                 lookback: int = 252, linear: bool = False):
        super().__init__()
        self.linear = linear
        self.num_heads = num_heads
        self.lookback = lookback

        if linear:
            # Rung 4: single linear functional. Input (N, in_ch, L); output (N, 1).
            # `Conv1d(in_ch, 1, kernel_size=lookback, bias=False)` with input length lookback
            # gives a single scalar per sample = sum_l W[l] * input[l].
            self.linear_conv = nn.Conv1d(in_ch, 1, kernel_size=lookback, bias=False)
            return

        # Rungs 5 & 6: full nonlinear TCN
        if kernel_size % 2 == 0:
            raise ValueError("Use odd kernel size so padding keeps length stable.")

        layers = []
        ch = in_ch
        for dil in dilations:
            pad = (kernel_size - 1) * dil // 2
            layers += [
                nn.Conv1d(ch, hidden, kernel_size=kernel_size, dilation=dil, padding=pad),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.GroupNorm(num_groups=min(4, hidden), num_channels=hidden),
            ]
            ch = hidden
        self.net = nn.Sequential(*layers)

        # Multi-head additive attention scorer (paper Appx A: 2-layer MLP, hidden H/2, tanh)
        self.attention = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_heads),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * num_heads, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_ch, L)
        if self.linear:
            return self.linear_conv(x).squeeze(-1).squeeze(-1)  # (N,)

        z = self.net(x)                             # (N, hidden, L)
        z_t = z.transpose(1, 2)                     # (N, L, hidden)
        scores = self.attention(z_t)                # (N, L, K)
        weights = F.softmax(scores, dim=1)          # (N, L, K)
        feats = torch.einsum("ntk,nth->nkh", weights, z_t)  # (N, K, hidden)
        feats = feats.reshape(feats.size(0), -1)    # (N, K*hidden)
        h = self.proj(feats)                        # (N, d)
        return h


# =============================================================================
# Sharpe loss + portfolio-return computation (with weight drift, BUG-8 cash pad)
# =============================================================================

def two_softmax_weights(u: torch.Tensor, gross_leverage: float = 1.0,
                         z_cap: float | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Cross-sectional z-score → (optional winsor clip) → dual-softmax → dollar-neutral L/S.

    `z_cap` (None or positive float):
      - None  : raw dual-softmax (paper config). Concentrates heavily on fat-tailed scores.
      - 2.5   : winsorize z-scores to [-2.5, +2.5] before softmax. `torch.clamp` is
                differentiable (gradient = 1 inside, 0 at endpoints), so this is a
                drop-in replacement that prevents the 3-stock concentration we
                empirically observed for hand-crafted momentum factors.
    """
    u = u - u.mean()
    u = u / (u.std(unbiased=False) + eps)
    if z_cap is not None:
        u = torch.clamp(u, -z_cap, z_cap)
    long  = F.softmax(u, dim=0)
    short = F.softmax(-u, dim=0)
    w = long - short
    w = w - w.mean()
    w = w * (gross_leverage / (w.abs().sum() + 1e-12))
    return w


def portfolio_returns_drift(w_initial: torch.Tensor, daily_rets: torch.Tensor) -> torch.Tensor:
    """BUG-8: NaN returns (delisted stocks past their last day) treated as 0 (cash)."""
    H = daily_rets.shape[0]
    out = []
    w = w_initial
    for t in range(H):
        r = daily_rets[t]
        r = torch.where(torch.isnan(r), torch.zeros_like(r), r)
        port_ret = (w * r).sum()
        out.append(port_ret)
        w = w * (1 + r) / (1 + port_ret + 1e-8)
    return torch.stack(out)


# =============================================================================
# Build / cache TCNN panels per experiment
# =============================================================================

def get_or_build_tcnn_panels(cfg: dict, panel: pd.DataFrame) -> panels_mod.TCNNPanels:
    """Build TCNN panels matching the experiment's feature_cols + universe.

    Caches to disk per experiment so subsequent (year, seed) cells reuse.
    """
    in_ch = cfg["factor"].get("in_ch", 1)
    if in_ch == 1:
        feature_cols = ["ret"]
    elif in_ch == 3:
        feature_cols = ["ret", "vol_ewma", "ret_norm"]
    else:
        raise ValueError(f"in_ch={in_ch} not supported (must be 1 or 3)")

    cache_key = f"panel_{in_ch}ch_{cfg['factor'].get('lookback', config.LOOKBACK_DAYS)}lookback"
    cache_dir = config.PANELS_DIR / cache_key

    if (cache_dir / "X_panel.npy").exists():
        print(f"  loading cached panels from {cache_dir}")
        return panels_mod.load_panels(str(cache_dir), mmap=True)

    print(f"  building panels at {cache_dir}...")
    universe_col = cfg["universe"]["filters"][0]
    additional = tuple(c for c in cfg["universe"]["filters"][1:])
    p = panels_mod.build_tcnn_panels(
        panel,
        feature_cols=feature_cols,
        universe_col=universe_col,
        additional_filter_cols=additional,
        lookback=cfg["factor"].get("lookback", config.LOOKBACK_DAYS),
    )
    panels_mod.save_panels(p, str(cache_dir))
    return p


# =============================================================================
# Per-fold dataset slice
# =============================================================================

class FoldDataset:
    """View into TCNNPanels for one (train, val, test) fold."""
    def __init__(self, panels: panels_mod.TCNNPanels, start_idx: int, end_idx: int):
        self.X = panels.X[start_idx:end_idx]
        self.Y = panels.Y[start_idx:end_idx]
        self.holding_days = panels.holding_days[start_idx:end_idx]
        self.mask = panels.mask[start_idx:end_idx]
        self.rebal_dates = panels.rebal_dates[start_idx:end_idx]

    def __len__(self):
        return len(self.X)

    def get_month_data(self, idx):
        m = self.mask[idx]
        H = self.holding_days[idx]
        X_t = torch.tensor(np.array(self.X[idx, m]), dtype=torch.float32)
        Y_t = torch.tensor(np.array(self.Y[idx, m, :H]), dtype=torch.float32).T  # (H, N)
        return X_t, Y_t, self.rebal_dates[idx], int(H)


def find_fold_indices(rebal_dates, train_start, train_end, val_start, val_end, test_year):
    """Map year ranges to (start, end) panel indices."""
    train_start = pd.Timestamp(train_start); train_end = pd.Timestamp(train_end)
    val_start   = pd.Timestamp(val_start);   val_end   = pd.Timestamp(val_end)
    test_start  = pd.Timestamp(f"{test_year}-01-01")
    test_end    = pd.Timestamp(f"{test_year}-12-31")

    def find(start, end):
        idxs = [i for i, d in enumerate(rebal_dates)
                if start <= pd.Timestamp(d) <= end]
        return (min(idxs), max(idxs) + 1) if idxs else (None, None)

    return find(train_start, train_end), find(val_start, val_end), find(test_start, test_end)


# =============================================================================
# Training / evaluation
# =============================================================================

def train_one_epoch(encoder, head, dataset, opt, batch_size, device,
                     max_grad_norm=1.0, z_cap: float | None = None):
    encoder.train(); head.train()
    n = len(dataset)
    if n == 0: return [], []
    indices = list(range(n))
    np.random.shuffle(indices)
    n_full = max(1, n // batch_size)
    indices = indices[:n_full * batch_size]
    epoch_returns, grad_norms = [], []
    for batch_idx in range(n_full):
        batch_returns = []
        for idx in range(batch_idx * batch_size, (batch_idx + 1) * batch_size):
            X_t, Y_t, _, _ = dataset.get_month_data(indices[idx])
            X_t = X_t.to(device); Y_t = Y_t.to(device)
            h = encoder(X_t)
            if h.dim() > 1:
                u = head(h).squeeze(-1)
            else:
                u = h  # linear case: encoder already outputs (N,)
            w = two_softmax_weights(u, gross_leverage=1.0, z_cap=z_cap)
            batch_returns.append(portfolio_returns_drift(w, Y_t))
        batch_rets = torch.cat(batch_returns)
        batch_rets = torch.clamp(batch_rets, -config.DAILY_RET_CLIP, config.DAILY_RET_CLIP)
        mu = batch_rets.mean()
        sd = torch.clamp(batch_rets.std(unbiased=False), min=1e-4, max=1.0)
        sharpe = torch.clamp(mu / sd * np.sqrt(config.TRADING_DAYS_PER_YEAR), -5.0, 5.0)
        loss = -sharpe
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(head.parameters()),
            max_norm=max_grad_norm)
        grad_norms.append(float(gn))
        if torch.isnan(torch.tensor(gn)) or float(gn) > 500.0:
            opt.zero_grad(); continue
        opt.step()
        epoch_returns.extend([float(r) for r in batch_rets.cpu()])
    return epoch_returns, grad_norms


@torch.no_grad()
def evaluate_one(encoder, head, dataset, device, z_cap: float | None = None):
    encoder.eval(); head.eval()
    results = []
    for t in range(len(dataset)):
        X_t, Y_t, rebal_date, H = dataset.get_month_data(t)
        X_t = X_t.to(device); Y_t = Y_t.to(device)
        h = encoder(X_t)
        u = head(h).squeeze(-1) if h.dim() > 1 else h
        w = two_softmax_weights(u, z_cap=z_cap)
        port_rets = portfolio_returns_drift(w, Y_t)
        for di, ret in enumerate(port_rets.cpu().numpy()):
            results.append({"date": rebal_date + pd.Timedelta(days=di + 1),
                            "return": float(ret), "rebal_date": rebal_date})
    return results


def train_one_year(cfg: dict, panels: panels_mod.TCNNPanels, year: int, seed: int,
                    device: str = "cpu") -> tuple[pd.DataFrame, dict]:
    """Train TCNN on training window, evaluate on test year. Returns (returns_df, history)."""
    seed_everything(seed)
    rolling = cfg["rolling"]
    train_years_count = rolling["train_years"]
    val_years_count = rolling["val_years"]

    train_start = f"{year - train_years_count - val_years_count}-01-01"
    train_end   = f"{year - val_years_count - 1}-12-31"
    val_start   = f"{year - val_years_count}-01-01"
    val_end     = f"{year - 1}-12-31"

    (train_si, train_ei), (val_si, val_ei), (test_si, test_ei) = find_fold_indices(
        panels.rebal_dates, train_start, train_end, val_start, val_end, year
    )
    if train_si is None or test_si is None:
        return pd.DataFrame(columns=["date", "return"]), {"error": "no_fold"}

    train_ds = FoldDataset(panels, train_si, train_ei)
    val_ds   = FoldDataset(panels, val_si, val_ei)   if val_si is not None else None
    test_ds  = FoldDataset(panels, test_si, test_ei)
    if len(train_ds) == 0 or len(test_ds) == 0:
        return pd.DataFrame(columns=["date", "return"]), {"error": "empty_fold"}

    fcfg = cfg["factor"]
    is_linear = (fcfg["type"] == "linear_tcnn")
    in_ch = fcfg.get("in_ch", 1)
    encoder = TCNEncoder(
        in_ch=in_ch,
        hidden=fcfg.get("hidden", 48),
        d=fcfg.get("d", 96),
        kernel_size=fcfg.get("kernel_size", 5),
        dilations=tuple(fcfg.get("dilations", [1, 2, 4, 8, 16])),
        dropout=fcfg.get("dropout", 0.15),
        num_heads=fcfg.get("num_heads", 4),
        lookback=fcfg.get("lookback", config.LOOKBACK_DAYS),
        linear=is_linear,
    ).to(device)
    head = nn.Identity().to(device) if is_linear else nn.Linear(fcfg.get("d", 96), 1, bias=False).to(device)

    tcfg = cfg.get("training", {})
    # Defensive float coercion — PyYAML 1.1 parses `8e-4` as a string (needs `8.0e-4`).
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=float(tcfg.get("lr", 8e-4)),
        weight_decay=float(tcfg.get("weight_decay", 0.15)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5, min_lr=1e-6)

    max_epochs = tcfg.get("max_epochs", 35)
    patience = tcfg.get("patience", 10)
    warmup = tcfg.get("warmup_epochs", 7)
    ma_window = tcfg.get("ma_window", 7)
    batch_size = tcfg.get("batch_size", 15)
    # Optional winsor z-cap inside the in-training portfolio mapping (None = raw dual-softmax)
    z_cap = tcfg.get("portfolio_z_cap", None)

    best_val_ma = -1e9; best_epoch = 0; no_imp = 0
    best_es = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
    best_hs = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    history = {"train_sharpe": [], "val_sharpe": [], "val_ma": []}

    for epoch in range(max_epochs):
        ep_rets, _ = train_one_epoch(encoder, head, train_ds, opt, batch_size, device, z_cap=z_cap)
        train_sr = (np.mean(ep_rets) / (np.std(ep_rets) + 1e-8) * np.sqrt(252)) if ep_rets else 0.0
        history["train_sharpe"].append(float(train_sr))

        if val_ds is not None and len(val_ds) > 0:
            val_results = evaluate_one(encoder, head, val_ds, device, z_cap=z_cap)
            val_rets = np.array([r["return"] for r in val_results])
            val_sr = (val_rets.mean() / (val_rets.std() + 1e-8) * np.sqrt(252)) if len(val_rets) else -999.0
        else:
            val_sr = train_sr  # no val set: use train as proxy
        history["val_sharpe"].append(float(val_sr))

        ma = float(np.mean(history["val_sharpe"][-ma_window:])) if len(history["val_sharpe"]) >= ma_window else val_sr
        history["val_ma"].append(ma)
        scheduler.step(ma)

        if epoch >= warmup:
            if ma > best_val_ma:
                best_val_ma = ma; best_epoch = epoch; no_imp = 0
                best_es = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
                best_hs = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

    if best_epoch >= warmup:
        encoder.load_state_dict({k: v.to(device) for k, v in best_es.items()})
        head.load_state_dict({k: v.to(device) for k, v in best_hs.items()})

    test_results = evaluate_one(encoder, head, test_ds, device, z_cap=z_cap)
    test_df = pd.DataFrame(test_results)
    history["best_epoch"] = int(best_epoch)
    history["best_val_ma"] = float(best_val_ma)
    history["portfolio_z_cap"] = z_cap
    return test_df, history


# =============================================================================
# Public API: called by runner.run_cell for TCNN factor types
# =============================================================================

def train_and_evaluate(cfg: dict, panel: pd.DataFrame, year: int, seed: int) -> pd.DataFrame:
    """Runner entry point. Builds (or loads cached) panels then trains for (year, seed)."""
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    tcnn_panels = get_or_build_tcnn_panels(cfg, panel)
    test_df, history = train_one_year(cfg, tcnn_panels, year, seed, device=device)

    output_dir = Path(cfg["output_dir"]) / f"year_{year}" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_history.json", "w") as f:
        import json
        json.dump(history, f, indent=2, default=str)
    return test_df
