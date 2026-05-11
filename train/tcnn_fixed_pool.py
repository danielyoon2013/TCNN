"""
TCNN portfolio paper — fixed-pooling variant (paper Table 1 row "TCNN Fixed Pooling").

Same training pipeline and CLI as tcnn_attention.py, but the encoder uses
concat(last, mean, max) over the temporal dimension instead of attention.

Reproduces paper Sharpe ~0.53 (OOS 2010-2023, top 2000 US equities, gross of TC).

PATCHES applied vs. authoring repo: see tcnn_attention.py docstring.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Determinism
# =============================================================================

def seed_everything(seed: int = 0, deterministic: bool = True):
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
# Model
# =============================================================================

class TCNEncoderFixedPool(nn.Module):
    """Dilated 1D conv encoder with FIXED pooling.

    Pooling = concat(last_state, time_mean, time_max). This is the paper
    "Fixed Pooling" baseline (Sharpe ~0.53). All else equal to the
    attention variant.
    """

    def __init__(self, in_ch: int = 3, hidden: int = 48, d: int = 96,
                 k: int = 5, dilations=(1, 2, 4, 8, 16), dropout: float = 0.15):
        super().__init__()
        if k % 2 == 0:
            raise ValueError("Use odd kernel size so padding keeps length stable.")

        layers = []
        ch = in_ch
        for dil in dilations:
            pad = (k - 1) * dil // 2
            layers += [
                nn.Conv1d(ch, hidden, kernel_size=k, dilation=dil, padding=pad),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.GroupNorm(num_groups=min(4, hidden), num_channels=hidden),
            ]
            ch = hidden
        self.net = nn.Sequential(*layers)

        # Fixed pool: concat(last, mean, max) → 3*hidden → linear projection to d
        self.proj = nn.Sequential(
            nn.Linear(hidden * 3, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)               # (N, hidden, T)
        last = z[..., -1]             # (N, hidden)
        mean = z.mean(dim=-1)         # (N, hidden)
        mx = z.max(dim=-1).values     # (N, hidden)
        feats = torch.cat([last, mean, mx], dim=-1)  # (N, 3*hidden)
        h = self.proj(feats)          # (N, d)
        return h


def two_softmax_weights(u: torch.Tensor, gross_leverage: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    u = u - u.mean()
    u = u / (u.std(unbiased=False) + eps)
    long = F.softmax(u, dim=0)
    short = F.softmax(-u, dim=0)
    w = long - short
    w = w - w.mean()
    w = w * (gross_leverage / (w.abs().sum() + 1e-12))
    return w


def compute_portfolio_returns_daily(w_initial: torch.Tensor, daily_rets: torch.Tensor) -> torch.Tensor:
    H = daily_rets.shape[0]
    port_daily_rets = []
    w = w_initial
    for t in range(H):
        r_t = daily_rets[t]
        port_ret_t = (w * r_t).sum()
        port_daily_rets.append(port_ret_t)
        w = w * (1 + r_t) / (1 + port_ret_t + 1e-8)
    return torch.stack(port_daily_rets)


# =============================================================================
# Dataset / loaders / training (re-use logic from tcnn_attention.py)
# =============================================================================

# To keep this file self-contained without circular imports we inline the same
# helper classes/functions. They are 1:1 with tcnn_attention.py — if you patch
# one, patch both. (Future cleanup: extract into shared module.)

class PrecomputedDataset:
    def __init__(self, X_panel, Y_panel, holding_days, valid_mask, rebal_dates,
                 start_idx, end_idx, top_n_permnos=None, permno_list=None, min_stocks=50):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.min_stocks = min_stocks
        self.X = X_panel[start_idx:end_idx]
        self.Y = Y_panel[start_idx:end_idx]
        self.holding_days = holding_days[start_idx:end_idx]
        self.valid_mask = valid_mask[start_idx:end_idx].copy()
        self.rebal_dates = rebal_dates[start_idx:end_idx]
        if top_n_permnos is not None and permno_list is not None:
            stock_filter = np.array([p in top_n_permnos for p in permno_list])
            self.valid_mask = self.valid_mask & stock_filter[np.newaxis, :]
        self._valid_month_indices = [t for t in range(len(self.X))
                                       if self.valid_mask[t].sum() >= min_stocks]

    def __len__(self):
        return len(self._valid_month_indices)

    def get_month_data(self, idx):
        t = self._valid_month_indices[idx]
        valid_stocks = self.valid_mask[t]
        X_raw = np.array(self.X[t, valid_stocks])
        Y_raw = np.array(self.Y[t, valid_stocks, :self.holding_days[t]])
        X_t = torch.tensor(X_raw, dtype=torch.float32)
        Y_t = torch.tensor(Y_raw, dtype=torch.float32).T
        return X_t, Y_t, self.rebal_dates[t], int(self.holding_days[t])


def load_precomputed_data(tensor_dir: str, num_channels: int):
    print(f"Loading precomputed tensors from {tensor_dir}...")
    X_panel = np.load(os.path.join(tensor_dir, "X_panel.npy"), mmap_mode="r")
    Y_panel = np.load(os.path.join(tensor_dir, "Y_panel.npy"), mmap_mode="r")
    holding_days = np.load(os.path.join(tensor_dir, "holding_days.npy"))
    valid_mask = np.load(os.path.join(tensor_dir, "valid_mask.npy"))
    with open(os.path.join(tensor_dir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    with open(os.path.join(tensor_dir, "date_to_idx.pkl"), "rb") as f:
        date_to_idx = pickle.load(f)
    if X_panel.shape[2] < num_channels:
        raise ValueError(f"Tensor has {X_panel.shape[2]} channels but --in-ch={num_channels} requested.")
    return X_panel, Y_panel, holding_days, valid_mask, metadata, date_to_idx


def get_period_indices(date_to_idx, rebal_dates, start_date, end_date):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    valid_indices = [idx for date, idx in date_to_idx.items() if start_date <= date <= end_date]
    if not valid_indices:
        return None, None
    return min(valid_indices), max(valid_indices) + 1


def get_top_n_stocks_by_mktcap_from_parquet(parquet_path, reference_year, top_n=2000):
    df = pd.read_parquet(parquet_path, columns=["permno", "date", "mktcap_dollars"])
    df["date"] = pd.to_datetime(df["date"])
    ref_start = pd.Timestamp(f"{reference_year}-01-01")
    ref_end = pd.Timestamp(f"{reference_year}-12-31")
    df_ref = df[(df["date"] >= ref_start) & (df["date"] <= ref_end)]
    if len(df_ref) == 0:
        return set()
    avg_mktcap = df_ref.groupby("permno")["mktcap_dollars"].mean()
    return set(avg_mktcap.nlargest(top_n).index.tolist())


def train_one_epoch(encoder, head, dataset, opt, batch_size, device, in_ch,
                     clip_returns=True, max_grad_norm=1.0):
    encoder.train(); head.train()
    num_months = len(dataset)
    if num_months == 0:
        return [], []
    indices = list(range(num_months))
    np.random.shuffle(indices)
    n_full = num_months // batch_size
    indices = indices[:n_full * batch_size]
    epoch_returns, grad_norms = [], []
    for batch_idx in range(n_full):
        batch_returns = []
        for idx in range(batch_idx * batch_size, (batch_idx + 1) * batch_size):
            t = indices[idx]
            X_t, Y_t, _, _ = dataset.get_month_data(t)
            X_t = X_t[:, :in_ch, :].to(device, non_blocking=True)
            Y_t = Y_t.to(device, non_blocking=True)
            h = encoder(X_t)
            u = head(h).squeeze(-1)
            w = two_softmax_weights(u, gross_leverage=1.0)
            port_rets = compute_portfolio_returns_daily(w, Y_t)
            batch_returns.append(port_rets)
        batch_rets = torch.cat(batch_returns)
        if clip_returns:
            batch_rets = torch.clamp(batch_rets, -0.1, 0.1)
        mu = batch_rets.mean()
        sd = torch.clamp(batch_rets.std(unbiased=False), min=1e-4, max=1.0)
        batch_sharpe = torch.clamp(mu / sd * torch.sqrt(torch.tensor(252.0, device=device)),
                                     -5.0, 5.0)
        loss = -batch_sharpe
        opt.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(head.parameters()), max_norm=max_grad_norm)
        grad_norms.append(total_norm.item())
        if torch.isnan(torch.tensor(total_norm)) or total_norm > 500.0:
            opt.zero_grad()
            continue
        opt.step()
        epoch_returns.extend([r.item() for r in batch_rets.cpu()])
        del batch_returns, batch_rets, loss, batch_sharpe
        if device == "cuda":
            torch.cuda.empty_cache()
    return epoch_returns, grad_norms


def evaluate_model(encoder, head, dataset, device, in_ch):
    encoder.eval(); head.eval()
    results = []
    with torch.no_grad():
        for t in range(len(dataset)):
            X_t, Y_t, rebal_date, holding_days = dataset.get_month_data(t)
            X_t = X_t[:, :in_ch, :].to(device, non_blocking=True)
            Y_t = Y_t.to(device, non_blocking=True)
            h = encoder(X_t)
            u = head(h).squeeze(-1)
            w = two_softmax_weights(u)
            port_rets = compute_portfolio_returns_daily(w, Y_t)
            for day_idx, ret in enumerate(port_rets.cpu().numpy()):
                results.append({"date": rebal_date + pd.Timedelta(days=day_idx + 1),
                                "return": ret, "rebalance_date": rebal_date,
                                "holding_days": holding_days})
    return results


def compute_max_drawdown(returns):
    cum = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum)
    return float(((cum - running_max) / running_max).min())


def train_model_for_year(X_panel, Y_panel, holding_days, valid_mask, metadata, date_to_idx,
                          train_start, train_end, val_start, val_end,
                          model_config, year_label, top_n_permnos,
                          max_epochs, batch_size, device, patience, ma_window, min_stocks,
                          warmup_epochs, in_ch):
    print(f"\n{'=' * 80}\nTRAINING FOR YEAR {year_label}\n{'=' * 80}")
    rebal_dates = metadata["rebal_dates"]
    permno_list = metadata["permno_list"]
    train_si, train_ei = get_period_indices(date_to_idx, rebal_dates, train_start, train_end)
    val_si, val_ei = get_period_indices(date_to_idx, rebal_dates, val_start, val_end)
    if train_si is None or val_si is None:
        return None, None, None
    train_ds = PrecomputedDataset(X_panel, Y_panel, holding_days, valid_mask, rebal_dates,
                                    train_si, train_ei, top_n_permnos, permno_list, min_stocks)
    val_ds = PrecomputedDataset(X_panel, Y_panel, holding_days, valid_mask, rebal_dates,
                                  val_si, val_ei, top_n_permnos, permno_list, min_stocks)
    print(f"  train months: {len(train_ds)}    val months: {len(val_ds)}")
    if len(train_ds) == 0:
        return None, None, None

    encoder = TCNEncoderFixedPool(in_ch=in_ch, hidden=model_config["hidden"], d=model_config["d"],
                                    k=model_config["kernel_size"], dilations=model_config["dilations"],
                                    dropout=model_config["dropout"]).to(device)
    head = nn.Linear(model_config["d"], 1, bias=False).to(device)
    n_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"  model parameters: {n_params:,}")

    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()),
                            lr=model_config["lr"], weight_decay=model_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                             patience=5, min_lr=1e-6)

    best_val_sharpe_ma = -1e9
    best_epoch = 0
    epochs_without_improvement = 0
    best_encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
    best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    train_sharpes, val_sharpes, val_sharpes_ma, avg_grad_norms = [], [], [], []

    print(f"\n  {'Epoch':<7} {'Train SR':<10} {'Val SR':<10} {'Val MA':<10} {'GradN':<8} {'LR':<10} {'Status'}")
    for epoch in range(max_epochs):
        ep_returns, grad_norms = train_one_epoch(encoder, head, train_ds, opt, batch_size,
                                                   device, in_ch, max_grad_norm=1.0)
        avg_g = float(np.mean(grad_norms)) if grad_norms else 0.0
        avg_grad_norms.append(avg_g)
        train_sharpes.append(float(np.mean(ep_returns) / (np.std(ep_returns) + 1e-8) * np.sqrt(252)) if ep_returns else 0.0)

        val_results = evaluate_model(encoder, head, val_ds, device, in_ch)
        val_returns = np.array([r["return"] for r in val_results]) if val_results else np.array([])
        val_sharpe = float(val_returns.mean() / (val_returns.std() + 1e-8) * np.sqrt(252)) if len(val_returns) else -999.0
        val_sharpes.append(val_sharpe)
        val_ma = float(np.mean(val_sharpes[-ma_window:])) if len(val_sharpes) >= ma_window else val_sharpes[-1]
        val_sharpes_ma.append(val_ma)
        scheduler.step(val_ma)
        cur_lr = opt.param_groups[0]["lr"]

        if epoch >= warmup_epochs:
            if val_ma > best_val_sharpe_ma:
                best_val_sharpe_ma = val_ma
                best_epoch = epoch
                epochs_without_improvement = 0
                best_encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
                best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                status = "best!"
            else:
                epochs_without_improvement += 1
                status = f"no-imp {epochs_without_improvement}/{patience}"
        else:
            status = f"warmup {epoch + 1}/{warmup_epochs}"

        print(f"  {epoch + 1:<7} {train_sharpes[-1]:<10.3f} {val_sharpes[-1]:<10.3f} {val_ma:<10.3f} {avg_g:<8.3f} {cur_lr:<10.2e} {status}")
        if epoch >= warmup_epochs and epochs_without_improvement >= patience:
            print(f"\n  early stopping at epoch {epoch + 1}")
            break

    if best_epoch >= warmup_epochs:
        encoder.load_state_dict({k: v.to(device) for k, v in best_encoder_state.items()})
        head.load_state_dict({k: v.to(device) for k, v in best_head_state.items()})

    history = {"train_sharpes": train_sharpes, "val_sharpes": val_sharpes,
                "val_sharpes_ma": val_sharpes_ma, "grad_norms": avg_grad_norms,
                "best_epoch": best_epoch, "best_val_sharpe_ma": best_val_sharpe_ma}
    return encoder, head, history


def test_model_on_year(X_panel, Y_panel, holding_days, valid_mask, metadata, date_to_idx,
                        encoder, head, test_year, top_n_permnos, device, in_ch, min_stocks):
    rebal_dates = metadata["rebal_dates"]
    permno_list = metadata["permno_list"]
    si, ei = get_period_indices(date_to_idx, rebal_dates, f"{test_year}-01-01", f"{test_year}-12-31")
    if si is None:
        return []
    test_ds = PrecomputedDataset(X_panel, Y_panel, holding_days, valid_mask, rebal_dates,
                                   si, ei, top_n_permnos, permno_list, min_stocks)
    results = evaluate_model(encoder, head, test_ds, device, in_ch)
    if results:
        rets = np.array([r["return"] for r in results])
        sharpe = rets.mean() / (rets.std() + 1e-8) * np.sqrt(252)
        max_dd = compute_max_drawdown(rets)
        print(f"  {test_year}: Sharpe {sharpe:.3f}  MaxDD {max_dd * 100:+.2f}%  ({len(rets)} days)")
    return results


def run_rolling_experiment(tensor_dir, parquet_path, model_config, start_year, end_year,
                            train_years, val_years, top_n_stocks, cache_dir, device,
                            batch_size, max_epochs, patience, ma_window, min_stocks,
                            warmup_epochs, in_ch):
    X_panel, Y_panel, holding_days, valid_mask, metadata, date_to_idx = \
        load_precomputed_data(tensor_dir, num_channels=in_ch)
    os.makedirs(cache_dir, exist_ok=True)

    all_yearly_results = []
    all_training_histories = {}
    for test_year in range(start_year, end_year + 1):
        print(f"\n\n{'#' * 80}\n# YEAR {test_year}\n{'#' * 80}")
        train_start_year = test_year - train_years - val_years
        train_end_year = test_year - val_years - 1
        val_start_year = test_year - val_years
        val_end_year = test_year - 1
        top_n_permnos = get_top_n_stocks_by_mktcap_from_parquet(parquet_path, val_end_year, top_n_stocks)
        if not top_n_permnos:
            continue
        result = train_model_for_year(X_panel, Y_panel, holding_days, valid_mask,
                                        metadata, date_to_idx,
                                        f"{train_start_year}-01-01", f"{train_end_year}-12-31",
                                        f"{val_start_year}-01-01", f"{val_end_year}-12-31",
                                        model_config, test_year, top_n_permnos,
                                        max_epochs, batch_size, device, patience, ma_window,
                                        min_stocks, warmup_epochs, in_ch)
        if result[0] is None:
            continue
        encoder, head, history = result
        torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(),
                     "model_config": model_config, "training_history": history,
                     "test_year": test_year},
                    os.path.join(cache_dir, f"model_{test_year}.pt"))
        year_results = test_model_on_year(X_panel, Y_panel, holding_days, valid_mask,
                                            metadata, date_to_idx, encoder, head, test_year,
                                            top_n_permnos, device, in_ch, min_stocks)
        for r in year_results:
            r["test_year"] = test_year
        all_yearly_results.extend(year_results)
        all_training_histories[test_year] = history
        if year_results:
            pd.DataFrame(year_results).to_csv(os.path.join(cache_dir, f"returns_{test_year}.csv"), index=False)

    if all_yearly_results:
        df = pd.DataFrame(all_yearly_results).sort_values("date")
        df.to_csv(os.path.join(cache_dir, "all_daily_returns.csv"), index=False)
        rets = df["return"].values
        sharpe = rets.mean() / (rets.std() + 1e-8) * np.sqrt(252)
        max_dd = compute_max_drawdown(rets)
        print(f"\n{'=' * 80}\nTOTAL Sharpe={sharpe:.3f}  MaxDD={max_dd * 100:+.2f}%\n{'=' * 80}")

    with open(os.path.join(cache_dir, "training_histories.json"), "w") as f:
        json.dump({str(y): {k: ([float(x) for x in v] if isinstance(v, list) else v)
                              for k, v in h.items()}
                    for y, h in all_training_histories.items()}, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="TCNN fixed-pooling training (paper baseline variant).")
    p.add_argument("--tensor-dir", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--cache-dir", default="./rolling_yearly_cache_fixed_pool")
    p.add_argument("--in-ch", type=int, default=3)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--d", type=int, default=96)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--max-epochs", type=int, default=35)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--warmup-epochs", type=int, default=7)
    p.add_argument("--ma-window", type=int, default=7)
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=2023)
    p.add_argument("--train-years", type=int, default=8)
    p.add_argument("--val-years", type=int, default=2)
    p.add_argument("--top-n-stocks", type=int, default=2000)
    p.add_argument("--min-stocks", type=int, default=1200)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.start_year = max(args.start_year, 2022)
        args.end_year = min(args.end_year, 2022)
        args.max_epochs = 2
        args.patience = 1
        args.warmup_epochs = 0
        args.train_years = 4
        args.val_years = 1
        args.batch_size = min(args.batch_size, 8)
        args.min_stocks = 200
        args.device = "cpu"
        print(f"\n*** SMOKE TEST MODE ***  year {args.start_year}, max_epochs={args.max_epochs}, device=cpu\n")

    seed_everything(args.seed, deterministic=True)

    model_config = {"hidden": args.hidden, "d": args.d, "kernel_size": args.kernel_size,
                     "dilations": tuple(args.dilations), "lr": args.lr,
                     "weight_decay": args.weight_decay, "dropout": args.dropout}

    print(f"Device: {args.device}")
    print(f"Model config (fixed pool): {model_config}")
    print(f"Rolling: {args.start_year}-{args.end_year}, train={args.train_years}y, val={args.val_years}y")

    run_rolling_experiment(
        tensor_dir=args.tensor_dir, parquet_path=args.parquet, model_config=model_config,
        start_year=args.start_year, end_year=args.end_year,
        train_years=args.train_years, val_years=args.val_years,
        top_n_stocks=args.top_n_stocks, cache_dir=args.cache_dir, device=args.device,
        batch_size=args.batch_size, max_epochs=args.max_epochs,
        patience=args.patience, ma_window=args.ma_window,
        min_stocks=args.min_stocks, warmup_epochs=args.warmup_epochs, in_ch=args.in_ch,
    )


if __name__ == "__main__":
    main()
