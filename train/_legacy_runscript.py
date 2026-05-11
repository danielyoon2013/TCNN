"""
Rolling Yearly Retraining with Early Stopping + Precomputed Tensors
VERSION 3 with:
- Precomputed tensors (X_panel, Y_panel) for fast training
- 3-channel input (ret, vol_ewma, ret_norm) - all point-in-time at close
- Last business day of month rebalancing
- Variable holding period (until next month-end business day)
- Stability improvements (dropout, gradient monitoring, warmup)

Execution assumption: observe close, execute at close auction, hold until next month-end.

Workflow:
1. Run data_download.py (once)
2. Run prep_tensor.py (once)
3. Run this script for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def seed_everything(seed=123):
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(0)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class TCNEncoder(nn.Module):
    """Temporal Convolutional Network encoder with dropout.

    3-channel input: ret, vol_ewma, ret_norm (all point-in-time at close).
    """
    def __init__(self, in_ch=3, hidden=32, d=16, k=5, dilations=(1, 2, 4, 8), dropout=0.3):
        super().__init__()
        if k % 2 == 0:
            raise ValueError("Use an odd kernel size so padding keeps length stable.")

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
        # Attention pooling instead of concat
        self.attention = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        
        self.proj = nn.Sequential(
            nn.Linear(hidden, d),  # now just hidden → d
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)  # (batch, hidden, time)
        
        # Attention pooling
        z_t = z.transpose(1, 2)  # (batch, time, hidden)
        attn_scores = self.attention(z_t)  # (batch, time, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, time, 1)
        
        # Weighted sum
        feats = (z_t * attn_weights).sum(dim=1)  # (batch, hidden)
        
        h = self.proj(feats)
        return h


def two_softmax_weights_no_tau(u, gross_leverage=1.0, eps=1e-6):
    u = u - u.mean()
    u = u / (u.std(unbiased=False) + eps)

    long = F.softmax(u, dim=0)
    short = F.softmax(-u, dim=0)

    w = long - short
    w = w - w.mean()
    w = w * (gross_leverage / (w.abs().sum() + 1e-12))
    return w


def compute_portfolio_returns_daily(w_initial, daily_rets):
    """Compute portfolio returns with weight drift."""
    H, N = daily_rets.shape
    port_daily_rets = []

    w = w_initial

    for t in range(H):
        r_t = daily_rets[t]
        port_ret_t = (w * r_t).sum()
        port_daily_rets.append(port_ret_t)
        w = w * (1 + r_t) / (1 + port_ret_t + 1e-8)

    return torch.stack(port_daily_rets)


# ============================================================
# PRECOMPUTED DATA LOADING
# ============================================================

class PrecomputedDataset:
    """Dataset that loads from precomputed tensors."""

    def __init__(self, X_panel, Y_panel, holding_days, valid_mask, rebal_dates,
                 start_idx, end_idx, top_n_permnos=None, permno_list=None, min_stocks=50):
        """
        Args:
            X_panel: (num_months, num_stocks, 3, lookback) - full panel
            Y_panel: (num_months, num_stocks, max_holding) - full panel
            holding_days: (num_months,) - actual holding days
            valid_mask: (num_months, num_stocks) - valid stock mask
            rebal_dates: list of rebalancing dates
            start_idx: start index for this dataset
            end_idx: end index for this dataset (exclusive)
            top_n_permnos: set of permnos to filter to (optional)
            permno_list: ordered list of all permnos
            min_stocks: minimum valid stocks required
        """
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.min_stocks = min_stocks

        # Slice the panels for this period
        self.X = X_panel[start_idx:end_idx]
        self.Y = Y_panel[start_idx:end_idx]
        self.holding_days = holding_days[start_idx:end_idx]
        self.valid_mask = valid_mask[start_idx:end_idx].copy()
        self.rebal_dates = rebal_dates[start_idx:end_idx]

        # Apply stock filter if provided
        if top_n_permnos is not None and permno_list is not None:
            stock_filter = np.array([p in top_n_permnos for p in permno_list])
            # Combine with existing valid mask
            self.valid_mask = self.valid_mask & stock_filter[np.newaxis, :]

        # Precompute valid indices
        self._valid_month_indices = []
        for t in range(len(self.X)):
            if self.valid_mask[t].sum() >= min_stocks:
                self._valid_month_indices.append(t)

    def __len__(self):
        return len(self._valid_month_indices)

    def get_month_data(self, idx):
        """Get data for a specific valid month.

        Returns:
            X_t: (num_valid_stocks, 3, lookback) tensor
            Y_t: (holding_days, num_valid_stocks) tensor
            rebal_date: rebalancing date
            holding_days: number of holding days
        """
        t = self._valid_month_indices[idx]

        # Get valid stocks for this month
        valid_stocks = self.valid_mask[t]

        # Extract data for valid stocks only (copy from mmap to regular array)
        X_raw = np.array(self.X[t, valid_stocks])
        Y_raw = np.array(self.Y[t, valid_stocks, :self.holding_days[t]])

        # Debug: Check for NaN in data
        if np.isnan(X_raw).any():
            nan_count = np.isnan(X_raw).sum()
            print(f"  WARNING: NaN in X_raw at month {t}, count={nan_count}, shape={X_raw.shape}")
        if np.isnan(Y_raw).any():
            nan_count = np.isnan(Y_raw).sum()
            print(f"  WARNING: NaN in Y_raw at month {t}, count={nan_count}, shape={Y_raw.shape}")

        X_t = torch.tensor(X_raw, dtype=torch.float32)
        Y_t = torch.tensor(Y_raw, dtype=torch.float32).T

        return X_t, Y_t, self.rebal_dates[t], int(self.holding_days[t])


def load_precomputed_data(tensor_dir):
    """Load precomputed tensors and metadata."""
    print(f"Loading precomputed tensors from {tensor_dir}...")

    # Load panels (as memory-mapped for efficiency)
    X_panel = np.load(os.path.join(tensor_dir, "X_panel.npy"), mmap_mode='r')
    Y_panel = np.load(os.path.join(tensor_dir, "Y_panel.npy"), mmap_mode='r')
    holding_days = np.load(os.path.join(tensor_dir, "holding_days.npy"))
    valid_mask = np.load(os.path.join(tensor_dir, "valid_mask.npy"))

    # Load metadata
    with open(os.path.join(tensor_dir, "metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)

    with open(os.path.join(tensor_dir, "date_to_idx.pkl"), 'rb') as f:
        date_to_idx = pickle.load(f)

    print(f"  X_panel shape: {X_panel.shape}")
    print(f"  Y_panel shape: {Y_panel.shape}")
    print(f"  Months: {len(holding_days)}")
    print(f"  Stocks: {len(metadata['permno_list'])}")

    return X_panel, Y_panel, holding_days, valid_mask, metadata, date_to_idx


def get_period_indices(date_to_idx, rebal_dates, start_date, end_date):
    """Get start and end indices for a date range."""
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Find indices within range
    valid_indices = []
    for date, idx in date_to_idx.items():
        if start_date <= date <= end_date:
            valid_indices.append(idx)

    if not valid_indices:
        return None, None

    return min(valid_indices), max(valid_indices) + 1


# ============================================================
# MARKET CAP FILTERING
# ============================================================

def get_top_n_stocks_by_mktcap_from_parquet(parquet_path, reference_year, top_n=2000):
    """Get top N stocks by average market cap in the reference year."""
    df = pd.read_parquet(parquet_path, columns=['permno', 'date', 'mktcap_dollars'])
    df['date'] = pd.to_datetime(df['date'])

    ref_start = pd.Timestamp(f"{reference_year}-01-01")
    ref_end = pd.Timestamp(f"{reference_year}-12-31")

    df_ref = df[(df['date'] >= ref_start) & (df['date'] <= ref_end)]

    if len(df_ref) == 0:
        print(f"  Warning: No data in reference year {reference_year}")
        return set()

    avg_mktcap = df_ref.groupby('permno')['mktcap_dollars'].mean()
    top_stocks = avg_mktcap.nlargest(top_n).index.tolist()

    print(f"  Selected top {len(top_stocks)} stocks by avg market cap in {reference_year}")

    return set(top_stocks)


# ============================================================
# TRAINING
# ============================================================


def train_one_epoch(encoder, head, dataset, opt, batch_size, device, clip_returns=True, max_grad_norm=1.0):
    """Train for one epoch with stability improvements."""
    encoder.train()
    head.train()

    num_months = len(dataset)
    if num_months == 0:
        return [], []

    indices = list(range(num_months))
    np.random.shuffle(indices)

    # Keep only full batches (drop incomplete last batch)
    num_full_batches = num_months // batch_size
    indices = indices[:num_full_batches * batch_size]
    
    n_batches = num_full_batches

    epoch_returns = []
    grad_norms = []

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = batch_start + batch_size  # Always exactly batch_size

        batch_returns = []

        for idx in range(batch_start, batch_end):
            t = indices[idx]
            X_t, Y_t, _, _ = dataset.get_month_data(t)

            X_t = X_t.to(device, non_blocking=True)
            Y_t = Y_t.to(device, non_blocking=True)

            h = encoder(X_t)
            u = head(h).squeeze(-1)
            w = two_softmax_weights_no_tau(u, gross_leverage=1.0)

            port_rets = compute_portfolio_returns_daily(w, Y_t)
            batch_returns.append(port_rets)

        batch_rets = torch.cat(batch_returns)

        if clip_returns:
            batch_rets = torch.clamp(batch_rets, -0.1, 0.1)

        mu = batch_rets.mean()
        sd = batch_rets.std(unbiased=False)
        
        # STABILITY FIX 1: Clamp standard deviation to prevent division by tiny numbers
        sd = torch.clamp(sd, min=1e-4, max=1.0)
        
        batch_sharpe = mu / sd * torch.sqrt(torch.tensor(252.0, device=device))
        
        # STABILITY FIX 2: Clamp Sharpe ratio to prevent extreme loss values
        batch_sharpe = torch.clamp(batch_sharpe, -5.0, 5.0)

        loss = -batch_sharpe

        opt.zero_grad()
        loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(head.parameters()),
            max_norm=max_grad_norm
        )
        grad_norms.append(total_norm.item())

        # Only skip truly pathological batches (NaN or extremely large)
        if torch.isnan(torch.tensor(total_norm)) or total_norm > 500.0:
            print(f"  Skipping pathological batch {batch_idx}, gradient norm: {total_norm:.2f}")
            opt.zero_grad()
            continue
        
        # Log warning for large (but not pathological) gradients
        if total_norm > 50.0:
            print(f"  Large gradient (batch {batch_idx}): {total_norm:.2f} → clipped to {max_grad_norm}")

        opt.step()

        epoch_returns.extend([r.item() for r in batch_rets.cpu()])

        del batch_returns, batch_rets, loss, batch_sharpe
        if device == 'cuda':
            torch.cuda.empty_cache()

    return epoch_returns, grad_norms


def evaluate_model(encoder, head, dataset, device):
    """Evaluate model on dataset."""
    encoder.eval()
    head.eval()

    results = []

    with torch.no_grad():
        for t in range(len(dataset)):
            X_t, Y_t, rebal_date, holding_days = dataset.get_month_data(t)

            X_t = X_t.to(device, non_blocking=True)
            Y_t = Y_t.to(device, non_blocking=True)

            h = encoder(X_t)
            u = head(h).squeeze(-1)
            w = two_softmax_weights_no_tau(u)

            port_rets = compute_portfolio_returns_daily(w, Y_t)

            for day_idx, ret in enumerate(port_rets.cpu().numpy()):
                results.append({
                    'date': rebal_date + pd.Timedelta(days=day_idx+1),
                    'return': ret,
                    'rebalance_date': rebal_date,
                    'holding_days': holding_days,
                })

    return results


def train_model_for_year(
    X_panel, Y_panel, holding_days, valid_mask, metadata, date_to_idx,
    train_start, train_end, val_start, val_end,
    model_config, cache_dir, year_label, top_n_permnos,
    max_epochs=60, batch_size=60, device='cuda',
    patience=15, ma_window=5, min_stocks=50, warmup_epochs=5,
):
    """Train model for a specific year using precomputed tensors."""
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL FOR YEAR {year_label}")
    print(f"{'='*80}")
    print(f"Training period: {train_start} to {train_end}")
    print(f"Validation period: {val_start} to {val_end}")
    print(f"Universe: {len(top_n_permnos)} stocks")
    print(f"{'='*80}\n")

    rebal_dates = metadata['rebal_dates']
    permno_list = metadata['permno_list']

    # Get indices for train and val periods
    train_start_idx, train_end_idx = get_period_indices(date_to_idx, rebal_dates, train_start, train_end)
    val_start_idx, val_end_idx = get_period_indices(date_to_idx, rebal_dates, val_start, val_end)

    if train_start_idx is None or val_start_idx is None:
        print("  ERROR: Could not find valid indices for train/val periods")
        return None, None, None

    print(f"Train indices: {train_start_idx} to {train_end_idx} ({train_end_idx - train_start_idx} months)")
    print(f"Val indices: {val_start_idx} to {val_end_idx} ({val_end_idx - val_start_idx} months)")

    # Create datasets
    train_ds = PrecomputedDataset(
        X_panel, Y_panel, holding_days, valid_mask, rebal_dates,
        train_start_idx, train_end_idx, top_n_permnos, permno_list, min_stocks
    )
    val_ds = PrecomputedDataset(
        X_panel, Y_panel, holding_days, valid_mask, rebal_dates,
        val_start_idx, val_end_idx, top_n_permnos, permno_list, min_stocks
    )

    print(f"Train dataset: {len(train_ds)} valid months")
    print(f"Val dataset: {len(val_ds)} valid months")

    if len(train_ds) == 0:
        print("  ERROR: No valid training months!")
        return None, None, None

    # Initialize model
    print("\nInitializing model...")
    encoder = TCNEncoder(
        in_ch=model_config.get('in_ch', 3),
        hidden=model_config['hidden'],
        d=model_config['d'],
        k=model_config.get('kernel_size', 5),
        dilations=model_config['dilations'],
        dropout=model_config.get('dropout', 0.3)
    ).to(device)
    head = nn.Linear(model_config['d'], 1, bias=False).to(device)

    n_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"  Model parameters: {n_params:,}")

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=model_config['lr'],
        weight_decay=model_config['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    # Early stopping
    best_val_sharpe_ma = -999
    best_epoch = 0
    epochs_without_improvement = 0
    best_encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
    best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    train_sharpes = []
    val_sharpes = []
    val_sharpes_ma = []
    avg_grad_norms = []

    print(f"\nTraining...")
    print(f"{'Epoch':<8} {'Train SR':<12} {'Val SR':<12} {'Val MA':<12} {'Grad Norm':<12} {'LR':<12} {'Status':<20}")
    print("=" * 100)

    for epoch in range(max_epochs):
        epoch_returns, grad_norms = train_one_epoch(
            encoder, head, train_ds, opt, batch_size, device,
            clip_returns=True, max_grad_norm=1.0
        )

        avg_grad_norm = np.mean(grad_norms) if len(grad_norms) > 0 else 0.0
        avg_grad_norms.append(avg_grad_norm)

        if len(epoch_returns) > 0:
            epoch_sharpe = np.mean(epoch_returns) / (np.std(epoch_returns) + 1e-8) * np.sqrt(252)
            train_sharpes.append(epoch_sharpe)
        else:
            train_sharpes.append(0.0)

        val_results = evaluate_model(encoder, head, val_ds, device)
        if len(val_results) > 0:
            val_returns = np.array([r['return'] for r in val_results])
            val_sharpe = val_returns.mean() / (val_returns.std() + 1e-8) * np.sqrt(252)
            val_sharpes.append(val_sharpe)
        else:
            val_sharpes.append(-999)

        if len(val_sharpes) >= ma_window:
            val_sharpe_ma = np.mean(val_sharpes[-ma_window:])
        else:
            val_sharpe_ma = val_sharpes[-1]
        val_sharpes_ma.append(val_sharpe_ma)

        scheduler.step(val_sharpe_ma)
        current_lr = opt.param_groups[0]['lr']

        status = ""
        if epoch >= warmup_epochs:
            if val_sharpe_ma > best_val_sharpe_ma:
                best_val_sharpe_ma = val_sharpe_ma
                best_epoch = epoch
                epochs_without_improvement = 0
                best_encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
                best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                status = "New best!"
            else:
                epochs_without_improvement += 1
                status = f"No improve: {epochs_without_improvement}/{patience}"
        else:
            status = f"Warmup {epoch+1}/{warmup_epochs}"

        print(f"{epoch+1:<8} {train_sharpes[-1]:<12.3f} {val_sharpes[-1]:<12.3f} {val_sharpe_ma:<12.3f} {avg_grad_norm:<12.3f} {current_lr:<12.2e} {status:<20}")

        if epoch >= warmup_epochs and epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    # Load best model
    if best_epoch >= warmup_epochs:
        print(f"\nLoading best model from epoch {best_epoch + 1}...")
        encoder.load_state_dict({k: v.to(device) for k, v in best_encoder_state.items()})
        head.load_state_dict({k: v.to(device) for k, v in best_head_state.items()})
    else:
        print(f"\nWarning: Using final model (no improvement after warmup)")
        best_epoch = len(train_sharpes) - 1
        best_val_sharpe_ma = val_sharpes_ma[-1] if val_sharpes_ma else -999

    print(f"Training complete for year {year_label}")
    print(f"  Best epoch: {best_epoch + 1}")
    print(f"  Best val Sharpe (MA): {best_val_sharpe_ma:.3f}")

    training_history = {
        'train_sharpes': train_sharpes,
        'val_sharpes': val_sharpes,
        'val_sharpes_ma': val_sharpes_ma,
        'grad_norms': avg_grad_norms,
        'best_epoch': best_epoch,
        'best_val_sharpe_ma': best_val_sharpe_ma,
    }

    return encoder, head, training_history


def test_model_on_year(
    X_panel, Y_panel, holding_days, valid_mask, metadata, date_to_idx,
    encoder, head, test_year, top_n_permnos, device='cuda', min_stocks=50,
):
    """Test trained model on a specific year."""
    print(f"\n{'='*60}")
    print(f"Testing on Year {test_year}")
    print(f"{'='*60}")

    year_start = f"{test_year}-01-01"
    year_end = f"{test_year}-12-31"

    rebal_dates = metadata['rebal_dates']
    permno_list = metadata['permno_list']

    start_idx, end_idx = get_period_indices(date_to_idx, rebal_dates, year_start, year_end)

    if start_idx is None:
        print(f"  No data for {test_year}")
        return []

    print(f"Test indices: {start_idx} to {end_idx} ({end_idx - start_idx} months)")

    test_ds = PrecomputedDataset(
        X_panel, Y_panel, holding_days, valid_mask, rebal_dates,
        start_idx, end_idx, top_n_permnos, permno_list, min_stocks
    )

    print(f"Test dataset: {len(test_ds)} valid months")

    year_results = evaluate_model(encoder, head, test_ds, device)

    if len(year_results) > 0:
        year_returns = np.array([r['return'] for r in year_results])
        total_ret = (1 + year_returns).prod() - 1
        sharpe = year_returns.mean() / year_returns.std() * np.sqrt(252)
        max_dd = compute_max_drawdown(year_returns)

        print(f"\n  {test_year} Performance:")
        print(f"    Total Return: {total_ret*100:.2f}%")
        print(f"    Sharpe Ratio: {sharpe:.3f}")
        print(f"    Max Drawdown: {max_dd*100:.2f}%")
        print(f"    Trading days: {len(year_returns)}")

    return year_results


def compute_max_drawdown(returns):
    """Compute maximum drawdown from returns series."""
    cum_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_rolling_yearly_experiment(
    tensor_dir: str,
    parquet_path: str,
    model_config: dict,
    start_year: int = 2003,
    end_year: int = 2023,
    train_years: int = 10,
    val_years: int = 2,
    top_n_stocks: int = 2000,
    cache_dir: str = "./rolling_yearly_cache_v3",
    device: str = "cuda",
    batch_size: int = 60,
    max_epochs: int = 60,
    patience: int = 15,
    ma_window: int = 5,
    min_stocks: int = 50,
    warmup_epochs: int = 5,
):
    """Rolling yearly retraining with precomputed tensors."""
    print(f"\n{'='*80}")
    print(f"ROLLING YEARLY RETRAINING EXPERIMENT V3")
    print(f"{'='*80}")
    print(f"Test years: {start_year} to {end_year}")
    print(f"Train period: {train_years} years")
    print(f"Val period: {val_years} years")
    print(f"Universe: Top {top_n_stocks} stocks by market cap")
    print(f"Using precomputed tensors from: {tensor_dir}")
    print(f"{'='*80}\n")

    # Load precomputed data
    X_panel, Y_panel, holding_days, valid_mask, metadata, date_to_idx = \
        load_precomputed_data(tensor_dir)

    os.makedirs(cache_dir, exist_ok=True)

    all_yearly_results = []
    all_training_histories = {}

    for test_year in range(start_year, end_year + 1):
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING YEAR {test_year}")
        print(f"{'#'*80}")

        # Define periods
        train_start_year = test_year - train_years - val_years
        train_end_year = test_year - val_years - 1
        val_start_year = test_year - val_years
        val_end_year = test_year - 1

        train_start = f"{train_start_year}-01-01"
        train_end = f"{train_end_year}-12-31"
        val_start = f"{val_start_year}-01-01"
        val_end = f"{val_end_year}-12-31"

        # Get top N stocks for this test year
        reference_year = val_end_year
        print(f"\nSelecting stock universe from {reference_year}...")
        top_n_permnos = get_top_n_stocks_by_mktcap_from_parquet(
            parquet_path, reference_year, top_n_stocks
        )

        if len(top_n_permnos) == 0:
            print(f"Skipping year {test_year}")
            continue

        print(f"\nPeriods:")
        print(f"  Train: {train_start} to {train_end}")
        print(f"  Val:   {val_start} to {val_end}")
        print(f"  Test:  {test_year}")

        # Train model
        encoder, head, training_history = train_model_for_year(
            X_panel, Y_panel, holding_days, valid_mask, metadata, date_to_idx,
            train_start, train_end, val_start, val_end,
            model_config, cache_dir, test_year, top_n_permnos,
            max_epochs, batch_size, device, patience, ma_window, min_stocks, warmup_epochs,
        )

        if encoder is None:
            print(f"\nSkipping year {test_year}")
            continue

        # Save model
        model_save_path = f"{cache_dir}/model_{test_year}.pt"
        torch.save({
            'encoder': encoder.state_dict(),
            'head': head.state_dict(),
            'model_config': model_config,
            'training_history': training_history,
            'test_year': test_year,
        }, model_save_path)

        # Test
        year_results = test_model_on_year(
            X_panel, Y_panel, holding_days, valid_mask, metadata, date_to_idx,
            encoder, head, test_year, top_n_permnos, device, min_stocks,
        )

        for r in year_results:
            r['test_year'] = test_year

        all_yearly_results.extend(year_results)
        all_training_histories[test_year] = training_history

        # Save individual year
        year_df = pd.DataFrame(year_results)
        year_df.to_csv(f"{cache_dir}/returns_{test_year}.csv", index=False)

    # Combine results
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}\n")

    all_results_df = pd.DataFrame(all_yearly_results)
    if len(all_results_df) > 0:
        all_results_df = all_results_df.sort_values('date')
        all_results_df.to_csv(f"{cache_dir}/all_daily_returns.csv", index=False)

        all_returns = all_results_df['return'].values
        total_return = (1 + all_returns).prod() - 1
        sharpe = all_returns.mean() / all_returns.std() * np.sqrt(252)
        max_dd = compute_max_drawdown(all_returns)

        yearly_metrics = []
        for year in range(start_year, end_year + 1):
            year_data = all_results_df[all_results_df['test_year'] == year]
            if len(year_data) > 0:
                year_rets = year_data['return'].values
                yearly_metrics.append({
                    'year': year,
                    'total_return': (1 + year_rets).prod() - 1,
                    'sharpe': year_rets.mean() / year_rets.std() * np.sqrt(252),
                    'max_drawdown': compute_max_drawdown(year_rets),
                    'n_days': len(year_rets),
                })

        metrics_df = pd.DataFrame(yearly_metrics)
        metrics_df.to_csv(f"{cache_dir}/yearly_metrics.csv", index=False)

        print(f"Overall Performance ({start_year}-{end_year}):")
        print(f"  Total Return: {total_return*100:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.3f}")
        print(f"  Max Drawdown: {max_dd*100:.2f}%")
        print(f"\nYearly Breakdown:")
        print(metrics_df.to_string(index=False))

    # Save training histories
    with open(f"{cache_dir}/training_histories.json", 'w') as f:
        histories_serializable = {}
        for year, hist in all_training_histories.items():
            histories_serializable[year] = {
                'train_sharpes': [float(x) for x in hist['train_sharpes']],
                'val_sharpes': [float(x) for x in hist['val_sharpes']],
                'val_sharpes_ma': [float(x) for x in hist['val_sharpes_ma']],
                'grad_norms': [float(x) for x in hist['grad_norms']],
                'best_epoch': int(hist['best_epoch']),
                'best_val_sharpe_ma': float(hist['best_val_sharpe_ma']),
            }
        json.dump(histories_serializable, f, indent=2)

    return all_results_df, metrics_df if 'metrics_df' in dir() else None, all_training_histories


# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    #script_dir = os.path.dirname(os.path.abspath(__file__))

    PROJECT_DIR  = "/content/drive/MyDrive/iclr26"
    parquet_path = os.path.join(PROJECT_DIR, "daily_data_sf.parquet")
    tensor_dir   = os.path.join(PROJECT_DIR, "precomputed_tensors")
    cache_dir = os.path.join(PROJECT_DIR, "rolling_yearly_cache_v3_1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    model_config = {
        'in_ch': 3,
        'hidden': 48,
        'd': 96,
        'kernel_size': 5,
        'dilations': (1, 2, 4, 8, 16),
        'lr': 0.0008,
        'weight_decay': 0.5,
        'dropout': 0.3,
    }

    start_year = 2010
    end_year = 2023
    train_years = 15
    val_years = 3
    top_n_stocks = 2000

    max_epochs = 50
    patience = 10
    ma_window = 5
    batch_size = 15
    min_stocks = 1200
    warmup_epochs = 7

    print("\nExperiment Configuration V3:")
    print(f"  OOS years: {start_year} to {end_year}")
    print(f"  Universe: Top {top_n_stocks} stocks")
    print(f"  Min stocks: {min_stocks}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Patience: {patience}")
    print("\nModel Configuration:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    print("\nKey Features in V3:")
    print("  - Precomputed tensors for fast training")
    print("  - 3-channel input (ret, vol_ewma, ret_norm)")
    print("  - Last business day of month rebalancing")
    print("  - Variable holding period (until next month-end)")
    print("  - Point-in-time: all features known at close, execute at close")

    results_df, metrics_df, training_histories = run_rolling_yearly_experiment(
        tensor_dir=tensor_dir,
        parquet_path=parquet_path,
        model_config=model_config,
        start_year=start_year,
        end_year=end_year,
        train_years=train_years,
        val_years=val_years,
        top_n_stocks=top_n_stocks,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        ma_window=ma_window,
        min_stocks=min_stocks,
        warmup_epochs=warmup_epochs,
    )

    if results_df is not None and len(results_df) > 0:
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*80}")