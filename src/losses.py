"""Differentiable listwise ranking losses for cross-sectional portfolio training.

Currently exports:
    approx_ndcg_loss_batched(scores, fwd_rets, stock_mask, k_top, k_bot, alpha=10.0)

ApproxNDCG (Qin et al. 2010) replaces the non-differentiable rank operation with
a sigmoid-based soft rank, making the whole NDCG computation differentiable.
This module's variant computes symmetric long+short NDCG: rank the universe by
score, score the top-K (long sleeve) AND the bottom-K (short sleeve), then
average. Each rebal date (month) in the batch is computed independently — there
is no cross-period merging — and per-month NDCG values are averaged for the
final batch loss.

Why this beats the Sharpe-loss baseline (per Poh-Lim-Zohren-Roberts 2020, Wang
2021, and recent 2024-25 LTR papers): gradient signal is denser (all pairs of
stocks contribute, not just the realized portfolio P&L) and naturally focused
on the top/bottom-K positions, which are the only ones that drive long-short
portfolio returns. Standard LTR convention is per-list aggregation, which here
means per-rebal-date — see `train_tcnn.train_one_epoch` for how this is plumbed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _percentile_rank_returns(fwd_rets: torch.Tensor, stock_mask: torch.Tensor,
                              eps: float = 1e-12) -> torch.Tensor:
    """Per-month percentile rank of forward returns, in [0, 1] for valid stocks; 0 for invalid.

    Used as the relevance gain in NDCG: higher-return stocks get higher gain.

    Args:
        fwd_rets:   (K, N_max) forward returns per stock
        stock_mask: (K, N_max) bool

    Returns:
        (K, N_max) float in [0, 1] for valid stocks (top-return → ~1, bottom → ~0); zero for invalid.
    """
    mask_f = stock_mask.float()
    n_valid = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)             # (K, 1)
    # For invalid entries, push returns to -inf so they sort to the bottom and don't
    # contribute to the rank of valid entries.
    NEG = torch.finfo(fwd_rets.dtype).min / 2
    r_masked = fwd_rets.masked_fill(~stock_mask, NEG)
    # Hard rank position via argsort-argsort (descending by return): rank 0 = highest.
    # Non-differentiable, but this is OUTSIDE the gradient path (gain values are constants).
    with torch.no_grad():
        order = torch.argsort(r_masked, dim=1, descending=True)
        rank_pos = torch.empty_like(order)
        rank_pos.scatter_(1, order, torch.arange(r_masked.size(1), device=r_masked.device).expand_as(order))
    # Convert to percentile in (0, 1]: top stock → ~1, bottom valid → ~1/n_valid.
    pct = 1.0 - rank_pos.float() / n_valid
    pct = pct.clamp(0.0, 1.0) * mask_f                                    # zero out invalid
    return pct


def _soft_rank(scores: torch.Tensor, stock_mask: torch.Tensor, alpha: float,
                eps: float = 1e-6) -> torch.Tensor:
    """Differentiable soft rank: rank_i ≈ 1 + Σ_{j ≠ i} σ(α (s_j − s_i)).

    Scale-invariant via internal z-scoring (within each month, over valid stocks).
    Without this, the sigmoid would be very smeared for small-magnitude scores
    (e.g. raw returns of order 1e-2), pushing nearly all stocks into the middle
    of the rank distribution. Z-scoring makes α=10 mean a consistent sharpness
    regardless of the scale of the upstream TCNN scores.

    Args:
        scores:     (K, N_max) raw model scores
        stock_mask: (K, N_max) bool
        alpha:      temperature for sigmoid sharpness on z-scored values

    Returns:
        (K, N_max) soft rank in [1, n_valid]; invalid entries have undefined values.
    """
    mask_f = stock_mask.float()
    n_valid = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
    # Z-score within month, valid stocks only
    s_sum = (scores * mask_f).sum(dim=1, keepdim=True)
    s_mean = s_sum / n_valid
    s_centered = (scores - s_mean) * mask_f
    s_var = (s_centered ** 2).sum(dim=1, keepdim=True) / n_valid
    s_std = s_var.sqrt() + eps
    s_norm = (scores - s_mean) / s_std

    NEG = torch.finfo(scores.dtype).min / 2
    s_masked = s_norm.masked_fill(~stock_mask, NEG)
    # Pairwise diffs: diff[k, i, j] = s_j - s_i
    diff = s_masked.unsqueeze(1) - s_masked.unsqueeze(2)                  # (K, N, N)
    sig = torch.sigmoid(alpha * diff)                                      # (K, N, N)
    # Subtract self-term (sig at i==j is sigmoid(0) = 0.5) so we sum only over j != i.
    eye = torch.eye(s_masked.size(1), device=s_masked.device, dtype=sig.dtype).unsqueeze(0)
    sig = sig - 0.5 * eye
    soft_rank = 1.0 + sig.sum(dim=2)                                      # (K, N) — sum over j
    return soft_rank


def _ndcg_one_direction(scores: torch.Tensor, fwd_rets: torch.Tensor,
                         stock_mask: torch.Tensor, k: int, alpha: float,
                         eps: float = 1e-12) -> torch.Tensor:
    """Compute ApproxNDCG@K for one direction (assume "high score = best").

    The gain is the percentile rank of the realized return (high returns →
    high gain). The discount is 1 / log2(soft_rank + 1). The numerator (DCG)
    uses the model's soft rank; the denominator (IdealDCG) uses the true rank
    ordered by realized return — both with the same gain values.

    A smoothed top-K indicator (sigmoid on a margin around K) restricts the
    gain to the top-K positions for a peaked NDCG@K rather than full-list NDCG.

    Args:
        scores:     (K, N_max)
        fwd_rets:   (K, N_max)
        stock_mask: (K, N_max) bool
        k:          top-K positions to focus on (e.g. 50)
        alpha:      soft-rank temperature
        eps:        numerical stabilizer

    Returns:
        (K,) per-month NDCG@K in [0, 1].
    """
    mask_f = stock_mask.float()
    gain = _percentile_rank_returns(fwd_rets, stock_mask)                 # (K, N_max)

    # Soft rank per stock (differentiable through scores).
    s_rank = _soft_rank(scores, stock_mask, alpha)                        # (K, N_max)

    # Smoothed top-K indicator: ~1 if soft_rank ≤ k, ~0 otherwise. Sigmoid centered at k+0.5.
    topk_indicator = torch.sigmoid(alpha * (k + 0.5 - s_rank))             # (K, N_max)
    # Hard zero outside the valid universe.
    topk_indicator = topk_indicator * mask_f

    # Standard NDCG gain-times-discount, masked.
    discount = 1.0 / torch.log2(s_rank + 1.0 + eps)                       # (K, N_max)
    dcg = (gain * topk_indicator * discount).sum(dim=1)                   # (K,)

    # IdealDCG: ranks assigned by true return order (top → rank 1). Top-K cutoff applies too.
    with torch.no_grad():
        order = torch.argsort(fwd_rets.masked_fill(~stock_mask, torch.finfo(fwd_rets.dtype).min / 2),
                              dim=1, descending=True)
        ideal_rank = torch.empty_like(order, dtype=scores.dtype)
        ideal_rank.scatter_(1, order, 1.0 + torch.arange(scores.size(1), device=scores.device,
                                                          dtype=scores.dtype).expand_as(order))
    ideal_topk = (ideal_rank <= k).float() * mask_f
    ideal_discount = 1.0 / torch.log2(ideal_rank + 1.0 + eps)
    ideal_dcg = (gain * ideal_topk * ideal_discount).sum(dim=1)           # (K,)

    return dcg / (ideal_dcg + eps)


def approx_ndcg_loss_batched(scores: torch.Tensor, fwd_rets: torch.Tensor,
                              stock_mask: torch.Tensor,
                              k_top: int = 50, k_bot: int = 50,
                              alpha: float = 10.0) -> torch.Tensor:
    """Symmetric ApproxNDCG@K listwise ranking loss for cross-sectional portfolios.

    Loss = − mean_k [ (NDCG_long_k + NDCG_short_k) / 2 ]
    where NDCG_long is computed by ranking high-score stocks against high-return
    relevance, and NDCG_short by ranking low-score (negated) against low-return
    (negated). Each month k in the batch is computed independently; the asset
    dimension never crosses month boundaries.

    Args:
        scores:     (K, N_max) raw model scores (TCNN output)
        fwd_rets:   (K, N_max) per-stock 21-day cumulative forward return (sum of Y window)
        stock_mask: (K, N_max) bool, True for valid stocks per month
        k_top:      top-K for long sleeve (default 50)
        k_bot:      bottom-K for short sleeve (default 50)
        alpha:      temperature for sigmoid soft-rank (default 10.0; lower=smoother, higher=sharper)

    Returns:
        scalar loss in [-1, 0] (negative NDCG; lower is better).
    """
    ndcg_long  = _ndcg_one_direction(scores,  fwd_rets,  stock_mask, k_top, alpha)
    ndcg_short = _ndcg_one_direction(-scores, -fwd_rets, stock_mask, k_bot, alpha)
    per_month = (ndcg_long + ndcg_short) / 2.0                            # (K,)
    return -per_month.mean()
