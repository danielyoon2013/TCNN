"""Synthetic sanity checks for src/losses.py:approx_ndcg_loss_batched.

Run: python scripts/test_listwise_loss.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.losses import approx_ndcg_loss_batched, _soft_rank, _percentile_rank_returns


def assert_close(actual, expected, tol, msg):
    diff = abs(float(actual) - float(expected))
    assert diff < tol, f"{msg}: |{actual} - {expected}| = {diff} >= {tol}"
    print(f"  OK  {msg}: |diff|={diff:.4e} < {tol}")


def test_perfect_ranking_gives_ndcg_one():
    """If scores correlate perfectly with returns, NDCG should be ~1.0."""
    torch.manual_seed(0)
    K, N = 4, 200
    fwd_rets = torch.randn(K, N) * 0.05
    scores = fwd_rets.clone()
    mask = torch.ones(K, N, dtype=torch.bool)
    loss = approx_ndcg_loss_batched(scores, fwd_rets, mask, k_top=20, k_bot=20, alpha=20.0)
    # loss = -ndcg; perfect → ndcg ~ 1.0 → loss ~ -1.0
    assert_close(loss.item(), -1.0, tol=0.05, msg="perfect ranking loss ~ -1.0")


def test_reverse_ranking_gives_low_ndcg():
    """If scores anti-correlate with returns, NDCG should be near 0 (worst case)."""
    torch.manual_seed(0)
    K, N = 4, 200
    fwd_rets = torch.randn(K, N) * 0.05
    scores = -fwd_rets.clone()
    mask = torch.ones(K, N, dtype=torch.bool)
    loss = approx_ndcg_loss_batched(scores, fwd_rets, mask, k_top=20, k_bot=20, alpha=20.0)
    # Reverse: long sleeve gets bottom stocks (low gain), short gets top → very low ndcg
    # Symmetric form gives near 0, loss → 0 (close to zero, NOT -1)
    assert loss.item() > -0.20, f"reverse should give NDCG near 0, loss near 0 — got {loss.item()}"
    print(f"  OK  reverse ranking loss = {loss.item():.4f}  (>= -0.20)")


def test_random_scores_give_mid_ndcg():
    """Random uncorrelated scores should give NDCG between perfect-reverse and perfect-correct."""
    torch.manual_seed(0)
    K, N = 4, 200
    fwd_rets = torch.randn(K, N) * 0.05
    scores = torch.randn(K, N)
    mask = torch.ones(K, N, dtype=torch.bool)
    loss = approx_ndcg_loss_batched(scores, fwd_rets, mask, k_top=20, k_bot=20, alpha=20.0)
    # Random should land somewhere between -1 (perfect) and 0 (reverse). Typical range -0.2 to -0.5.
    assert -1.0 <= loss.item() <= 0.0
    print(f"  OK  random scores loss = {loss.item():.4f}  (in [-1, 0])")


def test_gradient_flows_through_scores():
    """Loss should be differentiable with non-zero gradient w.r.t. scores."""
    torch.manual_seed(0)
    K, N = 2, 100
    fwd_rets = torch.randn(K, N) * 0.05
    scores = torch.randn(K, N, requires_grad=True)
    mask = torch.ones(K, N, dtype=torch.bool)
    loss = approx_ndcg_loss_batched(scores, fwd_rets, mask, k_top=10, k_bot=10, alpha=10.0)
    loss.backward()
    grad_norm = scores.grad.abs().sum().item()
    assert grad_norm > 0, "gradient is zero — soft rank or top-K indicator may be broken"
    print(f"  OK  gradient flows: |grad|_1 = {grad_norm:.4f}")


def test_padding_does_not_change_result():
    """Adding padded invalid stocks should NOT change the NDCG for the valid universe."""
    torch.manual_seed(0)
    K, N_real = 3, 100
    fwd_rets_real = torch.randn(K, N_real) * 0.05
    scores_real = fwd_rets_real + torch.randn_like(fwd_rets_real) * 0.5   # noisy signal
    mask_real = torch.ones(K, N_real, dtype=torch.bool)
    loss_no_pad = approx_ndcg_loss_batched(scores_real, fwd_rets_real, mask_real,
                                            k_top=10, k_bot=10, alpha=10.0)

    # Same data but padded with 50 invalid entries
    N_padded = N_real + 50
    fwd_rets_padded = torch.zeros(K, N_padded)
    scores_padded = torch.zeros(K, N_padded)
    mask_padded = torch.zeros(K, N_padded, dtype=torch.bool)
    fwd_rets_padded[:, :N_real] = fwd_rets_real
    scores_padded[:, :N_real] = scores_real
    mask_padded[:, :N_real] = True
    # Pad values are intentionally weird (huge scores) to test mask robustness
    scores_padded[:, N_real:] = 1000.0
    fwd_rets_padded[:, N_real:] = -1000.0
    loss_padded = approx_ndcg_loss_batched(scores_padded, fwd_rets_padded, mask_padded,
                                            k_top=10, k_bot=10, alpha=10.0)
    assert_close(loss_padded.item(), loss_no_pad.item(), tol=1e-3,
                 msg="padding does not change NDCG")


def test_optimization_converges():
    """Gradient descent on scores toward fwd_rets should drive NDCG to 1."""
    torch.manual_seed(0)
    K, N = 2, 200
    fwd_rets = torch.randn(K, N) * 0.05
    # Initialize with small random noise (a real model's outputs are never identical).
    scores = (torch.randn(K, N) * 0.01).requires_grad_(True)
    mask = torch.ones(K, N, dtype=torch.bool)
    opt = torch.optim.Adam([scores], lr=0.05)
    initial = approx_ndcg_loss_batched(scores, fwd_rets, mask, k_top=20, k_bot=20, alpha=10.0).item()
    for step in range(200):
        opt.zero_grad()
        loss = approx_ndcg_loss_batched(scores, fwd_rets, mask, k_top=20, k_bot=20, alpha=10.0)
        loss.backward()
        opt.step()
    final = loss.item()
    # Random-init NDCG is ~0.4-0.5 (not 0) due to the top-K cutoff geometry;
    # we just need to see meaningful improvement from optimization.
    assert final < initial - 0.15, f"loss did not decrease meaningfully: {initial} -> {final}"
    assert final < -0.5, f"loss should be clearly negative after 200 steps: got {final}"
    print(f"  OK  optimization converges: {initial:.4f} -> {final:.4f}")


def main():
    print("=== Listwise loss synthetic sanity checks ===\n")
    tests = [
        ("perfect ranking", test_perfect_ranking_gives_ndcg_one),
        ("reverse ranking", test_reverse_ranking_gives_low_ndcg),
        ("random scores",   test_random_scores_give_mid_ndcg),
        ("gradient flows",  test_gradient_flows_through_scores),
        ("padding safe",    test_padding_does_not_change_result),
        ("optimization",    test_optimization_converges),
    ]
    for name, fn in tests:
        print(f"\n[{name}]")
        fn()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
