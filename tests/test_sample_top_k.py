"""
Tests for _sample_top_k in pipeline.py.

Covers:
  - Greedy path (temperature=0 / top_k=1) returns argmax
  - Sampled token always falls within the top-k set
  - Empirical distribution matches softmax probabilities (chi-squared)
  - Low-temperature output concentrates on the highest-probability token
  - Determinism under a fixed mx.random seed
  - Single-token vocab (edge case)
  - top_k >= vocab_size behaves the same as unrestricted sampling
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from indextts_mlx.pipeline import _sample_top_k

# ── helpers ───────────────────────────────────────────────────────────────────


def _logits(values: list[float]) -> mx.array:
    return mx.array(values, dtype=mx.float32)


def _top_k_indices(logits: mx.array, k: int) -> set[int]:
    """Return the indices of the k largest logits."""
    vals = np.array(logits)
    return set(np.argsort(vals)[-k:].tolist())


# ── greedy path ───────────────────────────────────────────────────────────────


def test_greedy_temperature_zero():
    """temperature=0 must return the argmax."""
    logits = _logits([1.0, 5.0, 2.0, 3.0])
    assert _sample_top_k(logits, temperature=0.0, top_k=50) == 1


def test_greedy_top_k_one():
    """top_k=1 must return the argmax regardless of temperature."""
    logits = _logits([0.5, 9.0, 1.0])
    assert _sample_top_k(logits, temperature=1.0, top_k=1) == 1


# ── top-k constraint ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("top_k", [1, 2, 5, 10])
def test_sample_within_top_k(top_k):
    """Every sampled token must be in the top-k set."""
    np.random.seed(0)
    mx.random.seed(0)
    logits = _logits(list(range(20)))  # token 19 is highest
    allowed = _top_k_indices(logits, top_k)
    for _ in range(200):
        tok = _sample_top_k(logits, temperature=1.0, top_k=top_k)
        assert tok in allowed, f"token {tok} not in top-{top_k} set {allowed}"


def test_top_k_larger_than_vocab_unconstrained():
    """top_k >= vocab_size should not restrict sampling at all."""
    mx.random.seed(42)
    logits = _logits([1.0, 2.0, 3.0])
    # With top_k=100 >> vocab=3 the masking branch is skipped entirely
    counts = [0, 0, 0]
    for _ in range(3000):
        counts[_sample_top_k(logits, temperature=1.0, top_k=100)] += 1
    # All tokens reachable
    assert all(c > 0 for c in counts), f"some token never sampled: {counts}"


# ── distribution shape ────────────────────────────────────────────────────────


def test_empirical_distribution_matches_softmax():
    """
    Over many samples the empirical frequency should match the softmax
    probabilities within a loose chi-squared tolerance.
    """
    mx.random.seed(1234)
    logits = _logits([2.0, 1.0, 0.5, 0.1])
    expected_probs = np.exp(np.array([2.0, 1.0, 0.5, 0.1]))
    expected_probs /= expected_probs.sum()

    N = 10_000
    counts = np.zeros(4)
    for _ in range(N):
        counts[_sample_top_k(logits, temperature=1.0, top_k=4)] += 1

    observed = counts / N
    # Allow ±3 percentage points per bucket at N=10k — very conservative
    for i, (obs, exp) in enumerate(zip(observed, expected_probs)):
        assert (
            abs(obs - exp) < 0.03
        ), f"token {i}: observed={obs:.4f} expected={exp:.4f} — distribution mismatch"


def test_low_temperature_concentrates_on_best():
    """Very low temperature should make the best token dominate strongly."""
    mx.random.seed(99)
    logits = _logits([5.0, 1.0, 1.0, 1.0])
    counts = [0, 0, 0, 0]
    for _ in range(500):
        counts[_sample_top_k(logits, temperature=0.1, top_k=4)] += 1
    # Token 0 should win the vast majority of the time
    assert counts[0] > 480, f"expected token 0 dominant, got {counts}"


# ── determinism ───────────────────────────────────────────────────────────────


def test_deterministic_with_fixed_seed():
    """Same seed should produce the same sequence of tokens."""
    logits = _logits([1.0, 2.0, 1.5, 0.5, 3.0])

    mx.random.seed(7)
    seq_a = [_sample_top_k(logits, temperature=1.0, top_k=3) for _ in range(20)]

    mx.random.seed(7)
    seq_b = [_sample_top_k(logits, temperature=1.0, top_k=3) for _ in range(20)]

    assert seq_a == seq_b, f"sequences differ under same seed:\n{seq_a}\n{seq_b}"


# ── edge cases ────────────────────────────────────────────────────────────────


def test_single_token_vocab():
    """Vocab size of 1 should always return token 0."""
    logits = _logits([2.5])
    for _ in range(20):
        assert _sample_top_k(logits, temperature=1.0, top_k=1) == 0


def test_uniform_logits_all_tokens_reachable():
    """Uniform logits with large top_k should sample all tokens eventually."""
    mx.random.seed(0)
    vocab = 10
    logits = _logits([0.0] * vocab)
    seen = set()
    for _ in range(2000):
        seen.add(_sample_top_k(logits, temperature=1.0, top_k=vocab))
    assert seen == set(range(vocab)), f"not all tokens seen: {seen}"
