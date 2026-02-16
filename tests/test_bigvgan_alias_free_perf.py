"""
Parity + performance test for the batched depthwise conv in BigVGAN
alias-free activation (LowPassFilter1d and UpSample1d).

MLX has no groups= parameter on conv1d/conv_transpose1d, so both ops fold the
channel dimension into the batch dimension to dispatch a single kernel instead
of one per channel.  These tests verify parity against a naive per-channel
reference loop and measure the speedup.

Run parity tests only (default):
    pytest tests/test_bigvgan_alias_free_perf.py -v

Include timing benchmarks:
    pytest tests/test_bigvgan_alias_free_perf.py -v -s --bench
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from indextts_mlx.models.bigvgan_alias_free import LowPassFilter1d, UpSample1d

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_input(B: int, T: int, C: int, seed: int = 7) -> mx.array:
    np.random.seed(seed)
    return mx.array(np.random.randn(B, T, C).astype(np.float32))


def _lowpass(stride: int = 1, kernel_size: int = 12) -> LowPassFilter1d:
    m = LowPassFilter1d(cutoff=0.4, half_width=0.6, stride=stride, kernel_size=kernel_size)
    # Give it a real filter so outputs are non-trivial
    np.random.seed(42)
    m.filter = mx.array(np.random.randn(1, 1, kernel_size).astype(np.float32))
    return m


def _upsample(ratio: int = 2) -> UpSample1d:
    m = UpSample1d(ratio=ratio)
    np.random.seed(42)
    m.filter = mx.array(np.random.randn(1, 1, m.kernel_size).astype(np.float32))
    return m


# ── reference (per-channel loop) implementations ─────────────────────────────


def _lowpass_loop(x: mx.array, m: LowPassFilter1d) -> mx.array:
    """Naive per-channel reference (matches the old implementation)."""
    B, T, C = x.shape
    xt = x.transpose(0, 2, 1)
    if m.padding:
        xt = mx.pad(xt, [(0, 0), (0, 0), (m.pad_left, m.pad_right)], mode="edge")
    xt = xt.transpose(0, 2, 1)
    filt_t = m.filter.transpose(0, 2, 1)
    return mx.concatenate(
        [mx.conv1d(xt[:, :, c : c + 1], filt_t, stride=m.stride) for c in range(C)],
        axis=2,
    )


def _upsample_loop(x: mx.array, m: UpSample1d) -> mx.array:
    """Naive per-channel reference (matches the old implementation)."""
    B, T, C = x.shape
    xt = x.transpose(0, 2, 1)
    xt = mx.pad(xt, [(0, 0), (0, 0), (m.pad, m.pad)], mode="edge")
    xt = xt.transpose(0, 2, 1)
    filt_t = m.filter.transpose(0, 2, 1)
    result = mx.concatenate(
        [mx.conv_transpose1d(xt[:, :, c : c + 1], filt_t, stride=m.stride) for c in range(C)],
        axis=2,
    )
    if m.pad_right > 0:
        result = result[:, m.pad_left : -m.pad_right, :]
    else:
        result = result[:, m.pad_left :, :]
    return m.ratio * result


# ── parity tests ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("C", [1, 4, 64, 512])
@pytest.mark.parametrize("stride", [1, 2])
def test_lowpass_parity(C, stride):
    """Batched lowpass must produce bit-identical output to the loop reference."""
    m = _lowpass(stride=stride)
    x = _make_input(B=1, T=200, C=C)

    ref = _lowpass_loop(x, m)
    got = m(x)
    mx.eval(ref, got)

    assert ref.shape == got.shape
    assert float(mx.max(mx.abs(ref - got)).item()) < 1e-5


@pytest.mark.parametrize("C", [1, 4, 64, 512])
@pytest.mark.parametrize("ratio", [2, 4])
def test_upsample_parity(C, ratio):
    """Batched upsample must produce bit-identical output to the loop reference."""
    m = _upsample(ratio=ratio)
    x = _make_input(B=1, T=100, C=C)

    ref = _upsample_loop(x, m)
    got = m(x)
    mx.eval(ref, got)

    assert ref.shape == got.shape
    assert float(mx.max(mx.abs(ref - got)).item()) < 1e-5


def test_lowpass_batch_gt1_parity():
    """Ensure batched version handles B > 1 correctly (channels fold into batch)."""
    m = _lowpass(stride=1)
    x = _make_input(B=3, T=150, C=128)

    ref = _lowpass_loop(x, m)
    got = m(x)
    mx.eval(ref, got)

    assert float(mx.max(mx.abs(ref - got)).item()) < 1e-5


# ── benchmarks (opt-in with --bench) ─────────────────────────────────────────


def _bench(fn, *args, warmup: int = 3, reps: int = 10, **kwargs) -> float:
    """Return median wall-clock seconds over `reps` evaluated runs."""
    for _ in range(warmup):
        mx.eval(fn(*args, **kwargs))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn(*args, **kwargs))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


@pytest.mark.parametrize(
    "C,T,label",
    [
        (64, 1000, "C=64  T=1000"),
        (256, 2000, "C=256 T=2000"),
        (512, 4000, "C=512 T=4000"),
    ],
)
def test_lowpass_benchmark(run_bench, C, T, label):
    """Timing: loop vs batched lowpass."""
    if not run_bench:
        pytest.skip("pass --bench to enable")

    m = _lowpass(stride=2)
    x = _make_input(B=1, T=T, C=C)

    t_loop = _bench(_lowpass_loop, x, m)
    t_fast = _bench(m, x)
    speedup = t_loop / t_fast if t_fast > 0 else float("inf")
    print(
        f"\n  lowpass [{label}]  loop={t_loop*1000:.1f}ms  batched={t_fast*1000:.1f}ms  speedup={speedup:.2f}x"
    )
    assert t_fast <= t_loop * 1.1


@pytest.mark.parametrize(
    "C,T,label",
    [
        (64, 500, "C=64  T=500"),
        (256, 1000, "C=256 T=1000"),
        (512, 2000, "C=512 T=2000"),
    ],
)
def test_upsample_benchmark(run_bench, C, T, label):
    """Timing: loop vs batched upsample."""
    if not run_bench:
        pytest.skip("pass --bench to enable")

    m = _upsample(ratio=2)
    x = _make_input(B=1, T=T, C=C)

    t_loop = _bench(_upsample_loop, x, m)
    t_fast = _bench(m, x)
    speedup = t_loop / t_fast if t_fast > 0 else float("inf")
    print(
        f"\n  upsample [{label}]  loop={t_loop*1000:.1f}ms  batched={t_fast*1000:.1f}ms  speedup={speedup:.2f}x"
    )
    assert t_fast <= t_loop * 1.1
