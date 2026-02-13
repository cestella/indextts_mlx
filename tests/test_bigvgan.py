"""
BigVGAN vocoder shape tests.
"""

import pytest
import numpy as np
import mlx.core as mx


def test_bigvgan_shape(bigvgan_model):
    # Fake mel: (1, 80, T)
    T = 50
    mel = mx.zeros((1, 80, T))
    audio = bigvgan_model(mel)
    mx.eval(audio)
    arr = np.array(audio)
    # Expected: (1, 1, T * hop_length) where hop_length=256
    assert arr.ndim == 3
    assert arr.shape[0] == 1
    assert arr.shape[-1] > T  # upsampled
    assert np.isfinite(arr).all()


def test_bigvgan_range(bigvgan_model):
    """Audio samples should be in roughly [-1, 1]."""
    mel = mx.zeros((1, 80, 100))
    audio = bigvgan_model(mel)
    mx.eval(audio)
    arr = np.array(audio)
    assert arr.min() > -10.0
    assert arr.max() < 10.0
