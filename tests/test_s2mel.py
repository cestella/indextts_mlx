"""
S2Mel regulator and CFM diffusion shape tests.
"""

import pytest
import numpy as np
import mlx.core as mx


def test_s2mel_regulator_shape(s2mel_pipeline):
    # Fake semantic tokens: (1, T_in, 1024)
    T_in = 50
    tokens = mx.zeros((1, T_in, 1024))
    target_len = 200
    ylens = mx.array([target_len], dtype=mx.int32)
    mu, olens = s2mel_pipeline.regulator(tokens, ylens=ylens, f0=None)
    mx.eval(mu)
    arr = np.array(mu)
    # Expected: (1, target_len, 512)
    assert arr.shape == (1, target_len, 512), f"Unexpected shape: {arr.shape}"
    assert np.isfinite(arr).all()


def test_s2mel_cfm_shape(s2mel_pipeline):
    """CFM inference: fake mu + zero prompt → mel spectrogram shape."""
    T = 100
    batch = 1
    n_mels = 80
    style_dim = 192

    mu = mx.zeros((batch, T, 512))
    x_lens = mx.array([T], dtype=mx.int32)
    prompt = mx.zeros((batch, n_mels, 10))
    style = mx.zeros((batch, style_dim))

    mel = s2mel_pipeline.cfm.inference(
        mu=mu,
        x_lens=x_lens,
        prompt=prompt,
        style=style,
        f0=None,
        n_timesteps=2,   # minimal steps for speed
        temperature=1.0,
        inference_cfg_rate=0.0,
    )
    mx.eval(mel)
    arr = np.array(mel)
    # Expected: (batch, n_mels, T)
    assert arr.shape == (batch, n_mels, T), f"Unexpected mel shape: {arr.shape}"
    assert np.isfinite(arr).all()
