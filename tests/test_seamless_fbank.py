"""
Test SeamlessM4T filterbank feature extraction.
Parity test against transformers is skipped if transformers not installed.
"""

import pytest
import numpy as np

from indextts_mlx.audio.seamless_fbank import compute_seamless_fbank


def make_sine_audio(duration_s=1.0, sr=16000, freq=440.0):
    t = np.linspace(0, duration_s, int(duration_s * sr), dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_seamless_fbank_shape():
    audio = make_sine_audio(duration_s=1.0, sr=16000)
    feats = compute_seamless_fbank(audio)
    # Expected: (T, 80)
    assert feats.ndim == 2
    assert feats.shape[1] == 160
    assert feats.shape[0] > 0


def test_seamless_fbank_range():
    audio = make_sine_audio(duration_s=1.0, sr=16000)
    feats = compute_seamless_fbank(audio)
    assert np.isfinite(feats).all(), "Features contain NaN/Inf"


@pytest.mark.parity
def test_seamless_fbank_parity_transformers():
    """Parity test against HF SeamlessM4TFeatureExtractor."""
    from transformers import SeamlessM4TFeatureExtractor

    audio = make_sine_audio(duration_s=1.0, sr=16000)
    mlx_feats = compute_seamless_fbank(audio)

    fe = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    hf_out = fe(audio, sampling_rate=16000, return_tensors="np")
    hf_feats = hf_out["input_features"][0]  # (T, 80)

    min_len = min(mlx_feats.shape[0], hf_feats.shape[0])
    diff = np.abs(mlx_feats[:min_len] - hf_feats[:min_len]).max()
    assert diff < 1.0, f"Max absolute difference vs transformers: {diff:.4f}"
