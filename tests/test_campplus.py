"""
CAMPPlus speaker embedding smoke tests.
"""

import pytest
import numpy as np
import mlx.core as mx

from indextts_mlx.audio.kaldi_fbank import compute_kaldi_fbank_mlx


def test_campplus_shape(campplus_model, reference_audio_np):
    _, audio_22k = reference_audio_np
    feat = compute_kaldi_fbank_mlx(mx.array(audio_22k), num_mel_bins=80, sample_frequency=22050)
    feat = feat - feat.mean(axis=0, keepdims=True)
    emb = campplus_model(feat[None])
    mx.eval(emb)
    arr = np.array(emb)
    # Expected: (1, 192)
    assert arr.shape == (1, 192), f"Unexpected shape: {arr.shape}"


def test_campplus_range(campplus_model, reference_audio_np):
    _, audio_22k = reference_audio_np
    feat = compute_kaldi_fbank_mlx(mx.array(audio_22k), num_mel_bins=80, sample_frequency=22050)
    feat = feat - feat.mean(axis=0, keepdims=True)
    emb = campplus_model(feat[None])
    mx.eval(emb)
    arr = np.array(emb)
    assert np.isfinite(arr).all(), "Embedding contains NaN/Inf"
