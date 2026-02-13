"""
W2V-BERT feature extraction smoke tests.
"""

import pytest
import numpy as np
import mlx.core as mx

from indextts_mlx.audio.seamless_fbank import compute_seamless_fbank


def test_w2vbert_shape(w2vbert_model, reference_audio_np):
    audio_16k, _ = reference_audio_np
    feats_np = compute_seamless_fbank(audio_16k)
    mlx_feats = mx.array(feats_np[None])
    T = feats_np.shape[0]
    mask = mx.ones((1, T), dtype=mx.int32)
    out = w2vbert_model(input_features=mlx_feats, attention_mask=mask, output_hidden_states=True)
    # hidden_states[17] used for semantic features
    h17 = out.hidden_states[17]
    mx.eval(h17)
    arr = np.array(h17)
    # Expected: (1, T', 1024)
    assert arr.ndim == 3
    assert arr.shape[0] == 1
    assert arr.shape[2] == 1024


def test_w2vbert_range(w2vbert_model, reference_audio_np):
    audio_16k, _ = reference_audio_np
    feats_np = compute_seamless_fbank(audio_16k)
    mlx_feats = mx.array(feats_np[None])
    T = feats_np.shape[0]
    mask = mx.ones((1, T), dtype=mx.int32)
    out = w2vbert_model(input_features=mlx_feats, attention_mask=mask, output_hidden_states=True)
    h17 = out.hidden_states[17]
    mx.eval(h17)
    arr = np.array(h17)
    assert np.isfinite(arr).all(), "W2V-BERT output contains NaN/Inf"
