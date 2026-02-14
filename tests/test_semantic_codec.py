"""
SemanticCodec quantize → vq2emb roundtrip tests.
"""

import pytest
import numpy as np
import mlx.core as mx

from indextts_mlx.audio.seamless_fbank import compute_seamless_fbank


def test_semantic_codec_quantize_shape(
    semantic_codec_model, w2vbert_model, reference_audio_np, weights_dir
):
    audio_16k, _ = reference_audio_np
    feats_np = compute_seamless_fbank(audio_16k)
    mlx_feats = mx.array(feats_np[None])
    T = feats_np.shape[0]
    mask = mx.ones((1, T), dtype=mx.int32)
    out = w2vbert_model(input_features=mlx_feats, attention_mask=mask, output_hidden_states=True)

    stats = np.load(str(weights_dir / "semantic_stats.npz"))
    mean = mx.array(stats["mean"])
    std = mx.array(stats["std"])
    semantic_features = (out.hidden_states[17] - mean) / std

    codes, _ = semantic_codec_model.quantize(semantic_features)
    mx.eval(codes)
    arr = np.array(codes)
    # Expected: (1, T', codebook_size) or (1, T')
    assert arr.ndim >= 2
    assert arr.shape[0] == 1


def test_semantic_codec_vq2emb(
    semantic_codec_model, w2vbert_model, reference_audio_np, weights_dir
):
    audio_16k, _ = reference_audio_np
    feats_np = compute_seamless_fbank(audio_16k)
    mlx_feats = mx.array(feats_np[None])
    T = feats_np.shape[0]
    mask = mx.ones((1, T), dtype=mx.int32)
    out = w2vbert_model(input_features=mlx_feats, attention_mask=mask, output_hidden_states=True)

    stats = np.load(str(weights_dir / "semantic_stats.npz"))
    mean = mx.array(stats["mean"])
    std = mx.array(stats["std"])
    semantic_features = (out.hidden_states[17] - mean) / std

    codes, _ = semantic_codec_model.quantize(semantic_features)
    codes_for_vq = codes[np.newaxis, :, :]
    emb = semantic_codec_model.vq2emb(codes_for_vq)
    mx.eval(emb)
    arr = np.array(emb)
    assert np.isfinite(arr).all(), "vq2emb output contains NaN/Inf"
    assert arr.ndim == 3  # (1, T', dim)
