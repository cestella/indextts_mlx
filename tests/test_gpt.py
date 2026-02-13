"""
GPT conditioning and prepare_inputs shape tests.
"""

import pytest
import numpy as np
import mlx.core as mx

from indextts_mlx.audio.seamless_fbank import compute_seamless_fbank


@pytest.fixture(scope="module")
def semantic_features_fixture(w2vbert_model, reference_audio_np, weights_dir):
    audio_16k, _ = reference_audio_np
    feats_np = compute_seamless_fbank(audio_16k)
    mlx_feats = mx.array(feats_np[None])
    T = feats_np.shape[0]
    mask = mx.ones((1, T), dtype=mx.int32)
    out = w2vbert_model(input_features=mlx_feats, attention_mask=mask, output_hidden_states=True)
    stats = np.load(str(weights_dir / "semantic_stats.npz"))
    mean = mx.array(stats['mean'])
    std = mx.array(stats['std'])
    feats = (out.hidden_states[17] - mean) / std
    mx.eval(feats)
    return feats


def test_gpt_conditioning_shape(gpt_model, semantic_features_fixture):
    cond = gpt_model.get_full_conditioning_34(semantic_features_fixture)
    mx.eval(cond)
    arr = np.array(cond)
    # Expected: (1, 34, 1280)
    assert arr.shape == (1, 34, 1280), f"Unexpected conditioning shape: {arr.shape}"
    assert np.isfinite(arr).all()


def test_gpt_conditioning_emotion_scale(gpt_model, semantic_features_fixture):
    cond_neutral = np.array(gpt_model.get_full_conditioning_34(semantic_features_fixture, emotion_scale=0.0))
    cond_default = np.array(gpt_model.get_full_conditioning_34(semantic_features_fixture, emotion_scale=1.0))
    cond_expressive = np.array(gpt_model.get_full_conditioning_34(semantic_features_fixture, emotion_scale=2.0))
    # Neutral and default should differ
    assert not np.allclose(cond_neutral, cond_default, atol=1e-4)
    # Default and expressive should differ
    assert not np.allclose(cond_default, cond_expressive, atol=1e-4)


def test_gpt_prepare_inputs_shape(gpt_model, semantic_features_fixture, bpe_model_path):
    from sentencepiece import SentencePieceProcessor
    sp = SentencePieceProcessor(model_file=str(bpe_model_path))
    text_tokens = mx.array([sp.encode("HELLO WORLD")])
    cond = gpt_model.get_full_conditioning_34(semantic_features_fixture)
    inputs_embeds, _ = gpt_model.prepare_inputs(cond, text_tokens)
    mx.eval(inputs_embeds)
    arr = np.array(inputs_embeds)
    assert arr.ndim == 3  # (1, seq_len, hidden)
    assert arr.shape[0] == 1
    assert np.isfinite(arr).all()
