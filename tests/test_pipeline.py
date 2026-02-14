"""
End-to-end pipeline test: synthesize() produces valid audio.
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture(scope="module")
def pipeline_instance(weights_dir, bpe_model_path):
    from indextts_mlx.config import WeightsConfig
    from indextts_mlx.pipeline import IndexTTS2

    cfg = WeightsConfig(weights_dir=weights_dir, bpe_model=bpe_model_path)
    return IndexTTS2(config=cfg)


def test_synthesize_short(pipeline_instance, reference_audio_np):
    _, audio_22k = reference_audio_np
    # Use a very short text to keep test fast
    audio_out = pipeline_instance.synthesize(
        text="Hello.",
        reference_audio=audio_22k,
        sample_rate=22050,
        cfm_steps=5,
        max_codes=200,
    )
    assert isinstance(audio_out, np.ndarray)
    assert audio_out.dtype == np.float32
    assert audio_out.ndim == 1
    assert audio_out.shape[0] > 0
    assert np.isfinite(audio_out).all()


def test_synthesize_emotion_param(pipeline_instance, reference_audio_np):
    """Different emotion values should produce different audio."""
    _, audio_22k = reference_audio_np
    kwargs = dict(
        reference_audio=audio_22k,
        sample_rate=22050,
        text="Hello.",
        cfm_steps=5,
        max_codes=200,
    )
    audio_neutral = pipeline_instance.synthesize(**kwargs, emotion=0.0)
    audio_expressive = pipeline_instance.synthesize(**kwargs, emotion=2.0)
    # They should differ
    min_len = min(len(audio_neutral), len(audio_expressive))
    assert not np.allclose(audio_neutral[:min_len], audio_expressive[:min_len], atol=1e-5)


def test_synthesize_output_file(pipeline_instance, reference_audio_np, tmp_path):
    """synthesize() can write output to a WAV file."""
    import soundfile as sf

    _, audio_22k = reference_audio_np
    out_path = tmp_path / "test_out.wav"

    audio_out = pipeline_instance.synthesize(
        text="Hello.",
        reference_audio=audio_22k,
        sample_rate=22050,
        cfm_steps=5,
        max_codes=200,
    )
    sf.write(str(out_path), audio_out, 22050)
    assert out_path.exists()
    audio_read, sr = sf.read(str(out_path))
    assert sr == 22050
    assert len(audio_read) == len(audio_out)
