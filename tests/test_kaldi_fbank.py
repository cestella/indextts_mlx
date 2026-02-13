"""
Test Kaldi filterbank feature extraction.
Parity test against torchaudio is skipped if torchaudio not installed.
"""

import pytest
import numpy as np
import mlx.core as mx

from indextts_mlx.audio.kaldi_fbank import compute_kaldi_fbank_mlx


def make_sine_audio(duration_s=1.0, sr=22050, freq=440.0):
    t = np.linspace(0, duration_s, int(duration_s * sr), dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_kaldi_fbank_shape():
    audio = make_sine_audio(duration_s=0.5)
    feats = compute_kaldi_fbank_mlx(mx.array(audio), num_mel_bins=80, sample_frequency=22050)
    # Expected: (T, 80)
    assert feats.ndim == 2
    assert feats.shape[1] == 80
    T = feats.shape[0]
    assert T > 0


def test_kaldi_fbank_range():
    audio = make_sine_audio(duration_s=0.5)
    feats = compute_kaldi_fbank_mlx(mx.array(audio), num_mel_bins=80, sample_frequency=22050)
    arr = np.array(feats)
    assert np.isfinite(arr).all(), "Features contain NaN/Inf"
    # Log-mel features are typically in [-20, 20] range
    assert arr.min() > -100
    assert arr.max() < 100


@pytest.mark.parity
def test_kaldi_fbank_parity_torchaudio():
    """Parity test against torchaudio."""
    import torch
    import torchaudio.compliance.kaldi as kaldi

    audio = make_sine_audio(duration_s=1.0)
    mlx_feats = np.array(
        compute_kaldi_fbank_mlx(mx.array(audio), num_mel_bins=80, sample_frequency=22050)
    )

    audio_t = torch.tensor(audio).unsqueeze(0)
    torch_feats = kaldi.fbank(
        audio_t,
        num_mel_bins=80,
        sample_frequency=22050,
        use_log_fbank=True,
        use_energy=False,
        window_type="hanning",
        dither=0.0,
    ).numpy()

    min_len = min(mlx_feats.shape[0], torch_feats.shape[0])
    diff = np.abs(mlx_feats[:min_len] - torch_feats[:min_len]).max()
    assert diff < 5.0, f"Max absolute difference vs torchaudio: {diff:.4f}"
