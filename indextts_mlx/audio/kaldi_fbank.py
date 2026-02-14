"""
Kaldi-compatible FBANK feature extraction in MLX.

Matches the output of torchaudio.compliance.kaldi.fbank with default parameters.
"""

import math
import numpy as np
import mlx.core as mx
from typing import Optional


def povey_window(window_length: int) -> np.ndarray:
    """Create Povey window = (0.5 - 0.5*cos(2*pi*n/N))^0.85

    Matches torchaudio.compliance.kaldi._feature_window_function with POVEY type.
    """
    n = np.arange(window_length, dtype=np.float64)
    window = (0.5 - 0.5 * np.cos(2.0 * math.pi / window_length * n)) ** 0.85
    return window.astype(np.float32)


def mel_filterbank_kaldi(
    num_bins: int,
    fft_size: int,
    sample_rate: int,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
) -> np.ndarray:
    """Create Kaldi-style mel filterbank matrix.

    Matches torchaudio.functional.melscale_fbanks with norm=None (no normalization).

    Returns:
        filterbank: (num_bins, fft_size // 2 + 1)
    """
    if high_freq <= 0:
        high_freq = sample_rate / 2.0

    # HTK mel formula (used by torchaudio melscale_fbanks)
    def hz_to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)

    # num_bins + 2 center points (including low/high endpoints)
    mel_points = np.linspace(low_mel, high_mel, num_bins + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])

    num_fft_bins = fft_size // 2 + 1
    freq_bins = np.linspace(0, sample_rate / 2.0, num_fft_bins)

    filterbank = np.zeros((num_bins, num_fft_bins), dtype=np.float32)

    for i in range(num_bins):
        f_left = hz_points[i]
        f_center = hz_points[i + 1]
        f_right = hz_points[i + 2]

        # Rising slope
        mask_up = (freq_bins >= f_left) & (freq_bins < f_center)
        if f_center > f_left:
            filterbank[i, mask_up] = (freq_bins[mask_up] - f_left) / (f_center - f_left)

        # Falling slope
        mask_down = (freq_bins >= f_center) & (freq_bins < f_right)
        if f_right > f_center:
            filterbank[i, mask_down] = (f_right - freq_bins[mask_down]) / (f_right - f_center)

    return filterbank


def compute_kaldi_fbank(
    waveform: np.ndarray,
    num_mel_bins: int = 80,
    sample_frequency: float = 16000.0,
    frame_length: float = 25.0,  # ms
    frame_shift: float = 10.0,  # ms
    dither: float = 0.0,
    preemphasis_coefficient: float = 0.97,
    remove_dc_offset: bool = True,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    use_log_fbank: bool = True,
    energy_floor: float = 1.1920929e-07,  # std::numeric_limits<float>::epsilon() = np.finfo(np.float32).eps
) -> np.ndarray:
    """Compute Kaldi-compatible FBANK features.

    Matches torchaudio.compliance.kaldi.fbank output exactly.

    Processing order (per-frame, matching torchaudio):
    1. Extract frame from waveform
    2. Remove DC offset (subtract frame mean)
    3. Apply preemphasis per-frame
    4. Apply Povey window
    5. Compute power spectrum (|FFT|^2)
    6. Apply mel filterbank
    7. Take log

    Args:
        waveform: 1D float32 numpy array
        num_mel_bins: Number of mel bins
        sample_frequency: Sample rate in Hz
        frame_length: Frame length in ms (default 25ms)
        frame_shift: Frame shift in ms (default 10ms)
        dither: Dither noise (0 = no dither)
        preemphasis_coefficient: Preemphasis filter coeff
        remove_dc_offset: Subtract per-frame mean
        low_freq: Low frequency for mel filterbank
        high_freq: High frequency cutoff (0 = Nyquist)
        use_log_fbank: Apply log to filterbank energies
        energy_floor: Floor for log computation

    Returns:
        fbank: (num_frames, num_mel_bins) float32 array
    """
    waveform = waveform.astype(np.float32)

    # Convert ms to samples — use int() truncation matching torchaudio MILLISECONDS_TO_SECONDS
    frame_length_samples = int(sample_frequency * frame_length * 0.001)
    frame_shift_samples = int(sample_frequency * frame_shift * 0.001)

    # FFT size: next power of 2 >= frame_length_samples
    fft_size = 1
    while fft_size < frame_length_samples:
        fft_size *= 2

    # Apply dither
    if dither > 0:
        waveform = waveform + dither * np.random.randn(len(waveform)).astype(np.float32)

    num_samples = len(waveform)
    if num_samples < frame_length_samples:
        return np.zeros((0, num_mel_bins), dtype=np.float32)

    # Number of frames (snip_edges=True, which is torchaudio default)
    num_frames = 1 + (num_samples - frame_length_samples) // frame_shift_samples

    # Precompute window
    window = povey_window(frame_length_samples)  # (frame_length_samples,)

    # Precompute mel filterbank
    mel_fb = mel_filterbank_kaldi(
        num_mel_bins, fft_size, int(sample_frequency), low_freq, high_freq
    )  # (num_mel_bins, fft_size//2+1)

    # Process all frames
    # Build frame matrix: (num_frames, frame_length_samples)
    indices = (
        np.arange(num_frames)[:, None] * frame_shift_samples
        + np.arange(frame_length_samples)[None, :]
    )
    frames = waveform[indices]  # (num_frames, frame_length_samples)

    # 1. Remove DC offset per frame
    if remove_dc_offset:
        frames = frames - frames.mean(axis=1, keepdims=True)

    # 2. Apply preemphasis per frame: y[0] = x[0], y[n] = x[n] - coeff*x[n-1]
    if preemphasis_coefficient != 0.0:
        frames_preemph = np.empty_like(frames)
        frames_preemph[:, 0] = frames[:, 0]
        frames_preemph[:, 1:] = frames[:, 1:] - preemphasis_coefficient * frames[:, :-1]
        frames = frames_preemph

    # 3. Apply window
    frames = frames * window[None, :]  # (num_frames, frame_length_samples)

    # 4. Zero-pad to FFT size and compute power spectrum
    if fft_size > frame_length_samples:
        pad_width = fft_size - frame_length_samples
        frames = np.pad(frames, ((0, 0), (0, pad_width)), mode="constant")

    # |FFT|^2 — use float64 for FFT accuracy then cast back
    fft_result = np.fft.rfft(frames.astype(np.float64), n=fft_size, axis=1)
    power_spectrum = (fft_result.real**2 + fft_result.imag**2).astype(np.float32)
    # (num_frames, fft_size//2+1)

    # 5. Apply mel filterbank
    mel_energies = power_spectrum @ mel_fb.T  # (num_frames, num_mel_bins)

    # 6. Apply floor and log
    mel_energies = np.maximum(mel_energies, energy_floor)
    if use_log_fbank:
        mel_energies = np.log(mel_energies)

    return mel_energies.astype(np.float32)


def compute_kaldi_fbank_mlx(
    waveform: mx.array, num_mel_bins: int = 80, sample_frequency: float = 16000.0, **kwargs
) -> mx.array:
    """MLX wrapper for Kaldi FBANK computation.

    Args:
        waveform: MLX array (1D) or (batch, T)
        num_mel_bins: Number of mel bins
        sample_frequency: Sample rate
        **kwargs: Passed to compute_kaldi_fbank

    Returns:
        fbank: (num_frames, num_mel_bins) or (batch, num_frames, num_mel_bins)
    """
    waveform_np = np.array(waveform)

    if waveform_np.ndim == 1:
        fbank = compute_kaldi_fbank(
            waveform_np, num_mel_bins=num_mel_bins, sample_frequency=sample_frequency, **kwargs
        )
        return mx.array(fbank)

    elif waveform_np.ndim == 2:
        batch_size = waveform_np.shape[0]
        fbank_list = [
            compute_kaldi_fbank(
                waveform_np[i],
                num_mel_bins=num_mel_bins,
                sample_frequency=sample_frequency,
                **kwargs,
            )
            for i in range(batch_size)
        ]
        max_len = max(f.shape[0] for f in fbank_list)
        batch_fbank = np.zeros((batch_size, max_len, num_mel_bins), dtype=np.float32)
        for i, fbank in enumerate(fbank_list):
            batch_fbank[i, : fbank.shape[0], :] = fbank
        return mx.array(batch_fbank)

    else:
        raise ValueError(f"Waveform must be 1D or 2D, got shape {waveform_np.shape}")
