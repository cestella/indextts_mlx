"""
Pure numpy/MLX implementation of SeamlessM4TFeatureExtractor.

Computes 160-dim log mel filterbank features for W2V-BERT input,
matching SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
at >120 dB SNR.

Key differences from standard Kaldi FBANK:
- mel_scale="kaldi" (1127*ln(1+f/700)) with triangularize_in_mel_space=True
- Preemphasis: buffer[0] *= (1-coeff), buffer[1:] -= coeff*buffer[:-1]
- Povey window: np.hanning(400)^0.85, periodic=False (no trim)
- waveform scaled by 2^15 (Kaldi int16 compliance)
- stride=2: output is (T//2, 160) by concatenating frame pairs
- Per-mel-bin normalization with ddof=1 variance
"""

import numpy as np
import mlx.core as mx


def _build_mel_filters_kaldi(
    num_frequency_bins: int = 257,
    num_mel_filters: int = 80,
    min_frequency: float = 20.0,
    max_frequency: float = 8000.0,
    sampling_rate: int = 16000,
) -> np.ndarray:
    """Build kaldi-scale mel filterbank triangularized in mel space.

    Returns:
        mel_filters: (num_frequency_bins, num_mel_filters)
    """
    def hertz_to_mel(freq):
        return 1127.0 * np.log(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)

    mel_min = hertz_to_mel(min_frequency)
    mel_max = hertz_to_mel(max_frequency)
    filter_freqs_mel = np.linspace(mel_min, mel_max, num_mel_filters + 2)

    # FFT bins in mel space (triangularize_in_mel_space=True)
    fft_bin_width = sampling_rate / ((num_frequency_bins - 1) * 2)
    fft_freqs_mel = hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins))

    f_diff = filter_freqs_mel[1:] - filter_freqs_mel[:-1]
    slopes = filter_freqs_mel[np.newaxis, :] - fft_freqs_mel[:, np.newaxis]  # (n_freqs, n_mels+2)
    down_slopes = (-slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]         # (n_freqs, n_mels)
    mel_filters = np.maximum(0.0, np.minimum(down_slopes, up_slopes))  # (n_freqs, n_mels)

    return mel_filters.astype(np.float32)


# Precomputed mel filters (constant for all calls with default params)
_MEL_FILTERS = _build_mel_filters_kaldi()

# Precomputed Povey window: hanning(400)^0.85, periodic=False
_POVEY_WINDOW = np.power(np.hanning(400), 0.85).astype(np.float64)


def compute_seamless_fbank(
    audio_16k: np.ndarray,
    mel_floor: float = 1.192092955078125e-07,  # float32 epsilon
    stride: int = 2,
) -> np.ndarray:
    """Compute SeamlessM4TFeatureExtractor features.

    Matches the output of:
        SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")(
            audio_16k, sampling_rate=16000
        )["input_features"][0]

    at >120 dB SNR.

    Args:
        audio_16k: 1D float32 audio at 16kHz
        mel_floor: Floor for mel energies before log
        stride: Frame striding (default 2, output has T//stride frames)

    Returns:
        features: (T//stride, 80*stride) float32 array
    """
    frame_length = 400
    hop_length = 160
    fft_length = 512
    preemphasis_coeff = 0.97

    # Scale to int16 range (Kaldi compliance)
    waveform = audio_16k.astype(np.float64) * 32768.0  # 2^15

    num_frames = int(1 + np.floor((len(waveform) - frame_length) / hop_length))

    # Allocate output
    num_fft_bins = fft_length // 2 + 1  # 257
    specs = np.zeros((num_frames, num_fft_bins), dtype=np.float64)
    buffer = np.zeros(fft_length, dtype=np.float64)

    for i in range(num_frames):
        buffer[:frame_length] = waveform[i * hop_length:i * hop_length + frame_length]
        # DC removal
        buffer[:frame_length] -= buffer[:frame_length].mean()
        # Preemphasis (SeamlessM4T formula: first sample scaled by (1-coeff))
        buffer[1:frame_length] -= preemphasis_coeff * buffer[:frame_length - 1]
        buffer[0] *= (1.0 - preemphasis_coeff)
        # Povey window
        buffer[:frame_length] *= _POVEY_WINDOW
        # Power spectrum
        specs[i] = np.abs(np.fft.rfft(buffer, n=fft_length)) ** 2

    # Apply mel filterbank
    mel_spec = np.maximum(mel_floor, specs @ _MEL_FILTERS)  # (T, 80)

    # Natural log
    log_mel = np.log(mel_spec).astype(np.float32)

    # Per-mel-bin normalization with ddof=1 (matching torch.std behavior)
    log_mel = (log_mel - log_mel.mean(0)) / np.sqrt(log_mel.var(0, ddof=1) + 1e-7)

    # Stride: trim to multiple of stride, reshape to (T//stride, 80*stride)
    T = log_mel.shape[0]
    T_trim = T - (T % stride)
    log_mel = log_mel[:T_trim].reshape(T_trim // stride, 80 * stride)

    return log_mel.astype(np.float32)


def compute_seamless_fbank_mlx(audio_16k: mx.array) -> mx.array:
    """MLX wrapper for SeamlessM4T feature extraction.

    Args:
        audio_16k: 1D MLX array at 16kHz

    Returns:
        features: (T//2, 160) MLX array
    """
    audio_np = np.array(audio_16k)
    features = compute_seamless_fbank(audio_np)
    return mx.array(features)
