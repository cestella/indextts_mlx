import numpy as np
import librosa
import mlx.core as mx

def compute_mel_s2mel(audio_22k: np.ndarray) -> mx.array:
    """Compute 80-bin log mel spectrogram for S2Mel reference conditioning.
    
    Returns: mx.array shape (1, 80, T)
    """
    n_fft, hop_length, win_length, n_mels = 1024, 256, 1024, 80
    pad = (n_fft - hop_length) // 2
    audio_padded = np.pad(audio_22k, pad, mode='reflect')
    stft = librosa.stft(audio_padded, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window='hann', center=False)
    spec = np.sqrt(stft.real**2 + stft.imag**2 + 1e-9)
    mel_basis = librosa.filters.mel(sr=22050, n_fft=n_fft, n_mels=n_mels,
                                    fmin=0, fmax=None, norm='slaney', htk=False)
    mel_log = np.log(np.clip(mel_basis @ spec, a_min=1e-5, a_max=None))
    return mx.array(mel_log[np.newaxis, :, :])
