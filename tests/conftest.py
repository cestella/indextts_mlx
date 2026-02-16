"""
Shared pytest fixtures for indextts_mlx tests.
"""

import os
import pytest
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# --llm flag: opt-in to tests that load the real LLM
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--llm",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.llm (loads the real LLM; slow).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llm: tests that load the real LLM (skipped by default; run with --llm)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--llm"):
        skip_llm = pytest.mark.skip(reason="LLM tests skipped by default; use --llm to run")
        for item in items:
            if item.get_closest_marker("llm"):
                item.add_marker(skip_llm)


@pytest.fixture(scope="session")
def classifier_llm():
    """Session-scoped real EmotionClassifier with the LLM loaded once."""
    from indextts_mlx.emotion_classifier import ClassifierConfig, EmotionClassifier
    cfg = ClassifierConfig(use_boundary_detection=False)
    return EmotionClassifier(cfg)

# Default weights dir — can override with env var
WEIGHTS_DIR = Path(
    os.environ.get(
        "INDEXTTS_MLX_WEIGHTS_DIR",
        str(Path.home() / "code/index-tts-m3-port/prototypes/s2mel_mlx/mlx_weights"),
    )
)

BPE_MODEL = Path(
    os.environ.get(
        "INDEXTTS_MLX_BPE_MODEL",
        str(Path.home() / "code/tts/index-tts/checkpoints/bpe.model"),
    )
)

REFERENCE_AUDIO = Path.home() / "audiobooks/voices/prunella_scales.wav"


# Session-scoped fixtures so models are loaded once per test session


@pytest.fixture(scope="session")
def weights_dir():
    assert (
        WEIGHTS_DIR.exists() and (WEIGHTS_DIR / "gpt.npz").exists()
    ), f"Weights not found at {WEIGHTS_DIR}"
    return WEIGHTS_DIR


@pytest.fixture(scope="session")
def bpe_model_path():
    assert BPE_MODEL.exists(), f"BPE model not found at {BPE_MODEL}"
    return BPE_MODEL


@pytest.fixture(scope="session")
def reference_audio_np():
    """Load reference audio as float32 numpy arrays at 16k and 22k."""
    assert REFERENCE_AUDIO.exists(), f"Reference audio not found at {REFERENCE_AUDIO}"
    import soundfile as sf
    import librosa

    audio, sr = sf.read(str(REFERENCE_AUDIO))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)[: 3 * 16000]
    audio_22k = librosa.resample(audio, orig_sr=sr, target_sr=22050).astype(np.float32)[: 3 * 22050]
    return audio_16k, audio_22k


@pytest.fixture(scope="session")
def campplus_model(weights_dir):
    from indextts_mlx.models.campplus import CAMPPlus
    from indextts_mlx.loaders.campplus_loader import load_campplus_model

    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model = load_campplus_model(model, str(weights_dir / "campplus.npz"))
    return model


@pytest.fixture(scope="session")
def w2vbert_model(weights_dir):
    from indextts_mlx.models.w2vbert import create_w2vbert_model
    from indextts_mlx.loaders.w2vbert_loader import load_w2vbert_model

    model = create_w2vbert_model()
    model = load_w2vbert_model(model, str(weights_dir / "w2vbert.npz"))
    return model


@pytest.fixture(scope="session")
def semantic_codec_model(weights_dir):
    from indextts_mlx.models.semantic_codec import RepCodec
    from indextts_mlx.loaders.semantic_codec_loader import load_semantic_codec_model

    model = RepCodec()
    model = load_semantic_codec_model(model, str(weights_dir / "semantic_codec.npz"))
    return model


@pytest.fixture(scope="session")
def gpt_model(weights_dir):
    from indextts_mlx.models.gpt import create_unifiedvoice
    from indextts_mlx.loaders.gpt_loader import load_gpt_model

    model = create_unifiedvoice()
    model = load_gpt_model(model, str(weights_dir / "gpt.npz"))
    return model


@pytest.fixture(scope="session")
def s2mel_pipeline(weights_dir):
    from indextts_mlx.models.s2mel import create_mlx_s2mel_pipeline

    return create_mlx_s2mel_pipeline(checkpoint_path=str(weights_dir / "s2mel_pytorch.npz"))


@pytest.fixture(scope="session")
def bigvgan_model(weights_dir):
    from indextts_mlx.models.bigvgan import BigVGAN
    from indextts_mlx.loaders.bigvgan_loader import load_bigvgan_model

    model = BigVGAN()
    model = load_bigvgan_model(model, str(weights_dir / "bigvgan.npz"))
    return model
