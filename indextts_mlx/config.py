"""Configuration and weight path management."""
import os
from pathlib import Path
from dataclasses import dataclass, field

_DEFAULT_WEIGHTS = Path.home() / "code/index-tts-m3-port/prototypes/s2mel_mlx/mlx_weights"
_DEFAULT_BPE = Path.home() / "code/tts/index-tts/checkpoints/bpe.model"
_DEFAULT_QWEN_EMO = Path.home() / "code/tts/index-tts/checkpoints/qwen0.6bemo4-merge"


def _default_weights_dir() -> Path:
    return Path(os.environ.get("INDEXTTS_MLX_WEIGHTS_DIR", str(_DEFAULT_WEIGHTS)))


def _default_bpe_model() -> Path:
    return Path(os.environ.get("INDEXTTS_MLX_BPE_MODEL", str(_DEFAULT_BPE)))


def _default_qwen_emo() -> Path:
    return Path(os.environ.get("INDEXTTS_MLX_QWEN_EMO", str(_DEFAULT_QWEN_EMO)))


@dataclass
class WeightsConfig:
    """Paths to model weights and resources.
    
    Can be configured via:
    - Constructor arguments
    - Environment variables: INDEXTTS_MLX_WEIGHTS_DIR, INDEXTTS_MLX_BPE_MODEL
    """
    weights_dir: Path = field(default_factory=_default_weights_dir)
    bpe_model: Path = field(default_factory=_default_bpe_model)
    qwen_emo: Path = field(default_factory=_default_qwen_emo)

    def __post_init__(self):
        self.weights_dir = Path(self.weights_dir)
        self.bpe_model = Path(self.bpe_model)
        self.qwen_emo = Path(self.qwen_emo)

    @property
    def gpt(self) -> Path:
        return self.weights_dir / "gpt.npz"

    @property
    def w2vbert(self) -> Path:
        return self.weights_dir / "w2vbert.npz"

    @property
    def campplus(self) -> Path:
        return self.weights_dir / "campplus.npz"

    @property
    def semantic_codec(self) -> Path:
        return self.weights_dir / "semantic_codec.npz"

    @property
    def semantic_stats(self) -> Path:
        return self.weights_dir / "semantic_stats.npz"

    @property
    def bigvgan(self) -> Path:
        return self.weights_dir / "bigvgan.npz"

    @property
    def s2mel(self) -> Path:
        return self.weights_dir / "s2mel_pytorch.npz"

    @property
    def emotion_matrix(self) -> Path:
        return self.weights_dir / "emotion_matrix.npz"

    @property
    def speaker_matrix(self) -> Path:
        return self.weights_dir / "speaker_matrix.npz"

    def validate(self):
        """Raise FileNotFoundError if any required weight file is missing."""
        required = [self.gpt, self.w2vbert, self.campplus, self.semantic_codec,
                    self.semantic_stats, self.bigvgan, self.bpe_model]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")
