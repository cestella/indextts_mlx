"""IndexTTS-2 MLX -- Apple Silicon TTS inference."""
from .pipeline import IndexTTS2, synthesize
from .config import WeightsConfig

__all__ = ["IndexTTS2", "WeightsConfig", "synthesize"]
__version__ = "0.1.0"
