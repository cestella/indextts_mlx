"""IndexTTS-2 MLX -- Apple Silicon TTS inference."""

from .pipeline import IndexTTS2, synthesize
from .config import WeightsConfig
from .voices import list_voices, resolve_voice, parse_emo_vector
from .renderer import render_segments_jsonl
from .normalizer import Normalizer, NormalizerConfig
from .segmenter import Segmenter, SegmenterConfig
from .synthesize_long import synthesize_long, LongSynthesisConfig

__all__ = [
    "IndexTTS2",
    "WeightsConfig",
    "synthesize",
    "list_voices",
    "resolve_voice",
    "parse_emo_vector",
    "render_segments_jsonl",
    "Normalizer",
    "NormalizerConfig",
    "Segmenter",
    "SegmenterConfig",
    "synthesize_long",
    "LongSynthesisConfig",
]
__version__ = "0.1.0"
