"""NeMo-based text normalization for TTS preprocessing.

Converts written text to spoken form — numbers, currency, dates, times, etc.
Lazy-loads NeMo on first use so the import cost is only paid when normalization
is actually requested.

Requires nemo_text_processing + pynini. See README.md for install instructions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional

# Maps friendly language names to NeMo language codes
_LANG_TO_NEMO: dict[str, str] = {
    "english": "en",
    "en": "en",
    "french": "fr",
    "fr": "fr",
    "spanish": "es",
    "es": "es",
    "italian": "it",
    "it": "it",
    "german": "de",
    "de": "de",
    "portuguese": "pt",
    "pt": "pt",
    "russian": "ru",
    "ru": "ru",
    "vietnamese": "vi",
    "vi": "vi",
}

_SUPPORTED_LANGUAGES = sorted(
    {k for k in _LANG_TO_NEMO if len(k) > 2}
)  # ["english", "french", ...]


def _to_nemo_code(language: str) -> str:
    code = _LANG_TO_NEMO.get(language.lower())
    if code is None:
        raise ValueError(
            f"Unsupported language '{language}'. " f"Supported: {_SUPPORTED_LANGUAGES}"
        )
    return code


@dataclass
class NormalizerConfig:
    """Configuration for NeMo text normalization.

    Args:
        language: Language name or ISO code. Supported: english/en, french/fr,
                  spanish/es, italian/it, german/de, portuguese/pt, russian/ru,
                  vietnamese/vi.
        input_case: 'cased' (default) or 'lower_cased'.
        cache_dir: Directory to cache compiled NeMo FST grammars. First run
                   compiles grammars (slow); subsequent runs load from cache.
        verbose: Print NeMo normalization debug info.
    """

    language: str = "english"
    input_case: str = "cased"
    cache_dir: Optional[str] = None
    verbose: bool = False

    def __post_init__(self) -> None:
        _to_nemo_code(self.language)  # validate early
        if self.input_case not in ("cased", "lower_cased"):
            raise ValueError(
                f"input_case must be 'cased' or 'lower_cased', got '{self.input_case}'"
            )

    @property
    def nemo_code(self) -> str:
        return _to_nemo_code(self.language)


class Normalizer:
    """Text normalizer backed by NeMo Text Processing.

    Converts written text to spoken form for TTS:
      - Numbers:   "123"        → "one hundred twenty three"
      - Currency:  "$45.50"     → "forty five dollars fifty cents"
      - Dates:     "Jan 1 2024" → "january first twenty twenty four"
      - Times:     "3:30pm"     → "three thirty p m"

    NeMo is lazy-loaded on first use. If nemo_text_processing is not
    installed, normalize() returns the original text unchanged with a
    one-time warning.

    Example::

        norm = Normalizer(NormalizerConfig(language="english"))
        text = norm.normalize("The price is $123.45 on Jan 1, 2024.")
        # "The price is one hundred twenty three dollars forty five cents
        #  on january first twenty twenty four."
    """

    def __init__(self, config: NormalizerConfig = None):
        if config is None:
            config = NormalizerConfig()
        self.config = config
        self._nemo: Any = None
        self._unavailable = False  # set True if import fails, to warn only once

    def _load(self) -> Any:
        if self._nemo is not None:
            return self._nemo
        if self._unavailable:
            return None
        try:
            from nemo_text_processing.text_normalization.normalize import (  # type: ignore[import-untyped]
                Normalizer as _NeMo,
            )
        except ImportError:
            warnings.warn(
                "nemo_text_processing is not installed — text normalization skipped. "
                "See README.md for install instructions (requires pynini + OpenFst).",
                stacklevel=3,
            )
            self._unavailable = True
            return None

        try:
            self._nemo = _NeMo(
                input_case=self.config.input_case,
                lang=self.config.nemo_code,
                cache_dir=self.config.cache_dir,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NeMo normalizer: {e}") from e

        return self._nemo

    def normalize(self, text: str) -> str:
        """Normalize a block of text to spoken form.

        Preserves newline structure (each line is normalized independently).
        If NeMo is unavailable, returns text unchanged.
        """
        if not text or not text.strip():
            return text

        nemo = self._load()
        if nemo is None:
            return text

        try:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            normalized_lines = []
            for line in lines:
                sentences = nemo.split_text_into_sentences(line)
                normalized = nemo.normalize_list(
                    sentences,
                    verbose=self.config.verbose,
                    punct_post_process=True,
                )
                normalized_lines.append(" ".join(normalized))
            return "\n".join(normalized_lines)
        except Exception as e:
            warnings.warn(
                f"NeMo normalization failed: {e}. Returning original text.",
                stacklevel=2,
            )
            return text

    @property
    def available(self) -> bool:
        """True if nemo_text_processing is installed and loaded successfully."""
        return self._load() is not None

    def __repr__(self) -> str:
        return (
            f"Normalizer(language='{self.config.language}', "
            f"input_case='{self.config.input_case}')"
        )
