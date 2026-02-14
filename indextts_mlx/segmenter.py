"""Sentence segmentation for long-text TTS synthesis.

Splits arbitrary-length text into chunks suitable for synthesis using spaCy
sentence boundary detection. Chunks respect sentence boundaries and can be
sized by character count or BPE token count.

Requires: spacy + a downloaded language model.
Optional: pysbd (better sentence boundary detection for abbreviations).

Install::

    pip install spacy pysbd
    python -m spacy download en_core_web_sm   # English
    python -m spacy download fr_core_news_sm  # French
    python -m spacy download es_core_news_sm  # Spanish
    python -m spacy download it_core_news_sm  # Italian
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Literal, Optional

try:
    from ftfy import fix_text as _ftfy_fix_text
except ImportError:
    _ftfy_fix_text = None

# Default spaCy model names per language
_SPACY_MODELS: dict[str, str] = {
    "english": "en_core_web_sm",
    "en": "en_core_web_sm",
    "french": "fr_core_news_sm",
    "fr": "fr_core_news_sm",
    "spanish": "es_core_news_sm",
    "es": "es_core_news_sm",
    "italian": "it_core_news_sm",
    "it": "it_core_news_sm",
    "german": "de_core_news_sm",
    "de": "de_core_news_sm",
    "portuguese": "pt_core_news_sm",
    "pt": "pt_core_news_sm",
}

# ISO codes used by pySBD
_PYSBD_LANG: dict[str, str] = {
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
}


@dataclass
class SegmenterConfig:
    """Configuration for sentence segmentation.

    Args:
        language: Language name or ISO code (english/en, french/fr,
                  spanish/es, italian/it, german/de, portuguese/pt).
        strategy: How to size chunks:
                  - 'char_count': stay under max_chars per chunk (default)
                  - 'token_count': stay under token_target BPE tokens per chunk
                  - 'sentence_count': fixed sentences_per_chunk sentences
        max_chars: Max characters per chunk for 'char_count' strategy. Default 300.
        token_target: Max BPE tokens per chunk for 'token_count' strategy.
                      120 is recommended for IndexTTS-2.
        sentences_per_chunk: Sentences per chunk for 'sentence_count' strategy. Default 3.
        min_chars: Chunks smaller than this are merged with a neighbour. Default 3.
        spacy_model: Override the spaCy model name (e.g. 'en_core_web_lg').
        use_pysbd: Use pySBD for sentence boundaries (better for abbreviations). Default True.
        bpe_model_path: Path to SentencePiece .model file for token counting.
                        Only needed for 'token_count' strategy. Uses the IndexTTS-2
                        BPE model from WeightsConfig by default.
    """

    language: str = "english"
    strategy: Literal["char_count", "token_count", "sentence_count"] = "char_count"
    max_chars: int = 300
    token_target: Optional[int] = None
    sentences_per_chunk: int = 3
    min_chars: int = 3
    spacy_model: Optional[str] = None
    use_pysbd: bool = True
    bpe_model_path: Optional[str] = None
    disable_pipes: List[str] = field(default_factory=lambda: ["ner", "lemmatizer"])

    def __post_init__(self) -> None:
        if self.language.lower() not in _SPACY_MODELS:
            raise ValueError(
                f"Unsupported language '{self.language}'. "
                f"Supported: {sorted(set(k for k in _SPACY_MODELS if len(k) > 2))}"
            )
        if self.token_target is not None:
            self.strategy = "token_count"
        if self.strategy == "token_count" and self.token_target is None:
            raise ValueError("token_target must be set when strategy='token_count'")

    @property
    def resolved_spacy_model(self) -> str:
        return self.spacy_model or _SPACY_MODELS[self.language.lower()]

    @property
    def pysbd_lang(self) -> str:
        return _PYSBD_LANG.get(self.language.lower(), "en")


class Segmenter:
    """Splits long text into TTS-sized chunks respecting sentence boundaries.

    Uses spaCy for sentence detection (with optional pySBD for better
    handling of abbreviations and edge cases).

    Example::

        seg = Segmenter(SegmenterConfig(language="english", max_chars=300))
        chunks = seg.segment("Long book text goes here...")
        # ["First few sentences.", "Next few sentences.", ...]
    """

    def __init__(self, config: SegmenterConfig = None):
        if config is None:
            config = SegmenterConfig()
        self.config = config
        self._nlp: Any = None
        self._tokenizer: Any = None

    # ── lazy loading ──────────────────────────────────────────────────────────

    @property
    def nlp(self) -> Any:
        if self._nlp is not None:
            return self._nlp

        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "spacy is required for text segmentation. " "Install with: pip install spacy"
            ) from e

        model = self.config.resolved_spacy_model
        try:
            self._nlp = spacy.load(model, disable=self.config.disable_pipes)
        except OSError as e:
            raise OSError(
                f"spaCy model '{model}' not found. "
                f"Install with: python -m spacy download {model}"
            ) from e

        if self.config.use_pysbd:
            self._add_pysbd(self._nlp)

        return self._nlp

    def _add_pysbd(self, nlp: Any) -> None:
        try:
            import pysbd
            from spacy.language import Language
        except ImportError:
            warnings.warn(
                "pysbd not installed — falling back to spaCy sentence detection. "
                "Install with: pip install pysbd",
                stacklevel=3,
            )
            return

        component_name = f"pysbd_sentencizer_{self.config.pysbd_lang}"
        if not Language.has_factory(component_name):
            lang = self.config.pysbd_lang

            @Language.component(component_name)
            def _pysbd_component(doc: Any) -> Any:
                seg = pysbd.Segmenter(language=lang, clean=False)
                sentences = seg.segment(doc.text)
                char_index = 0
                sent_starts: set[int] = set()
                for sent in sentences:
                    start = doc.text.find(sent, char_index)
                    if start != -1:
                        sent_starts.add(start)
                        char_index = start + len(sent)
                for token in doc:
                    token.is_sent_start = token.idx in sent_starts
                return doc

        if component_name not in nlp.pipe_names:
            nlp.add_pipe(component_name, first=True)

    def _get_tokenizer(self, bpe_model_path: Optional[str]) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError("sentencepiece is required for token_count strategy.") from e
        proc = spm.SentencePieceProcessor()
        proc.Load(bpe_model_path)
        self._tokenizer = proc
        return proc

    # ── public API ────────────────────────────────────────────────────────────

    def segment(self, text: str, bpe_model_path: Optional[str] = None) -> List[str]:
        """Split text into synthesis-sized chunks.

        Args:
            text: Input text of any length.
            bpe_model_path: Path to BPE model (only needed for token_count strategy
                            if not set in config).

        Returns:
            List of text chunks, each respecting sentence boundaries.
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        sents = list(doc.sents)

        if self.config.strategy == "token_count":
            path = bpe_model_path or self.config.bpe_model_path
            if path is None:
                raise ValueError(
                    "bpe_model_path required for token_count strategy. "
                    "Pass it to segment() or set in SegmenterConfig."
                )
            tokenizer = self._get_tokenizer(path)
            chunks = self._by_token_count(sents, tokenizer)
        elif self.config.strategy == "sentence_count":
            chunks = self._by_sentence_count(sents)
        else:
            chunks = self._by_char_count(sents)
            chunks = self._merge_small(chunks)

        return chunks

    # ── chunking strategies ───────────────────────────────────────────────────

    def _by_char_count(self, sents: list) -> List[str]:
        chunks: List[str] = []
        current = ""
        for sent in sents:
            s = self._normalize_sentence(sent.text.strip())
            if not s:
                continue
            if len(s) > self.config.max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._hard_split(s))
                continue
            if current and len(current) + 1 + len(s) > self.config.max_chars:
                chunks.append(current)
                current = s
            else:
                current = (current + " " + s).strip() if current else s
        if current:
            chunks.append(current)
        return chunks

    def _by_token_count(self, sents: list, tokenizer: Any) -> List[str]:
        chunks: List[str] = []
        current_sents: List[str] = []
        current_count = 0
        for sent in sents:
            s = self._normalize_sentence(sent.text.strip())
            if not s:
                continue
            n = len(tokenizer.Encode(s))
            if not current_sents:
                current_sents.append(s)
                current_count = n
            elif current_count + n <= self.config.token_target:
                current_sents.append(s)
                current_count += n
            else:
                chunks.append(" ".join(current_sents))
                current_sents = [s]
                current_count = n
        if current_sents:
            chunks.append(" ".join(current_sents))
        return chunks

    def _by_sentence_count(self, sents: list) -> List[str]:
        chunks: List[str] = []
        current: List[str] = []
        for sent in sents:
            s = self._normalize_sentence(sent.text.strip())
            if not s:
                continue
            current.append(s)
            if len(current) >= self.config.sentences_per_chunk:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _merge_small(self, chunks: List[str]) -> List[str]:
        if not chunks or self.config.min_chars == 0:
            return chunks
        merged: List[str] = []
        for chunk in chunks:
            if merged and len(chunk.strip()) < self.config.min_chars:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)
        return merged

    # ── punctuation normalization ─────────────────────────────────────────────

    @staticmethod
    def _normalize_sentence(s: str) -> str:
        """Normalize a sentence for TTS consumption.

        - Replaces em-dash / en-dash / horizontal bar / swung dash variants
          with `` -- `` (readable pause) or strips soft hyphens.
        - Replaces non-breaking hyphens with regular hyphens.
        - Normalizes curly quotes to ASCII quotes and inserts a missing space
          before opening quotes when they stick to a preceding word.
        - Collapses multiple spaces.
        - Ensures the sentence ends with terminal punctuation (. ? !).
        """
        if _ftfy_fix_text is not None:
            s = _ftfy_fix_text(s)

        # Soft hyphen (U+00AD) — invisible, just remove
        s = s.replace("\u00ad", "")

        # Non-breaking hyphen (U+2011) → regular hyphen
        s = s.replace("\u2011", "-")

        # Add a space before opening quotes stuck to the prior word.
        #
        # U+2018 (LEFT single quote) and U+201C/U+201D/ASCII " are unambiguously
        # quotes — always insert the space when they follow a word character.
        s = re.sub(
            r'(?<=[A-Za-z0-9])([\u2018\u201c\u201d"])',
            r" \1",
            s,
        )
        # Straight apostrophe (') and U+2019 (RIGHT single quote) are ambiguous:
        # they appear in contractions (aren't, don't) AND as opening/closing quotes.
        # Only insert a space when NOT immediately followed by a letter (contraction).
        s = re.sub(
            r"(?<=[A-Za-z0-9])(['\u2019])(?![A-Za-z])(?=[^'\u2019]*['\u2019])",
            r" \1",
            s,
        )

        # Curly single/double quotes and primes → ASCII
        quote_map = {
            "\u2018": "'",
            "\u2019": "'",
            "\u2032": "'",
            "\u2035": "'",
            "\u201c": '"',
            "\u201d": '"',
        }
        for k, v in quote_map.items():
            s = s.replace(k, v)

        # Em-dash (U+2014), en-dash (U+2013), horizontal bar (U+2015),
        # swung dash (U+2053), figure dash (U+2012) → " -- "
        s = re.sub(r"[\u2012\u2013\u2014\u2015\u2053]+", " -- ", s)

        # Collapse any runs of whitespace produced by the above
        s = re.sub(r"  +", " ", s).strip()

        # Ensure terminal punctuation
        if s and s[-1] not in ".?!:":
            s = s + "."

        return s

    def _hard_split(self, text: str) -> List[str]:
        warnings.warn(
            f"Hard-splitting {len(text)}-char sentence (max {self.config.max_chars}): "
            f"{text[:80]}...",
            stacklevel=3,
        )
        chunks: List[str] = []
        words = text.split()
        current = ""
        for word in words:
            if len(word) > self.config.max_chars:
                if current:
                    chunks.append(current.strip())
                    current = ""
                for i in range(0, len(word), self.config.max_chars):
                    chunks.append(word[i : i + self.config.max_chars])
                continue
            candidate = (current + " " + word).strip() if current else word
            if len(candidate) <= self.config.max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = word
        if current:
            chunks.append(current.strip())
        return chunks

    def __repr__(self) -> str:
        if self.config.strategy == "token_count":
            detail = f"token_target={self.config.token_target}"
        elif self.config.strategy == "sentence_count":
            detail = f"sentences_per_chunk={self.config.sentences_per_chunk}"
        else:
            detail = f"max_chars={self.config.max_chars}"
        return (
            f"Segmenter(language='{self.config.language}', "
            f"strategy='{self.config.strategy}', {detail})"
        )
