"""CPU-only endpoints — no GPU contention.

Unlike the CLI, the srv endpoints fail loudly if dependencies are missing.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/cpu", tags=["cpu"])


def _require_nemo():
    """Import and return the NeMo Normalizer class, or raise 503."""
    try:
        from nemo_text_processing.text_normalization.normalize import (
            Normalizer as NeMoNormalizer,
        )
        return NeMoNormalizer
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="nemo_text_processing is not installed. "
            "See README.md for macOS install instructions (requires pynini + OpenFst).",
        )


def _require_spacy():
    """Import and return spacy, or raise 503."""
    try:
        import spacy
        return spacy
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="spacy is not installed. Install with: uv sync --extra long",
        )


@router.post("/normalize")
async def normalize(body: dict) -> dict:
    """Normalize text to spoken form (NeMo-based)."""
    NeMoNormalizer = _require_nemo()

    text = body.get("text", "")
    language = body.get("language", "en")
    if not text or not text.strip():
        return {"text": text}

    normalizer = NeMoNormalizer(input_case="cased", lang=language)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    normalized_lines = []
    for line in lines:
        sentences = normalizer.split_text_into_sentences(line)
        normalized = normalizer.normalize_list(sentences, punct_post_process=True)
        normalized_lines.append(" ".join(normalized))
    return {"text": "\n".join(normalized_lines)}


@router.post("/segment")
async def segment(body: dict) -> dict:
    """Segment text into TTS-sized chunks."""
    _require_spacy()
    from indextts_mlx.segmenter import Segmenter, SegmenterConfig

    text = body.get("text", "")
    language = body.get("language", "english")
    strategy = body.get("strategy", "char_count")
    max_chars = body.get("max_chars", 300)

    config = SegmenterConfig(language=language, strategy=strategy, max_chars=max_chars)
    seg = Segmenter(config)
    segments = seg.segment(text)
    return {"segments": segments}
