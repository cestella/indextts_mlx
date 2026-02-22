"""Tests for the ``...`` paragraph break marker pipeline.

These tests are fast — no model loading required.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── 1. Extraction: _to_sentences inserts `...` between paragraphs ──────────


class TestExtractInsertsParagraphMarkers:
    """Verify that _to_sentences() inserts ``...`` between paragraphs."""

    @pytest.fixture()
    def to_sentences(self):
        """Build a minimal _to_sentences closure using a real spaCy nlp."""
        try:
            import spacy
        except ImportError:
            pytest.skip("spacy not installed")

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("en_core_web_sm model not installed")

        try:
            import ftfy as _ftfy
            _fix_text = _ftfy.fix_text
        except ImportError:
            _fix_text = lambda t: t  # noqa: E731

        def _to_sentences(text: str) -> str:
            text = _fix_text(text)
            paragraphs = re.split(r"\n{2,}", text)
            sentences: list[str] = []
            for para in paragraphs:
                flat = re.sub(r"\s*\n\s*", " ", para).strip()
                flat = re.sub(r"  +", " ", flat)
                if not flat:
                    continue
                if sentences:
                    sentences.append("...")
                is_bare_heading = not re.search(r"[.!?…]$", flat)
                doc = nlp(flat)
                sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                if is_bare_heading and len(sents) == 1:
                    sents[0] = sents[0] + "."
                sentences.extend(sents)
            return "\n".join(sentences)

        return _to_sentences

    def test_two_paragraphs_have_marker(self, to_sentences):
        text = "First paragraph sentence one. Sentence two.\n\nSecond paragraph sentence."
        result = to_sentences(text)
        lines = result.split("\n")
        assert "..." in lines, f"Expected '...' marker in lines: {lines}"
        idx = lines.index("...")
        assert idx > 0, "Marker should not be the first line"
        assert idx < len(lines) - 1, "Marker should not be the last line"

    def test_single_paragraph_no_marker(self, to_sentences):
        text = "Just one paragraph. With two sentences."
        result = to_sentences(text)
        lines = result.split("\n")
        assert "..." not in lines

    def test_three_paragraphs_two_markers(self, to_sentences):
        text = "Para one.\n\nPara two.\n\nPara three."
        result = to_sentences(text)
        lines = result.split("\n")
        assert lines.count("...") == 2

    def test_empty_paragraphs_ignored(self, to_sentences):
        text = "Para one.\n\n\n\nPara two."
        result = to_sentences(text)
        lines = result.split("\n")
        assert lines.count("...") == 1


# ── 2. Classifier: strips markers and upgrades pause ───────────────────────


class TestClassifierHandlesMarkers:
    """Verify that classify_text strips ``...`` and upgrades pause at boundaries."""

    def test_strips_markers_and_upgrades_pause(self):
        from indextts_mlx.emotion_classifier import EmotionClassifier, ClassifierConfig

        clf = EmotionClassifier(ClassifierConfig())

        sentences = ["Sentence one.", "Sentence two.", "Sentence three.", "Sentence four."]
        paragraph_breaks = {1}  # break after sentence index 1

        with patch.object(clf, "_get_sentences", return_value=(sentences, paragraph_breaks)):
            with patch.object(clf, "_classify_one_emotion", return_value=0):
                with patch.object(clf, "_classify_one_pause", return_value=2):
                    clf.config.use_boundary_detection = False
                    records = clf.classify_text("dummy text", verbose=False)

        assert len(records) == 4
        assert records[1].pause_idx >= 3, (
            f"Expected pause_idx >= 3 at paragraph break, got {records[1].pause_idx}"
        )
        assert records[0].pause_idx == 2
        assert records[2].pause_idx == 2
        assert records[3].pause_idx == 2

    def test_get_sentences_filters_markers(self):
        """_get_sentences should strip ``...`` markers and record break positions."""
        from indextts_mlx.emotion_classifier import EmotionClassifier, ClassifierConfig

        clf = EmotionClassifier(ClassifierConfig())

        raw_segments = ["Hello world.", "...", "Goodbye world."]

        mock_seg = MagicMock()
        mock_seg.segment.return_value = raw_segments

        with patch("indextts_mlx.segmenter.Segmenter", return_value=mock_seg):
            # Re-import to pick up the patched Segmenter inside _get_sentences
            # Actually, _get_sentences imports Segmenter locally, so we patch
            # at the source module level.
            sentences, breaks = clf._get_sentences("dummy")

        assert sentences == ["Hello world.", "Goodbye world."]
        assert breaks == {0}


# ── 3. synthesize_long: splits on `...` lines and inserts silence ──────────


def _make_mock_segmenter():
    """Create a mock Segmenter that just returns the input text as a single chunk."""
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.segment.side_effect = lambda text: [text] if text.strip() else []
    mock_cls.return_value = mock_instance
    return mock_cls


class TestSynthesizeLongParagraphPause:
    """Verify that synthesize_long splits on ``...`` and inserts paragraph pauses."""

    def test_splits_on_markers(self):
        from indextts_mlx.synthesize_long import synthesize_long, OUTPUT_SAMPLE_RATE

        text = "Sentence one.\n...\nSentence two."

        mock_tts = MagicMock()
        chunk_len = int(OUTPUT_SAMPLE_RATE * 0.5)
        mock_tts.synthesize.return_value = np.ones(chunk_len, dtype=np.float32)

        with patch("indextts_mlx.synthesize_long.Segmenter", _make_mock_segmenter()):
            audio = synthesize_long(
                text,
                tts=mock_tts,
                spk_audio_prompt="speaker.wav",
                normalize=False,
                crossfade_ms=0,
                silence_between_chunks_ms=0,
                paragraph_pause_ms=1100,
            )

        # Should have called synthesize twice (once per paragraph block)
        assert mock_tts.synthesize.call_count == 2

        # Audio should be: chunk1 + paragraph_silence + chunk2
        paragraph_silence_samples = int(OUTPUT_SAMPLE_RATE * 1100 / 1000)
        expected_len = chunk_len + paragraph_silence_samples + chunk_len
        assert len(audio) == expected_len

    def test_no_markers_no_extra_pause(self):
        from indextts_mlx.synthesize_long import synthesize_long, OUTPUT_SAMPLE_RATE

        text = "Sentence one. Sentence two."

        mock_tts = MagicMock()
        chunk_len = int(OUTPUT_SAMPLE_RATE * 0.5)
        mock_tts.synthesize.return_value = np.ones(chunk_len, dtype=np.float32)

        with patch("indextts_mlx.synthesize_long.Segmenter", _make_mock_segmenter()):
            audio = synthesize_long(
                text,
                tts=mock_tts,
                spk_audio_prompt="speaker.wav",
                normalize=False,
                crossfade_ms=0,
                silence_between_chunks_ms=0,
            )

        # Single block, single chunk — just the audio, no extra pause
        assert mock_tts.synthesize.call_count == 1
        assert len(audio) == chunk_len

    def test_paragraph_pause_ms_config(self):
        from indextts_mlx.synthesize_long import LongSynthesisConfig

        cfg = LongSynthesisConfig(paragraph_pause_ms=2000)
        assert cfg.paragraph_pause_ms == 2000

        cfg_default = LongSynthesisConfig()
        assert cfg_default.paragraph_pause_ms == 1100


# ── 4. EPUB extractor: <p> tags produce paragraph boundaries ──────────────


class TestEPUBParagraphBoundaries:
    """Verify that consecutive <p> tags produce \\n\\n paragraph boundaries."""

    def test_p_tags_produce_double_newlines(self):
        from indextts_mlx.epub_extractor import EPUBParser

        html = "<html><body><p>First paragraph.</p><p>Second paragraph.</p></body></html>"
        # Use extract_text_from_html directly (no EPUB file needed)
        parser = EPUBParser.__new__(EPUBParser)
        text = parser.extract_text_from_html(html)
        assert "\n\n" in text, f"Expected \\n\\n between <p> tags, got: {text!r}"
        parts = text.split("\n\n")
        assert any("First paragraph." in p for p in parts)
        assert any("Second paragraph." in p for p in parts)

    def test_single_p_no_double_newline(self):
        from indextts_mlx.epub_extractor import EPUBParser

        html = "<html><body><p>Only one paragraph.</p></body></html>"
        parser = EPUBParser.__new__(EPUBParser)
        text = parser.extract_text_from_html(html)
        assert "\n\n" not in text
        assert "Only one paragraph." in text

    def test_three_p_tags(self):
        from indextts_mlx.epub_extractor import EPUBParser

        html = (
            "<html><body>"
            "<p>Para one.</p><p>Para two.</p><p>Para three.</p>"
            "</body></html>"
        )
        parser = EPUBParser.__new__(EPUBParser)
        text = parser.extract_text_from_html(html)
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        assert len(parts) == 3, f"Expected 3 paragraphs, got {len(parts)}: {parts}"
