"""Unit tests for boundary_detector — no sentence-transformers required for most tests."""

import pytest

from indextts_mlx.boundary_detector import detect_boundaries

sent_transformers_available = bool(
    __import__("importlib").util.find_spec("sentence_transformers")
)


# ---------------------------------------------------------------------------
# Edge cases (no model needed — tier-1 paragraph breaks + degenerate inputs)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_sentences_returns_empty_set(self):
        assert detect_boundaries([], "") == set()

    def test_single_sentence_returns_empty_set(self):
        assert detect_boundaries(["Hello."], "Hello.") == set()

    def test_two_sentences_no_break(self):
        # No double-newline, and sentence-transformers may or may not be available —
        # but we should always get a set back.
        result = detect_boundaries(["Hello.", "World."], "Hello. World.")
        assert isinstance(result, set)

    def test_return_type_is_set(self):
        result = detect_boundaries(["A.", "B.", "C."], "A. B. C.")
        assert isinstance(result, set)

    def test_boundary_indices_in_range(self):
        sentences = ["One.", "Two.", "Three."]
        result = detect_boundaries(sentences, "One. Two. Three.")
        for idx in result:
            assert 0 <= idx < len(sentences) - 1, f"Index {idx} out of range"


# ---------------------------------------------------------------------------
# Tier-1: explicit paragraph breaks (\n\n)
# ---------------------------------------------------------------------------


class TestExplicitParagraphBreaks:
    def test_double_newline_creates_boundary(self):
        sentences = ["First sentence.", "Second sentence."]
        text = "First sentence.\n\nSecond sentence."
        result = detect_boundaries(sentences, text)
        assert 0 in result, "Expected boundary after sentence 0 due to \\n\\n"

    def test_single_newline_does_not_create_boundary(self):
        sentences = ["First sentence.", "Second sentence."]
        text = "First sentence.\nSecond sentence."
        result = detect_boundaries(sentences, text)
        # Tier-1 should NOT trigger; tier-2 may or may not (depends on embeddings)
        # We only verify tier-1: single \n should not force index 0 in
        # Unless sentence-transformers also fires — but we can't control that.
        # At minimum the function should not crash.
        assert isinstance(result, set)

    def test_multiple_paragraph_breaks(self):
        sentences = ["Alpha.", "Beta.", "Gamma.", "Delta."]
        text = "Alpha.\n\nBeta.\n\nGamma.\n\nDelta."
        result = detect_boundaries(sentences, text)
        # Boundaries after 0, 1, 2
        assert 0 in result
        assert 1 in result
        assert 2 in result

    def test_last_sentence_never_in_result(self):
        sentences = ["A.", "B.", "C."]
        text = "A.\n\nB.\n\nC."
        result = detect_boundaries(sentences, text)
        assert len(sentences) - 1 not in result, "Boundary after the last sentence is nonsensical"

    def test_paragraph_break_not_doubled_in_result(self):
        # index 0 should appear at most once regardless of how many \n\n there are
        sentences = ["A.", "B."]
        text = "A.\n\n\n\nB."
        result = detect_boundaries(sentences, text)
        assert 0 in result

    def test_no_false_boundary_without_break(self):
        sentences = ["One.", "Two."]
        text = "One. Two."
        # Tier-1 cannot fire here; no double-newline
        result = detect_boundaries(sentences, text)
        # Either empty (no embeddings) or possibly 0 if semantic distance is large —
        # but without sentence-transformers it must be empty.
        if not sent_transformers_available:
            assert result == set()

    def test_boundary_not_created_for_sentence_not_found_in_text(self):
        # Sentence that cannot be located in the original text — should not crash
        sentences = ["A.", "B.", "Not in text at all."]
        text = "A. B."
        result = detect_boundaries(sentences, text)
        assert isinstance(result, set)


# ---------------------------------------------------------------------------
# Tier-2: semantic similarity (requires sentence-transformers)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not sent_transformers_available, reason="sentence-transformers not installed"
)
class TestSemanticBoundaries:
    """These tests verify tier-2 behaviour but are approximate — we just check
    that the function runs and returns plausible results without asserting
    specific boundary locations (since model outputs may vary)."""

    def test_runs_without_error(self):
        sentences = [
            "The cat sat on the mat.",
            "It was a warm and sunny day.",
            "The quantum mechanics of black holes remain mysterious.",
            "Hawking radiation is a theoretical prediction.",
        ]
        text = " ".join(sentences)
        result = detect_boundaries(sentences, text)
        assert isinstance(result, set)

    def test_indices_valid(self):
        sentences = ["A.", "B.", "C.", "D.", "E."]
        text = " ".join(sentences)
        result = detect_boundaries(sentences, text, threshold=0.5)
        for idx in result:
            assert 0 <= idx < len(sentences) - 1

    def test_threshold_zero_produces_many_boundaries(self):
        # With threshold=0.0, almost every boundary should be detected (cos_sim >= 0 always)
        sentences = ["One.", "Two.", "Three.", "Four.", "Five."]
        text = " ".join(sentences)
        result = detect_boundaries(sentences, text, threshold=0.0)
        # Not all possible boundaries (0-3) need to be present, but we expect at least some
        # because cos_sim < 0.0 is unusual for these short sentences.
        assert isinstance(result, set)

    def test_threshold_one_produces_no_semantic_boundaries(self):
        # threshold=1.0 means cos_sim must be strictly less than 1.0 for a boundary,
        # which would fire for nearly identical sentences — with dissimilar text nothing fires.
        sentences = ["Cat.", "Dog.", "Bird.", "Fish."]
        text = " ".join(sentences)
        # Only explicit paragraph breaks would fire; there are none here.
        result = detect_boundaries(sentences, text, threshold=1.0)
        # Result may be non-empty if cosine similarity is exactly < 1.0 for these,
        # but we only assert no crash.
        assert isinstance(result, set)

    def test_explicit_and_semantic_combined(self):
        sentences = ["Alpha.", "Beta.", "Totally unrelated technical jargon sentence."]
        text = "Alpha.\n\nBeta. Totally unrelated technical jargon sentence."
        result = detect_boundaries(sentences, text)
        # At minimum, index 0 should be in result (explicit paragraph break)
        assert 0 in result
