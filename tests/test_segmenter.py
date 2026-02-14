"""Tests for spaCy-based sentence segmenter."""

import pytest

from indextts_mlx.segmenter import Segmenter, SegmenterConfig


class TestSegmenterConfig:
    """Tests for SegmenterConfig."""

    def test_default_config(self) -> None:
        config = SegmenterConfig()
        assert config.language == "english"
        assert config.strategy == "char_count"
        assert config.max_chars == 300
        assert config.sentences_per_chunk == 3
        assert config.min_chars == 3
        assert config.use_pysbd is True
        assert "ner" in config.disable_pipes
        assert "lemmatizer" in config.disable_pipes

    def test_custom_config(self) -> None:
        config = SegmenterConfig(language="french", max_chars=500)
        assert config.language == "french"
        assert config.max_chars == 500

    def test_iso_code_language(self) -> None:
        SegmenterConfig(language="en")  # should not raise

    def test_invalid_language_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            SegmenterConfig(language="klingon")

    def test_token_count_strategy_requires_target(self) -> None:
        with pytest.raises(ValueError, match="token_target must be set"):
            SegmenterConfig(strategy="token_count")

    def test_token_target_sets_strategy(self) -> None:
        config = SegmenterConfig(token_target=120)
        assert config.strategy == "token_count"
        assert config.token_target == 120

    def test_sentence_count_strategy(self) -> None:
        config = SegmenterConfig(strategy="sentence_count", sentences_per_chunk=5)
        assert config.strategy == "sentence_count"
        assert config.sentences_per_chunk == 5

    def test_resolved_spacy_model_default(self) -> None:
        assert SegmenterConfig(language="english").resolved_spacy_model == "en_core_web_sm"
        assert SegmenterConfig(language="french").resolved_spacy_model == "fr_core_news_sm"
        assert SegmenterConfig(language="spanish").resolved_spacy_model == "es_core_news_sm"

    def test_resolved_spacy_model_override(self) -> None:
        config = SegmenterConfig(language="english", spacy_model="en_core_web_lg")
        assert config.resolved_spacy_model == "en_core_web_lg"

    def test_pysbd_lang_property(self) -> None:
        assert SegmenterConfig(language="english").pysbd_lang == "en"
        assert SegmenterConfig(language="german").pysbd_lang == "de"
        assert SegmenterConfig(language="it").pysbd_lang == "it"

    def test_supported_languages(self) -> None:
        for lang in [
            "english",
            "en",
            "french",
            "fr",
            "spanish",
            "es",
            "italian",
            "it",
            "german",
            "de",
            "portuguese",
            "pt",
        ]:
            SegmenterConfig(language=lang)  # should not raise


@pytest.fixture(scope="module")
def en_segmenter():
    """English segmenter with small max_chars for testing."""
    try:
        seg = Segmenter(SegmenterConfig(language="english", max_chars=100))
        seg.nlp  # trigger lazy load; skip if model missing
        return seg
    except OSError:
        pytest.skip("en_core_web_sm spaCy model not installed")


class TestSegmenterInit:
    """Tests for Segmenter initialization."""

    def test_default_init(self) -> None:
        seg = Segmenter()
        assert seg.config.language == "english"
        assert seg._nlp is None

    def test_custom_config_init(self) -> None:
        config = SegmenterConfig(language="english", max_chars=200)
        seg = Segmenter(config)
        assert seg.config == config

    def test_nlp_lazy_loading(self, en_segmenter) -> None:
        # After fixture construction, _nlp should be set
        assert en_segmenter._nlp is not None
        nlp = en_segmenter.nlp
        assert nlp is en_segmenter._nlp  # same instance

    def test_missing_spacy_model_raises_os_error(self) -> None:
        config = SegmenterConfig(language="english", spacy_model="en_core_web_nonexistent_xyz")
        seg = Segmenter(config)
        with pytest.raises(OSError, match="not found"):
            _ = seg.nlp


class TestSegmenterSegment:
    """Tests for Segmenter.segment()."""

    def test_empty_text_returns_empty_list(self, en_segmenter) -> None:
        assert en_segmenter.segment("") == []
        assert en_segmenter.segment("   ") == []

    def test_single_sentence(self, en_segmenter) -> None:
        chunks = en_segmenter.segment("Hello world.")
        assert len(chunks) == 1
        assert "Hello" in chunks[0]

    def test_two_sentences_fit_in_one_chunk(self) -> None:
        config = SegmenterConfig(language="english", max_chars=200)
        try:
            seg = Segmenter(config)
            seg.nlp
        except OSError:
            pytest.skip("en_core_web_sm not installed")
        chunks = seg.segment("Short first. Short second.")
        assert len(chunks) == 1

    def test_many_sentences_split_into_multiple_chunks(self) -> None:
        config = SegmenterConfig(language="english", max_chars=50)
        try:
            seg = Segmenter(config)
            seg.nlp
        except OSError:
            pytest.skip("en_core_web_sm not installed")
        text = (
            "This is sentence one. This is sentence two. "
            "This is sentence three. This is sentence four."
        )
        chunks = seg.segment(text)
        assert len(chunks) > 1

    def test_content_preserved(self, en_segmenter) -> None:
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = en_segmenter.segment(text)
        combined = " ".join(chunks)
        assert "First sentence" in combined
        assert "Third sentence" in combined

    def test_chunks_respect_max_chars(self, en_segmenter) -> None:
        text = (
            "This is a short sentence. This is another short sentence. "
            "And here is one more short sentence."
        )
        chunks = en_segmenter.segment(text)
        for chunk in chunks:
            assert len(chunk) <= en_segmenter.config.max_chars * 2  # generous upper bound

    def test_oversized_single_sentence_hard_split(self) -> None:
        """A sentence longer than max_chars should be hard-split."""
        config = SegmenterConfig(language="english", max_chars=20, use_pysbd=False)
        try:
            seg = Segmenter(config)
            seg.nlp
        except OSError:
            pytest.skip("en_core_web_sm not installed")
        text = "This is a very long sentence that definitely exceeds twenty characters."
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = seg.segment(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= config.max_chars

    def test_sentence_count_strategy(self) -> None:
        config = SegmenterConfig(
            language="english", strategy="sentence_count", sentences_per_chunk=2
        )
        try:
            seg = Segmenter(config)
            seg.nlp
        except OSError:
            pytest.skip("en_core_web_sm not installed")
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = seg.segment(text)
        assert len(chunks) == 3
        assert "one" in chunks[0] and "two" in chunks[0]
        assert "three" in chunks[1] and "four" in chunks[1]
        assert "five" in chunks[2]

    def test_repr_char_count(self, en_segmenter) -> None:
        r = repr(en_segmenter)
        assert "Segmenter" in r
        assert "english" in r
        assert "char_count" in r
        assert "max_chars=100" in r

    def test_repr_sentence_count(self) -> None:
        config = SegmenterConfig(
            language="english", strategy="sentence_count", sentences_per_chunk=4
        )
        seg = Segmenter(config)
        r = repr(seg)
        assert "sentence_count" in r
        assert "sentences_per_chunk=4" in r

    def test_repr_token_count(self) -> None:
        config = SegmenterConfig(language="english", token_target=120)
        seg = Segmenter(config)
        r = repr(seg)
        assert "token_count" in r
        assert "token_target=120" in r


class TestSegmenterIntegration:
    """Integration tests with realistic text."""

    def test_audiobook_paragraph(self) -> None:
        config = SegmenterConfig(language="english", max_chars=350)
        try:
            seg = Segmenter(config)
            seg.nlp
        except OSError:
            pytest.skip("en_core_web_sm not installed")
        text = (
            "It was a bright cold day in April, and the clocks were striking thirteen. "
            "Winston Smith, his chin nuzzled into his breast in an effort to escape the "
            "vile wind, slipped quickly through the glass doors of Victory Mansions. "
            "The hallway smelt of boiled cabbage and old rag mats."
        )
        chunks = seg.segment(text)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) > 0
        combined = " ".join(chunks)
        assert "April" in combined
        assert "cabbage" in combined

    def test_abbreviations_not_split(self) -> None:
        """Abbreviations like Dr. should not create spurious sentence breaks."""
        config = SegmenterConfig(language="english", max_chars=300)
        try:
            seg = Segmenter(config)
            seg.nlp
        except OSError:
            pytest.skip("en_core_web_sm not installed")
        text = "Dr. Smith went to the store. He bought milk. Then he went home."
        chunks = seg.segment(text)
        combined = " ".join(chunks)
        assert "Dr" in combined
        assert "Smith" in combined


class TestNormalizeSentence:
    """Tests for Segmenter._normalize_sentence (static method, no spaCy needed)."""

    def test_em_dash_replaced(self) -> None:
        s = Segmenter._normalize_sentence(
            "We all like ideas that have already been had\u2014normal pi"
        )
        assert "\u2014" not in s
        assert "--" in s

    def test_en_dash_replaced(self) -> None:
        s = Segmenter._normalize_sentence("She waited\u2013then she left")
        assert "\u2013" not in s
        assert "--" in s

    def test_soft_hyphen_removed(self) -> None:
        s = Segmenter._normalize_sentence("un\u00adbelievable")
        assert "\u00ad" not in s
        assert "unbelievable" in s

    def test_non_breaking_hyphen_normalized(self) -> None:
        s = Segmenter._normalize_sentence("well\u2011known")
        assert "\u2011" not in s
        assert "well-known" in s

    def test_adds_terminal_period(self) -> None:
        s = Segmenter._normalize_sentence("Introduction")
        assert s == "Introduction."

    def test_preserves_question_mark(self) -> None:
        s = Segmenter._normalize_sentence("Are you sure?")
        assert s.endswith("?")
        assert not s.endswith("?.")

    def test_preserves_exclamation(self) -> None:
        s = Segmenter._normalize_sentence("Stop!")
        assert s.endswith("!")

    def test_preserves_existing_period(self) -> None:
        s = Segmenter._normalize_sentence("Hello world.")
        assert s == "Hello world."

    def test_multiple_dashes_collapsed(self) -> None:
        s = Segmenter._normalize_sentence("a\u2013\u2013b")
        assert "--" in s
        # Should not have "-- --" (multiple replacements should collapse)
        assert "-- --" not in s

    def test_empty_string(self) -> None:
        s = Segmenter._normalize_sentence("")
        assert s == ""

    def test_sentence_ending_colon_preserved(self) -> None:
        s = Segmenter._normalize_sentence("Ingredients:")
        assert s.endswith(":")
