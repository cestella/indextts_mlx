"""Tests for NeMo-based text normalizer."""

import pytest

from indextts_mlx.normalizer import Normalizer, NormalizerConfig


class TestNormalizerConfig:
    """Tests for NormalizerConfig."""

    def test_default_config(self) -> None:
        config = NormalizerConfig()
        assert config.language == "english"
        assert config.input_case == "cased"
        assert config.cache_dir is None
        assert config.verbose is False

    def test_custom_config(self) -> None:
        config = NormalizerConfig(
            language="french",
            input_case="lower_cased",
            verbose=True,
        )
        assert config.language == "french"
        assert config.input_case == "lower_cased"
        assert config.verbose is True

    def test_iso_code_language(self) -> None:
        config = NormalizerConfig(language="en")
        assert config.nemo_code == "en"

    def test_nemo_code_property(self) -> None:
        assert NormalizerConfig(language="english").nemo_code == "en"
        assert NormalizerConfig(language="french").nemo_code == "fr"
        assert NormalizerConfig(language="spanish").nemo_code == "es"
        assert NormalizerConfig(language="italian").nemo_code == "it"
        assert NormalizerConfig(language="german").nemo_code == "de"

    def test_invalid_language_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            NormalizerConfig(language="klingon")

    def test_invalid_input_case_raises_error(self) -> None:
        with pytest.raises(ValueError, match="must be 'cased' or 'lower_cased'"):
            NormalizerConfig(input_case="UPPER")

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
            "russian",
            "ru",
            "vietnamese",
            "vi",
        ]:
            NormalizerConfig(language=lang)  # should not raise


class TestNormalizerNoNemo:
    """Tests that work even when NeMo is not installed."""

    def test_init(self) -> None:
        norm = Normalizer()
        assert norm.config.language == "english"
        assert norm._nemo is None
        assert norm._unavailable is False

    def test_init_with_config(self) -> None:
        config = NormalizerConfig(language="french")
        norm = Normalizer(config)
        assert norm.config == config

    def test_normalize_empty_string(self) -> None:
        norm = Normalizer()
        assert norm.normalize("") == ""
        assert norm.normalize("   ") == "   "

    def test_normalize_returns_str_when_nemo_unavailable(self) -> None:
        """If NeMo is not installed, normalize() returns original text."""
        norm = Normalizer()
        text = "The price is $42 today."
        result = norm.normalize(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_available_property(self) -> None:
        norm = Normalizer()
        # available is a bool regardless of whether NeMo is installed
        assert isinstance(norm.available, bool)

    def test_repr(self) -> None:
        norm = Normalizer(NormalizerConfig(language="spanish"))
        r = repr(norm)
        assert "Normalizer" in r
        assert "spanish" in r
        assert "cased" in r


@pytest.mark.skipif(
    not Normalizer().available,
    reason="nemo_text_processing not installed",
)
class TestNormalizerWithNemo:
    """Tests that require NeMo to be installed."""

    @pytest.fixture(scope="class")
    def normalizer(self):
        return Normalizer(NormalizerConfig(language="english"))

    def test_normalize_currency(self, normalizer) -> None:
        text = "The price is $123.45"
        result = normalizer.normalize(text)
        assert "dollar" in result.lower()

    def test_normalize_numbers(self, normalizer) -> None:
        text = "I have 42 apples"
        result = normalizer.normalize(text)
        assert "42" not in result or "forty" in result.lower()

    def test_normalize_plain_text_unchanged(self, normalizer) -> None:
        text = "This is a simple sentence"
        result = normalizer.normalize(text)
        assert "simple" in result.lower()
        assert "sentence" in result.lower()

    def test_normalize_preserves_structure(self, normalizer) -> None:
        text = "Line one.\nLine two."
        result = normalizer.normalize(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalize_audiobook_text(self, normalizer) -> None:
        text = (
            "Dr. Smith arrived at 3:30pm with $1,234.56 in his pocket. "
            "He had exactly 42 reasons to be there."
        )
        result = normalizer.normalize(text)
        assert isinstance(result, str)
        assert len(result) >= len(text)
        assert "dollar" in result.lower() or "cent" in result.lower()
