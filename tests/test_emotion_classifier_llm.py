"""LLM-backed tests for EmotionClassifier rules.

These tests load the real model and assert on actual classifier output.
They are skipped by default — run with:

    pytest tests/test_emotion_classifier_llm.py --llm -v

Each test is paired with the prompt-content test it validates, and uses
fresh sentences that do NOT appear as examples in the prompt so we are
testing generalisation, not memorisation.
"""

import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _classify_pause(clf, sentence: str, context: str = "") -> int:
    ctx = context or sentence
    return clf._classify_one_pause(ctx, sentence)


def _classify_emotion(clf, sentence: str, context: str = "") -> int:
    ctx = context or sentence
    return clf._classify_one_emotion(ctx, sentence)


# ---------------------------------------------------------------------------
# Pause: attribution-tag rule
# Corresponds to: TestPausePromptAttributionTagRule
#
# Rule: sentences whose main verb is a speech verb (said, asked, replied, …)
# must receive short (1) or neutral (2) — never long (3) or dramatic (4).
#
# Test sentences are deliberately different from the prompt examples so we
# verify generalisation.
# ---------------------------------------------------------------------------


@pytest.mark.llm
class TestPauseLLMAttributionTag:
    """Attribution-tag sentences must never get long (3) or dramatic (4)."""

    @pytest.mark.parametrize("sentence", [
        # said
        '"We leave at dawn," the general said.',
        # whispered
        '"Be quiet," she whispered, glancing at the door.',
        # announced
        '"The vote has been counted," the chairman announced.',
        # called out
        '"Over here!" Marcus called out from across the street.',
        # replied
        '"I don\'t know what you mean," he replied stiffly.',
        # muttered
        '"This is hopeless," she muttered under her breath.',
        # shouted
        '"Get down!" Captain Torres shouted.',
        # continued — mid-speech attribution
        '"We have very little time left," he continued, his voice steady.',
    ])
    def test_attribution_tag_not_long_or_dramatic(self, classifier_llm, sentence):
        result = _classify_pause(classifier_llm, sentence)
        assert result <= 2, (
            f"Attribution tag got pause={result} (expected <=2/neutral)\n"
            f"  sentence: {sentence!r}"
        )


# ---------------------------------------------------------------------------
# Pause: long/dramatic reserved for structural transitions, not dialogue
# Corresponds to: TestPausePromptAttributionTagRule.test_long_excluded_for_attribution
# ---------------------------------------------------------------------------


@pytest.mark.llm
class TestPauseLLMLongReserved:
    """Long (3) and dramatic (4) should only fire on genuine structural breaks,
    not on attribution tags or ordinary mid-scene sentences."""

    @pytest.mark.parametrize("sentence,max_expected", [
        # Genuine structural break — long is fine
        ("The war ended that afternoon.", 4),
        # Section-ending revelation — long or dramatic fine
        ("Nothing would ever be the same again.", 4),
        # Ordinary prose — neutral at most
        ("He walked across the room and sat down.", 2),
        # Attribution tag — never long
        ('"I see," she said.', 2),
        ('"You\'re right," he admitted.', 2),
    ])
    def test_pause_within_expected_range(self, classifier_llm, sentence, max_expected):
        result = _classify_pause(classifier_llm, sentence)
        assert result <= max_expected, (
            f"Pause={result} exceeds max expected {max_expected}\n"
            f"  sentence: {sentence!r}"
        )


# ---------------------------------------------------------------------------
# Suspense: dialogue must never be suspense
# Corresponds to: TestSuspensePromptContent.test_dialogue_exclusion_present
# ---------------------------------------------------------------------------


@pytest.mark.llm
class TestEmotionLLMDialogueNotSuspense:
    """Sentences that are clearly dialogue must not be classified as suspense (4)."""

    @pytest.mark.parametrize("sentence", [
        # Calm robot/assistant dialogue about a threat
        '"Unit seven has been destroyed," the system reported.',
        # Character calmly stating danger
        '"The bridge is out," Elena said.',
        # Urgent but still quoted speech
        '"They\'re inside the building," he told her.',
        # Question inside quotes
        '"Where did everyone go?" Dani asked.',
        # Calm informational quote despite alarming content
        '"The building caught fire last night," the officer said.',
    ])
    def test_dialogue_not_suspense(self, classifier_llm, sentence):
        result = _classify_emotion(classifier_llm, sentence)
        assert result != 4, (
            f"Dialogue sentence classified as suspense (4)\n"
            f"  sentence: {sentence!r}"
        )


# ---------------------------------------------------------------------------
# Suspense: statements of fact / status reports must not be suspense
# Corresponds to: TestSuspensePromptContent.test_fact_report_exclusion_present
# ---------------------------------------------------------------------------


@pytest.mark.llm
class TestEmotionLLMFactNotSuspense:
    """Factual status-report sentences must not be classified as suspense (4),
    even when the content is alarming."""

    @pytest.mark.parametrize("sentence", [
        # Alarming fact delivered flatly
        "Three of the reactors had gone offline.",
        # Status report
        "The population of the city had fallen by half.",
        # Historical fact
        "The attack had destroyed the northern grid.",
        # Medical fact
        "Forty percent of the crew showed symptoms.",
    ])
    def test_status_report_not_suspense(self, classifier_llm, sentence):
        result = _classify_emotion(classifier_llm, sentence)
        assert result != 4, (
            f"Status-report sentence classified as suspense (4)\n"
            f"  sentence: {sentence!r}"
        )


# ---------------------------------------------------------------------------
# Suspense: physical movement/approach without explicit fear language
# Corresponds to: TestSuspensePromptContent.test_physical_approach_exclusion_present
# ---------------------------------------------------------------------------


@pytest.mark.llm
class TestEmotionLLMMovementNotSuspense:
    """Neutral physical movement descriptions must not be suspense (4)."""

    @pytest.mark.parametrize("sentence", [
        # Approach without horror language
        "The tall figure moved toward the window.",
        "A drone drifted slowly across the courtyard.",
        "The car pulled up alongside the curb.",
        # Exit/departure
        "He walked out of the room without a word.",
    ])
    def test_movement_not_suspense(self, classifier_llm, sentence):
        result = _classify_emotion(classifier_llm, sentence)
        assert result != 4, (
            f"Physical movement sentence classified as suspense (4)\n"
            f"  sentence: {sentence!r}"
        )


# ---------------------------------------------------------------------------
# Suspense: when in doubt the classifier should fall back to neutral/mild
# Corresponds to: TestSuspensePromptContent.test_doubt_fallback_to_neutral_present
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Positive emotion cases: non-neutral labels that SHOULD fire
# ---------------------------------------------------------------------------


@pytest.mark.llm
class TestEmotionLLMPositiveCases:
    """Verify the classifier actually assigns non-neutral labels when warranted.
    These are the complement of the over-triggering tests — we need to confirm
    the rules haven't suppressed everything into neutral.
    """

    @pytest.mark.parametrize("sentence,expected_label,label_name", [
        # mild_emphasis: structural punch sentence after buildup
        ("Admit it.", 1, "mild_emphasis"),
        # mild_emphasis: short isolated command
        ("Stop.", 1, "mild_emphasis"),
        # joyful: sustained happiness
        (
            "She laughed and threw her arms around him, tears of joy streaming down her face.",
            3,
            "joyful",
        ),
        # melancholic: genuine sorrow
        (
            "He stood alone at the graveside long after everyone else had gone home.",
            5,
            "melancholic",
        ),
        # suspense: pure narration, active unfolding moment of dread
        (
            "The footsteps grew louder, and the door handle slowly began to turn.",
            4,
            "suspense",
        ),
    ])
    def test_non_neutral_fires_when_warranted(self, classifier_llm, sentence, expected_label, label_name):
        result = _classify_emotion(classifier_llm, sentence)
        assert result == expected_label, (
            f"Expected {label_name} ({expected_label}), got {result}\n"
            f"  sentence: {sentence!r}"
        )


@pytest.mark.llm
class TestEmotionLLMAmbiguousFallback:
    """Ambiguous sentences that could go either way should not be suspense (4)."""

    @pytest.mark.parametrize("sentence", [
        # Tense situation but described analytically
        "The situation was becoming increasingly difficult to manage.",
        # Mentions danger but in retrospect
        "It had been a dangerous week.",
        # Vague unease — narrator would read flatly
        "Something felt off about the whole arrangement.",
    ])
    def test_ambiguous_not_suspense(self, classifier_llm, sentence):
        result = _classify_emotion(classifier_llm, sentence)
        assert result != 4, (
            f"Ambiguous sentence classified as suspense (4) — expected fallback to neutral/mild\n"
            f"  sentence: {sentence!r}"
        )
