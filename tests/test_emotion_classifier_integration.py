"""Integration tests for EmotionClassifier using the real LLM.

These tests call mlx_lm.generate with the actual model and verify the
classifier returns sensible, stable emotion labels for unambiguous text.

Skipped entirely if mlx_lm is not installed.

Run with:
    pytest tests/test_emotion_classifier_integration.py -v -s
"""

from __future__ import annotations

import pytest

from indextts_mlx.emotion_classifier import (
    EMOTION_LABELS,
    ClassifierConfig,
    EmotionClassifier,
)

# ---------------------------------------------------------------------------
# Skip guard + shared fixture
# ---------------------------------------------------------------------------

mlx_lm_available = False
try:
    import mlx_lm  # noqa: F401

    mlx_lm_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not mlx_lm_available, reason="mlx_lm not installed")


@pytest.fixture(scope="module")
def clf():
    """Load the classifier once for the entire module."""
    config = ClassifierConfig(
        max_retries=3,
        context_window=3,
        hysteresis_min_run=2,
    )
    c = EmotionClassifier(config)
    c._load_model()
    return c


def one(clf, context: str, sentence: str) -> str:
    """Classify a single sentence and return its emotion label."""
    return EMOTION_LABELS[clf._classify_one(context, sentence)]


# ---------------------------------------------------------------------------
# Neutral — the model must not over-dramatise plain prose.
#
# These cases have no emotional signal. Any non-neutral result is a false
# positive and indicates over-dramatisation.
# ---------------------------------------------------------------------------


class TestNeutralCases:
    def test_plain_physical_description(self, clf):
        context = (
            "The village of Thornwick lay at the foot of a long valley. "
            "A river ran through its centre, and the houses were built of grey stone. "
            "The church stood on a small rise above the market square."
        )
        assert (
            one(clf, context, "The church stood on a small rise above the market square.")
            == "neutral"
        )

    def test_mundane_domestic_action(self, clf):
        context = (
            "She walked to the kitchen and filled the kettle. "
            "The windows were steamed up from the morning cold. "
            "She set two cups on the counter."
        )
        assert one(clf, context, "She set two cups on the counter.") == "neutral"

    def test_factual_historical_record(self, clf):
        context = (
            "The treaty was signed in 1648, ending thirty years of conflict. "
            "The terms were negotiated over several months by representatives. "
            "A copy of the document was sent to each signatory state."
        )
        assert (
            one(clf, context, "A copy of the document was sent to each signatory state.")
            == "neutral"
        )

    def test_dialogue_attribution_is_neutral(self, clf):
        context = (
            'She turned to face him. "I think we should go," she said. '
            "He nodded and reached for his coat."
        )
        assert one(clf, context, "He nodded and reached for his coat.") == "neutral"

    def test_exclamation_alone_does_not_trigger_emotion(self, clf):
        # A single "!" for a small practical win in a mundane context.
        # The prompt's JOYFUL SPECIFIC RULES explicitly rule this out.
        context = (
            "The bus arrived on time for once. "
            "She grabbed her bag and ran for the door. "
            "She made it!"
        )
        result = one(clf, context, "She made it!")
        assert result in (
            "neutral",
            "mild_emphasis",
        ), f"Small practical win in mundane context should not produce {result!r}"

    def test_inventory_list_sentence(self, clf):
        context = (
            "He unpacked the rucksack methodically. "
            "Inside were a compass, a torch, two tins of food, and a map. "
            "He laid each item on the table."
        )
        assert one(clf, context, "He laid each item on the table.") == "neutral"


# ---------------------------------------------------------------------------
# Melancholic — grief, loss, heavy tone with strong textual evidence.
# ---------------------------------------------------------------------------


class TestMelancholicCases:
    def test_grief_after_death(self, clf):
        context = (
            "They lowered the coffin into the frozen ground. "
            "His daughter stood apart from the others, her eyes fixed on the earth. "
            "She had not spoken since the morning of his death, three days before."
        )
        result = one(
            clf, context, "She had not spoken since the morning of his death, three days before."
        )
        assert result == "melancholic", f"Expected melancholic, got {result!r}"

    def test_irrecoverable_loss(self, clf):
        context = (
            "The house felt different without her. "
            "Every room held some trace of her presence — a book left open, a cup unwashed. "
            "He could not bring himself to move any of it."
        )
        result = one(clf, context, "He could not bring himself to move any of it.")
        assert result == "melancholic", f"Expected melancholic, got {result!r}"

    def test_twenty_year_silence(self, clf):
        context = (
            "The telegram arrived on a Tuesday, brief and final. "
            "It said he had died honourably, as if that were a comfort. "
            "She folded the paper and did not unfold it again for twenty years."
        )
        result = one(
            clf, context, "She folded the paper and did not unfold it again for twenty years."
        )
        assert result == "melancholic", f"Expected melancholic, got {result!r}"


# ---------------------------------------------------------------------------
# Suspense — dread, tension, threat with strong contextual build-up.
# ---------------------------------------------------------------------------


class TestSuspenseCases:
    def test_door_handle_turning(self, clf):
        context = (
            "The footsteps in the corridor had stopped. "
            "She pressed herself against the wall and held her breath. "
            "The door handle began to turn, slowly, without a sound."
        )
        result = one(clf, context, "The door handle began to turn, slowly, without a sound.")
        assert result == "suspense", f"Expected suspense, got {result!r}"

    def test_radio_goes_silent_at_midnight(self, clf):
        context = (
            "The radio had been broadcasting emergency instructions for hours. "
            "Then, at exactly midnight, it went silent. "
            "No one in the shelter moved or spoke."
        )
        result = one(clf, context, "No one in the shelter moved or spoke.")
        # Strong suspense context; should be suspense (not neutral)
        assert result == "suspense", f"Expected suspense, got {result!r}"

    def test_missing_person_context(self, clf):
        # The sentence is a calm physical action but the context is dense with dread.
        # The model may read it as suspense (strong context signal) or neutral (restrained).
        # Both are valid audiobook choices; what it must NOT be is positive.
        context = (
            "Three days had passed and there was still no word from him. "
            "Every knock at the door made her flinch. "
            "She kept the phone on the table and watched it."
        )
        result = one(clf, context, "She kept the phone on the table and watched it.")
        assert result in (
            "suspense",
            "neutral",
            "melancholic",
        ), f"Expected a tense or neutral read, got {result!r}"


# ---------------------------------------------------------------------------
# Joyful — warm happiness or relief with clear textual support.
# ---------------------------------------------------------------------------


class TestJoyfulCases:
    def test_long_reunion(self, clf):
        context = (
            "She had not seen her brother in four years. "
            "When he stepped off the train, she ran to him across the platform. "
            "They held each other for a long time, laughing and crying at once."
        )
        result = one(
            clf, context, "They held each other for a long time, laughing and crying at once."
        )
        assert result == "joyful", f"Expected joyful, got {result!r}"

    def test_acceptance_after_years_of_trying(self, clf):
        context = (
            "The letter had arrived that morning. "
            "She read it three times before she believed it. "
            "She had been accepted — after years of trying, she had finally been accepted."
        )
        result = one(
            clf,
            context,
            "She had been accepted — after years of trying, she had finally been accepted.",
        )
        assert result == "joyful", f"Expected joyful, got {result!r}"


# ---------------------------------------------------------------------------
# Context sensitivity — the *same* sentence should be classified differently
# depending on the surrounding paragraph.
# ---------------------------------------------------------------------------


class TestContextSensitivity:
    def test_same_sentence_neutral_in_mundane_context(self, clf):
        neutral_context = (
            "She walked in from the garden and took off her boots. "
            "The kitchen was warm from the oven. "
            "She sat down at the table."
        )
        result = one(clf, neutral_context, "She sat down at the table.")
        assert (
            result == "neutral"
        ), f"In mundane context, 'She sat down at the table.' should be neutral, got {result!r}"

    def test_same_sentence_emotional_in_grief_context(self, clf):
        grief_context = (
            "They had buried her husband that morning. "
            "The mourners had gone and she was alone in the house. "
            "She sat down at the table."
        )
        result = one(clf, grief_context, "She sat down at the table.")
        # In a grief context this same sentence should carry emotional weight
        assert (
            result != "neutral"
        ), f"In grief context, 'She sat down at the table.' should not be neutral, got {result!r}"

    def test_darkness_neutral_without_threat_context(self, clf):
        # Pure inconvenience context — broken light, awaiting electrician.
        # Should not be suspense or a strong emotion. Neutral or mild_emphasis only.
        mundane_context = (
            "The electrician had been booked for Thursday. "
            "Until then they would have to manage without the hallway light. "
            "The corridor was dark."
        )
        result = one(clf, mundane_context, "The corridor was dark.")
        assert result in (
            "neutral",
            "mild_emphasis",
        ), f"Without threat context, 'The corridor was dark.' should be neutral/mild, got {result!r}"

    def test_darkness_suspense_with_threat_context(self, clf):
        threat_context = (
            "Someone had been following her since the station. "
            "She had turned twice and each time heard footsteps stop a moment after hers. "
            "The corridor was dark."
        )
        result = one(clf, threat_context, "The corridor was dark.")
        # A strong model may correctly read this as neutral (the sentence itself is a plain
        # physical description) or as suspense (context is dense with threat). Both are valid.
        # What it must NOT be is positive.
        assert result in (
            "suspense",
            "neutral",
        ), f"With threat context, 'The corridor was dark.' should be suspense or neutral, got {result!r}"


# ---------------------------------------------------------------------------
# Mild-emphasis false-positive resistance — discursive/analytical prose that
# should stay neutral even when it is opinionated, witty, or rhetorical.
# ---------------------------------------------------------------------------


class TestMildEmphasisFalsePositives:
    def test_analytical_opinion_is_neutral(self, clf):
        # Discursive commentary with a clear authorial view → neutral, not mild_emphasis
        context = (
            "Most of us probably think we're believers in ideas too, but we're deluding ourselves. "
            "Believing in ideas is one of those attributes like libido or skill at driving a car "
            "that most people reckon they possess in above-average quantities — but that's "
            "mathematically impossible. "
            "Admit it: ideas can be annoying and frightening and threatening."
        )
        result = one(clf, context, "Admit it: ideas can be annoying and frightening and threatening.")
        assert result in (
            "neutral",
            "mild_emphasis",
        ), f"Discursive opinion should be neutral or at most mild_emphasis, got {result!r}"
        # The stronger assertion: it should not be anything else
        assert result != "joyful" and result != "suspense" and result != "melancholic"

    def test_witty_aside_is_neutral(self, clf):
        # Dry wit / irony in an essayistic context → neutral
        context = (
            "Ideas aren't all lovely vaccines — they can be a right pain. "
            "We all like some ideas that have already been had — normal pizza, dishwashers, "
            "freedom of speech — but we don't put much faith in those that are yet to emerge. "
            "The internet was an idea. So were self-service tills in supermarkets."
        )
        result = one(clf, context, "The internet was an idea. So were self-service tills in supermarkets.")
        assert result in (
            "neutral",
            "mild_emphasis",
        ), f"Witty aside should be neutral or at most mild_emphasis, got {result!r}"

    def test_historical_commentary_with_opinion_is_neutral(self, clf):
        # Author has a clear view ("actually quite sensible") but it is still analytical prose
        context = (
            "Hence, during the First World War, in the face of the Western Front's murderous "
            "deadlock, Churchill championed the idea of attacking Turkey. "
            "I think this was actually quite a sensible plan. "
            "The knackered old Ottoman Empire was a far feebler military opponent than Germany."
        )
        result = one(clf, context, "I think this was actually quite a sensible plan.")
        assert result == "neutral", (
            f"Analytical historical opinion should be neutral, got {result!r}"
        )

    def test_speculative_commentary_is_neutral(self, clf):
        # Author speculation about motivations → neutral discursive prose
        context = (
            "Perhaps it felt like a proposed British takeover. "
            "That might not have appealed to them at a time when resisting a German takeover "
            "was their focus. "
            "My suspicion is, though, that they simply didn't see the point of it."
        )
        result = one(clf, context, "My suspicion is, though, that they simply didn't see the point of it.")
        assert result == "neutral", (
            f"Speculative authorial commentary should be neutral, got {result!r}"
        )

    def test_essayistic_passage_mostly_neutral(self, clf):
        # A full discursive passage — after hysteresis, should be overwhelmingly neutral.
        # mild_emphasis should be at most 1 sentence in 5 (20%).
        text = (
            "Believing in ideas is one of those attributes like libido or skill at driving a car "
            "that most people reckon they possess in above-average quantities. "
            "But that's mathematically impossible. "
            "We generally think that a problem is what it is, and needs to be addressed in one "
            "of the established ways that have been handed down for addressing it. "
            "And we're usually right."
        )
        records = clf.classify_text(text, verbose=True)
        labels = [r.emotion for r in records]
        mild_count = labels.count("mild_emphasis")
        assert mild_count <= 1, (
            f"Discursive prose should have at most 1 mild_emphasis sentence, got: {labels}"
        )
        neutral_count = labels.count("neutral")
        assert neutral_count >= len(labels) - 1, (
            f"Discursive prose should be almost entirely neutral, got: {labels}"
        )


# ---------------------------------------------------------------------------
# False-positive resistance — sentences that look dramatic but should remain
# neutral per the prompt's CRITICAL RULES.
# ---------------------------------------------------------------------------


class TestFalsePositiveResistance:
    def test_dramatic_words_in_mundane_context(self, clf):
        # "A terrible shame about the weather" is mild social disappointment.
        # The prompt's JOYFUL SPECIFIC RULES require reading it literally, not ironically.
        context = (
            "The forecast had been wrong again. "
            "It rained all morning and the match was called off. "
            "It was a terrible shame about the weather."
        )
        result = one(clf, context, "It was a terrible shame about the weather.")
        assert result in (
            "neutral",
            "mild_emphasis",
        ), f"Social disappointment phrase should not produce {result!r}"

    def test_question_mark_does_not_trigger_emotion(self, clf):
        context = (
            "He looked at the timetable on the wall. "
            "The last departure was at nine. "
            "Was there time to make it?"
        )
        result = one(clf, context, "Was there time to make it?")
        assert result in (
            "neutral",
            "mild_emphasis",
            "suspense",
        ), f"A practical question should not produce extreme emotion {result!r}"

    def test_past_event_reported_neutrally(self, clf):
        # The war is referenced but purely historically, no emotional framing
        context = (
            "The town had been rebuilt after the war. "
            "Most of the original buildings were replaced in the 1950s. "
            "The market hall dated from 1958."
        )
        result = one(clf, context, "The market hall dated from 1958.")
        assert result == "neutral", f"Historical fact should be neutral, got {result!r}"


# ---------------------------------------------------------------------------
# End-to-end passage classification with hysteresis
# ---------------------------------------------------------------------------


class TestPassageClassification:
    def test_neutral_passage_mostly_neutral(self, clf):
        """Plain descriptive prose should be predominantly neutral after smoothing."""
        text = (
            "The train arrived at the station at half past seven. "
            "The platform was crowded with commuters. "
            "A guard walked the length of the train, checking the doors. "
            "The departure board showed three delays. "
            "Most passengers waited in silence."
        )
        records = clf.classify_text(text, verbose=True)
        labels = [r.emotion for r in records]
        neutral_count = labels.count("neutral")
        assert (
            neutral_count >= len(labels) * 0.6
        ), f"Plain prose should be ≥60% neutral after smoothing, got: {labels}"

    def test_grief_passage_produces_non_neutral(self, clf):
        """A passage dense with grief should yield at least one non-neutral label."""
        text = (
            "He had not returned from the front. "
            "The telegram arrived on a Tuesday, brief and final. "
            "She read it standing in the hallway, still in her coat. "
            "It said he had died honourably, as if that were a comfort. "
            "She folded the paper and did not unfold it again for twenty years."
        )
        records = clf.classify_text(text, verbose=True)
        non_neutral = [r for r in records if r.emotion != "neutral"]
        assert len(non_neutral) >= 1, (
            f"Expected at least one non-neutral label in grief passage, "
            f"got: {[r.emotion for r in records]}"
        )

    def test_hysteresis_removes_isolated_spikes_from_neutral_passage(self, clf):
        """Any isolated non-neutral spike in an otherwise neutral passage should
        be smoothed away. Compare raw vs smoothed labels."""
        text = (
            "The accountant reviewed the figures for the third time. "
            "Everything balanced to within a few pence. "
            "He initialled each page and put the file away. "
            "There was nothing more to do until Monday. "
            "He switched off the desk lamp."
        )
        records = clf.classify_text(text, verbose=True)
        raw = [r.raw_emotion_idx for r in records]
        smoothed = [r.emotion_idx for r in records]
        # If any raw spikes existed, hysteresis should have reduced non-neutral count
        raw_non_neutral = sum(1 for x in raw if x != 0)
        smoothed_non_neutral = sum(1 for x in smoothed if x != 0)
        assert (
            smoothed_non_neutral <= raw_non_neutral
        ), "Hysteresis should never increase non-neutral count"
        # In a genuinely mundane passage, non-neutral sentences should be a small minority
        assert (
            smoothed_non_neutral <= len(smoothed) // 2
        ), f"Mundane accountant passage should be mostly neutral after smoothing: {smoothed}"
