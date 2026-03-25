"""Tests for signal extractors — register shift detection between casual and technical modes."""

import json
import time
from pathlib import Path

import pytest

from ghost_sense.extractor.register import RegisterExtractor
from ghost_sense.models import SignalType

FIXTURES = Path(__file__).parent / "fixtures" / "sample_messages.json"


@pytest.fixture
def extractor():
    return RegisterExtractor()


@pytest.fixture
def sample_messages():
    return json.loads(FIXTURES.read_text())


def _metadata(ts=None):
    return {
        "timestamp": ts or time.time(),
        "local_time": "14:00",
        "message_index_in_session": 0,
        "time_since_last_message_seconds": None,
    }


class TestRegisterExtractor:
    def test_returns_all_dimensions(self, extractor):
        events = extractor.extract("hello world this is a test", _metadata())
        dims = {e.dimension for e in events}
        expected_dims = {
            "register.avg_word_length",
            "register.slang_ratio",
            "register.sentence_count",
            "register.punctuation_density",
            "register.capitalization_ratio",
            "register.emoji_density",
            "register.formality_score",
        }
        assert dims == expected_dims

    def test_all_events_are_register_type(self, extractor):
        events = extractor.extract("testing signal types", _metadata())
        assert all(e.signal_type == SignalType.REGISTER for e in events)

    def test_empty_message_returns_nothing(self, extractor):
        events = extractor.extract("", _metadata())
        assert events == []

    def test_fixture_messages(self, extractor, sample_messages):
        """Validate all fixture messages against their expected ranges."""
        for msg in sample_messages:
            events = extractor.extract(msg["text"], _metadata())
            event_map = {e.dimension: e.value for e in events}

            for dim, bounds in msg["expected"].items():
                assert dim in event_map, f"[{msg['id']}] missing dimension {dim}"
                val = event_map[dim]
                assert bounds["min"] <= val <= bounds["max"], (
                    f"[{msg['id']}] {dim} = {val:.3f}, expected [{bounds['min']}, {bounds['max']}]"
                )

    def test_casual_vs_technical_register_shift(self, extractor):
        """Core test: casual and technical messages should produce clearly different formality scores."""
        casual = extractor.extract("yo bruh ngl that was lowkey mid tbh lol", _metadata())
        technical = extractor.extract(
            "The implementation leverages exponential decay with configurable parameters "
            "to maintain temporal coherence across the state vector dimensions.",
            _metadata(),
        )

        casual_formality = next(e.value for e in casual if e.dimension == "register.formality_score")
        technical_formality = next(e.value for e in technical if e.dimension == "register.formality_score")

        # Technical should be meaningfully more formal
        assert technical_formality - casual_formality > 0.2, (
            f"Formality gap too small: technical={technical_formality:.3f}, casual={casual_formality:.3f}"
        )

    def test_slang_detection(self, extractor):
        events = extractor.extract("ngl bruh idk tbh lmao", _metadata())
        slang = next(e for e in events if e.dimension == "register.slang_ratio")
        assert slang.value >= 0.8  # nearly all slang

    def test_no_slang_in_formal(self, extractor):
        events = extractor.extract(
            "The architectural considerations necessitate a comprehensive review of the subsystem interfaces.",
            _metadata(),
        )
        slang = next(e for e in events if e.dimension == "register.slang_ratio")
        assert slang.value == 0.0

    def test_emoji_detection(self, extractor):
        events = extractor.extract("nice one :) love it :D XD", _metadata())
        emoji = next(e for e in events if e.dimension == "register.emoji_density")
        assert emoji.value > 0.0


class TestRegisterWithAccumulator:
    """End-to-end: message -> extractor -> bus -> state store -> readable state."""

    def test_end_to_end_pipeline(self, tmp_path):
        from ghost_sense.main import GhostSense


        db_path = str(tmp_path / "test.db")
        engine = GhostSense(db_path=db_path)

        # Process a casual message
        state = engine.process_message("yo bruh ngl that was mid lmao")
        assert "register.formality_score" in state.fields
        assert "register.slang_ratio" in state.fields

        casual_formality = state.fields["register.formality_score"].value

        # Process a technical message
        state = engine.process_message(
            "The implementation requires careful consideration of the exponential decay parameters "
            "to ensure temporal coherence in the state accumulation pipeline."
        )
        technical_formality = state.fields["register.formality_score"].value

        # State should shift toward more formal after technical message
        assert technical_formality > casual_formality

        engine.close()

    def test_state_persists_across_instances(self, tmp_path):
        from ghost_sense.main import GhostSense

        db_path = str(tmp_path / "persist.db")

        engine1 = GhostSense(db_path=db_path)
        engine1.process_message("yo what up dawg lol")
        engine1.close()

        engine2 = GhostSense(db_path=db_path)
        state = engine2.get_state()
        assert len(state.fields) > 0
        engine2.close()

    def test_events_stored_in_db(self, tmp_path):
        from ghost_sense.main import GhostSense

        db_path = str(tmp_path / "events.db")
        engine = GhostSense(db_path=db_path)
        engine.process_message("testing event storage bruh")

        events = engine.store.get_events(dimension="register.slang_ratio")
        assert len(events) >= 1
        assert events[0].dimension == "register.slang_ratio"

        engine.close()
