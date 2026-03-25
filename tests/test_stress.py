"""Tests for stress extractor."""

import time

import pytest

from ghost_sense.extractor.stress import StressExtractor
from ghost_sense.models import SignalType


@pytest.fixture
def extractor():
    return StressExtractor()


def _meta(**kw):
    m = {
        "timestamp": time.time(),
        "local_time": "14:00",
        "message_index_in_session": 0,
        "time_since_last_message_seconds": None,
    }
    m.update(kw)
    return m


class TestStressExtractor:
    def test_direct_stress(self, extractor):
        events = extractor.extract("so stressed out right now", _meta())
        level = [e for e in events if e.dimension == "stress.level"]
        assert len(level) == 1
        assert level[0].value > 0.5

    def test_overwhelmed(self, extractor):
        events = extractor.extract("completely overwhelmed with everything", _meta())
        level = [e for e in events if e.dimension == "stress.level"]
        assert len(level) == 1
        assert level[0].value > 0.5

    def test_work_deadline(self, extractor):
        events = extractor.extract("deadline is tomorrow and nothing works", _meta())
        level = [e for e in events if e.dimension == "stress.level"]
        source = [e for e in events if e.dimension == "stress.source"]
        assert len(level) == 1
        assert level[0].value > 0.5
        assert len(source) == 1
        assert source[0].metadata["source_type"] == "professional"

    def test_production_incident(self, extractor):
        events = extractor.extract("prod is down, oncall pager going off", _meta())
        level = [e for e in events if e.dimension == "stress.level"]
        assert len(level) == 1
        assert level[0].value > 0.6  # multiple hits = higher

    def test_emotional_distress(self, extractor):
        events = extractor.extract("feeling really frustrated and burned out", _meta())
        level = [e for e in events if e.dimension == "stress.level"]
        source = [e for e in events if e.dimension == "stress.source"]
        assert len(level) == 1
        assert level[0].value > 0.5
        assert len(source) == 1
        assert source[0].metadata["source_type"] == "personal"

    def test_calm_signal(self, extractor):
        events = extractor.extract("just relaxing, no stress at all today", _meta())
        level = [e for e in events if e.dimension == "stress.level"]
        assert len(level) == 1
        assert level[0].value < 0.3

    def test_no_stress_in_neutral(self, extractor):
        events = extractor.extract("can you review this pull request", _meta())
        assert len(events) == 0

    def test_multiple_stressors_increase_level(self, extractor):
        events = extractor.extract(
            "deadline tomorrow, stressed out, boss wants it done, working overtime", _meta()
        )
        level = [e for e in events if e.dimension == "stress.level"]
        assert len(level) == 1
        assert level[0].value > 0.7
        assert level[0].confidence > 0.7

    def test_all_events_are_stress_type(self, extractor):
        events = extractor.extract("stressed about the deadline, frustrated", _meta())
        assert all(e.signal_type == SignalType.STRESS for e in events)

    def test_burnout(self, extractor):
        events = extractor.extract("completely burned out at this point", _meta())
        level = [e for e in events if e.dimension == "stress.level"]
        assert len(level) == 1
        assert level[0].value > 0.5
