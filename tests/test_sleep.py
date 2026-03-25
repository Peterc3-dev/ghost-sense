"""Tests for sleep extractor."""

import time

import pytest

from ghost_sense.extractor.sleep import SleepExtractor
from ghost_sense.models import SignalType


@pytest.fixture
def extractor():
    return SleepExtractor()


def _meta(local_time="14:00", **kw):
    m = {
        "timestamp": time.time(),
        "local_time": local_time,
        "message_index_in_session": 0,
        "time_since_last_message_seconds": None,
    }
    m.update(kw)
    return m


class TestSleepExtractor:
    def test_poor_sleep_explicit(self, extractor):
        events = extractor.extract("barely slept last night man", _meta())
        quality = [e for e in events if e.dimension == "sleep.quality"]
        assert len(quality) == 1
        assert quality[0].value < 0.5

    def test_poor_sleep_hours(self, extractor):
        events = extractor.extract("got 3 hours of sleep", _meta())
        quality = [e for e in events if e.dimension == "sleep.quality"]
        assert len(quality) == 1
        assert quality[0].value < 0.5

    def test_good_sleep(self, extractor):
        events = extractor.extract("slept like a rock last night, feeling great", _meta())
        quality = [e for e in events if e.dimension == "sleep.quality"]
        assert len(quality) == 1
        assert quality[0].value > 0.7

    def test_all_nighter(self, extractor):
        events = extractor.extract("pulled an all-nighter to finish the project", _meta())
        quality = [e for e in events if e.dimension == "sleep.quality"]
        assert len(quality) == 1
        assert quality[0].value < 0.5

    def test_caffeine_normal_hours(self, extractor):
        events = extractor.extract("grabbing a coffee real quick", _meta(local_time="10:00"))
        caffeine = [e for e in events if e.dimension == "sleep.caffeine_proxy"]
        assert len(caffeine) == 1
        assert caffeine[0].value > 0.5  # normal hours = not alarming

    def test_caffeine_late_night(self, extractor):
        events = extractor.extract("need another coffee", _meta(local_time="23:00"))
        caffeine = [e for e in events if e.dimension == "sleep.caffeine_proxy"]
        assert len(caffeine) == 1
        assert caffeine[0].value < 0.5  # late = bad sleep proxy
        assert caffeine[0].metadata["late_caffeine"] is True

    def test_fatigue_language(self, extractor):
        events = extractor.extract("brain is completely fried right now", _meta())
        energy = [e for e in events if e.dimension == "sleep.energy_level"]
        assert len(energy) == 1
        assert energy[0].value < 0.5

    def test_high_energy(self, extractor):
        events = extractor.extract("feeling wired and locked in today", _meta())
        energy = [e for e in events if e.dimension == "sleep.energy_level"]
        assert len(energy) == 1
        assert energy[0].value > 0.7

    def test_no_sleep_signals_in_unrelated(self, extractor):
        events = extractor.extract("the API endpoint returns a 404 for that route", _meta())
        assert len(events) == 0

    def test_all_events_are_sleep_type(self, extractor):
        events = extractor.extract("exhausted, barely slept, third coffee today", _meta())
        assert all(e.signal_type == SignalType.SLEEP for e in events)

    def test_insomnia(self, extractor):
        events = extractor.extract("insomnia hit hard last night", _meta())
        quality = [e for e in events if e.dimension == "sleep.quality"]
        assert len(quality) == 1
        assert quality[0].value < 0.5

    def test_need_nap(self, extractor):
        events = extractor.extract("i need a nap so bad", _meta())
        energy = [e for e in events if e.dimension == "sleep.energy_level"]
        assert len(energy) == 1
        assert energy[0].value < 0.5
