"""Tests for cadence extractor."""

import time

import pytest

from ghost_sense.extractor.cadence import CadenceExtractor
from ghost_sense.models import SignalType


@pytest.fixture
def extractor():
    return CadenceExtractor()


def _meta(interval=None, index=0):
    return {
        "timestamp": time.time(),
        "local_time": "14:00",
        "message_index_in_session": index,
        "time_since_last_message_seconds": interval,
    }


class TestCadenceExtractor:
    def test_first_message_no_events(self, extractor):
        """No cadence data on first message (no interval)."""
        events = extractor.extract("hello", _meta(interval=None))
        assert len(events) == 0

    def test_burst_mode(self, extractor):
        """Rapid messages should produce high burst score."""
        # Need at least one prior message
        extractor.extract("first", _meta(interval=5, index=1))
        events = extractor.extract("second", _meta(interval=3, index=2))
        burst = [e for e in events if e.dimension == "cadence.burst_score"]
        assert len(burst) == 1
        assert burst[0].value > 0.8

    def test_sparse_mode(self, extractor):
        """Long gaps should produce low burst score."""
        extractor.extract("first", _meta(interval=900, index=1))
        events = extractor.extract("second", _meta(interval=900, index=2))
        burst = [e for e in events if e.dimension == "cadence.burst_score"]
        assert len(burst) == 1
        assert burst[0].value < 0.2

    def test_avg_interval_computed(self, extractor):
        """After 2+ messages, avg interval should be reported."""
        extractor.extract("a", _meta(interval=10, index=1))
        events = extractor.extract("b", _meta(interval=20, index=2))
        avg = [e for e in events if e.dimension == "cadence.avg_interval"]
        assert len(avg) == 1
        assert 10 <= avg[0].value <= 20

    def test_confidence_grows_with_data(self, extractor):
        """More messages = higher confidence."""
        for i in range(1, 6):
            events = extractor.extract(f"msg {i}", _meta(interval=15, index=i))
        burst = [e for e in events if e.dimension == "cadence.burst_score"]
        assert burst[0].confidence > 0.7

    def test_moderate_interval(self, extractor):
        """Middle-range interval should produce moderate burst score."""
        extractor.extract("a", _meta(interval=300, index=1))
        events = extractor.extract("b", _meta(interval=300, index=2))
        burst = [e for e in events if e.dimension == "cadence.burst_score"]
        assert 0.2 < burst[0].value < 0.8

    def test_all_events_are_cadence_type(self, extractor):
        extractor.extract("a", _meta(interval=10, index=1))
        events = extractor.extract("b", _meta(interval=10, index=2))
        assert all(e.signal_type == SignalType.CADENCE for e in events)
