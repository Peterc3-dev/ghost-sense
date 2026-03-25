"""Tests for absence extractor — calibration gating and gap detection."""

import time

import pytest

from ghost_sense.config import ABSENCE_CALIBRATION_HOURS
from ghost_sense.extractor.absence import ABSENCE_MULTIPLIER, AbsenceExtractor
from ghost_sense.models import SignalType


@pytest.fixture
def extractor():
    return AbsenceExtractor()


def _meta(ts=None):
    return {
        "timestamp": ts or time.time(),
        "local_time": "14:00",
        "message_index_in_session": 0,
        "time_since_last_message_seconds": None,
    }


class TestAbsenceExtractor:
    def test_no_events_before_calibration(self, extractor):
        """Should not flag absence during calibration period."""
        now = time.time()
        # Record a few observations but not enough time has passed
        extractor.record_signal("sleep.quality", now)
        extractor.record_signal("sleep.quality", now + 3600)
        extractor.record_signal("sleep.quality", now + 7200)

        # Check way later — but calibration hasn't completed (not enough hours)
        events = extractor.extract("anything", _meta(ts=now + 86400 * 30))
        assert len(events) == 0
        assert not extractor.is_calibrated

    def test_calibration_completes(self, extractor):
        """After enough time and observations, calibration completes."""
        now = time.time()
        cal_seconds = ABSENCE_CALIBRATION_HOURS * 3600

        # Simulate observations over the calibration period
        extractor.record_signal("sleep.quality", now)
        extractor.record_signal("sleep.quality", now + cal_seconds * 0.3)
        extractor.record_signal("sleep.quality", now + cal_seconds * 0.6)
        extractor.record_signal("sleep.quality", now + cal_seconds + 1)

        baseline = extractor.get_baseline("sleep.quality")
        assert baseline.calibrated is True
        assert extractor.is_calibrated

    def test_flags_absence_after_calibration(self, extractor):
        """Once calibrated, missing signals beyond threshold are flagged."""
        now = time.time()
        cal_seconds = ABSENCE_CALIBRATION_HOURS * 3600

        # Calibrate with ~24h interval
        for i in range(5):
            extractor.record_signal("sleep.quality", now + i * 86400)

        # Force calibration
        extractor.record_signal("sleep.quality", now + cal_seconds + 1)

        baseline = extractor.get_baseline("sleep.quality")
        assert baseline.calibrated

        # Now check way past the expected interval
        gap_hours = baseline.avg_interval_hours * (ABSENCE_MULTIPLIER + 1)
        gap_ts = baseline.last_seen + gap_hours * 3600

        events = extractor.extract("checking in", _meta(ts=gap_ts))
        absence = [e for e in events if "sleep.quality" in e.dimension]
        assert len(absence) == 1
        assert absence[0].signal_type == SignalType.ABSENCE
        assert absence[0].value > ABSENCE_MULTIPLIER

    def test_no_flag_within_threshold(self, extractor):
        """Within normal interval after calibration — no absence signal."""
        now = time.time()
        cal_seconds = ABSENCE_CALIBRATION_HOURS * 3600

        for i in range(5):
            extractor.record_signal("sleep.quality", now + i * 86400)
        extractor.record_signal("sleep.quality", now + cal_seconds + 1)

        baseline = extractor.get_baseline("sleep.quality")

        # Check just slightly after last observation — within threshold
        check_ts = baseline.last_seen + baseline.avg_interval_hours * 1.5 * 3600
        events = extractor.extract("normal check", _meta(ts=check_ts))
        absence = [e for e in events if "sleep.quality" in e.dimension]
        assert len(absence) == 0

    def test_multiple_dimensions_independent(self, extractor):
        """Each tracked dimension calibrates independently."""
        now = time.time()
        cal_seconds = ABSENCE_CALIBRATION_HOURS * 3600

        # Only calibrate sleep, not nutrition
        for i in range(5):
            extractor.record_signal("sleep.quality", now + i * 86400)
        extractor.record_signal("sleep.quality", now + cal_seconds + 1)

        sleep_bl = extractor.get_baseline("sleep.quality")
        nutrition_bl = extractor.get_baseline("nutrition.last_meal")

        assert sleep_bl.calibrated is True
        assert nutrition_bl.calibrated is False

    def test_is_calibrated_property(self, extractor):
        assert not extractor.is_calibrated

    def test_uncalibrated_dimensions_ignored(self, extractor):
        """Even with huge gaps, uncalibrated dims produce no events."""
        now = time.time()
        extractor.record_signal("nutrition.last_meal", now)
        events = extractor.extract("test", _meta(ts=now + 86400 * 365))
        assert len(events) == 0

    def test_absence_metadata(self, extractor):
        """Absence events include useful metadata."""
        now = time.time()
        cal_seconds = ABSENCE_CALIBRATION_HOURS * 3600

        for i in range(5):
            extractor.record_signal("stress.level", now + i * 86400)
        extractor.record_signal("stress.level", now + cal_seconds + 1)

        baseline = extractor.get_baseline("stress.level")
        gap_ts = baseline.last_seen + baseline.avg_interval_hours * 5 * 3600
        events = extractor.extract("yo", _meta(ts=gap_ts))

        absence = [e for e in events if "stress.level" in e.dimension]
        if absence:
            assert "tracked_dimension" in absence[0].metadata
            assert "hours_since_last" in absence[0].metadata
            assert "baseline_interval_hours" in absence[0].metadata
