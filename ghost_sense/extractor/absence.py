"""Absence extractor — tracks expected-but-missing signals (silence as data).

Requires a calibration period before flagging. During calibration, it only
records baseline frequencies. After calibration, deviations from baseline
trigger absence signals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ghost_sense.config import ABSENCE_CALIBRATION_HOURS
from ghost_sense.extractor.base import BaseExtractor
from ghost_sense.models import SignalEvent, SignalType

# Dimensions to track absence for, with default expected intervals (hours)
# These get overridden by calibrated baselines
TRACKED_DIMENSIONS = {
    "sleep.quality": {"default_interval_hours": 24.0},
    "sleep.energy_level": {"default_interval_hours": 12.0},
    "nutrition.last_meal": {"default_interval_hours": 12.0},
    "stress.level": {"default_interval_hours": 24.0},
}

# How many multiples of the baseline interval before flagging absence
ABSENCE_MULTIPLIER = 2.5


@dataclass
class DimensionBaseline:
    """Tracks observation frequency for a single dimension during calibration."""

    dimension: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    observation_count: int = 0
    avg_interval_hours: float = 0.0
    calibrated: bool = False

    def record_observation(self, timestamp: float) -> None:
        if self.first_seen == 0.0:
            self.first_seen = timestamp
        if self.last_seen > 0 and self.observation_count >= 1:
            # Running average of intervals
            new_interval = (timestamp - self.last_seen) / 3600
            self.avg_interval_hours = (
                (self.avg_interval_hours * (self.observation_count - 1) + new_interval)
                / self.observation_count
            )
        self.last_seen = timestamp
        self.observation_count += 1

        # Check if calibration period has passed
        hours_tracked = (timestamp - self.first_seen) / 3600
        if hours_tracked >= ABSENCE_CALIBRATION_HOURS and self.observation_count >= 3:
            self.calibrated = True


class AbsenceExtractor(BaseExtractor):
    """Detects when expected signals are missing beyond baseline frequency.

    This extractor is stateful — it accumulates baselines across calls.
    During calibration (first 14 days), it only records observations.
    After calibration, it flags when a dimension hasn't been seen in
    ABSENCE_MULTIPLIER * baseline_interval hours.
    """

    def __init__(self) -> None:
        self._baselines: dict[str, DimensionBaseline] = {
            dim: DimensionBaseline(dimension=dim)
            for dim in TRACKED_DIMENSIONS
        }

    @property
    def is_calibrated(self) -> bool:
        """True if at least one dimension has completed calibration."""
        return any(b.calibrated for b in self._baselines.values())

    def get_baseline(self, dimension: str) -> DimensionBaseline | None:
        return self._baselines.get(dimension)

    def record_signal(self, dimension: str, timestamp: float) -> None:
        """Call this when a signal is observed (from bus), to update baselines."""
        if dimension in self._baselines:
            self._baselines[dimension].record_observation(timestamp)

    def extract(self, message: str, metadata: dict) -> list[SignalEvent]:
        """Check for absent signals. Called on every message to evaluate timing gaps.

        Note: This extractor doesn't parse the message text itself.
        It relies on record_signal() being called when other extractors
        fire, and checks for gaps here.
        """
        ts = metadata.get("timestamp", time.time())
        events: list[SignalEvent] = []

        for dim, baseline in self._baselines.items():
            if not baseline.calibrated:
                continue
            if baseline.last_seen == 0.0:
                continue

            hours_since = (ts - baseline.last_seen) / 3600
            expected_interval = baseline.avg_interval_hours
            if expected_interval <= 0:
                continue

            threshold = expected_interval * ABSENCE_MULTIPLIER

            if hours_since > threshold:
                # Signal is overdue — emit absence event
                overdue_ratio = hours_since / expected_interval
                confidence = min(0.4 + (overdue_ratio - ABSENCE_MULTIPLIER) * 0.1, 0.85)
                confidence = max(0.4, confidence)

                events.append(SignalEvent(
                    signal_type=SignalType.ABSENCE,
                    dimension=f"absence.{dim}",
                    value=overdue_ratio,  # how many multiples overdue
                    confidence=confidence,
                    source_text=f"{dim} not seen in {hours_since:.1f}h (baseline: {expected_interval:.1f}h)",
                    timestamp=ts,
                    metadata={
                        "tracked_dimension": dim,
                        "hours_since_last": hours_since,
                        "baseline_interval_hours": expected_interval,
                    },
                ))

        return events
