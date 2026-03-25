"""Cadence extractor — inter-message timing, burst patterns, response latency."""

from __future__ import annotations

import time

from ghost_sense.extractor.base import BaseExtractor
from ghost_sense.models import SignalEvent, SignalType

# Thresholds for burst detection (seconds)
BURST_THRESHOLD = 30  # messages < 30s apart = burst mode
SPARSE_THRESHOLD = 600  # messages > 10min apart = sparse mode


class CadenceExtractor(BaseExtractor):
    """Extracts cadence signals from message timing metadata.

    Requires metadata fields:
        - time_since_last_message_seconds: float | None
        - message_index_in_session: int
    """

    def __init__(self) -> None:
        # Rolling window of recent intervals for burst scoring
        self._recent_intervals: list[float] = []
        self._max_window = 20

    def extract(self, message: str, metadata: dict) -> list[SignalEvent]:
        ts = metadata.get("timestamp", time.time())
        events: list[SignalEvent] = []

        interval = metadata.get("time_since_last_message_seconds")
        msg_index = metadata.get("message_index_in_session", 0)

        if interval is None:
            # First message in session — no cadence data yet
            return events

        # Track interval history
        self._recent_intervals.append(interval)
        if len(self._recent_intervals) > self._max_window:
            self._recent_intervals = self._recent_intervals[-self._max_window:]

        # --- Average interval ---
        if len(self._recent_intervals) >= 2:
            avg_interval = sum(self._recent_intervals) / len(self._recent_intervals)
            events.append(SignalEvent(
                signal_type=SignalType.CADENCE,
                dimension="cadence.avg_interval",
                value=avg_interval,
                confidence=min(0.5 + len(self._recent_intervals) * 0.05, 0.9),
                source_text=f"avg_interval={avg_interval:.1f}s over {len(self._recent_intervals)} msgs",
                timestamp=ts,
            ))

        # --- Burst score ---
        # 0.0 = very sparse, 1.0 = rapid-fire burst
        if interval <= BURST_THRESHOLD:
            burst_score = 1.0 - (interval / BURST_THRESHOLD)
        elif interval >= SPARSE_THRESHOLD:
            burst_score = 0.0
        else:
            # Linear interpolation between burst and sparse
            burst_score = 1.0 - (interval - BURST_THRESHOLD) / (SPARSE_THRESHOLD - BURST_THRESHOLD)

        burst_score = max(0.0, min(1.0, burst_score))

        # Confidence grows with more data points
        confidence = min(0.5 + len(self._recent_intervals) * 0.05, 0.85)

        events.append(SignalEvent(
            signal_type=SignalType.CADENCE,
            dimension="cadence.burst_score",
            value=burst_score,
            confidence=confidence,
            source_text=f"interval={interval:.1f}s, burst={burst_score:.2f}",
            timestamp=ts,
            metadata={"interval_seconds": interval, "message_index": msg_index},
        ))

        return events
