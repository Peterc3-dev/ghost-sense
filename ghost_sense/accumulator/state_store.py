"""SQLite-backed state vector with time-series history and trend analysis."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from ghost_sense.accumulator.decay import apply_decay, merge_with_decay
from ghost_sense.accumulator.trend import Trend, compute_all_trends, cache_trends, load_cached_trends
from ghost_sense.config import CONFIDENCE_THRESHOLD, get_lambda
from ghost_sense.models import FieldState, SignalEvent, SignalType, StateVector

SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "schema.sql"

# Dimensions that get trend analysis
TRENDED_DIMENSIONS = [
    "sleep.quality", "sleep.energy_level", "sleep.caffeine_proxy",
    "nutrition.last_meal", "nutrition.quality",
    "stress.level",
    "register.formality_score", "register.slang_ratio",
    "cadence.burst_score",
]

# Only recompute trends every N state updates (not every single event)
TREND_RECOMPUTE_INTERVAL = 5


class StateStore:
    """Persistent state backed by SQLite in WAL mode."""

    def __init__(self, db_path: str = "ghost_sense.db") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        self._update_count = 0

    def _init_schema(self) -> None:
        schema = SCHEMA_PATH.read_text()
        self._conn.executescript(schema)

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose connection for trend queries."""
        return self._conn

    def record_event(self, event: SignalEvent) -> None:
        """Persist a signal event."""
        self._conn.execute(
            "INSERT INTO signal_events (signal_type, dimension, value, confidence, source_text, timestamp, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.signal_type.value,
                event.dimension,
                event.value,
                event.confidence,
                event.source_text,
                event.timestamp,
                json.dumps(event.metadata),
            ),
        )
        self._conn.commit()

    def update_state(self, event: SignalEvent) -> None:
        """Update state from a new signal event, persist snapshot, maybe recompute trends."""
        self.record_event(event)

        # Load current state, apply decay, merge new event
        state = self.load_state()

        existing = state.get(event.dimension)
        if existing is not None:
            hours_elapsed = (event.timestamp - existing.last_updated) / 3600
            lam = get_lambda(event.dimension)
            merged_value, merged_confidence = merge_with_decay(
                existing.value, existing.confidence,
                event.value, event.confidence,
                hours_elapsed, lam,
            )
        else:
            merged_value = event.value
            merged_confidence = event.confidence

        state.set(FieldState(
            dimension=event.dimension,
            signal_type=event.signal_type,
            value=merged_value,
            confidence=merged_confidence,
            last_updated=event.timestamp,
        ))

        self._save_snapshot(state)

        # Periodic trend recomputation
        self._update_count += 1
        if self._update_count % TREND_RECOMPUTE_INTERVAL == 0:
            self.recompute_trends()

    def load_state(self, at_time: float | None = None) -> StateVector:
        """Load the latest state snapshot, applying decay at read-time.

        Args:
            at_time: If provided, decay is computed relative to this timestamp
                     instead of wall-clock now. Useful for testing.
        """
        row = self._conn.execute(
            "SELECT timestamp, fields FROM state_snapshots ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

        if row is None:
            return StateVector()

        snap_ts, fields_json = row
        fields_data = json.loads(fields_json)
        now = at_time if at_time is not None else time.time()
        state = StateVector()

        for dim, fdata in fields_data.items():
            hours_elapsed = (now - fdata["last_updated"]) / 3600
            lam = get_lambda(dim)
            decayed_confidence = apply_decay(fdata["confidence"], hours_elapsed, lam)

            if decayed_confidence >= CONFIDENCE_THRESHOLD:
                state.set(FieldState(
                    dimension=dim,
                    signal_type=SignalType(fdata["signal_type"]),
                    value=fdata["value"],
                    confidence=decayed_confidence,
                    last_updated=fdata["last_updated"],
                ))

        return state

    def _save_snapshot(self, state: StateVector) -> None:
        """Persist a state snapshot."""
        now = time.time()
        fields = {}
        for dim, fs in state.fields.items():
            fields[dim] = {
                "signal_type": fs.signal_type.value,
                "value": fs.value,
                "confidence": fs.confidence,
                "last_updated": fs.last_updated,
            }
        self._conn.execute(
            "INSERT INTO state_snapshots (timestamp, fields) VALUES (?, ?)",
            (now, json.dumps(fields)),
        )
        self._conn.commit()

    def recompute_trends(self, now: float | None = None) -> dict[str, dict[int, Trend]]:
        """Recompute and cache trends for all tracked dimensions."""
        trends = compute_all_trends(self._conn, TRENDED_DIMENSIONS, now=now)
        cache_trends(self._conn, trends)
        return trends

    def get_trends(self) -> dict[str, dict[int, Trend]]:
        """Load cached trends."""
        return load_cached_trends(self._conn)

    def get_events(self, dimension: str | None = None, since: float | None = None, limit: int = 100) -> list[SignalEvent]:
        """Query signal events, optionally filtered."""
        query = "SELECT signal_type, dimension, value, confidence, source_text, timestamp, metadata FROM signal_events WHERE 1=1"
        params: list = []

        if dimension:
            query += " AND dimension = ?"
            params.append(dimension)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [
            SignalEvent(
                signal_type=SignalType(r[0]),
                dimension=r[1],
                value=r[2],
                confidence=r[3],
                source_text=r[4],
                timestamp=r[5],
                metadata=json.loads(r[6]),
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
