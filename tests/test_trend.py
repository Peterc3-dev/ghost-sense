"""Tests for trend detection across synthetic time-series data."""

import json
import sqlite3
import time
from pathlib import Path

import pytest

from ghost_sense.accumulator.trend import (
    STABLE_THRESHOLD,
    Trend,
    TrendDirection,
    cache_trends,
    compute_all_trends,
    compute_trend,
    load_cached_trends,
)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"


@pytest.fixture
def db():
    """In-memory SQLite with schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_PATH.read_text())
    yield conn
    conn.close()


def _insert_snapshots(conn: sqlite3.Connection, dimension: str, data: list[tuple[float, float]]):
    """Insert synthetic snapshots. data = [(timestamp, value), ...]"""
    for ts, val in data:
        fields = {dimension: {
            "signal_type": "sleep",
            "value": val,
            "confidence": 0.9,
            "last_updated": ts,
        }}
        conn.execute(
            "INSERT INTO state_snapshots (timestamp, fields) VALUES (?, ?)",
            (ts, json.dumps(fields)),
        )
    conn.commit()


class TestComputeTrend:
    def test_rising_trend(self, db):
        now = time.time()
        # 7 data points over 3 days, steadily rising 0.2 -> 0.9
        data = [(now - 3 * 86400 + i * 43200, 0.2 + i * 0.1) for i in range(7)]
        _insert_snapshots(db, "sleep.quality", data)

        trend = compute_trend(db, "sleep.quality", window_days=3, now=now)
        assert trend.direction == TrendDirection.RISING
        assert trend.slope > 0
        assert trend.data_points == 7
        assert trend.is_significant

    def test_falling_trend(self, db):
        now = time.time()
        data = [(now - 3 * 86400 + i * 43200, 0.9 - i * 0.1) for i in range(7)]
        _insert_snapshots(db, "stress.level", data)

        trend = compute_trend(db, "stress.level", window_days=3, now=now)
        assert trend.direction == TrendDirection.FALLING
        assert trend.slope < 0
        assert trend.is_significant

    def test_stable_trend(self, db):
        now = time.time()
        # Flat line at 0.5
        data = [(now - 3 * 86400 + i * 43200, 0.5) for i in range(7)]
        _insert_snapshots(db, "sleep.quality", data)

        trend = compute_trend(db, "sleep.quality", window_days=3, now=now)
        assert trend.direction == TrendDirection.STABLE
        assert trend.magnitude < STABLE_THRESHOLD

    def test_near_stable_with_noise(self, db):
        now = time.time()
        # Tiny oscillation around 0.5
        data = [(now - 3 * 86400 + i * 43200, 0.5 + (i % 2) * 0.001) for i in range(7)]
        _insert_snapshots(db, "sleep.quality", data)

        trend = compute_trend(db, "sleep.quality", window_days=3, now=now)
        assert trend.direction == TrendDirection.STABLE

    def test_insufficient_data(self, db):
        now = time.time()
        _insert_snapshots(db, "sleep.quality", [(now - 3600, 0.5)])

        trend = compute_trend(db, "sleep.quality", window_days=3, now=now)
        assert trend.direction == TrendDirection.STABLE
        assert trend.data_points == 1
        assert not trend.is_significant

    def test_no_data(self, db):
        trend = compute_trend(db, "sleep.quality", window_days=3)
        assert trend.data_points == 0
        assert trend.direction == TrendDirection.STABLE

    def test_window_excludes_old_data(self, db):
        now = time.time()
        # Old data (outside 3-day window) trending up
        old_data = [(now - 10 * 86400 + i * 86400, 0.1 + i * 0.1) for i in range(5)]
        # Recent data (inside window) trending down
        recent_data = [(now - 2 * 86400 + i * 43200, 0.9 - i * 0.15) for i in range(5)]
        _insert_snapshots(db, "stress.level", old_data + recent_data)

        trend = compute_trend(db, "stress.level", window_days=3, now=now)
        assert trend.direction == TrendDirection.FALLING
        assert trend.data_points == 5  # only recent data

    def test_7day_window_includes_more(self, db):
        now = time.time()
        # Data spanning 6 days
        data = [(now - 6 * 86400 + i * 86400, 0.3 + i * 0.05) for i in range(7)]
        _insert_snapshots(db, "sleep.quality", data)

        trend_3d = compute_trend(db, "sleep.quality", window_days=3, now=now)
        trend_7d = compute_trend(db, "sleep.quality", window_days=7, now=now)
        assert trend_7d.data_points > trend_3d.data_points


class TestComputeAllTrends:
    def test_multiple_dimensions(self, db):
        now = time.time()
        data = [(now - 3 * 86400 + i * 43200, 0.3 + i * 0.08) for i in range(7)]
        _insert_snapshots(db, "sleep.quality", data)
        _insert_snapshots(db, "stress.level", data)

        results = compute_all_trends(
            db, ["sleep.quality", "stress.level"], windows=[3, 7], now=now
        )
        assert "sleep.quality" in results
        assert "stress.level" in results
        assert 3 in results["sleep.quality"]
        assert 7 in results["sleep.quality"]

    def test_default_windows(self, db):
        now = time.time()
        data = [(now - 5 * 86400 + i * 86400, 0.5) for i in range(6)]
        _insert_snapshots(db, "sleep.quality", data)

        results = compute_all_trends(db, ["sleep.quality"], now=now)
        assert 3 in results["sleep.quality"]
        assert 7 in results["sleep.quality"]


class TestTrendCache:
    def test_cache_roundtrip(self, db):
        now = time.time()
        data = [(now - 3 * 86400 + i * 43200, 0.2 + i * 0.1) for i in range(7)]
        _insert_snapshots(db, "sleep.quality", data)

        trends = compute_all_trends(db, ["sleep.quality"], now=now)
        cache_trends(db, trends)

        loaded = load_cached_trends(db)
        assert "sleep.quality" in loaded
        assert 3 in loaded["sleep.quality"]
        orig = trends["sleep.quality"][3]
        cached = loaded["sleep.quality"][3]
        assert orig.direction == cached.direction
        assert abs(orig.slope - cached.slope) < 1e-10

    def test_cache_overwrites(self, db):
        now = time.time()
        data1 = [(now - 3 * 86400 + i * 43200, 0.2 + i * 0.1) for i in range(7)]
        _insert_snapshots(db, "sleep.quality", data1)
        trends1 = compute_all_trends(db, ["sleep.quality"], now=now)
        cache_trends(db, trends1)

        # Overwrite with new data
        data2 = [(now - 3 * 86400 + i * 43200, 0.9 - i * 0.1) for i in range(7)]
        _insert_snapshots(db, "sleep.quality", data2)
        trends2 = compute_all_trends(db, ["sleep.quality"], now=now)
        cache_trends(db, trends2)

        loaded = load_cached_trends(db)
        # Should reflect the latest cache, not the first
        assert loaded["sleep.quality"][3].direction != trends1["sleep.quality"][3].direction


class TestTrendProperties:
    def test_is_significant_requires_data(self):
        trend = Trend("x", 3, TrendDirection.RISING, 0.05, 0.05, data_points=1)
        assert not trend.is_significant

    def test_is_significant_requires_magnitude(self):
        trend = Trend("x", 3, TrendDirection.STABLE, 0.001, 0.001, data_points=10)
        assert not trend.is_significant

    def test_borderline_not_significant(self):
        trend = Trend("x", 3, TrendDirection.RISING, 0.004, 0.004, data_points=5)
        assert not trend.is_significant

    def test_is_significant_when_both(self):
        trend = Trend("x", 3, TrendDirection.RISING, 0.05, 0.05, data_points=5)
        assert trend.is_significant
