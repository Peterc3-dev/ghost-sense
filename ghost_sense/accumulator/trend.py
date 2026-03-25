"""Sliding window trend detection over state snapshot history.

Computes trend direction and magnitude for each dimension over
configurable windows (default: 3-day and 7-day).

Trend is computed via simple linear regression (OLS) on the
(timestamp, value) pairs within each window. The slope sign gives
direction, the magnitude gives rate of change per hour.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum


class TrendDirection(Enum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"


@dataclass
class Trend:
    dimension: str
    window_days: int
    direction: TrendDirection
    magnitude: float  # absolute rate of change per hour
    slope: float  # raw slope (value per hour), signed
    data_points: int

    @property
    def is_significant(self) -> bool:
        """A trend is significant if it has enough data and non-trivial magnitude."""
        return self.data_points >= 3 and self.magnitude > 0.005


# Slope threshold: below this absolute slope, call it stable
STABLE_THRESHOLD = 0.005  # value change per hour


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Simple OLS: returns (slope, intercept). xs are hours-since-epoch."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0)

    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0, sum_y / n

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def compute_trend(
    conn: sqlite3.Connection,
    dimension: str,
    window_days: int,
    now: float | None = None,
) -> Trend:
    """Compute trend for a dimension over a time window.

    Reads state_snapshots from SQLite, extracts the dimension's value
    at each snapshot within the window, and fits a line.
    """
    now = now or time.time()
    window_start = now - window_days * 86400

    rows = conn.execute(
        "SELECT timestamp, fields FROM state_snapshots WHERE timestamp >= ? ORDER BY timestamp ASC",
        (window_start,),
    ).fetchall()

    timestamps: list[float] = []
    values: list[float] = []

    for ts, fields_json in rows:
        fields = json.loads(fields_json)
        if dimension in fields:
            fdata = fields[dimension]
            # Use raw stored value (decay not applied here — we want the trend
            # of observations, not of decayed readings)
            timestamps.append(ts)
            values.append(fdata["value"])

    n = len(timestamps)
    if n < 2:
        return Trend(
            dimension=dimension,
            window_days=window_days,
            direction=TrendDirection.STABLE,
            magnitude=0.0,
            slope=0.0,
            data_points=n,
        )

    # Normalize timestamps to hours relative to window start (numerical stability)
    hours = [(t - window_start) / 3600 for t in timestamps]
    slope, _ = _linear_regression(hours, values)

    if abs(slope) < STABLE_THRESHOLD:
        direction = TrendDirection.STABLE
    elif slope > 0:
        direction = TrendDirection.RISING
    else:
        direction = TrendDirection.FALLING

    return Trend(
        dimension=dimension,
        window_days=window_days,
        direction=direction,
        magnitude=abs(slope),
        slope=slope,
        data_points=n,
    )


def compute_all_trends(
    conn: sqlite3.Connection,
    dimensions: list[str],
    windows: list[int] | None = None,
    now: float | None = None,
) -> dict[str, dict[int, Trend]]:
    """Compute trends for multiple dimensions across multiple windows.

    Returns: {dimension: {window_days: Trend}}
    """
    if windows is None:
        windows = [3, 7]

    results: dict[str, dict[int, Trend]] = {}
    for dim in dimensions:
        results[dim] = {}
        for w in windows:
            results[dim][w] = compute_trend(conn, dim, w, now=now)
    return results


def cache_trends(conn: sqlite3.Connection, trends: dict[str, dict[int, Trend]]) -> None:
    """Write computed trends to the trend_cache table."""
    now = time.time()
    for dim, windows in trends.items():
        for window_days, trend in windows.items():
            conn.execute(
                "INSERT OR REPLACE INTO trend_cache (dimension, window_days, direction, magnitude, computed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (dim, window_days, trend.slope, trend.magnitude, now),
            )
    conn.commit()


def load_cached_trends(conn: sqlite3.Connection) -> dict[str, dict[int, Trend]]:
    """Load trends from cache. Returns {dimension: {window_days: Trend}}."""
    rows = conn.execute(
        "SELECT dimension, window_days, direction, magnitude, computed_at FROM trend_cache"
    ).fetchall()

    results: dict[str, dict[int, Trend]] = {}
    for dim, window_days, slope, magnitude, _ in rows:
        if abs(slope) < STABLE_THRESHOLD:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.RISING
        else:
            direction = TrendDirection.FALLING

        if dim not in results:
            results[dim] = {}
        results[dim][window_days] = Trend(
            dimension=dim,
            window_days=window_days,
            direction=direction,
            magnitude=magnitude,
            slope=slope,
            data_points=0,  # not stored in cache
        )
    return results
