-- ghost-sense SQLite schema (WAL mode set at connection time)

CREATE TABLE IF NOT EXISTS signal_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_type TEXT    NOT NULL,
    dimension   TEXT    NOT NULL,
    value       REAL    NOT NULL,
    confidence  REAL    NOT NULL,
    source_text TEXT    NOT NULL,
    timestamp   REAL    NOT NULL,
    metadata    TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_signal_events_dimension_ts
    ON signal_events (dimension, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_signal_events_type_ts
    ON signal_events (signal_type, timestamp DESC);

CREATE TABLE IF NOT EXISTS state_snapshots (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL    NOT NULL,
    fields    TEXT    NOT NULL  -- JSON blob of {dimension: {value, confidence, last_updated, signal_type}}
);

CREATE INDEX IF NOT EXISTS idx_state_snapshots_ts
    ON state_snapshots (timestamp DESC);

CREATE TABLE IF NOT EXISTS trend_cache (
    dimension   TEXT    NOT NULL,
    window_days INTEGER NOT NULL,
    direction   REAL    NOT NULL,  -- positive = increasing, negative = decreasing
    magnitude   REAL    NOT NULL,
    computed_at REAL    NOT NULL,
    PRIMARY KEY (dimension, window_days)
);
