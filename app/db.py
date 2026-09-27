"""SQLite persistence layer.

The database path is read from the DATABASE_PATH environment variable at
connect time (default: ./data/app.db), so tests and deployments can point
the app at a different file without touching code.
"""

import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = "./data/app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    device_id    TEXT PRIMARY KEY,
    threshold_h  REAL NOT NULL,
    threshold_l  REAL NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    CHECK (threshold_h > threshold_l)
);

CREATE TABLE IF NOT EXISTS readings (
    device_id   TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    sampled_at  TEXT NOT NULL,  -- UTC ISO-8601, fixed-width microseconds
    temperature REAL NOT NULL,
    received_at TEXT NOT NULL,  -- UTC ISO-8601, when the server stored it
    PRIMARY KEY (device_id, sampled_at)
);

-- Alarms are derived data: they are fully recomputed from readings +
-- thresholds whenever either changes, then stored here for querying.
CREATE TABLE IF NOT EXISTS alarms (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id  TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    opened_at  TEXT NOT NULL,  -- UTC ISO-8601 sample time of the 3rd high reading
    closed_at  TEXT            -- UTC ISO-8601 sample time of the 2nd low reading, NULL while open
);

CREATE INDEX IF NOT EXISTS idx_alarms_device ON alarms(device_id);

-- Alarm revision audit: one row per threshold PUT or readings batch POST
-- that actually changed the device's thresholds or some stored temperature.
-- Revision numbers are per-device and consecutive from 1; the row is written
-- in the same transaction as the mutation itself. Databases created before
-- this feature simply have no rows until the first new write.
CREATE TABLE IF NOT EXISTS alarm_revisions (
    device_id        TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    revision         INTEGER NOT NULL,  -- per-device, consecutive from 1
    op               TEXT NOT NULL CHECK (op IN ('thresholds', 'readings')),
    created_at       TEXT NOT NULL,     -- UTC ISO-8601, when the change was applied
    old_threshold_h  REAL,              -- NULL for a newly created device
    old_threshold_l  REAL,
    new_threshold_h  REAL,              -- set iff op = 'thresholds'
    new_threshold_l  REAL,
    PRIMARY KEY (device_id, revision)
);

-- Per-revision reading changes (op = 'readings'): one row per sample point
-- whose stored temperature actually changed in that batch (after collapsing
-- duplicate sample instants to their last value).
CREATE TABLE IF NOT EXISTS alarm_revision_readings (
    device_id       TEXT NOT NULL,
    revision        INTEGER NOT NULL,
    sampled_at      TEXT NOT NULL,  -- UTC ISO-8601, fixed-width microseconds
    old_temperature REAL,           -- NULL for a first-time sample point
    new_temperature REAL NOT NULL,
    PRIMARY KEY (device_id, revision, sampled_at),
    FOREIGN KEY (device_id, revision) REFERENCES alarm_revisions(device_id, revision) ON DELETE CASCADE
);

-- Per-revision alarm interval diff, compared by (opened_at, closed_at) rather
-- than alarm row id: a moved alarm appears as one 'withdrawn' row (old
-- interval) plus one 'added' row (new interval).
CREATE TABLE IF NOT EXISTS alarm_revision_alarms (
    device_id  TEXT NOT NULL,
    revision   INTEGER NOT NULL,
    change     TEXT NOT NULL CHECK (change IN ('added', 'withdrawn')),
    opened_at  TEXT NOT NULL,  -- UTC ISO-8601
    closed_at  TEXT,           -- NULL while the alarm is open
    PRIMARY KEY (device_id, revision, change, opened_at),
    FOREIGN KEY (device_id, revision) REFERENCES alarm_revisions(device_id, revision) ON DELETE CASCADE
);

-- Alarm acknowledgements: one row per acknowledged alarm, keyed by the
-- alarm's open instant (opened_at). Written by
-- POST /devices/{device_id}/alarm-acknowledgements and kept in sync by every
-- alarm recompute: an acknowledgement survives only while an alarm with the
-- same opened_at exists (a changed closed_at does not detach it), so a moved
-- or withdrawn alarm loses its acknowledgement in the same transaction.
CREATE TABLE IF NOT EXISTS alarm_acknowledgements (
    device_id       TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    opened_at       TEXT NOT NULL,  -- UTC ISO-8601 open instant of the acknowledged alarm
    operator        TEXT NOT NULL,  -- who acknowledged; non-empty
    acknowledged_at TEXT NOT NULL,  -- UTC ISO-8601, when the acknowledgement was recorded
    PRIMARY KEY (device_id, opened_at)
);

-- Per-device sampling-interruption config (independent of alarm thresholds).
-- Gaps are computed on query from current readings, so nothing derived is
-- stored here; the config itself is persisted and survives restarts.
CREATE TABLE IF NOT EXISTS sampling_configs (
    device_id     TEXT PRIMARY KEY REFERENCES devices(device_id) ON DELETE CASCADE,
    max_interval  INTEGER NOT NULL CHECK (max_interval >= 1),  -- M seconds
    grace         INTEGER NOT NULL CHECK (grace >= 0),         -- G seconds
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

-- Per-device alarm response deadline D (seconds). Independent of both the
-- alarm thresholds and the sampling config: only the policy itself is
-- persisted, the per-alarm response status is computed on query, so late
-- arrivals, corrections, recomputes, acknowledgements and M/D changes all
-- show up in the next GET .../response-status query.
CREATE TABLE IF NOT EXISTS response_policies (
    device_id   TEXT PRIMARY KEY REFERENCES devices(device_id) ON DELETE CASCADE,
    deadline    INTEGER NOT NULL CHECK (deadline >= 1),  -- D seconds
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Device groups: a named member set plus K, the minimum number of members
-- that must be in alarm at the same time to open a group alarm. Only the
-- configuration is persisted; group alarms are computed on query from the
-- members' current alarms, so nothing derived is stored here.
CREATE TABLE IF NOT EXISTS device_groups (
    group_id    TEXT PRIMARY KEY,
    k           INTEGER NOT NULL CHECK (k >= 1),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS group_members (
    group_id   TEXT NOT NULL REFERENCES device_groups(group_id) ON DELETE CASCADE,
    device_id  TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    position   INTEGER NOT NULL,  -- submission order, used when echoing members back
    PRIMARY KEY (group_id, device_id)
);
"""


def database_path() -> str:
    return os.environ.get("DATABASE_PATH", DEFAULT_DB_PATH)


def connect() -> sqlite3.Connection:
    path = Path(database_path())
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = connect()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def get_db():
    """FastAPI dependency: one connection per request."""
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()
