"""Device temperature alarm, sampling-interruption, heat-exposure and group-alarm API.

Alarms are a pure function of a device's readings and thresholds, so every
mutation (reading write/correction/late arrival, threshold change) triggers
a full recompute for that device: stale alarms are removed and shifted
alarms are persisted with their new times.

Sampling gaps, by contrast, are never stored: each GET /gaps request
recomputes them from the device's current readings, its current M/G
config and the request's as_of instant, so corrections, late arrivals and
config changes all show up in the next query. Heat exposure
(GET /heat-exposure) follows the same recompute-on-query approach, reusing
the readings and the sampling config's maximum interval M.

Device groups persist only their member list and the minimum concurrent
count K; group alarms (GET /groups/{group_id}/alarms) are computed on
query from the members' current stored alarms, so membership/K edits and
any member alarm recompute (late/corrected readings, threshold changes)
all show up in the next query.

Peer deviations (GET /groups/{group_id}/peer-deviations) diagnose one
member's sustained temperature deviation from its group peers over a
query window: each of the target's sample points in [start, end) is
compared against the median of the other members' values at the same
instant (exact readings, or linear interpolation bounded by each peer's
own maximum sampling interval M; members without a configured M never
participate), and runs of at least min_samples consecutive deviating
points are reported. The group's members, the members' sampling configs
and all their readings are read in one SQLite read snapshot and nothing
is stored, so late arrivals, corrections and membership/config changes
show up in the next query.

Temperature lag (GET /groups/{group_id}/lag) estimates the integer-second
clock/temperature offset between two members of a group over a query
window: for each of the target's sample points in [start, end) and every
whole-second lag in [-max_lag, max_lag], the reference temperature at
t + lag is taken from an exact reading, or linearly interpolated between
adjacent reference readings at most the reference's maximum sampling
interval M apart (never extrapolated). Shifts backed by at least
min_pairs pairs are ranked by smallest median absolute temperature
difference, then most pairs, then smallest |lag|, then smallest lag. The
group's members, the reference's sampling config and both members'
readings are read in one SQLite read snapshot and nothing is stored, so
late arrivals, corrections and membership/config changes show up in the
next query.

Every threshold PUT or readings batch POST that actually changes the
device's thresholds or some stored temperature additionally appends one
alarm revision (per-device, consecutive from 1) inside the same SQLite
write transaction: the operation type, the threshold before/after values
(or, for a readings batch, each changed sample point's temperature
before/after) and the alarm interval diff of the recompute, compared by
(opened_at, closed_at) rather than alarm row id. The audit trail is
served by GET /devices/{device_id}/alarm-revisions with cursor
pagination (exclusive ``after``, positive ``limit``).

Alarm acknowledgements are stored per (device, alarm open instant):
POST /devices/{device_id}/alarm-acknowledgements acknowledges a
currently open alarm, matched by its opened_at instant, guarded by the
device's current alarm revision (``expected_revision``; a mismatch
returns 409, checked inside the same write transaction). Repeating the
acknowledgement is idempotent: the first operator and time are kept.
GET /devices/{device_id}/alarms returns each alarm's acknowledgement
(operator and time) in the same read snapshot and the current revision
in the X-Alarm-Revision response header. Acknowledgements follow the
alarm's open instant across recomputes: kept while an alarm with the
same opened_at exists (even if its closed_at changes), dropped in the
same transaction when the open instant moves or the alarm is withdrawn.

POST /devices/{device_id}/alarm-preview is a dry run for planning a
threshold or reading change: it takes candidate thresholds and/or
candidate readings, reads the device's current thresholds, readings,
alarms and revision in one SQLite read snapshot, overlays the
candidates in memory (candidate readings replace stored ones at the
same UTC sample instant; the last batch entry for an instant wins) and
recomputes the alarms with the usual rules. The response carries the
base revision, the previewed intervals and the added/withdrawn
interval diff against the current alarms, compared by
(opened_at, closed_at). Nothing is written: readings, thresholds,
alarms, acknowledgements and the revision history are all untouched.

GET /devices/{device_id}/alarm-revisions/{revision}/snapshot
reconstructs the device's state right after a given revision
completed: the current thresholds, the current readings and the
changes recorded by every later revision are read in one SQLite read
snapshot, then rolled back in descending revision order (a reading
change whose old value is null removes that sample point, otherwise
the old temperature is restored; a threshold change restores the old
H/L). The restored readings, sorted by UTC sample instant, go through
the same alarm state machine, and the response carries the target
revision, the H/L in effect at that time, the restored readings and
the recomputed alarm intervals. The query is read-only and never
includes the current acknowledgements.

GET /devices/{device_id}/forecast projects when the temperature trend
will reach the device's current threshold H: the threshold, the
sampling config's maximum interval M and the readings are read in one
SQLite read snapshot (a device without a configured M returns 409),
and the latest contiguous sampling segment inside
[as_of - lookback, as_of] (adjacent intervals at most M, capped at 30
points) is fitted with as_of as the zero instant — the slope is the
median of all point-pair slopes, the intercept the median of
(temperature - slope * relative seconds). A newest sample older than
M or fewer than 3 points yields insufficient_data; an upward fit that
reaches H within ``horizon`` seconds yields predicted with the UTC
crossing instant, everything else no_crossing. Nothing is stored, so
late arrivals, corrections and config changes show up in the next
query.

PUT /devices/{device_id}/response-policy persists a device's alarm
response deadline D (whole positive seconds). GET
/devices/{device_id}/response-status takes a timezone-aware as_of and
lists the alarms that are still open, unacknowledged and opened at or
before as_of. For each one, observable time accumulates within
[opened_at, as_of] using only readings sampled at or before as_of:
adjacent readings more than M apart deduct [prev + M, next), and a
last reading more than M before as_of deducts [last + M, as_of], each
clipped to [opened_at, as_of] (exactly M is observable). The response
carries the cumulative observable seconds, whether D is reached and
the first UTC instant at which that happened (null otherwise). A
device without a configured D or M returns 409; the alarms,
acknowledgements, M, readings and D are read in one SQLite read
snapshot, and nothing is stored, so late arrivals, corrections,
recomputes, acknowledgements and M/D changes show up in the next
query.

GET /devices/{device_id}/alarm-consistency audits the stored alarm
rows as derived data: the current thresholds and readings are read in
one SQLite read snapshot, the alarms are recomputed, and the stored
rows are compared item-wise by (opened_at, closed_at) multiplicity —
missing intervals, surplus extra intervals and duplicate copies are
all listed, as are acknowledgements whose opened_at has no
corresponding recomputed alarm open instant. POST
/devices/{device_id}/alarm-reconcile rechecks inside a BEGIN IMMEDIATE
transaction and, when the alarms differ, rebuilds them and removes
only the orphan acknowledgements (the first acknowledgement of every
still-matching open instant is kept). Readings, thresholds and the
revision history are untouched and the reconcile records no revision;
an already-consistent device leaves every alarm row and
acknowledgement unchanged.
"""

import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from sqlite3 import Connection
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Response

from .db import get_db, init_db
from .logic import (
    compare_alarm_consistency,
    compute_alarms,
    compute_forecast,
    compute_gaps,
    compute_group_alarms,
    compute_heat_exposure,
    compute_peer_deviations,
    compute_response_status,
    compute_temperature_lag,
    diff_alarm_intervals,
)
from .schemas import (
    AlarmAcknowledgementIn,
    AlarmAcknowledgementOut,
    AlarmConsistencyOut,
    AlarmIntervalOut,
    AlarmOut,
    AlarmPreviewIn,
    AlarmPreviewOut,
    AlarmReconcileOut,
    AlarmRevisionOut,
    AlarmRevisionSnapshotOut,
    DeviceOut,
    ForecastOut,
    GapOut,
    GroupAlarmOut,
    GroupIn,
    GroupOut,
    HeatExposureOut,
    LagEstimateOut,
    OrphanAcknowledgementOut,
    PeerDeviationOut,
    ReadingChangeOut,
    ReadingOut,
    ReadingsBatchIn,
    ResponsePolicyIn,
    ResponsePolicyOut,
    ResponseStatusOut,
    SamplingConfigIn,
    SamplingConfigOut,
    SnapshotReadingOut,
    ThresholdChangeOut,
    ThresholdsIn,
    WriteResult,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Device Temperature Alarm, Sampling Gap, Heat Exposure, Group Alarm, Peer Deviation, Temperature Lag, Threshold Forecast & Alarm Response Deadline API",
    version="1.12.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------- helpers

def utc_iso(dt: datetime) -> str:
    """Normalize an aware datetime to a fixed-width UTC ISO-8601 string.

    Fixed width (microseconds always present) makes the string
    lexicographically sortable, which is how readings are ordered in SQLite.
    """
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def now_iso() -> str:
    return utc_iso(datetime.now(timezone.utc))


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value is not None else None


def get_device_or_404(conn: Connection, device_id: str):
    row = conn.execute(
        "SELECT * FROM devices WHERE device_id = ?", (device_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"device '{device_id}' not found; set thresholds first")
    return row


def current_alarm_revision(conn: Connection, device_id: str) -> int:
    """The device's latest alarm revision number (0 when none is recorded yet)."""
    return conn.execute(
        "SELECT COALESCE(MAX(revision), 0) AS revision"
        " FROM alarm_revisions WHERE device_id = ?",
        (device_id,),
    ).fetchone()["revision"]


def recompute_alarms(conn: Connection, device_id: str):
    """Rebuild the device's alarms from its current readings and thresholds.

    Returns (old_intervals, new_intervals), each a list of
    (opened_at, closed_at) in opened_at order, so the caller can record the
    revision diff of the recompute.
    """
    device = get_device_or_404(conn, device_id)
    rows = conn.execute(
        "SELECT sampled_at, temperature FROM readings WHERE device_id = ? ORDER BY sampled_at ASC",
        (device_id,),
    ).fetchall()
    alarms = compute_alarms(
        [(r["sampled_at"], r["temperature"]) for r in rows],
        device["threshold_h"],
        device["threshold_l"],
    )
    old_intervals = [
        (r["opened_at"], r["closed_at"])
        for r in conn.execute(
            "SELECT opened_at, closed_at FROM alarms WHERE device_id = ?"
            " ORDER BY opened_at ASC, id ASC",
            (device_id,),
        ).fetchall()
    ]
    conn.execute("DELETE FROM alarms WHERE device_id = ?", (device_id,))
    conn.executemany(
        "INSERT INTO alarms (device_id, opened_at, closed_at) VALUES (?, ?, ?)",
        [(device_id, a.opened_at, a.closed_at) for a in alarms],
    )
    # Acknowledgements are tied to the alarm's open instant: keep those whose
    # opened_at still exists (a changed closed_at does not detach them) and
    # drop the rest — moved or withdrawn alarms — in the same transaction.
    new_opened = [a.opened_at for a in alarms]
    if new_opened:
        placeholders = ", ".join("?" for _ in new_opened)
        conn.execute(
            "DELETE FROM alarm_acknowledgements"
            f" WHERE device_id = ? AND opened_at NOT IN ({placeholders})",
            [device_id, *new_opened],
        )
    else:
        conn.execute(
            "DELETE FROM alarm_acknowledgements WHERE device_id = ?", (device_id,)
        )
    return old_intervals, [(a.opened_at, a.closed_at) for a in alarms]


def alarm_consistency_report(conn: Connection, device_id: str, device) -> AlarmConsistencyOut:
    """Compare the device's stored alarms against a fresh recompute.

    The device row, its thresholds/readings, the stored alarm rows and the
    stored acknowledgements must all come from the caller's current SQLite
    snapshot (a read snapshot for GET, the write transaction for POST).
    Alarms are recomputed with the current thresholds and readings, then
    compared item-wise with the stored rows by (opened_at, closed_at)
    multiplicity: missing/extra intervals and surplus duplicate copies are
    listed. Acknowledgements whose opened_at has no recomputed alarm open
    instant are listed as orphans (their acknowledgement details are kept
    for the report).
    """
    rows = conn.execute(
        "SELECT sampled_at, temperature FROM readings WHERE device_id = ? ORDER BY sampled_at ASC",
        (device_id,),
    ).fetchall()
    recomputed = compute_alarms(
        [(r["sampled_at"], r["temperature"]) for r in rows],
        device["threshold_h"],
        device["threshold_l"],
    )
    recomputed_intervals = [(a.opened_at, a.closed_at) for a in recomputed]
    stored_rows = conn.execute(
        "SELECT opened_at, closed_at FROM alarms WHERE device_id = ?"
        " ORDER BY opened_at ASC, id ASC",
        (device_id,),
    ).fetchall()
    stored_intervals = [(r["opened_at"], r["closed_at"]) for r in stored_rows]
    comparison = compare_alarm_consistency(stored_intervals, recomputed_intervals)

    recomputed_open_instants = {a.opened_at for a in recomputed}
    orphan_rows = conn.execute(
        "SELECT opened_at, operator, acknowledged_at FROM alarm_acknowledgements"
        " WHERE device_id = ? ORDER BY opened_at ASC",
        (device_id,),
    ).fetchall()
    orphans = [r for r in orphan_rows if r["opened_at"] not in recomputed_open_instants]

    def interval_out(interval):
        opened, closed = interval
        return AlarmIntervalOut(opened_at=parse_iso(opened), closed_at=parse_iso(closed))

    return AlarmConsistencyOut(
        device_id=device_id,
        consistent=(
            not comparison.missing
            and not comparison.extra
            and not comparison.duplicate
            and not orphans
        ),
        stored_alarm_count=len(stored_intervals),
        recomputed_alarm_count=len(recomputed_intervals),
        missing_alarms=[interval_out(iv) for iv in comparison.missing],
        extra_alarms=[interval_out(iv) for iv in comparison.extra],
        duplicate_alarms=[interval_out(iv) for iv in comparison.duplicate],
        orphan_acknowledgements=[
            OrphanAcknowledgementOut(
                opened_at=parse_iso(r["opened_at"]),
                operator=r["operator"],
                acknowledged_at=parse_iso(r["acknowledged_at"]),
            )
            for r in orphans
        ],
    )


def record_alarm_revision(
    conn: Connection,
    device_id: str,
    op: str,
    now: str,
    *,
    threshold_change=None,   # (old_h, old_l, new_h, new_l); old_* None for a new device
    reading_changes=(),      # (sampled_at, old_temperature, new_temperature)
    old_alarms=(),           # (opened_at, closed_at) before the recompute
    new_alarms=(),           # (opened_at, closed_at) after the recompute
) -> None:
    """Append one alarm revision with its input changes and alarm interval diff.

    Must run inside the same write transaction as the mutation itself (the
    caller takes the write lock up front with BEGIN IMMEDIATE), so the
    per-device revision number — consecutive from 1 — and the recorded
    before/after states stay consistent under concurrent writers.
    """
    revision = conn.execute(
        "SELECT COALESCE(MAX(revision), 0) + 1 AS next_revision"
        " FROM alarm_revisions WHERE device_id = ?",
        (device_id,),
    ).fetchone()["next_revision"]
    old_h = old_l = new_h = new_l = None
    if threshold_change is not None:
        old_h, old_l, new_h, new_l = threshold_change
    conn.execute(
        "INSERT INTO alarm_revisions"
        " (device_id, revision, op, created_at,"
        "  old_threshold_h, old_threshold_l, new_threshold_h, new_threshold_l)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (device_id, revision, op, now, old_h, old_l, new_h, new_l),
    )
    conn.executemany(
        "INSERT INTO alarm_revision_readings"
        " (device_id, revision, sampled_at, old_temperature, new_temperature)"
        " VALUES (?, ?, ?, ?, ?)",
        [
            (device_id, revision, sampled_at, old_temp, new_temp)
            for sampled_at, old_temp, new_temp in sorted(reading_changes, key=lambda c: c[0])
        ],
    )
    added, withdrawn = diff_alarm_intervals(old_alarms, new_alarms)
    conn.executemany(
        "INSERT INTO alarm_revision_alarms"
        " (device_id, revision, change, opened_at, closed_at)"
        " VALUES (?, ?, ?, ?, ?)",
        [
            *[(device_id, revision, "added", opened, closed) for opened, closed in added],
            *[(device_id, revision, "withdrawn", opened, closed) for opened, closed in withdrawn],
        ],
    )


def device_out(row) -> DeviceOut:
    return DeviceOut(
        device_id=row["device_id"],
        threshold_h=row["threshold_h"],
        threshold_l=row["threshold_l"],
        created_at=parse_iso(row["created_at"]),
        updated_at=parse_iso(row["updated_at"]),
    )


def alarm_out(row) -> AlarmOut:
    """Shape one alarm row, joined with its acknowledgement (if any)."""
    return AlarmOut(
        id=row["id"],
        device_id=row["device_id"],
        opened_at=parse_iso(row["opened_at"]),
        closed_at=parse_iso(row["closed_at"]),
        status="open" if row["closed_at"] is None else "closed",
        acknowledged_by=row["ack_operator"],
        acknowledged_at=parse_iso(row["ack_acknowledged_at"]),
    )


def sampling_config_out(row) -> SamplingConfigOut:
    return SamplingConfigOut(
        device_id=row["device_id"],
        max_interval=row["max_interval"],
        grace=row["grace"],
        created_at=parse_iso(row["created_at"]),
        updated_at=parse_iso(row["updated_at"]),
    )


def response_policy_out(row) -> ResponsePolicyOut:
    return ResponsePolicyOut(
        device_id=row["device_id"],
        deadline=row["deadline"],
        created_at=parse_iso(row["created_at"]),
        updated_at=parse_iso(row["updated_at"]),
    )


# ---------------------------------------------------------------- endpoints

@app.get("/health")
def health():
    return {"status": "ok"}


@app.put("/devices/{device_id}/thresholds", response_model=DeviceOut)
def set_thresholds(
    body: ThresholdsIn,
    device_id: str = Path(min_length=1, max_length=200),
    conn: Connection = Depends(get_db),
):
    """Create a device or update its thresholds (requires h > l).

    Updating thresholds recomputes the device's alarms from its stored
    readings: alarms that no longer apply are withdrawn, others are moved.
    Whenever the thresholds actually change (a new device counts, with null
    old thresholds), one alarm revision is recorded in the same transaction;
    re-setting the same values records nothing.
    """
    now = now_iso()
    # BEGIN IMMEDIATE takes the write lock up front, so concurrent writers
    # are serialized and revision numbers / before-after states stay
    # consistent. `with conn:` commits on success, rolls back on error.
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        existing = conn.execute(
            "SELECT * FROM devices WHERE device_id = ?", (device_id,)
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO devices (device_id, threshold_h, threshold_l, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (device_id, body.h, body.l, now, now),
            )
            threshold_change = (None, None, body.h, body.l)
        else:
            conn.execute(
                "UPDATE devices SET threshold_h = ?, threshold_l = ?, updated_at = ?"
                " WHERE device_id = ?",
                (body.h, body.l, now, device_id),
            )
            threshold_change = (
                (existing["threshold_h"], existing["threshold_l"], body.h, body.l)
                if (existing["threshold_h"], existing["threshold_l"]) != (body.h, body.l)
                else None
            )
        old_alarms, new_alarms = recompute_alarms(conn, device_id)
        if threshold_change is not None:
            record_alarm_revision(
                conn,
                device_id,
                "thresholds",
                now,
                threshold_change=threshold_change,
                old_alarms=old_alarms,
                new_alarms=new_alarms,
            )
    return device_out(get_device_or_404(conn, device_id))


@app.get("/devices/{device_id}", response_model=DeviceOut)
def get_device(device_id: str, conn: Connection = Depends(get_db)):
    return device_out(get_device_or_404(conn, device_id))


@app.post("/devices/{device_id}/readings", response_model=WriteResult, status_code=201)
def write_readings(
    body: ReadingsBatchIn,
    device_id: str = Path(min_length=1, max_length=200),
    conn: Connection = Depends(get_db),
):
    """Write one or more readings (upsert semantics).

    A reading for an already-stored (device_id, sampled_at) replaces the
    previous value (correction); readings may arrive out of order (late
    arrivals). After the write, alarms are recomputed from all current
    readings. Whenever at least one sample point's temperature actually
    changes, one alarm revision is recorded in the same transaction,
    listing each changed point's old/new temperature; a batch that only
    repeats stored values records nothing.
    """
    get_device_or_404(conn, device_id)
    received = now_iso()
    # BEGIN IMMEDIATE serializes concurrent writers (see set_thresholds).
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        # Duplicate sample instants within the batch: the last entry wins.
        by_instant = {utc_iso(r.sampled_at): r.temperature for r in body.readings}
        instants = sorted(by_instant)
        placeholders = ", ".join("?" for _ in instants)
        stored = {
            row["sampled_at"]: row["temperature"]
            for row in conn.execute(
                "SELECT sampled_at, temperature FROM readings"
                f" WHERE device_id = ? AND sampled_at IN ({placeholders})",
                [device_id, *instants],
            ).fetchall()
        }
        changes = [
            (sampled_at, stored.get(sampled_at), by_instant[sampled_at])
            for sampled_at in instants
            if sampled_at not in stored or stored[sampled_at] != by_instant[sampled_at]
        ]
        conn.executemany(
            "INSERT INTO readings (device_id, sampled_at, temperature, received_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(device_id, sampled_at) DO UPDATE SET"
            "   temperature = excluded.temperature, received_at = excluded.received_at",
            [
                (device_id, sampled_at, by_instant[sampled_at], received)
                for sampled_at in instants
            ],
        )
        old_alarms, new_alarms = recompute_alarms(conn, device_id)
        if changes:
            record_alarm_revision(
                conn,
                device_id,
                "readings",
                received,
                reading_changes=changes,
                old_alarms=old_alarms,
                new_alarms=new_alarms,
            )
    return WriteResult(device_id=device_id, written=len(body.readings))


@app.get("/devices/{device_id}/readings", response_model=List[ReadingOut])
def list_readings(device_id: str, conn: Connection = Depends(get_db)):
    get_device_or_404(conn, device_id)
    rows = conn.execute(
        "SELECT * FROM readings WHERE device_id = ? ORDER BY sampled_at ASC",
        (device_id,),
    ).fetchall()
    return [
        ReadingOut(
            device_id=r["device_id"],
            sampled_at=parse_iso(r["sampled_at"]),
            temperature=r["temperature"],
            received_at=parse_iso(r["received_at"]),
        )
        for r in rows
    ]


@app.get("/devices/{device_id}/alarms", response_model=List[AlarmOut])
def list_alarms(
    device_id: str,
    response: Response,
    status: Optional[Literal["open", "closed"]] = Query(default=None),
    from_: Optional[datetime] = Query(default=None, alias="from", description="Only alarms opened at or after this time (timezone-aware)"),
    to: Optional[datetime] = Query(default=None, description="Only alarms opened at or before this time (timezone-aware)"),
    conn: Connection = Depends(get_db),
):
    """List the device's alarms with their acknowledgements.

    Each item carries ``acknowledged_by`` / ``acknowledged_at`` (null while
    unacknowledged). The response header ``X-Alarm-Revision`` returns the
    device's current alarm revision; the alarm rows, their acknowledgements
    and the revision are read inside one explicit read transaction, so they
    all come from the same snapshot.
    """
    conn.execute("BEGIN")
    try:
        get_device_or_404(conn, device_id)
        clauses = ["a.device_id = ?"]
        params: list = [device_id]
        if status == "open":
            clauses.append("a.closed_at IS NULL")
        elif status == "closed":
            clauses.append("a.closed_at IS NOT NULL")
        if from_ is not None:
            if from_.tzinfo is None:
                raise HTTPException(status_code=422, detail="'from' must include a timezone offset")
            clauses.append("a.opened_at >= ?")
            params.append(utc_iso(from_))
        if to is not None:
            if to.tzinfo is None:
                raise HTTPException(status_code=422, detail="'to' must include a timezone offset")
            clauses.append("a.opened_at <= ?")
            params.append(utc_iso(to))
        rows = conn.execute(
            "SELECT a.*, k.operator AS ack_operator, k.acknowledged_at AS ack_acknowledged_at"
            " FROM alarms a"
            " LEFT JOIN alarm_acknowledgements k"
            "   ON k.device_id = a.device_id AND k.opened_at = a.opened_at"
            f" WHERE {' AND '.join(clauses)}"
            " ORDER BY a.opened_at ASC, a.id ASC",
            params,
        ).fetchall()
        revision = current_alarm_revision(conn, device_id)
    finally:
        conn.rollback()  # read-only snapshot: nothing to commit
    response.headers["X-Alarm-Revision"] = str(revision)
    return [alarm_out(r) for r in rows]


@app.post("/devices/{device_id}/alarm-preview", response_model=AlarmPreviewOut)
def preview_alarms(
    body: AlarmPreviewIn,
    device_id: str = Path(min_length=1, max_length=200),
    conn: Connection = Depends(get_db),
):
    """Dry-run the device's alarms against candidate thresholds/readings.

    At least one of ``thresholds`` (finite, h > l) and ``readings``
    (timezone-aware, finite temperatures) must be given. The device's
    current thresholds, readings, alarms and alarm revision are read in
    one SQLite read snapshot; the candidate readings are then overlaid
    on the stored ones in memory (a candidate replaces the stored value
    at the same UTC sample instant, and within the batch the last entry
    for an instant wins), the thresholds default to the current ones
    when not supplied, and the alarms are recomputed with the usual
    rules. The response carries the base (current) revision, the
    previewed intervals and the interval diff against the current
    alarms, compared by (opened_at, closed_at) — a moved interval shows
    up as one withdrawn plus one added interval. The request is
    read-only: readings, thresholds, alarms, acknowledgements and the
    revision history are never modified. Unknown device: 404.
    """
    # One explicit read transaction: thresholds, readings, stored alarms and
    # the current revision all come from the same snapshot. The transaction
    # is only ever rolled back, so the preview cannot persist anything.
    conn.execute("BEGIN")
    try:
        device = get_device_or_404(conn, device_id)
        rows = conn.execute(
            "SELECT sampled_at, temperature FROM readings WHERE device_id = ? ORDER BY sampled_at ASC",
            (device_id,),
        ).fetchall()
        current_intervals = [
            (r["opened_at"], r["closed_at"])
            for r in conn.execute(
                "SELECT opened_at, closed_at FROM alarms WHERE device_id = ?"
                " ORDER BY opened_at ASC, id ASC",
                (device_id,),
            ).fetchall()
        ]
        revision = current_alarm_revision(conn, device_id)
    finally:
        conn.rollback()  # read-only preview: nothing to commit

    threshold_h = body.thresholds.h if body.thresholds is not None else device["threshold_h"]
    threshold_l = body.thresholds.l if body.thresholds is not None else device["threshold_l"]

    # Overlay the candidates on the stored readings in memory, keyed by UTC
    # sample instant: a candidate replaces the stored value at its instant,
    # and a later batch entry replaces an earlier one at the same instant.
    merged = {r["sampled_at"]: r["temperature"] for r in rows}
    for reading in body.readings:
        merged[utc_iso(reading.sampled_at)] = reading.temperature

    preview = compute_alarms(sorted(merged.items()), threshold_h, threshold_l)
    preview_intervals = [(a.opened_at, a.closed_at) for a in preview]
    added, withdrawn = diff_alarm_intervals(current_intervals, preview_intervals)

    def interval_out(opened, closed):
        return AlarmIntervalOut(opened_at=parse_iso(opened), closed_at=parse_iso(closed))

    return AlarmPreviewOut(
        device_id=device_id,
        base_revision=revision,
        alarms=[interval_out(opened, closed) for opened, closed in preview_intervals],
        added_alarms=[interval_out(opened, closed) for opened, closed in added],
        withdrawn_alarms=[interval_out(opened, closed) for opened, closed in withdrawn],
    )


@app.post(
    "/devices/{device_id}/alarm-acknowledgements",
    response_model=AlarmAcknowledgementOut,
    status_code=201,
)
def acknowledge_alarm(
    body: AlarmAcknowledgementIn,
    response: Response,
    device_id: str = Path(min_length=1, max_length=200),
    conn: Connection = Depends(get_db),
):
    """Acknowledge a currently open alarm, matched by its opened_at instant.

    ``opened_at`` must be timezone-aware and is matched as a UTC instant
    against the device's still-open alarms; ``operator`` must be non-empty.
    The write runs in one SQLite transaction (BEGIN IMMEDIATE) that first
    checks the device's latest alarm revision against ``expected_revision``
    (mismatch: 409 — re-fetch the alarms and retry), then the alarm's
    existence (unknown device or no such open alarm: 404). Re-acknowledging
    the same open instant is idempotent: the first operator and time are
    kept (200 instead of 201). The acknowledgement itself does not create
    an alarm revision.
    """
    get_device_or_404(conn, device_id)
    opened_at = utc_iso(body.opened_at)
    now = now_iso()
    # BEGIN IMMEDIATE takes the write lock up front, so the revision check
    # and the acknowledgement write are atomic against concurrent writers.
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        revision = current_alarm_revision(conn, device_id)
        if revision != body.expected_revision:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"alarm revision mismatch: expected {body.expected_revision},"
                    f" current is {revision}; re-fetch the alarms and retry"
                ),
            )
        alarm = conn.execute(
            "SELECT id FROM alarms"
            " WHERE device_id = ? AND opened_at = ? AND closed_at IS NULL",
            (device_id, opened_at),
        ).fetchone()
        if alarm is None:
            raise HTTPException(
                status_code=404,
                detail=f"device '{device_id}' has no open alarm opened at {opened_at}",
            )
        # Idempotent: an existing acknowledgement for this open instant wins,
        # so a repeat keeps the first operator and time.
        cursor = conn.execute(
            "INSERT OR IGNORE INTO alarm_acknowledgements"
            " (device_id, opened_at, operator, acknowledged_at)"
            " VALUES (?, ?, ?, ?)",
            (device_id, opened_at, body.operator, now),
        )
        created = cursor.rowcount > 0
        row = conn.execute(
            "SELECT * FROM alarm_acknowledgements WHERE device_id = ? AND opened_at = ?",
            (device_id, opened_at),
        ).fetchone()
    if not created:
        response.status_code = 200
    return AlarmAcknowledgementOut(
        device_id=row["device_id"],
        opened_at=parse_iso(row["opened_at"]),
        operator=row["operator"],
        acknowledged_at=parse_iso(row["acknowledged_at"]),
    )


# ------------------------------------------------------------- alarm revisions

@app.get("/devices/{device_id}/alarm-revisions", response_model=List[AlarmRevisionOut])
def list_alarm_revisions(
    device_id: str,
    after: int = Query(
        default=0, ge=0,
        description="Exclusive cursor: only revisions numbered greater than this are returned",
    ),
    limit: int = Query(
        default=100, ge=1,
        description="Maximum number of revisions to return (positive integer)",
    ),
    conn: Connection = Depends(get_db),
):
    """List the device's alarm revision audit trail, ascending by revision number.

    One entry exists per threshold PUT or readings batch POST that actually
    changed the device's thresholds or some stored temperature (a new device
    starts at revision 1 with null old thresholds; same-value retries record
    nothing). Each entry carries the input change — threshold before/after
    values or the batch's changed sample points with old/new temperatures —
    plus the alarm interval diff of the recompute: intervals are compared by
    (opened_at, closed_at), so a moved alarm appears as one withdrawn and one
    added interval. Pages are fetched with the exclusive ``after`` cursor
    (pass the last seen revision number) and a positive ``limit``.
    """
    get_device_or_404(conn, device_id)
    rev_rows = conn.execute(
        "SELECT * FROM alarm_revisions WHERE device_id = ? AND revision > ?"
        " ORDER BY revision ASC LIMIT ?",
        (device_id, after, limit),
    ).fetchall()
    if not rev_rows:
        return []
    revisions = [r["revision"] for r in rev_rows]
    placeholders = ", ".join("?" for _ in revisions)
    reading_rows = conn.execute(
        "SELECT * FROM alarm_revision_readings"
        f" WHERE device_id = ? AND revision IN ({placeholders})"
        " ORDER BY revision ASC, sampled_at ASC",
        [device_id, *revisions],
    ).fetchall()
    alarm_rows = conn.execute(
        "SELECT * FROM alarm_revision_alarms"
        f" WHERE device_id = ? AND revision IN ({placeholders})"
        " ORDER BY revision ASC, opened_at ASC",
        [device_id, *revisions],
    ).fetchall()
    readings_by_revision = {rev: [] for rev in revisions}
    for row in reading_rows:
        readings_by_revision[row["revision"]].append(row)
    alarms_by_revision = {rev: {"added": [], "withdrawn": []} for rev in revisions}
    for row in alarm_rows:
        alarms_by_revision[row["revision"]][row["change"]].append(row)
    return [
        AlarmRevisionOut(
            device_id=device_id,
            revision=r["revision"],
            op=r["op"],
            created_at=parse_iso(r["created_at"]),
            threshold_change=(
                ThresholdChangeOut(
                    old_h=r["old_threshold_h"],
                    old_l=r["old_threshold_l"],
                    new_h=r["new_threshold_h"],
                    new_l=r["new_threshold_l"],
                )
                if r["op"] == "thresholds"
                else None
            ),
            reading_changes=[
                ReadingChangeOut(
                    sampled_at=parse_iso(x["sampled_at"]),
                    old_temperature=x["old_temperature"],
                    new_temperature=x["new_temperature"],
                )
                for x in readings_by_revision[r["revision"]]
            ],
            added_alarms=[
                AlarmIntervalOut(
                    opened_at=parse_iso(a["opened_at"]),
                    closed_at=parse_iso(a["closed_at"]),
                )
                for a in alarms_by_revision[r["revision"]]["added"]
            ],
            withdrawn_alarms=[
                AlarmIntervalOut(
                    opened_at=parse_iso(a["opened_at"]),
                    closed_at=parse_iso(a["closed_at"]),
                )
                for a in alarms_by_revision[r["revision"]]["withdrawn"]
            ],
        )
        for r in rev_rows
    ]


@app.get(
    "/devices/{device_id}/alarm-revisions/{revision}/snapshot",
    response_model=AlarmRevisionSnapshotOut,
)
def alarm_revision_snapshot(
    device_id: str,
    revision: int = Path(
        ge=1,
        description="Target revision number: a positive integer not beyond the"
        " device's current alarm revision",
    ),
    conn: Connection = Depends(get_db),
):
    """Reconstruct the device's state right after the given revision completed.

    ``revision`` must be a positive integer (422 otherwise) and must not
    exceed the device's current alarm revision (a revision that does not
    exist: 404, as for an unknown device). The device's current thresholds,
    all current readings and the changes recorded by every revision after
    the target are read in one SQLite read snapshot, then rolled back in
    memory in descending revision order: a reading change whose old value
    is null removes that sample point, otherwise the old temperature is
    restored; a threshold change restores the old H/L. The restored
    readings, sorted by UTC sample instant, are fed through the usual alarm
    state machine. The response carries the target revision, the H/L in
    effect at that time, the restored readings and the recomputed alarm
    intervals — the current acknowledgements are not part of the snapshot.
    The request is read-only: readings, thresholds, alarms,
    acknowledgements and the revision history are never modified.
    """
    # One explicit read transaction: current thresholds, current readings and
    # the post-target revision changes all come from the same snapshot. The
    # transaction is only ever rolled back, so the query cannot persist
    # anything.
    conn.execute("BEGIN")
    try:
        device = get_device_or_404(conn, device_id)
        current_revision = current_alarm_revision(conn, device_id)
        if revision > current_revision:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"device '{device_id}' has no revision {revision};"
                    f" current revision is {current_revision}"
                ),
            )
        reading_rows = conn.execute(
            "SELECT sampled_at, temperature FROM readings WHERE device_id = ?",
            (device_id,),
        ).fetchall()
        revision_rows = conn.execute(
            "SELECT revision, op, old_threshold_h, old_threshold_l"
            " FROM alarm_revisions WHERE device_id = ? AND revision > ?"
            " ORDER BY revision DESC",
            (device_id, revision),
        ).fetchall()
        change_rows = conn.execute(
            "SELECT revision, sampled_at, old_temperature"
            " FROM alarm_revision_readings"
            " WHERE device_id = ? AND revision > ?"
            " ORDER BY revision DESC, sampled_at ASC",
            (device_id, revision),
        ).fetchall()
    finally:
        conn.rollback()  # read-only snapshot: nothing to commit

    # Roll the current state back to the target revision, newest revision
    # first: a threshold revision restores its old H/L; a readings revision
    # removes the sample points it created (old value null) and restores the
    # temperatures it overwrote.
    threshold_h = device["threshold_h"]
    threshold_l = device["threshold_l"]
    readings = {row["sampled_at"]: row["temperature"] for row in reading_rows}
    changes_by_revision: dict = {}
    for row in change_rows:
        changes_by_revision.setdefault(row["revision"], []).append(row)
    for rev in revision_rows:
        if rev["op"] == "thresholds":
            threshold_h = rev["old_threshold_h"]
            threshold_l = rev["old_threshold_l"]
        else:
            for change in changes_by_revision.get(rev["revision"], []):
                if change["old_temperature"] is None:
                    readings.pop(change["sampled_at"], None)
                else:
                    readings[change["sampled_at"]] = change["old_temperature"]

    # Fixed-width UTC ISO strings sort lexicographically by sample instant.
    restored = sorted(readings.items())
    alarms = compute_alarms(restored, threshold_h, threshold_l)
    return AlarmRevisionSnapshotOut(
        device_id=device_id,
        revision=revision,
        threshold_h=threshold_h,
        threshold_l=threshold_l,
        readings=[
            SnapshotReadingOut(sampled_at=parse_iso(sampled_at), temperature=temperature)
            for sampled_at, temperature in restored
        ],
        alarms=[
            AlarmIntervalOut(
                opened_at=parse_iso(a.opened_at),
                closed_at=parse_iso(a.closed_at),
            )
            for a in alarms
        ],
    )


# ------------------------------------------------- alarm consistency / reconcile

@app.get("/devices/{device_id}/alarm-consistency", response_model=AlarmConsistencyOut)
def get_alarm_consistency(
    device_id: str = Path(min_length=1, max_length=200),
    conn: Connection = Depends(get_db),
):
    """Reconcile the device's stored alarms against a fresh recompute (read-only).

    Inside one SQLite read snapshot the device's current thresholds and all
    readings are fed through the usual alarm state machine, and the
    recomputed intervals are compared item-wise with the stored alarm rows by
    (opened_at, closed_at) multiplicity — duplicated stored rows are reported
    rather than hidden. The response lists ``missing_alarms`` (recomputed but
    not stored), ``extra_alarms`` (stored beyond the recomputed multiplicity)
    and ``duplicate_alarms`` (every surplus copy of an interval stored more
    than once), all sorted by opened_at. It also lists
    ``orphan_acknowledgements``: stored acknowledgements whose opened_at has
    no corresponding recomputed alarm open instant. Unknown device: 404.
    Nothing is written.
    """
    conn.execute("BEGIN")
    try:
        device = get_device_or_404(conn, device_id)
        report = alarm_consistency_report(conn, device_id, device)
    finally:
        conn.rollback()  # read-only snapshot: nothing to commit
    return report


@app.post("/devices/{device_id}/alarm-reconcile", response_model=AlarmReconcileOut)
def reconcile_alarms(
    device_id: str = Path(min_length=1, max_length=200),
    conn: Connection = Depends(get_db),
):
    """Recheck the derived alarms and repair any inconsistency in one write transaction.

    The recheck runs inside one BEGIN IMMEDIATE transaction. When the stored
    alarms differ item-wise from the recomputed alarms (missing, extra or
    duplicate intervals), the alarm rows are rebuilt from the current
    thresholds and readings; acknowledgements whose opened_at no longer
    matches a recomputed alarm open instant are deleted in the same
    transaction, while the first acknowledgement of every open instant that
    still matches is kept (the acknowledgements table holds at most one row
    per open instant). Readings, thresholds and the alarm revision history
    are never modified, and a reconcile creates no new alarm revision. When
    everything is already consistent, no alarm row or acknowledgement is
    touched. The response carries the before/after consistency reports;
    ``fixed`` is true exactly when something was repaired. Unknown device:
    404.
    """
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        device = get_device_or_404(conn, device_id)
        before = alarm_consistency_report(conn, device_id, device)
        if before.consistent:
            # Nothing to repair: leave every alarm row and acknowledgement
            # exactly as it is (the transaction only commits no-ops).
            return AlarmReconcileOut(device_id=device_id, fixed=False, before=before, after=before)

        if before.missing_alarms or before.extra_alarms or before.duplicate_alarms:
            # Rebuild the alarm rows from the current thresholds/readings.
            device_row = conn.execute(
                "SELECT * FROM devices WHERE device_id = ?", (device_id,)
            ).fetchone()
            rows = conn.execute(
                "SELECT sampled_at, temperature FROM readings WHERE device_id = ?"
                " ORDER BY sampled_at ASC",
                (device_id,),
            ).fetchall()
            alarms = compute_alarms(
                [(r["sampled_at"], r["temperature"]) for r in rows],
                device_row["threshold_h"],
                device_row["threshold_l"],
            )
            conn.execute("DELETE FROM alarms WHERE device_id = ?", (device_id,))
            conn.executemany(
                "INSERT INTO alarms (device_id, opened_at, closed_at) VALUES (?, ?, ?)",
                [(device_id, a.opened_at, a.closed_at) for a in alarms],
            )
        # Delete acknowledgements tied to open instants the recompute no
        # longer produces; the first acknowledgement of every open instant
        # that still matches is kept (the table holds at most one per open
        # instant). This also repairs orphan acknowledgements when the alarm
        # rows themselves were already consistent.
        valid_open_instants = [
            r["opened_at"]
            for r in conn.execute(
                "SELECT DISTINCT opened_at FROM alarms WHERE device_id = ?", (device_id,)
            ).fetchall()
        ]
        if valid_open_instants:
            placeholders = ", ".join("?" for _ in valid_open_instants)
            conn.execute(
                "DELETE FROM alarm_acknowledgements"
                f" WHERE device_id = ? AND opened_at NOT IN ({placeholders})",
                [device_id, *valid_open_instants],
            )
        else:
            conn.execute(
                "DELETE FROM alarm_acknowledgements WHERE device_id = ?", (device_id,)
            )
        after = alarm_consistency_report(
            conn,
            device_id,
            conn.execute(
                "SELECT * FROM devices WHERE device_id = ?", (device_id,)
            ).fetchone(),
        )
    return AlarmReconcileOut(device_id=device_id, fixed=True, before=before, after=after)


# ------------------------------------------------- sampling interruption

@app.put("/devices/{device_id}/sampling-config", response_model=SamplingConfigOut)
def set_sampling_config(
    body: SamplingConfigIn,
    device_id: str = Path(min_length=1, max_length=200),
    conn: Connection = Depends(get_db),
):
    """Configure the sampling-interruption parameters for a device.

    ``max_interval`` (M, positive whole seconds) is the maximum allowed
    interval between adjacent samples; ``grace`` (G, non-negative whole
    seconds) is the confirmation grace for an ongoing interruption. The
    config is persisted; gap intervals are always computed on query from
    the current readings and the current M/G, so changing either takes
    effect immediately.
    """
    get_device_or_404(conn, device_id)
    now = now_iso()
    with conn:
        conn.execute(
            "INSERT INTO sampling_configs (device_id, max_interval, grace, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(device_id) DO UPDATE SET"
            "   max_interval = excluded.max_interval, grace = excluded.grace,"
            "   updated_at = excluded.updated_at",
            (device_id, body.max_interval, body.grace, now, now),
        )
    row = conn.execute(
        "SELECT * FROM sampling_configs WHERE device_id = ?", (device_id,)
    ).fetchone()
    return sampling_config_out(row)


@app.get("/devices/{device_id}/sampling-config", response_model=SamplingConfigOut)
def get_sampling_config(device_id: str, conn: Connection = Depends(get_db)):
    get_device_or_404(conn, device_id)
    row = conn.execute(
        "SELECT * FROM sampling_configs WHERE device_id = ?", (device_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=409,
            detail=f"device '{device_id}' has no sampling config; set max_interval/grace first",
        )
    return sampling_config_out(row)


@app.get("/devices/{device_id}/gaps", response_model=List[GapOut])
def list_gaps(
    device_id: str,
    as_of: datetime = Query(
        ..., description="Query instant; must include a timezone offset. Only readings sampled at or before this time are used."
    ),
    conn: Connection = Depends(get_db),
):
    """List missing-sample intervals as of the given (timezone-aware) time.

    Only readings whose sample time is <= ``as_of`` (compared in UTC) take
    part. For adjacent readings whose interval is strictly greater than M,
    a ``recovered`` gap is returned from (prev sample + M) to the next
    sample. If the last reading is more than M + G before ``as_of``, an
    ``ongoing`` gap starting at (last sample + M) with a null ``ended_at``
    is appended. No readings means no inferred gaps; an interval of exactly
    M is not a gap.
    """
    get_device_or_404(conn, device_id)
    if as_of.tzinfo is None:
        raise HTTPException(status_code=422, detail="'as_of' must include a timezone offset")
    config = conn.execute(
        "SELECT * FROM sampling_configs WHERE device_id = ?", (device_id,)
    ).fetchone()
    if config is None:
        raise HTTPException(
            status_code=409,
            detail=f"device '{device_id}' has no sampling config; set max_interval/grace first",
        )
    rows = conn.execute(
        "SELECT sampled_at, temperature FROM readings WHERE device_id = ? ORDER BY sampled_at ASC",
        (device_id,),
    ).fetchall()
    gaps = compute_gaps(
        [(r["sampled_at"], r["temperature"]) for r in rows],
        config["max_interval"],
        config["grace"],
        as_of,
        utc_format=utc_iso,
    )
    return [
        GapOut(
            started_at=parse_iso(g.started_at),
            ended_at=parse_iso(g.ended_at),
            status=g.status,
        )
        for g in gaps
    ]


# ----------------------------------------------------- alarm response deadline

@app.put("/devices/{device_id}/response-policy", response_model=ResponsePolicyOut)
def set_response_policy(
    body: ResponsePolicyIn,
    device_id: str = Path(min_length=1, max_length=200),
    conn: Connection = Depends(get_db),
):
    """Save the device's alarm response deadline D (whole positive seconds).

    Only the policy is persisted; every GET .../response-status query
    recomputes the per-alarm status from the current alarms,
    acknowledgements, sampling config M, readings and D, so replacing the
    policy takes effect on the next query. Unknown device (no thresholds):
    404; a non-positive/non-integer D: 422.
    """
    get_device_or_404(conn, device_id)
    now = now_iso()
    with conn:
        conn.execute(
            "INSERT INTO response_policies (device_id, deadline, created_at, updated_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(device_id) DO UPDATE SET"
            "   deadline = excluded.deadline, updated_at = excluded.updated_at",
            (device_id, body.deadline, now, now),
        )
    row = conn.execute(
        "SELECT * FROM response_policies WHERE device_id = ?", (device_id,)
    ).fetchone()
    return response_policy_out(row)


@app.get("/devices/{device_id}/response-policy", response_model=ResponsePolicyOut)
def get_response_policy(device_id: str, conn: Connection = Depends(get_db)):
    get_device_or_404(conn, device_id)
    row = conn.execute(
        "SELECT * FROM response_policies WHERE device_id = ?", (device_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=409,
            detail=f"device '{device_id}' has no response policy; set deadline D first",
        )
    return response_policy_out(row)


@app.get("/devices/{device_id}/response-status", response_model=List[ResponseStatusOut])
def list_response_status(
    device_id: str,
    as_of: datetime = Query(
        ..., description="Query instant; must include a timezone offset. Only readings sampled at or before this time are used."
    ),
    conn: Connection = Depends(get_db),
):
    """List still-open, unacknowledged alarms opened at or before ``as_of``.

    Only readings whose sample time is <= ``as_of`` (compared in UTC) take
    part. Observable time accumulates from the alarm's opened_at, except for
    unobservable sampling stretches clipped to [opened_at, as_of]: for
    adjacent readings more than M apart, [prev sample + M, next sample) is
    deducted; if the last reading is more than M before ``as_of``,
    [last sample + M, as_of] is deducted (an interval of exactly M is
    observable). Each item carries the cumulative observable seconds,
    whether the configured deadline D has been reached, and — if so — the
    first UTC instant at which it was reached (null otherwise). The
    cumulative measure, the reached check and the first-reached instant are
    all computed in one exact (microsecond) precision, so an accumulation
    of exactly D seconds always reports reached=true with the true instant,
    never reached=true with a null instant. The alarms,
    their acknowledgements, the sampling config M, the readings and the
    response policy D are all read inside one SQLite read snapshot; a device
    without a configured D or M returns 409, an unknown device 404 and a
    timezone-naive ``as_of`` 422. Nothing is stored, so late arrivals,
    corrections, alarm recomputes, acknowledgements and M/D changes show up
    in the next query.
    """
    if as_of.tzinfo is None:
        raise HTTPException(status_code=422, detail="'as_of' must include a timezone offset")
    as_of_utc = as_of.astimezone(timezone.utc)
    as_of_key = utc_iso(as_of_utc)
    # One explicit read transaction: the open alarms, their
    # acknowledgements, the sampling config M, the readings and the response
    # policy D all come from the same snapshot. The transaction is only ever
    # rolled back, so the query cannot persist anything.
    conn.execute("BEGIN")
    try:
        get_device_or_404(conn, device_id)
        policy = conn.execute(
            "SELECT * FROM response_policies WHERE device_id = ?", (device_id,)
        ).fetchone()
        config = conn.execute(
            "SELECT * FROM sampling_configs WHERE device_id = ?", (device_id,)
        ).fetchone()
        if policy is None or config is None:
            missing = []
            if policy is None:
                missing.append("response deadline D (PUT /devices/{device_id}/response-policy)")
            if config is None:
                missing.append("sampling interval M (PUT /devices/{device_id}/sampling-config)")
            raise HTTPException(
                status_code=409,
                detail=f"device '{device_id}' is missing: {'; '.join(missing)}",
            )
        alarm_rows = conn.execute(
            "SELECT a.opened_at FROM alarms a"
            " LEFT JOIN alarm_acknowledgements k"
            "   ON k.device_id = a.device_id AND k.opened_at = a.opened_at"
            " WHERE a.device_id = ? AND a.closed_at IS NULL"
            "   AND k.opened_at IS NULL AND a.opened_at <= ?"
            " ORDER BY a.opened_at ASC, a.id ASC",
            (device_id, as_of_key),
        ).fetchall()
        reading_rows = conn.execute(
            "SELECT sampled_at, temperature FROM readings"
            " WHERE device_id = ? AND sampled_at <= ? ORDER BY sampled_at ASC",
            (device_id, as_of_key),
        ).fetchall()
    finally:
        conn.rollback()  # read-only snapshot: nothing to commit

    readings = [(r["sampled_at"], r["temperature"]) for r in reading_rows]
    results = []
    for alarm in alarm_rows:
        status = compute_response_status(
            alarm["opened_at"],
            readings,
            config["max_interval"],
            policy["deadline"],
            as_of_utc,
            utc_format=utc_iso,
        )
        results.append(
            ResponseStatusOut(
                device_id=device_id,
                opened_at=parse_iso(alarm["opened_at"]),
                as_of=as_of_utc,
                deadline=policy["deadline"],
                observable_seconds=round(status.observable_seconds, 6),
                deadline_reached=status.deadline_reached,
                deadline_reached_at=parse_iso(status.deadline_reached_at),
            )
        )
    return results


# ------------------------------------------------------------- heat exposure

@app.get("/devices/{device_id}/heat-exposure", response_model=HeatExposureOut)
def heat_exposure(
    device_id: str,
    start: datetime = Query(..., description="Window start (inclusive); must include a timezone offset"),
    end: datetime = Query(..., description="Window end (exclusive overlap); must include a timezone offset, must be after start"),
    T: float = Query(..., alias="T", description="Finite temperature baseline"),
    conn: Connection = Depends(get_db),
):
    """Integrate temperature exposure to baseline T over a time window.

    Adjacent readings whose sampling interval is at most the configured
    maximum interval M are joined by linear temperature interpolation;
    longer gaps and the time outside the first/last reading are not
    extrapolated. Segments are clipped to [start, end] (a window boundary
    may be interpolated from the readings on either side of it) and split
    wherever the temperature crosses T. A clip boundary is interpolated
    from exact integer microsecond offsets within the segment's span in
    exact rational arithmetic (correctly rounded to float64 only at the
    end), so a boundary one microsecond before the end of a huge span —
    or a microsecond-wide window astride the span's midpoint, whose
    fraction of the span rounds to exactly 1.0 / 0.5 in float64 from
    either endpoint — keeps its true interpolated temperature instead of
    snapping to the endpoint or midpoint value; for opposite extreme
    endpoints around T = 0 that midpoint snap would otherwise cancel a
    genuine above-T sliver (and its area) to zero, on rising and falling
    segments alike. The crossing fraction and the
    (temperature - T) area are evaluated on one common power-of-two
    scale derived from the largest magnitude involved (the scale comes
    from frexp's binary exponent, so choosing it never overflows even
    for the largest finite float), and the scale is folded back into
    the area last, so legitimate finite extremes never overflow and
    the smallest positive readings never underflow the crossing
    division: a sliver of above-T time keeps its finite degree-second
    area over any window length even when the rounded above_seconds
    display is 0, and 0 -> 5e-324 at T = 0 reports full coverage with
    an area that rounds to 0. The area is infinite only when the true
    integral itself exceeds the float range. All time math is done in
    UTC, and the result is always recomputed from the current readings
    and M, so late arrivals, corrections and M changes take effect
    immediately. A device without a sampling config returns 409.
    """
    get_device_or_404(conn, device_id)
    if start.tzinfo is None:
        raise HTTPException(status_code=422, detail="'start' must include a timezone offset")
    if end.tzinfo is None:
        raise HTTPException(status_code=422, detail="'end' must include a timezone offset")
    if not math.isfinite(T):
        raise HTTPException(status_code=422, detail="'T' must be a finite number")
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    if not start_utc < end_utc:
        raise HTTPException(status_code=422, detail="'start' must be strictly before 'end'")

    config = conn.execute(
        "SELECT * FROM sampling_configs WHERE device_id = ?", (device_id,)
    ).fetchone()
    if config is None:
        raise HTTPException(
            status_code=409,
            detail=f"device '{device_id}' has no sampling config; set max_interval/grace first",
        )
    rows = conn.execute(
        "SELECT sampled_at, temperature FROM readings WHERE device_id = ? ORDER BY sampled_at ASC",
        (device_id,),
    ).fetchall()
    result = compute_heat_exposure(
        [(r["sampled_at"], r["temperature"]) for r in rows],
        config["max_interval"],
        start_utc,
        end_utc,
        T,
    )
    # Microsecond precision matches the stored timestamp resolution.
    return HeatExposureOut(
        device_id=device_id,
        start=start_utc,
        end=end_utc,
        baseline=T,
        covered_seconds=round(result.covered_seconds, 6),
        above_seconds=round(result.above_seconds, 6),
        excess_degree_seconds=round(result.excess_degree_seconds, 6),
    )


# ------------------------------------------------------------------ forecast

@app.get("/devices/{device_id}/forecast", response_model=ForecastOut)
def forecast_threshold_crossing(
    device_id: str,
    as_of: datetime = Query(
        ..., description="Query instant and zero point of the fit; must include a timezone offset"
    ),
    lookback: int = Query(
        ..., ge=1, description="Lookback window in whole seconds (>= 1): only samples in [as_of - lookback, as_of] are used"
    ),
    horizon: int = Query(
        ..., ge=1, description="Forecast horizon in whole seconds (>= 1)"
    ),
    conn: Connection = Depends(get_db),
):
    """Forecast when the temperature trend reaches the current threshold H.

    ``as_of`` must be timezone-aware; ``lookback`` and ``horizon`` must be
    positive whole seconds (422 otherwise). Only the latest contiguous
    sampling segment inside [as_of - lookback, as_of] is used: adjacent
    samples may be at most M apart and the segment is capped at 30 points
    (the newest ones). A newest sample more than M before ``as_of`` or
    fewer than 3 points yields ``insufficient_data``. Otherwise a line is
    fitted with ``as_of`` as the zero instant: the slope b is the median of
    every point pair's temperature difference divided by its time
    difference, the intercept a the median of (temperature - b * relative
    seconds). When b > 0, a < H and (H - a) / b <= ``horizon``, the status
    is ``predicted`` and the response carries the UTC crossing instant;
    every other fitted case is ``no_crossing``. The predicted instant is
    null unless the status is ``predicted``. H, M and the readings are read
    in one SQLite read snapshot and nothing is stored, so late arrivals,
    corrections and config changes take effect on the next query. Unknown
    device: 404; no sampling config (M): 409.
    """
    if as_of.tzinfo is None:
        raise HTTPException(status_code=422, detail="'as_of' must include a timezone offset")
    as_of_utc = as_of.astimezone(timezone.utc)
    # One explicit read transaction: the threshold H, the sampling config's
    # M and the readings all come from the same snapshot. The transaction is
    # only ever rolled back, so the query cannot persist anything.
    conn.execute("BEGIN")
    try:
        device = get_device_or_404(conn, device_id)
        config = conn.execute(
            "SELECT * FROM sampling_configs WHERE device_id = ?", (device_id,)
        ).fetchone()
        if config is None:
            raise HTTPException(
                status_code=409,
                detail=f"device '{device_id}' has no sampling config; set max_interval/grace first",
            )
        rows = conn.execute(
            "SELECT sampled_at, temperature FROM readings WHERE device_id = ? ORDER BY sampled_at ASC",
            (device_id,),
        ).fetchall()
    finally:
        conn.rollback()  # read-only snapshot: nothing to commit
    result = compute_forecast(
        [(r["sampled_at"], r["temperature"]) for r in rows],
        config["max_interval"],
        device["threshold_h"],
        as_of_utc,
        lookback,
        horizon,
        utc_format=utc_iso,
    )
    return ForecastOut(
        device_id=device_id,
        as_of=as_of_utc,
        lookback=lookback,
        horizon=horizon,
        threshold_h=device["threshold_h"],
        samples=result.samples,
        slope=result.slope,
        intercept=result.intercept,
        status=result.status,
        predicted_at=parse_iso(result.predicted_at),
    )


# ------------------------------------------------------------- device groups

def get_group_or_404(conn: Connection, group_id: str):
    row = conn.execute(
        "SELECT * FROM device_groups WHERE group_id = ?", (group_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"group '{group_id}' not found; save it with PUT /groups/{{group_id}} first",
        )
    return row


def group_member_ids(conn: Connection, group_id: str) -> List[str]:
    rows = conn.execute(
        "SELECT device_id FROM group_members WHERE group_id = ? ORDER BY position ASC",
        (group_id,),
    ).fetchall()
    return [r["device_id"] for r in rows]


def group_out(conn: Connection, row) -> GroupOut:
    return GroupOut(
        group_id=row["group_id"],
        device_ids=group_member_ids(conn, row["group_id"]),
        k=row["k"],
        created_at=parse_iso(row["created_at"]),
        updated_at=parse_iso(row["updated_at"]),
    )


@app.put("/groups/{group_id}", response_model=GroupOut)
def set_group(
    body: GroupIn,
    group_id: str = Path(min_length=1, max_length=200),
    conn: Connection = Depends(get_db),
):
    """Create or replace a device group (members + minimum concurrent count K).

    The member list must be non-empty and duplicate-free (422), and every
    member must be an existing device (404). K must be an integer with
    1 <= K <= member count (422). Calling again replaces the members and K
    as a whole; the config is persisted, and group alarms are recomputed
    from it on the next GET /groups/{group_id}/alarms query.
    """
    placeholders = ", ".join("?" for _ in body.device_ids)
    found = {
        r["device_id"]
        for r in conn.execute(
            f"SELECT device_id FROM devices WHERE device_id IN ({placeholders})",
            body.device_ids,
        ).fetchall()
    }
    missing = [d for d in body.device_ids if d not in found]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"unknown device(s): {', '.join(missing)}; set their thresholds first",
        )

    now = now_iso()
    with conn:
        conn.execute(
            "INSERT INTO device_groups (group_id, k, created_at, updated_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(group_id) DO UPDATE SET"
            "   k = excluded.k, updated_at = excluded.updated_at",
            (group_id, body.k, now, now),
        )
        conn.execute("DELETE FROM group_members WHERE group_id = ?", (group_id,))
        conn.executemany(
            "INSERT INTO group_members (group_id, device_id, position) VALUES (?, ?, ?)",
            [
                (group_id, device_id, position)
                for position, device_id in enumerate(body.device_ids)
            ],
        )
    return group_out(conn, get_group_or_404(conn, group_id))


@app.get("/groups/{group_id}", response_model=GroupOut)
def get_group(group_id: str, conn: Connection = Depends(get_db)):
    return group_out(conn, get_group_or_404(conn, group_id))


@app.get("/groups/{group_id}/alarms", response_model=List[GroupAlarmOut])
def list_group_alarms(group_id: str, conn: Connection = Depends(get_db)):
    """List the group's concurrent-alarm intervals, recomputed on query.

    Each current member's alarms are half-open [opened_at, closed_at)
    intervals; an unclosed alarm is still active. A group alarm opens at
    the first instant when at least K members are simultaneously in alarm
    and closes when the count drops below K. Opens and closes at the same
    instant are netted first, so a count that never actually drops below
    K does not split the interval and no zero-length interval is returned.
    Results are ordered by opened_at; an ongoing group alarm has a null
    closed_at. All times are UTC.
    """
    group = get_group_or_404(conn, group_id)
    members = group_member_ids(conn, group_id)
    if not members:
        return []
    placeholders = ", ".join("?" for _ in members)
    rows = conn.execute(
        f"SELECT opened_at, closed_at FROM alarms WHERE device_id IN ({placeholders})",
        members,
    ).fetchall()
    alarms = compute_group_alarms(
        [(r["opened_at"], r["closed_at"]) for r in rows],
        group["k"],
        utc_format=utc_iso,
    )
    return [
        GroupAlarmOut(
            opened_at=parse_iso(a.opened_at),
            closed_at=parse_iso(a.closed_at),
            status="open" if a.closed_at is None else "closed",
        )
        for a in alarms
    ]


# ------------------------------------------------------------ peer deviations

@app.get("/groups/{group_id}/peer-deviations", response_model=List[PeerDeviationOut])
def list_peer_deviations(
    group_id: str,
    device_id: str = Query(
        ..., min_length=1, max_length=200,
        description="Target device; must be a member of the group",
    ),
    start: datetime = Query(..., description="Window start (inclusive); must include a timezone offset"),
    end: datetime = Query(..., description="Window end (exclusive); must include a timezone offset, must be after start"),
    delta: float = Query(..., description="Finite positive deviation threshold"),
    min_samples: int = Query(
        ..., ge=2,
        description="Minimum number of consecutive deviation points that makes a segment (>= 2)",
    ),
    conn: Connection = Depends(get_db),
):
    """Diagnose a member's sustained temperature deviation from its group peers.

    The target's own sample points inside [start, end) are processed one by
    one in UTC order. At each point, every other member contributes its
    temperature at that instant: the stored reading when one exists,
    otherwise a linear interpolation between its adjacent readings — but
    only when their interval does not exceed that member's configured
    maximum sampling interval M (time outside a member's first/last reading
    is never extrapolated, and members without a configured M never
    participate). When at least two other members have a value, their
    median is the reference; the point deviates when the absolute
    difference exceeds ``delta``. Too few comparable members, a
    non-deviating point or a target sampling gap beyond the target's own M
    all break the consecutive run; every run of at least ``min_samples``
    deviation points is returned with its first/last instants, point count
    and maximum absolute deviation, ordered by UTC. The target must be a
    group member (404 otherwise) and have a sampling config (409
    otherwise). Members, configs and readings are read in one SQLite read
    snapshot and nothing is stored, so late arrivals, corrections and
    membership/config changes are reflected on the next query.
    """
    if start.tzinfo is None:
        raise HTTPException(status_code=422, detail="'start' must include a timezone offset")
    if end.tzinfo is None:
        raise HTTPException(status_code=422, detail="'end' must include a timezone offset")
    if not math.isfinite(delta) or delta <= 0:
        raise HTTPException(status_code=422, detail="'delta' must be a finite positive number")
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    if not start_utc < end_utc:
        raise HTTPException(status_code=422, detail="'start' must be strictly before 'end'")

    # One explicit read transaction: the group's members, their sampling
    # configs and all their readings come from the same snapshot. The
    # transaction is only ever rolled back, so the query cannot persist
    # anything.
    conn.execute("BEGIN")
    try:
        get_group_or_404(conn, group_id)
        members = group_member_ids(conn, group_id)
        if device_id not in members:
            raise HTTPException(
                status_code=404,
                detail=f"device '{device_id}' is not a member of group '{group_id}'",
            )
        placeholders = ", ".join("?" for _ in members)
        max_intervals = {
            row["device_id"]: row["max_interval"]
            for row in conn.execute(
                "SELECT device_id, max_interval FROM sampling_configs"
                f" WHERE device_id IN ({placeholders})",
                members,
            ).fetchall()
        }
        if device_id not in max_intervals:
            raise HTTPException(
                status_code=409,
                detail=f"device '{device_id}' has no sampling config; set max_interval/grace first",
            )
        reading_rows = conn.execute(
            "SELECT device_id, sampled_at, temperature FROM readings"
            f" WHERE device_id IN ({placeholders})"
            " ORDER BY sampled_at ASC",
            members,
        ).fetchall()
    finally:
        conn.rollback()  # read-only snapshot: nothing to commit

    readings_by_device: dict = {member: [] for member in members}
    for row in reading_rows:
        readings_by_device[row["device_id"]].append((row["sampled_at"], row["temperature"]))

    # Only members with a configured M take part in the peer comparison.
    peers = [
        (readings_by_device[member], max_intervals[member])
        for member in members
        if member != device_id and member in max_intervals
    ]
    segments = compute_peer_deviations(
        readings_by_device[device_id],
        max_intervals[device_id],
        peers,
        start_utc,
        end_utc,
        delta,
        min_samples,
        utc_format=utc_iso,
    )
    # Microsecond precision matches the stored timestamp resolution.
    return [
        PeerDeviationOut(
            started_at=parse_iso(s.started_at),
            ended_at=parse_iso(s.ended_at),
            samples=s.samples,
            max_abs_deviation=round(s.max_abs_deviation, 6),
        )
        for s in segments
    ]


# ------------------------------------------------------------ temperature lag

@app.get("/groups/{group_id}/lag", response_model=LagEstimateOut)
def estimate_group_lag(
    group_id: str,
    target_id: str = Query(
        ..., min_length=1, max_length=200,
        description="Target device; must be a member of the group and differ from reference_id",
    ),
    reference_id: str = Query(
        ..., min_length=1, max_length=200,
        description="Reference device; must be a member of the group and have a sampling config M",
    ),
    start: datetime = Query(..., description="Window start (inclusive); must include a timezone offset"),
    end: datetime = Query(..., description="Window end (exclusive); must include a timezone offset, must be after start"),
    max_lag: int = Query(
        ..., ge=1, le=300,
        description="Whole-second shifts -max_lag..max_lag are tried (integer 1..300)",
    ),
    min_pairs: int = Query(
        ..., ge=3,
        description="A shift qualifies only with at least this many comparable pairs (integer >= 3)",
    ),
    conn: Connection = Depends(get_db),
):
    """Estimate the integer-second temperature offset between two group members.

    Each of the target's UTC sample points ``t`` inside ``[start, end)`` is
    paired with the reference temperature at ``t + lag`` for every integer
    ``lag`` in ``[-max_lag, max_lag]``: the stored reference reading at that
    exact instant is used, otherwise the value is linearly interpolated
    between the reference readings bracketing it — but only when their
    sampling interval does not exceed the reference's configured maximum
    interval M (an interval of exactly M is allowed). Time before the
    reference's first and after its last reading is never extrapolated, and
    such target points are skipped for that shift. Shifts backed by at least
    ``min_pairs`` comparable pairs are ranked by (1) smallest median absolute
    temperature difference, (2) most pairs, (3) smallest |lag|, (4) smallest
    lag. The winner's lag, pair count and median absolute difference are
    returned; when no shift reaches ``min_pairs`` the status is
    ``insufficient_data`` with a null lag. Both devices must be group members
    and distinct (404 / 422 respectively); the reference must have a sampling
    config M, otherwise 409 (the target needs none). Members, the reference's
    sampling config and both members' readings are read in one SQLite read
    snapshot and nothing is stored, so late arrivals, corrections and
    membership/config changes are reflected on the next query.
    """
    if start.tzinfo is None:
        raise HTTPException(status_code=422, detail="'start' must include a timezone offset")
    if end.tzinfo is None:
        raise HTTPException(status_code=422, detail="'end' must include a timezone offset")
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    if not start_utc < end_utc:
        raise HTTPException(status_code=422, detail="'start' must be strictly before 'end'")
    if target_id == reference_id:
        raise HTTPException(
            status_code=422, detail="'target_id' and 'reference_id' must be different devices"
        )

    # One explicit read transaction: the group's members, the reference's
    # sampling config and both members' readings come from the same snapshot.
    # The transaction is only ever rolled back, so the query cannot persist
    # anything.
    conn.execute("BEGIN")
    try:
        get_group_or_404(conn, group_id)
        members = group_member_ids(conn, group_id)
        if target_id not in members:
            raise HTTPException(
                status_code=404,
                detail=f"device '{target_id}' is not a member of group '{group_id}'",
            )
        if reference_id not in members:
            raise HTTPException(
                status_code=404,
                detail=f"device '{reference_id}' is not a member of group '{group_id}'",
            )
        ref_config = conn.execute(
            "SELECT max_interval FROM sampling_configs WHERE device_id = ?",
            (reference_id,),
        ).fetchone()
        if ref_config is None:
            raise HTTPException(
                status_code=409,
                detail=f"reference device '{reference_id}' has no sampling config; set max_interval/grace first",
            )
        reading_rows = conn.execute(
            "SELECT device_id, sampled_at, temperature FROM readings"
            " WHERE device_id IN (?, ?) ORDER BY sampled_at ASC",
            (target_id, reference_id),
        ).fetchall()
    finally:
        conn.rollback()  # read-only snapshot: nothing to commit

    readings_by_device: dict = {target_id: [], reference_id: []}
    for row in reading_rows:
        readings_by_device[row["device_id"]].append((row["sampled_at"], row["temperature"]))

    result = compute_temperature_lag(
        readings_by_device[target_id],
        readings_by_device[reference_id],
        ref_config["max_interval"],
        start_utc,
        end_utc,
        max_lag,
        min_pairs,
    )
    return LagEstimateOut(
        group_id=group_id,
        target_id=target_id,
        reference_id=reference_id,
        start=start_utc,
        end=end_utc,
        max_lag=max_lag,
        min_pairs=min_pairs,
        status=result.status,
        lag=result.lag,
        pairs=result.pairs,
        # Microsecond precision matches the stored timestamp resolution.
        median_abs_diff=(
            round(result.median_abs_diff, 6) if result.median_abs_diff is not None else None
        ),
    )
