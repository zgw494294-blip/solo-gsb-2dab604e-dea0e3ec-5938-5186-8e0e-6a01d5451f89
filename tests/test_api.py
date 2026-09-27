import math
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def set_thresholds(client, device="dev-1", h=30.0, l=20.0):
    return client.put(f"/devices/{device}/thresholds", json={"h": h, "l": l})


def post_readings(client, readings, device="dev-1"):
    return client.post(
        f"/devices/{device}/readings",
        json={"readings": [{"sampled_at": t, "temperature": v} for t, v in readings]},
    )


def get_alarms(client, device="dev-1", **params):
    return client.get(f"/devices/{device}/alarms", params=params)


def get_revisions(client, device="dev-1", **params):
    return client.get(f"/devices/{device}/alarm-revisions", params=params)


def set_sampling(client, max_interval, grace, device="dev-1"):
    return client.put(
        f"/devices/{device}/sampling-config",
        json={"max_interval": max_interval, "grace": grace},
    )


def get_sampling(client, device="dev-1"):
    return client.get(f"/devices/{device}/sampling-config")


def get_gaps(client, as_of, device="dev-1"):
    return client.get(f"/devices/{device}/gaps", params={"as_of": as_of})


def get_heat(client, start, end, T, device="dev-1"):
    return client.get(
        f"/devices/{device}/heat-exposure",
        params={"start": start, "end": end, "T": T},
    )


def ts(minute, hour=0, tz="+00:00"):
    return f"2026-01-01T{hour:02d}:{minute:02d}:00{tz}"


# ------------------------------------------------------------ validation

def test_thresholds_require_h_greater_than_l(client):
    assert set_thresholds(client, h=20.0, l=30.0).status_code == 422
    assert set_thresholds(client, h=30.0, l=30.0).status_code == 422


def test_readings_require_timezone_aware_timestamps(client):
    set_thresholds(client)
    resp = post_readings(client, [("2026-01-01T00:00:00", 31.0)])
    assert resp.status_code == 422


def test_readings_for_unknown_device_return_404(client):
    resp = post_readings(client, [(ts(0), 31.0)], device="ghost")
    assert resp.status_code == 404


def test_non_finite_values_rejected(client):
    set_thresholds(client)
    assert post_readings(client, [(ts(0), "NaN")]).status_code == 422
    assert set_thresholds(client, h="inf", l=20.0).status_code == 422


# ------------------------------------------------------------ open / close

def test_alarm_opens_at_third_consecutive_high(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 32.0), (ts(2), 30.0)])
    alarms = get_alarms(client).json()
    assert len(alarms) == 1
    assert alarms[0]["opened_at"].startswith("2026-01-01T00:02:00")
    assert alarms[0]["closed_at"] is None
    assert alarms[0]["status"] == "open"


def test_intermediate_reading_resets_open_streak(client):
    set_thresholds(client)
    post_readings(
        client,
        [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 10.0),
         (ts(3), 31.0), (ts(4), 31.0), (ts(5), 31.0)],
    )
    alarms = get_alarms(client).json()
    assert len(alarms) == 1
    assert alarms[0]["opened_at"].startswith("2026-01-01T00:05:00")


def test_alarm_closes_at_second_consecutive_low(client):
    set_thresholds(client)
    post_readings(
        client,
        [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0),
         (ts(3), 19.0), (ts(4), 20.0)],
    )
    alarms = get_alarms(client).json()
    assert len(alarms) == 1
    assert alarms[0]["opened_at"].startswith("2026-01-01T00:02:00")
    assert alarms[0]["closed_at"].startswith("2026-01-01T00:04:00")
    assert alarms[0]["status"] == "closed"


def test_intermediate_reading_resets_close_streak(client):
    set_thresholds(client)
    post_readings(
        client,
        [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0),
         (ts(3), 19.0), (ts(4), 25.0),  # breaks the low streak
         (ts(5), 19.0), (ts(6), 18.0)],
    )
    alarms = get_alarms(client).json()
    assert alarms[0]["closed_at"].startswith("2026-01-01T00:06:00")


def test_multiple_alarm_cycles(client):
    set_thresholds(client)
    post_readings(
        client,
        [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0),   # open  @00:02
         (ts(3), 10.0), (ts(4), 10.0),                  # close @00:04
         (ts(5), 31.0), (ts(6), 31.0), (ts(7), 31.0),   # open  @00:07
         (ts(8), 10.0), (ts(9), 10.0)],                 # close @00:09
    )
    alarms = get_alarms(client).json()
    assert [(a["opened_at"][11:19], a["closed_at"][11:19]) for a in alarms] == [
        ("00:02:00", "00:04:00"),
        ("00:07:00", "00:09:00"),
    ]


def test_boundary_values_are_inclusive(client):
    set_thresholds(client, h=30.0, l=20.0)
    post_readings(
        client,
        [(ts(0), 30.0), (ts(1), 30.0), (ts(2), 30.0),  # == H triggers
         (ts(3), 20.0), (ts(4), 20.0)],                # == L recovers
    )
    alarms = get_alarms(client).json()
    assert alarms[0]["opened_at"].startswith("2026-01-01T00:02:00")
    assert alarms[0]["closed_at"].startswith("2026-01-01T00:04:00")


# ------------------------------------------------------------ corrections & late arrivals

def test_correction_withdraws_alarm(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    assert len(get_alarms(client).json()) == 1

    # Correct the middle reading: the >= H streak is broken, alarm withdrawn.
    post_readings(client, [(ts(1), 10.0)])
    assert get_alarms(client).json() == []


def test_correction_moves_alarm(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    # Correct the first reading to a low value: streak is 00:01..00:03 now.
    post_readings(client, [(ts(0), 10.0), (ts(3), 31.0), (ts(4), 31.0)])
    alarms = get_alarms(client).json()
    assert len(alarms) == 1
    assert alarms[0]["opened_at"].startswith("2026-01-01T00:03:00")


def test_late_arriving_reading_moves_alarm_earlier(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(2), 31.0), (ts(3), 31.0)])
    assert get_alarms(client).json()[0]["opened_at"].startswith("2026-01-01T00:03:00")

    # Late reading fills the gap at 00:01 -> alarm moves to 00:02.
    post_readings(client, [(ts(1), 31.0)])
    alarms = get_alarms(client).json()
    assert len(alarms) == 1
    assert alarms[0]["opened_at"].startswith("2026-01-01T00:02:00")


def test_same_instant_different_offset_is_a_correction(client):
    set_thresholds(client)
    post_readings(client, [("2026-01-01T00:00:00+00:00", 31.0)])
    # Same instant expressed in +08:00 overwrites the previous value.
    post_readings(client, [("2026-01-01T08:00:00+08:00", 10.0)])
    readings = client.get("/devices/dev-1/readings").json()
    assert len(readings) == 1
    assert readings[0]["temperature"] == 10.0


# ------------------------------------------------------------ threshold changes

def test_threshold_change_recomputes_alarms(client):
    set_thresholds(client, h=30.0, l=20.0)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    assert len(get_alarms(client).json()) == 1

    # Raising H above all readings withdraws the alarm.
    set_thresholds(client, h=35.0, l=20.0)
    assert get_alarms(client).json() == []

    # Lowering H again brings the alarm back.
    set_thresholds(client, h=31.0, l=20.0)
    alarms = get_alarms(client).json()
    assert len(alarms) == 1
    assert alarms[0]["opened_at"].startswith("2026-01-01T00:02:00")


def test_threshold_change_can_move_alarm(client):
    set_thresholds(client, h=30.0, l=20.0)
    post_readings(
        client,
        [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0), (ts(3), 33.0), (ts(4), 33.0)],
    )
    assert get_alarms(client).json()[0]["opened_at"].startswith("2026-01-01T00:02:00")

    # H=32: only the last two readings qualify -> no alarm (streak of 2).
    set_thresholds(client, h=32.0, l=20.0)
    assert get_alarms(client).json() == []

    # H=33 with one more high reading -> alarm opens at 00:05.
    post_readings(client, [(ts(5), 33.0)])
    set_thresholds(client, h=33.0, l=20.0)
    alarms = get_alarms(client).json()
    assert len(alarms) == 1
    assert alarms[0]["opened_at"].startswith("2026-01-01T00:05:00")


# ------------------------------------------------------------ querying

def test_alarm_status_filter(client):
    set_thresholds(client)
    post_readings(
        client,
        [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0),
         (ts(3), 10.0), (ts(4), 10.0),
         (ts(5), 31.0), (ts(6), 31.0), (ts(7), 31.0)],
    )
    closed = get_alarms(client, status="closed").json()
    open_ = get_alarms(client, status="open").json()
    assert len(closed) == 1 and closed[0]["closed_at"] is not None
    assert len(open_) == 1 and open_[0]["closed_at"] is None


def test_alarm_time_range_filter(client):
    set_thresholds(client)
    post_readings(
        client,
        [(ts(m), 31.0) for m in range(6)]
        + [(ts(6), 10.0), (ts(7), 10.0)]
        + [(ts(m), 31.0) for m in range(8, 11)],
    )
    alarms = get_alarms(client, **{"from": "2026-01-01T00:04:00+00:00"}).json()
    assert len(alarms) == 1
    assert alarms[0]["opened_at"].startswith("2026-01-01T00:10:00")


def test_readings_sorted_by_instant_not_offset_string(client):
    set_thresholds(client)
    post_readings(
        client,
        [
            ("2026-01-01T10:02:00+08:00", 31.0),  # 02:02 UTC
            ("2026-01-01T01:00:00+00:00", 31.0),  # 01:00 UTC (earliest)
            ("2026-01-01T02:01:00+00:00", 31.0),  # 02:01 UTC
        ],
    )
    alarms = get_alarms(client).json()
    opened = datetime.fromisoformat(alarms[0]["opened_at"])
    assert opened == datetime(2026, 1, 1, 2, 2, tzinfo=timezone.utc)


def test_devices_are_independent(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b", h=50.0, l=40.0)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)], device="a")
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)], device="b")
    assert len(get_alarms(client, device="a").json()) == 1
    assert get_alarms(client, device="b").json() == []


def test_persistence_across_restarts(tmp_path, monkeypatch):
    db = tmp_path / "persist.db"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    with TestClient(app) as c1:
        set_thresholds(c1)
        post_readings(c1, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    with TestClient(app) as c2:
        assert len(get_alarms(c2).json()) == 1
        assert c2.get("/devices/dev-1").json()["threshold_h"] == 30.0


# ============================================== sampling interruption

# ------------------------------------------------------------ config

def test_sampling_config_validation(client):
    set_thresholds(client)
    assert set_sampling(client, max_interval=0, grace=0).status_code == 422
    assert set_sampling(client, max_interval=-5, grace=0).status_code == 422
    assert set_sampling(client, max_interval=60, grace=-1).status_code == 422
    assert set_sampling(client, max_interval=1.5, grace=0).status_code == 422
    assert set_sampling(client, max_interval="sixty", grace=0).status_code == 422


def test_sampling_config_requires_existing_device(client):
    assert set_sampling(client, max_interval=60, grace=10, device="ghost").status_code == 404


def test_sampling_config_persisted_and_updatable(client):
    set_thresholds(client)
    resp = set_sampling(client, max_interval=60, grace=10)
    assert resp.status_code == 200
    assert resp.json()["max_interval"] == 60
    assert resp.json()["grace"] == 10

    assert set_sampling(client, max_interval=120, grace=0).status_code == 200
    body = get_sampling(client).json()
    assert body["max_interval"] == 120 and body["grace"] == 0

    # Whole-number floats are accepted as integers.
    assert set_sampling(client, max_interval=90.0, grace=5.0).status_code == 200


def test_missing_sampling_config_conflicts(client):
    set_thresholds(client)
    assert get_sampling(client).status_code == 409
    post_readings(client, [(ts(0), 25.0)])
    assert get_gaps(client, ts(9)).status_code == 409


# ------------------------------------------------------------ gap rules

def test_no_readings_means_no_gaps(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    assert get_gaps(client, ts(10)).json() == []


def test_interval_equal_to_m_is_not_a_gap(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0)])
    assert get_gaps(client, ts(1)).json() == []


def test_gap_larger_than_m_is_recovered(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 25.0), (ts(3), 25.0)])
    gaps = get_gaps(client, ts(3)).json()
    assert gaps == [
        {
            "started_at": "2026-01-01T00:01:00Z",
            "ended_at": "2026-01-01T00:03:00Z",
            "status": "recovered",
        }
    ]


def test_grace_boundary_for_ongoing_gap(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=30)
    post_readings(client, [(ts(0), 25.0)])
    # as_of - last == M + G exactly -> still confirmed, no ongoing gap
    assert get_gaps(client, "2026-01-01T00:01:30+00:00").json() == []
    # strictly beyond M + G -> ongoing gap from last + M, end open
    gaps = get_gaps(client, "2026-01-01T00:01:31+00:00").json()
    assert gaps == [
        {"started_at": "2026-01-01T00:01:00Z", "ended_at": None, "status": "ongoing"}
    ]


def test_zero_grace_ongoing_gap(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 25.0)])
    # exactly M seconds later -> not a gap yet
    assert get_gaps(client, ts(1)).json() == []
    gaps = get_gaps(client, ts(2)).json()
    assert gaps == [
        {"started_at": "2026-01-01T00:01:00Z", "ended_at": None, "status": "ongoing"}
    ]


def test_recovered_and_ongoing_gaps_together_and_ordered(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 25.0), (ts(3), 25.0), (ts(5), 25.0)])
    gaps = get_gaps(client, ts(7)).json()
    assert [(g["started_at"][11:19], g["ended_at"][11:19] if g["ended_at"] else None, g["status"]) for g in gaps] == [
        ("00:01:00", "00:03:00", "recovered"),
        ("00:04:00", "00:05:00", "recovered"),
        ("00:06:00", None, "ongoing"),
    ]


def test_as_of_cutoff_ignores_later_readings(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 25.0), (ts(2), 25.0), (ts(9), 25.0)])
    # At as_of=00:02 the 00:09 reading must not count: one recovered gap,
    # and 00:02 is the last reading (no ongoing).
    gaps = get_gaps(client, ts(2)).json()
    assert gaps == [
        {
            "started_at": "2026-01-01T00:01:00Z",
            "ended_at": "2026-01-01T00:02:00Z",
            "status": "recovered",
        }
    ]


def test_same_instant_correction_adds_no_sample_point(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [("2026-01-01T00:00:00+00:00", 25.0)])
    # Same instant in +08:00 is a correction, not a second sample point.
    post_readings(client, [("2026-01-01T08:00:00+08:00", 26.0)])
    assert len(client.get("/devices/dev-1/readings").json()) == 1
    gaps = get_gaps(client, ts(2)).json()
    assert gaps == [
        {"started_at": "2026-01-01T00:01:00Z", "ended_at": None, "status": "ongoing"}
    ]


def test_late_arriving_reading_closes_gap(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 25.0), (ts(3), 25.0)])
    gaps = get_gaps(client, ts(3)).json()
    assert len(gaps) == 1 and gaps[0]["status"] == "recovered"
    assert gaps[0]["started_at"][11:19] == "00:01:00"
    assert gaps[0]["ended_at"][11:19] == "00:03:00"

    # Late arrival at 00:02 shrinks the interval; 00:02 -> 00:03 is
    # exactly M, so the single remaining gap ends at 00:02.
    post_readings(client, [(ts(2), 25.0)])
    gaps = get_gaps(client, ts(3)).json()
    assert [(g["started_at"][11:19], g["ended_at"][11:19], g["status"]) for g in gaps] == [
        ("00:01:00", "00:02:00", "recovered"),
    ]


def test_changing_m_and_g_updates_gaps(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 25.0), (ts(2), 25.0)])
    assert len(get_gaps(client, ts(2)).json()) == 1

    # Widen M to 120s: 2-minute interval is exactly M -> gap disappears.
    set_sampling(client, max_interval=120, grace=0)
    assert get_gaps(client, ts(2)).json() == []

    # Back to M=60 with a large grace: last reading 00:02 at as_of 00:04 is
    # exactly M+G=180s later -> no ongoing gap... use 00:05:01 to exceed it.
    set_sampling(client, max_interval=60, grace=120)
    assert get_gaps(client, ts(4)).json() == [
        {
            "started_at": "2026-01-01T00:01:00Z",
            "ended_at": "2026-01-01T00:02:00Z",
            "status": "recovered",
        }
    ]
    gaps = get_gaps(client, "2026-01-01T00:05:01+00:00").json()
    assert [g["status"] for g in gaps] == ["recovered", "ongoing"]


def test_gaps_require_timezone_aware_as_of(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    resp = client.get("/devices/dev-1/gaps", params={"as_of": "2026-01-01T00:00:00"})
    assert resp.status_code == 422


def test_gaps_as_of_missing_is_422(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    assert client.get("/devices/dev-1/gaps").status_code == 422


def test_gaps_for_unknown_device_return_404(client):
    assert get_gaps(client, ts(1), device="ghost").status_code == 404


def test_gap_times_are_utc_regardless_of_input_offset(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(
        client,
        [("2026-01-01T08:00:00+08:00", 25.0), ("2026-01-01T08:03:00+08:00", 25.0)],
    )
    gaps = get_gaps(client, "2026-01-01T08:03:00+08:00").json()
    assert gaps[0]["started_at"] == "2026-01-01T00:01:00Z"
    assert gaps[0]["ended_at"] == "2026-01-01T00:03:00Z"

    # as_of in a different offset names the same instant -> same result
    assert get_gaps(client, "2026-01-01T00:03:00+00:00").json() == gaps


def test_sampling_config_persists_across_restarts(tmp_path, monkeypatch):
    db = tmp_path / "persist-gap.db"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    with TestClient(app) as c1:
        set_thresholds(c1)
        set_sampling(c1, max_interval=60, grace=15)
        post_readings(c1, [(ts(0), 25.0)])
    with TestClient(app) as c2:
        cfg = get_sampling(c2).json()
        assert cfg["max_interval"] == 60 and cfg["grace"] == 15
        gaps = get_gaps(c2, "2026-01-01T00:01:16+00:00").json()
        assert gaps == [
            {"started_at": "2026-01-01T00:01:00Z", "ended_at": None, "status": "ongoing"}
        ]


def test_alarms_still_work_alongside_sampling(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(4), 31.0)])
    alarms = get_alarms(client).json()
    gaps = get_gaps(client, ts(4)).json()
    assert len(alarms) == 1 and alarms[0]["status"] == "open"
    assert gaps == [
        {
            "started_at": "2026-01-01T00:02:00Z",
            "ended_at": "2026-01-01T00:04:00Z",
            "status": "recovered",
        }
    ]


def test_sampling_configs_are_per_device(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b")
    set_sampling(client, max_interval=60, grace=0, device="a")
    # b has thresholds but no sampling config -> 409, not a's config
    assert get_sampling(client, device="b").status_code == 409
    post_readings(client, [(ts(0), 25.0)], device="b")
    assert get_gaps(client, ts(5), device="b").status_code == 409


# ============================================================= heat exposure

# ------------------------------------------------------------ validation

def test_heat_exposure_requires_sampling_config(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0)])
    assert get_heat(client, ts(0), ts(2), 20.0).status_code == 409


def test_heat_exposure_unknown_device_is_404(client):
    assert get_heat(client, ts(0), ts(2), 20.0, device="ghost").status_code == 404


def test_heat_exposure_validates_params(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0)])

    # naive start / end -> 422
    assert client.get(
        "/devices/dev-1/heat-exposure",
        params={"start": "2026-01-01T00:00:00", "end": ts(2), "T": 20.0},
    ).status_code == 422
    assert client.get(
        "/devices/dev-1/heat-exposure",
        params={"start": ts(0), "end": "2026-01-01T00:02:00", "T": 20.0},
    ).status_code == 422
    # start == end and start > end -> 422
    assert get_heat(client, ts(1), ts(1), 20.0).status_code == 422
    assert get_heat(client, ts(2), ts(0), 20.0).status_code == 422
    # non-finite / missing T -> 422
    assert get_heat(client, ts(0), ts(2), "NaN").status_code == 422
    assert get_heat(client, ts(0), ts(2), "inf").status_code == 422
    assert client.get(
        "/devices/dev-1/heat-exposure", params={"start": ts(0), "end": ts(2)}
    ).status_code == 422


# ------------------------------------------------------------ coverage rules

def test_heat_exposure_constant_above_and_below(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    # [00:00,00:01] steady 22; [00:01,00:02] ramps 22 -> 10 crossing T=20 at
    # 00:01:10; [00:02,00:03] steady 10.
    post_readings(client, [(ts(0), 22.0), (ts(1), 22.0), (ts(2), 10.0), (ts(3), 10.0)])
    body = get_heat(client, ts(0), ts(4), 20.0).json()
    assert body["covered_seconds"] == 180.0
    assert body["above_seconds"] == 70.0  # 60 steady + 10 s of the ramp
    # 2 deg * 60 s + triangle (2 deg * 10 s / 2) = 130 degree-seconds
    assert body["excess_degree_seconds"] == pytest.approx(130.0)
    # echoes the request, normalized
    assert body["baseline"] == 20.0
    assert body["start"] == "2026-01-01T00:00:00Z"
    assert body["end"] == "2026-01-01T00:04:00Z"


def test_heat_exposure_crossing_inside_segment(client):
    # Linear ramp 10 -> 30 over 120 s crosses T=20 at its midpoint (00:01).
    set_thresholds(client)
    set_sampling(client, max_interval=120, grace=0)
    post_readings(client, [(ts(0), 10.0), (ts(2), 30.0)])
    body = get_heat(client, ts(0), ts(2), 20.0).json()
    assert body["covered_seconds"] == 120.0
    assert body["above_seconds"] == 60.0
    # triangle: (30-20) * 60 / 2 = 300 degree-seconds
    assert body["excess_degree_seconds"] == pytest.approx(300.0)


def test_heat_exposure_downward_crossing(client):
    set_thresholds(client)
    set_sampling(client, max_interval=120, grace=0)
    post_readings(client, [(ts(0), 30.0), (ts(2), 10.0)])
    body = get_heat(client, ts(0), ts(2), 20.0).json()
    assert body["covered_seconds"] == 120.0
    assert body["above_seconds"] == 60.0
    assert body["excess_degree_seconds"] == pytest.approx(300.0)


def test_heat_exposure_temperature_equal_to_baseline_is_not_above(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 20.0), (ts(1), 20.0)])
    body = get_heat(client, ts(0), ts(1), 20.0).json()
    assert body["covered_seconds"] == 60.0
    assert body["above_seconds"] == 0.0
    assert body["excess_degree_seconds"] == 0.0


def test_heat_exposure_clips_segments_to_window(client):
    # Reading every minute, 25 deg throughout; window [00:01, 00:03] is fully
    # covered even though its boundaries sit on the inner segments.
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(m), 25.0) for m in range(5)])
    body = get_heat(client, ts(1), ts(3), 20.0).json()
    assert body["covered_seconds"] == 120.0
    assert body["above_seconds"] == 120.0
    assert body["excess_degree_seconds"] == 5.0 * 120.0


def test_heat_exposure_window_boundary_interpolated_from_outside(client):
    # Window starts before the first reading and ends after the last one:
    # only the reading span is covered, no extrapolation, and the boundary
    # temperatures are interpolated from the flanking readings.
    set_thresholds(client)
    set_sampling(client, max_interval=120, grace=0)
    post_readings(client, [(ts(1), 10.0), (ts(3), 30.0)])  # crosses at 00:02
    body = get_heat(client, ts(0), ts(4), 20.0).json()
    assert body["covered_seconds"] == 120.0
    assert body["above_seconds"] == 60.0
    assert body["excess_degree_seconds"] == pytest.approx(300.0)


def test_heat_exposure_crossing_fraction_of_clipped_segment(client):
    # Segment 00:00=10 -> 00:02=30 (M=120); window [00:01, 00:01:30] clips
    # it to temperatures 20 -> 25: above for the whole 30 s, excess is the
    # trapezoid ((20-20) + (25-20)) * 30 / 2 = 75.
    set_thresholds(client)
    set_sampling(client, max_interval=120, grace=0)
    post_readings(client, [(ts(0), 10.0), (ts(2), 30.0)])
    body = get_heat(
        client, "2026-01-01T00:01:00+00:00", "2026-01-01T00:01:30+00:00", 20.0
    ).json()
    assert body["covered_seconds"] == 30.0
    assert body["above_seconds"] == 30.0
    assert body["excess_degree_seconds"] == pytest.approx(75.0)


def test_heat_exposure_long_gap_is_not_covered(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    # 00:00 -> 00:03 is a 180 s gap: covered segments end at 00:01 / start at 00:03.
    post_readings(
        client, [(ts(0), 25.0), (ts(1), 25.0), (ts(3), 25.0), (ts(4), 25.0)]
    )
    body = get_heat(client, ts(0), ts(5), 20.0).json()
    assert body["covered_seconds"] == 120.0  # [00:00,00:01] and [00:03,00:04]
    assert body["above_seconds"] == 120.0
    assert body["excess_degree_seconds"] == 5.0 * 120.0


def test_heat_exposure_interval_equal_to_m_is_interpolated(client):
    set_thresholds(client)
    set_sampling(client, max_interval=120, grace=0)
    post_readings(client, [(ts(0), 30.0), (ts(2), 10.0)])  # exactly M
    body = get_heat(client, ts(0), ts(2), 20.0).json()
    assert body["covered_seconds"] == 120.0
    assert body["above_seconds"] == 60.0


def test_heat_exposure_window_inside_gap_covers_nothing(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 25.0), (ts(3), 25.0)])
    body = get_heat(client, ts(1), ts(2), 20.0).json()
    assert body == {
        "device_id": "dev-1",
        "start": "2026-01-01T00:01:00Z",
        "end": "2026-01-01T00:02:00Z",
        "baseline": 20.0,
        "covered_seconds": 0.0,
        "above_seconds": 0.0,
        "excess_degree_seconds": 0.0,
    }


def test_heat_exposure_no_readings_covers_nothing(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    body = get_heat(client, ts(0), ts(2), 20.0).json()
    assert body["covered_seconds"] == 0.0
    assert body["above_seconds"] == 0.0
    assert body["excess_degree_seconds"] == 0.0


def test_heat_exposure_extreme_temperatures_stay_finite(client):
    # Legitimate finite extremes: -1e308 at 00:00:00, +1e308 at 00:00:01
    # (M=1s). The ramp crosses T=0 at the midpoint; the endpoint difference
    # 2e308 overflows the slope form, but the results must stay finite.
    set_thresholds(client)
    set_sampling(client, max_interval=1, grace=0)
    post_readings(
        client,
        [("2026-01-01T00:00:00+00:00", -1e308), ("2026-01-01T00:00:01+00:00", 1e308)],
    )
    resp = get_heat(client, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00", 0.0)
    assert resp.status_code == 200
    body = resp.json()
    assert body["covered_seconds"] == 1.0
    assert body["above_seconds"] == 0.5
    # triangle: (1e308 - 0) * 0.5 s / 2 = 2.5e307 degree-seconds
    assert body["excess_degree_seconds"] == 2.5e307

    # The downward extreme ramp is symmetric: above for the first half.
    post_readings(
        client,
        [("2026-01-01T00:00:00+00:00", 1e308), ("2026-01-01T00:00:01+00:00", -1e308)],
    )
    body = get_heat(client, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00", 0.0).json()
    assert body["covered_seconds"] == 1.0
    assert body["above_seconds"] == 0.5
    assert body["excess_degree_seconds"] == 2.5e307


def test_heat_exposure_crossing_barely_inside_extreme_ramp(client):
    # -1e308 -> 1e308 over one second (M=1s) with a baseline so close to
    # the top endpoint that the above fraction rounds to ~1e-16. The old
    # "1 - crossing_fraction" form collapsed it (and the area) to 0;
    # both must survive as finite values on the float64 ramp.
    set_thresholds(client)
    set_sampling(client, max_interval=1, grace=0)
    post_readings(
        client,
        [("2026-01-01T00:00:00+00:00", -1e308), ("2026-01-01T00:00:01+00:00", 1e308)],
    )
    resp = get_heat(
        client,
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:01+00:00",
        9.999999999999998e307,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["covered_seconds"] == 1.0
    # ~9.98e-17 s above rounds to 0 at the API's six-decimal display.
    assert body["above_seconds"] == 0.0
    # Triangle over the sliver above T: about 9.9584e275 degree-seconds.
    assert body["excess_degree_seconds"] == pytest.approx(9.9584463529e275, rel=1e-10)


def test_heat_exposure_narrow_extreme_crossing_over_longer_span(client):
    # Same extreme ramp as above but with the two readings ten seconds
    # apart and M=10, queried over the whole span. Multiplying the scale
    # back into the area before the ~1e-16 crossing fraction overflowed
    # an intermediate product (scale * 10 is already beyond the float
    # range) and returned excess_degree_seconds=inf; the true triangle
    # area is ten times the one-second value and still finite.
    set_thresholds(client)
    set_sampling(client, max_interval=10, grace=0)
    post_readings(
        client,
        [("2026-01-01T00:00:00+00:00", -1e308), ("2026-01-01T00:00:10+00:00", 1e308)],
    )
    resp = get_heat(
        client,
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:10+00:00",
        9.999999999999998e307,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["covered_seconds"] == 10.0
    # ~9.98e-16 s above still rounds to 0 at the six-decimal display.
    assert body["above_seconds"] == 0.0
    assert math.isfinite(body["excess_degree_seconds"])
    # Ten seconds of ramp makes the triangle ten times the 1 s value.
    assert body["excess_degree_seconds"] == pytest.approx(9.9584463529e276, rel=1e-10)


def test_heat_exposure_ramp_to_max_finite_float_stays_finite(client):
    # Whole-second readings -1 -> sys.float_info.max with M=1 and T=0:
    # the scale derivation ldexp(1, floor(log2(max))) asked for
    # 2**1024 and raised OverflowError. With the scale taken from
    # frexp's binary exponent the ramp computes normally: above for
    # essentially the whole second, with a triangle area of about
    # max_float / 2 degree-seconds.
    max_float = sys.float_info.max
    set_thresholds(client)
    set_sampling(client, max_interval=1, grace=0)
    post_readings(
        client,
        [
            ("2026-01-01T00:00:00+00:00", -1.0),
            ("2026-01-01T00:00:01+00:00", max_float),
        ],
    )
    resp = get_heat(
        client,
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:01+00:00",
        0.0,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["covered_seconds"] == 1.0
    assert body["above_seconds"] == 1.0
    assert math.isfinite(body["excess_degree_seconds"])
    assert body["excess_degree_seconds"] == pytest.approx(max_float / 2, rel=1e-15)


def test_heat_exposure_smallest_subnormal_ramp_at_zero_baseline(client):
    # 0 -> 5e-324 (the smallest positive subnormal) over one second with
    # T = 0: the whole segment is strictly above T. The old halved
    # differences underflowed the denominator to zero (division by
    # zero); now it interpolates normally with a zero-rounding area.
    set_thresholds(client)
    set_sampling(client, max_interval=1, grace=0)
    post_readings(
        client,
        [("2026-01-01T00:00:00+00:00", 0.0), ("2026-01-01T00:00:01+00:00", 5e-324)],
    )
    resp = get_heat(
        client,
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:01+00:00",
        0.0,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["covered_seconds"] == 1.0
    assert body["above_seconds"] == 1.0
    assert body["excess_degree_seconds"] == 0.0


def test_heat_exposure_last_microsecond_of_huge_span_keeps_boundary_offset(client):
    # M = 1e11 s with the two readings exactly that far apart
    # (2026-01-01T00:00:00Z -> 5194-11-16T09:46:40Z): the negative/positive
    # maximum finite floats, T one representable step below the max. The
    # query window is the LAST MICROSECOND of the segment: its start sits
    # at fraction 1 - 1e-17 of the span, which rounds to exactly 1.0 in
    # float64, so a boundary temperature interpolated as a float64
    # fraction of the span (from either endpoint, or any "1 - f" form)
    # snaps to the +max endpoint and the sliver is misread as a
    # constant-max segment (area = one full endpoint ulp above T,
    # ~1.9958403095e286). Exact rational interpolation from the integer
    # microsecond offset keeps the offset: the true boundary temperature
    # is -max + 2*max*(1 - 1e-17), and the trapezoid over the microsecond
    # is ~1.8160709960e286.
    max_float = sys.float_info.max
    baseline = math.nextafter(max_float, 0.0)
    set_thresholds(client)
    set_sampling(client, max_interval=100000000000, grace=0)
    post_readings(
        client,
        [
            ("2026-01-01T00:00:00+00:00", -max_float),
            ("5194-11-16T09:46:40+00:00", max_float),
        ],
    )
    resp = get_heat(
        client,
        "5194-11-16T09:46:39.999999+00:00",
        "5194-11-16T09:46:40+00:00",
        baseline,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["covered_seconds"] == 0.000001
    assert body["above_seconds"] == 0.000001
    assert math.isfinite(body["excess_degree_seconds"])
    assert body["excess_degree_seconds"] == pytest.approx(1.8160709960e286, rel=1e-9)

    # The mirrored first-microsecond window with T one step above the
    # negative extreme: the crossing sits ~5.5 us after the first reading,
    # so the first microsecond is entirely below the baseline.
    resp = get_heat(
        client,
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:00.000001+00:00",
        math.nextafter(-max_float, 0.0),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["covered_seconds"] == 0.000001
    assert body["above_seconds"] == 0.0
    assert body["excess_degree_seconds"] == 0.0


def test_heat_exposure_mid_span_microsecond_window_keeps_crossing(client):
    # Same M = 1e11 s segment as above (2026-01-01T00:00:00Z negative max
    # finite float -> 5194-11-16T09:46:40Z positive max finite float,
    # T = 0), but the query window is the single microsecond starting at
    # the segment MIDPOINT (3610-06-09T16:53:20Z), exactly where the ramp
    # crosses T. The window's offsets from both endpoints are ~5e16 of
    # 1e17 microseconds, so every float64 fraction-of-span form rounds
    # them to exactly 0.5 and both boundary heights collapse to the
    # midpoint height 0.0: the window's genuine above-T sliver and its
    # area cancelled out (midpoint cancellation). Exact rational boundary
    # interpolation keeps the microsecond offset: the window start sits
    # exactly on T and the window end a sliver above it, so the whole
    # microsecond counts with a triangle area of max_float * 1e-23
    # degree-seconds.
    max_float = sys.float_info.max
    set_thresholds(client)
    set_sampling(client, max_interval=100000000000, grace=0)
    post_readings(
        client,
        [
            ("2026-01-01T00:00:00+00:00", -max_float),
            ("5194-11-16T09:46:40+00:00", max_float),
        ],
    )
    resp = get_heat(
        client,
        "3610-06-09T16:53:20+00:00",
        "3610-06-09T16:53:20.000001+00:00",
        0.0,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["covered_seconds"] == 0.000001
    assert body["above_seconds"] == 0.000001
    assert body["excess_degree_seconds"] == pytest.approx(1.7976931349e285, rel=1e-9)

    # The microsecond just BEFORE the midpoint is entirely below T.
    body = get_heat(
        client,
        "3610-06-09T16:53:19.999999+00:00",
        "3610-06-09T16:53:20+00:00",
        0.0,
    ).json()
    assert body["covered_seconds"] == 0.000001
    assert body["above_seconds"] == 0.0
    assert body["excess_degree_seconds"] == 0.0

    # The descending ramp is handled by the same rule: mirrored across
    # the midpoint, the microsecond just before it is entirely above T
    # with the same area, and the one starting at it is entirely below.
    post_readings(
        client,
        [
            ("2026-01-01T00:00:00+00:00", max_float),
            ("5194-11-16T09:46:40+00:00", -max_float),
        ],
    )
    body = get_heat(
        client,
        "3610-06-09T16:53:19.999999+00:00",
        "3610-06-09T16:53:20+00:00",
        0.0,
    ).json()
    assert body["covered_seconds"] == 0.000001
    assert body["above_seconds"] == 0.000001
    assert body["excess_degree_seconds"] == pytest.approx(1.7976931349e285, rel=1e-9)

    body = get_heat(
        client,
        "3610-06-09T16:53:20+00:00",
        "3610-06-09T16:53:20.000001+00:00",
        0.0,
    ).json()
    assert body["covered_seconds"] == 0.000001
    assert body["above_seconds"] == 0.0
    assert body["excess_degree_seconds"] == 0.0


# ------------------------------------------------------------ freshness & UTC

def test_heat_exposure_late_arrival_fills_gap(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 25.0), (ts(3), 25.0)])
    assert get_heat(client, ts(0), ts(3), 20.0).json()["covered_seconds"] == 0.0
    # Late readings fill the gap -> full coverage.
    post_readings(client, [(ts(1), 25.0), (ts(2), 25.0)])
    body = get_heat(client, ts(0), ts(3), 20.0).json()
    assert body["covered_seconds"] == 180.0
    assert body["above_seconds"] == 180.0


def test_heat_exposure_correction_changes_result(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0)])
    assert get_heat(client, ts(0), ts(1), 20.0).json()["above_seconds"] == 60.0
    # Correct the readings down below T.
    post_readings(client, [(ts(0), 10.0), (ts(1), 10.0)])
    body = get_heat(client, ts(0), ts(1), 20.0).json()
    assert body["above_seconds"] == 0.0
    assert body["excess_degree_seconds"] == 0.0


def test_heat_exposure_changing_m_updates_result(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 30.0), (ts(2), 10.0)])  # 120 s > M
    assert get_heat(client, ts(0), ts(2), 20.0).json()["covered_seconds"] == 0.0
    # Widen M to include the interval -> the crossing segment now counts.
    set_sampling(client, max_interval=120, grace=0)
    body = get_heat(client, ts(0), ts(2), 20.0).json()
    assert body["covered_seconds"] == 120.0
    assert body["above_seconds"] == 60.0


def test_heat_exposure_input_offsets_normalized_to_utc(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(
        client,
        [("2026-01-01T08:00:00+08:00", 25.0), ("2026-01-01T08:01:00+08:00", 25.0)],
    )
    # Window named in +08:00 and in UTC is the same instant range.
    in_offset = get_heat(
        client, "2026-01-01T08:00:00+08:00", "2026-01-01T08:01:00+08:00", 20.0
    ).json()
    in_utc = get_heat(
        client, "2026-01-01T00:00:00+00:00", "2026-01-01T00:01:00+00:00", 20.0
    ).json()
    assert in_offset == in_utc
    assert in_utc["start"] == "2026-01-01T00:00:00Z"
    assert in_utc["covered_seconds"] == 60.0
    assert in_utc["excess_degree_seconds"] == 300.0


# ============================================================= device groups

def set_group(client, device_ids, k, group="grp-1"):
    return client.put(
        f"/groups/{group}", json={"device_ids": device_ids, "k": k}
    )


def get_group(client, group="grp-1"):
    return client.get(f"/groups/{group}")


def get_group_alarms(client, group="grp-1"):
    return client.get(f"/groups/{group}/alarms")


def alarm_readings(open_min, close_min=None, device_h=31.0, device_l=10.0):
    """Readings that open a device alarm at minute open_min (3rd high) and,
    if close_min is given, close it at minute close_min (2nd low)."""
    readings = [(ts(m), device_h) for m in range(open_min - 2, open_min + 1)]
    if close_min is not None:
        readings += [(ts(m), device_l) for m in range(close_min - 1, close_min + 1)]
    return readings


# ------------------------------------------------------------ group config

def test_group_requires_nonempty_member_list(client):
    assert set_group(client, [], 1).status_code == 422


def test_group_rejects_duplicate_members(client):
    set_thresholds(client, device="a")
    assert set_group(client, ["a", "a"], 1).status_code == 422


def test_group_k_must_be_within_member_count(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b")
    assert set_group(client, ["a", "b"], 0).status_code == 422
    assert set_group(client, ["a", "b"], -1).status_code == 422
    assert set_group(client, ["a", "b"], 3).status_code == 422
    assert set_group(client, ["a", "b"], 1.5).status_code == 422
    assert set_group(client, ["a", "b"], "two").status_code == 422
    # 1 <= k <= member count is accepted (whole-number floats are coerced).
    assert set_group(client, ["a", "b"], 2.0).status_code == 200


def test_group_rejects_unknown_members(client):
    set_thresholds(client, device="a")
    resp = set_group(client, ["a", "ghost"], 1)
    assert resp.status_code == 404
    assert "ghost" in resp.json()["detail"]
    # The rejected save must not have created the group.
    assert get_group(client).status_code == 404


def test_group_config_saved_echoed_and_updated(client):
    for d in ("a", "b", "c"):
        set_thresholds(client, device=d)
    resp = set_group(client, ["a", "b"], 2)
    assert resp.status_code == 200
    body = resp.json()
    assert body["group_id"] == "grp-1"
    assert body["device_ids"] == ["a", "b"]
    assert body["k"] == 2
    assert get_group(client).json()["device_ids"] == ["a", "b"]

    # A second PUT replaces members and K wholesale; submission order is echoed.
    set_group(client, ["c", "b", "a"], 3)
    body = get_group(client).json()
    assert body["device_ids"] == ["c", "b", "a"]
    assert body["k"] == 3


def test_unknown_group_is_404(client):
    assert get_group(client).status_code == 404
    assert get_group_alarms(client).status_code == 404


def test_group_config_persists_across_restarts(tmp_path, monkeypatch):
    db = tmp_path / "persist-group.db"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    with TestClient(app) as c1:
        set_thresholds(c1, device="a")
        set_thresholds(c1, device="b")
        post_readings(c1, alarm_readings(2), device="a")  # open @00:02, ongoing
        post_readings(c1, alarm_readings(4), device="b")  # open @00:04, ongoing
        set_group(c1, ["a", "b"], 2)
    with TestClient(app) as c2:
        assert get_group(c2).json()["device_ids"] == ["a", "b"]
        assert get_group(c2).json()["k"] == 2
        assert get_group_alarms(c2).json() == [
            {"opened_at": "2026-01-01T00:04:00Z", "closed_at": None, "status": "open"}
        ]


# ------------------------------------------------------------ group alarm rules

def test_group_alarm_opens_only_while_k_concurrent(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b")
    post_readings(client, alarm_readings(2, 10), device="a")   # [00:02, 00:10)
    post_readings(client, alarm_readings(6, 14), device="b")   # [00:06, 00:14)
    set_group(client, ["a", "b"], 2)
    assert get_group_alarms(client).json() == [
        {
            "opened_at": "2026-01-01T00:06:00Z",
            "closed_at": "2026-01-01T00:10:00Z",
            "status": "closed",
        }
    ]


def test_group_alarm_not_opened_below_k(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b")
    post_readings(client, alarm_readings(2, 6), device="a")    # [00:02, 00:06)
    post_readings(client, alarm_readings(8, 12), device="b")   # [00:08, 00:12)
    set_group(client, ["a", "b"], 2)
    assert get_group_alarms(client).json() == []


def test_simultaneous_open_and_close_do_not_split_group_alarm(client):
    for d in ("a", "b", "c"):
        set_thresholds(client, device=d)
    post_readings(client, alarm_readings(2, 10), device="a")   # [00:02, 00:10)
    post_readings(client, alarm_readings(6, 14), device="b")   # [00:06, 00:14)
    post_readings(client, alarm_readings(10, 16), device="c")  # [00:10, 00:16)
    set_group(client, ["a", "b", "c"], 2)
    # At 00:10 a closes and c opens at the same instant: the concurrent
    # count never actually drops below K=2, so the interval is not split.
    assert get_group_alarms(client).json() == [
        {
            "opened_at": "2026-01-01T00:06:00Z",
            "closed_at": "2026-01-01T00:14:00Z",
            "status": "closed",
        }
    ]


def test_touching_intervals_merge_and_no_zero_length(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b")
    post_readings(client, alarm_readings(2, 6), device="a")    # [00:02, 00:06)
    post_readings(client, alarm_readings(6, 10), device="b")   # [00:06, 00:10)
    set_group(client, ["a", "b"], 1)
    # a closes exactly when b opens: one continuous interval, no split and
    # no zero-length interval at the touch point.
    assert get_group_alarms(client).json() == [
        {
            "opened_at": "2026-01-01T00:02:00Z",
            "closed_at": "2026-01-01T00:10:00Z",
            "status": "closed",
        }
    ]


def test_group_alarm_ongoing_when_enough_members_unclosed(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b")
    post_readings(client, alarm_readings(2), device="a")  # [00:02, ...)
    post_readings(client, alarm_readings(6), device="b")  # [00:06, ...)
    set_group(client, ["a", "b"], 2)
    assert get_group_alarms(client).json() == [
        {"opened_at": "2026-01-01T00:06:00Z", "closed_at": None, "status": "open"}
    ]


def test_k_equal_to_member_count_requires_all(client):
    for d in ("a", "b", "c"):
        set_thresholds(client, device=d)
    post_readings(client, alarm_readings(2, 30), device="a")   # [00:02, 00:30)
    post_readings(client, alarm_readings(6, 30), device="b")   # [00:06, 00:30)
    post_readings(client, alarm_readings(10, 30), device="c")  # [00:10, 00:30)
    set_group(client, ["a", "b", "c"], 3)
    assert get_group_alarms(client).json() == [
        {
            "opened_at": "2026-01-01T00:10:00Z",
            "closed_at": "2026-01-01T00:30:00Z",
            "status": "closed",
        }
    ]


def test_group_alarms_sorted_by_opened_at(client):
    set_thresholds(client)
    post_readings(
        client,
        [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0),      # open  @00:02
         (ts(3), 10.0), (ts(4), 10.0),                     # close @00:04
         (ts(10), 31.0), (ts(11), 31.0), (ts(12), 31.0),   # open  @00:12
         (ts(13), 10.0), (ts(14), 10.0)],                  # close @00:14
    )
    set_group(client, ["dev-1"], 1)
    alarms = get_group_alarms(client).json()
    assert [(a["opened_at"][11:19], a["closed_at"][11:19]) for a in alarms] == [
        ("00:02:00", "00:04:00"),
        ("00:12:00", "00:14:00"),
    ]


def test_alarms_of_non_members_are_ignored(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b")
    set_thresholds(client, device="outsider")
    post_readings(client, alarm_readings(2), device="a")       # [00:02, ...)
    post_readings(client, alarm_readings(4), device="b")       # [00:04, ...)
    post_readings(client, alarm_readings(3), device="outsider")
    set_group(client, ["a", "b"], 2)
    assert get_group_alarms(client).json() == [
        {"opened_at": "2026-01-01T00:04:00Z", "closed_at": None, "status": "open"}
    ]


# ------------------------------------------------------------ recompute on query

def test_correction_recomputes_group_alarms(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b")
    post_readings(client, alarm_readings(2, 10), device="a")
    post_readings(client, alarm_readings(6, 14), device="b")
    set_group(client, ["a", "b"], 2)
    assert len(get_group_alarms(client).json()) == 1

    # Correct one of b's three high readings: b's alarm is withdrawn and
    # the group alarm disappears on the next query.
    post_readings(client, [(ts(5), 10.0)], device="b")
    assert get_group_alarms(client).json() == []


def test_late_arrival_moves_group_alarm_earlier(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b")
    post_readings(client, alarm_readings(2, 30), device="a")  # [00:02, 00:30)
    post_readings(client, alarm_readings(6, 30), device="b")  # [00:06, 00:30)
    set_group(client, ["a", "b"], 2)
    assert get_group_alarms(client).json()[0]["opened_at"].startswith("2026-01-01T00:06:00")

    # Late high reading for b at 00:03: b's 3rd high is now 00:05.
    post_readings(client, [(ts(3), 31.0)], device="b")
    assert get_group_alarms(client).json()[0]["opened_at"].startswith("2026-01-01T00:05:00")


def test_threshold_change_recomputes_group_alarms(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b")
    post_readings(client, alarm_readings(2, 10), device="a")
    post_readings(client, alarm_readings(6, 14), device="b")
    set_group(client, ["a", "b"], 2)
    assert len(get_group_alarms(client).json()) == 1

    # Raising a's H above its readings withdraws a's alarm -> no group alarm.
    set_thresholds(client, device="a", h=35.0, l=20.0)
    assert get_group_alarms(client).json() == []

    # Lowering H again brings a's alarm and the group alarm back.
    set_thresholds(client, device="a", h=31.0, l=20.0)
    assert len(get_group_alarms(client).json()) == 1


def test_member_and_k_changes_recompute_on_query(client):
    for d in ("a", "b", "c"):
        set_thresholds(client, device=d)
    post_readings(client, alarm_readings(2, 30), device="a")   # [00:02, 00:30)
    post_readings(client, alarm_readings(6, 30), device="b")   # [00:06, 00:30)
    post_readings(client, alarm_readings(10, 30), device="c")  # [00:10, 00:30)
    set_group(client, ["a", "b", "c"], 2)
    assert get_group_alarms(client).json()[0]["opened_at"].startswith("2026-01-01T00:06:00")

    # Raising K to 3 narrows the window to the triple overlap.
    set_group(client, ["a", "b", "c"], 3)
    assert [
        (al["opened_at"][11:19], al["closed_at"][11:19])
        for al in get_group_alarms(client).json()
    ] == [("00:10:00", "00:30:00")]

    # Replacing the members recomputes from the remaining device only.
    set_group(client, ["a"], 1)
    assert [
        (al["opened_at"][11:19], al["closed_at"][11:19])
        for al in get_group_alarms(client).json()
    ] == [("00:02:00", "00:30:00")]


def test_group_alarm_times_are_utc_regardless_of_member_offsets(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b")
    # 08:00..08:02 +08:00 == 00:00..00:02 UTC: a opens at 00:02 UTC.
    post_readings(
        client,
        [("2026-01-01T08:00:00+08:00", 31.0),
         ("2026-01-01T08:01:00+08:00", 31.0),
         ("2026-01-01T08:02:00+08:00", 31.0)],
        device="a",
    )
    post_readings(client, alarm_readings(4), device="b")  # opens 00:04 UTC
    set_group(client, ["a", "b"], 2)
    assert get_group_alarms(client).json() == [
        {"opened_at": "2026-01-01T00:04:00Z", "closed_at": None, "status": "open"}
    ]


# ========================================================== alarm revisions

# ------------------------------------------------------------ recording rules

def test_device_creation_records_first_revision(client):
    set_thresholds(client, h=30.0, l=20.0)
    revs = get_revisions(client).json()
    assert len(revs) == 1
    rev = revs[0]
    assert rev["device_id"] == "dev-1"
    assert rev["revision"] == 1
    assert rev["op"] == "thresholds"
    assert rev["threshold_change"] == {
        "old_h": None, "old_l": None, "new_h": 30.0, "new_l": 20.0
    }
    assert rev["reading_changes"] == []
    assert rev["added_alarms"] == []
    assert rev["withdrawn_alarms"] == []
    datetime.fromisoformat(rev["created_at"])  # parseable timestamp


def test_same_thresholds_retry_records_no_revision(client):
    set_thresholds(client)
    set_thresholds(client)
    set_thresholds(client)
    assert [r["revision"] for r in get_revisions(client).json()] == [1]


def test_threshold_change_revision_records_old_new_and_alarm_diff(client):
    set_thresholds(client, h=30.0, l=20.0)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])

    # Raising H above all readings withdraws the alarm.
    set_thresholds(client, h=35.0, l=20.0)
    revs = get_revisions(client).json()
    assert [r["revision"] for r in revs] == [1, 2, 3]
    rev = revs[2]
    assert rev["op"] == "thresholds"
    assert rev["threshold_change"] == {
        "old_h": 30.0, "old_l": 20.0, "new_h": 35.0, "new_l": 20.0
    }
    assert rev["reading_changes"] == []
    assert rev["withdrawn_alarms"] == [
        {"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}
    ]
    assert rev["added_alarms"] == []

    # Lowering H again brings the same interval back as an addition.
    set_thresholds(client, h=31.0, l=20.0)
    rev = get_revisions(client).json()[3]
    assert rev["threshold_change"] == {
        "old_h": 35.0, "old_l": 20.0, "new_h": 31.0, "new_l": 20.0
    }
    assert rev["added_alarms"] == [
        {"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}
    ]
    assert rev["withdrawn_alarms"] == []


def test_threshold_change_without_alarm_effect_still_records_revision(client):
    set_thresholds(client, h=30.0, l=20.0)
    post_readings(client, [(ts(0), 25.0)])  # no alarm before or after
    set_thresholds(client, h=31.0, l=21.0)
    rev = get_revisions(client).json()[2]
    assert rev["op"] == "thresholds"
    assert rev["added_alarms"] == [] and rev["withdrawn_alarms"] == []


def test_readings_revision_records_changes_and_alarm_diff(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    revs = get_revisions(client).json()
    assert [r["revision"] for r in revs] == [1, 2]
    rev = revs[1]
    assert rev["op"] == "readings"
    assert rev["threshold_change"] is None
    assert rev["reading_changes"] == [
        {"sampled_at": "2026-01-01T00:00:00Z", "old_temperature": None, "new_temperature": 31.0},
        {"sampled_at": "2026-01-01T00:01:00Z", "old_temperature": None, "new_temperature": 31.0},
        {"sampled_at": "2026-01-01T00:02:00Z", "old_temperature": None, "new_temperature": 31.0},
    ]
    assert rev["added_alarms"] == [
        {"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}
    ]
    assert rev["withdrawn_alarms"] == []


def test_correction_revision_records_old_and_new_temperature(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    # Correct the middle reading: the >= H streak breaks, alarm withdrawn.
    post_readings(client, [(ts(1), 10.0)])
    rev = get_revisions(client).json()[2]
    assert rev["reading_changes"] == [
        {"sampled_at": "2026-01-01T00:01:00Z", "old_temperature": 31.0, "new_temperature": 10.0}
    ]
    assert rev["withdrawn_alarms"] == [
        {"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}
    ]
    assert rev["added_alarms"] == []


def test_moved_alarm_recorded_as_withdraw_and_add(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(2), 31.0), (ts(3), 31.0)])  # open @00:03
    # Late reading at 00:01 moves the alarm earlier: old interval withdrawn,
    # new interval added (compared by times, not alarm row id).
    post_readings(client, [(ts(1), 31.0)])
    rev = get_revisions(client).json()[2]
    assert rev["withdrawn_alarms"] == [
        {"opened_at": "2026-01-01T00:03:00Z", "closed_at": None}
    ]
    assert rev["added_alarms"] == [
        {"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}
    ]


def test_same_value_readings_retry_records_no_revision(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0)])
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0)])  # identical retry
    post_readings(client, [(ts(0), 31.0)])                # identical retry
    assert [r["revision"] for r in get_revisions(client).json()] == [1, 2]


def test_batch_records_only_changed_points(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0)])
    # One unchanged point, one correction, one new point in the same batch.
    post_readings(client, [(ts(0), 31.0), (ts(1), 10.0), (ts(2), 25.0)])
    rev = get_revisions(client).json()[2]
    assert rev["reading_changes"] == [
        {"sampled_at": "2026-01-01T00:01:00Z", "old_temperature": 31.0, "new_temperature": 10.0},
        {"sampled_at": "2026-01-01T00:02:00Z", "old_temperature": None, "new_temperature": 25.0},
    ]


def test_batch_same_instant_last_write_wins(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0)])
    # The same instant appears three times: only the last value counts.
    post_readings(client, [(ts(0), 10.0), (ts(0), 99.0), (ts(0), 25.0), (ts(1), 26.0)])
    rev = get_revisions(client).json()[2]
    assert rev["reading_changes"] == [
        {"sampled_at": "2026-01-01T00:00:00Z", "old_temperature": 31.0, "new_temperature": 25.0},
        {"sampled_at": "2026-01-01T00:01:00Z", "old_temperature": None, "new_temperature": 26.0},
    ]
    readings = client.get("/devices/dev-1/readings").json()
    assert [(r["sampled_at"], r["temperature"]) for r in readings] == [
        ("2026-01-01T00:00:00Z", 25.0),
        ("2026-01-01T00:01:00Z", 26.0),
    ]


def test_batch_settling_on_stored_value_records_no_revision(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0)])
    # Same instant twice, ending on the stored value: net no change.
    post_readings(client, [(ts(0), 10.0), (ts(0), 31.0)])
    assert [r["revision"] for r in get_revisions(client).json()] == [1, 2]
    assert client.get("/devices/dev-1/readings").json()[0]["temperature"] == 31.0


def test_revision_numbers_are_per_device(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b")
    post_readings(client, [(ts(0), 31.0)], device="a")
    post_readings(client, [(ts(0), 25.0), (ts(1), 26.0)], device="b")
    post_readings(client, [(ts(1), 32.0)], device="a")
    assert [r["revision"] for r in get_revisions(client, device="a").json()] == [1, 2, 3]
    assert [r["revision"] for r in get_revisions(client, device="b").json()] == [1, 2]


# ------------------------------------------------------------ querying

def test_revisions_for_unknown_device_return_404(client):
    assert get_revisions(client, device="ghost").status_code == 404


def test_revisions_paginated_with_after_and_limit(client):
    set_thresholds(client)
    for m in range(5):
        post_readings(client, [(ts(m), 20.0 + m)])  # revisions 2..6

    page1 = get_revisions(client, limit=2).json()
    assert [r["revision"] for r in page1] == [1, 2]
    # Exclusive cursor: the page resumes strictly after the last seen revision.
    page2 = get_revisions(client, after=page1[-1]["revision"], limit=2).json()
    assert [r["revision"] for r in page2] == [3, 4]
    page3 = get_revisions(client, after=page2[-1]["revision"], limit=2).json()
    assert [r["revision"] for r in page3] == [5, 6]
    assert get_revisions(client, after=6, limit=2).json() == []
    # A larger limit returns everything in one ascending page.
    assert [r["revision"] for r in get_revisions(client, limit=100).json()] == [1, 2, 3, 4, 5, 6]


def test_revisions_pagination_params_validated(client):
    set_thresholds(client)
    assert get_revisions(client, limit=0).status_code == 422
    assert get_revisions(client, limit=-3).status_code == 422
    assert get_revisions(client, limit=1.5).status_code == 422
    assert get_revisions(client, limit="ten").status_code == 422
    assert get_revisions(client, after=-1).status_code == 422
    assert get_revisions(client, after=1.5).status_code == 422
    assert get_revisions(client, after="x").status_code == 422


def test_revisions_persist_across_restarts(tmp_path, monkeypatch):
    db = tmp_path / "persist-rev.db"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    with TestClient(app) as c1:
        set_thresholds(c1)
        post_readings(c1, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    with TestClient(app) as c2:
        revs = get_revisions(c2).json()
        assert [r["revision"] for r in revs] == [1, 2]
        assert revs[1]["added_alarms"] == [
            {"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}
        ]
        # The history continues where it left off.
        post_readings(c2, [(ts(1), 10.0)])
        revs = get_revisions(c2).json()
        assert [r["revision"] for r in revs] == [1, 2, 3]
        assert revs[2]["reading_changes"] == [
            {"sampled_at": "2026-01-01T00:01:00Z", "old_temperature": 31.0, "new_temperature": 10.0}
        ]


def test_preexisting_database_starts_recording_at_first_new_write(tmp_path, monkeypatch):
    db = tmp_path / "pre-audit.db"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    with TestClient(app) as c1:
        set_thresholds(c1)
        post_readings(c1, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    # Simulate a database from before the audit feature existed.
    conn = sqlite3.connect(str(db))
    conn.execute("DROP TABLE alarm_revision_alarms")
    conn.execute("DROP TABLE alarm_revision_readings")
    conn.execute("DROP TABLE alarm_revisions")
    conn.commit()
    conn.close()

    with TestClient(app) as c2:
        # Old writes have no recorded history...
        assert get_revisions(c2).json() == []
        # ...but the first new write becomes revision 1, against the real
        # pre-existing state.
        post_readings(c2, [(ts(1), 10.0)])  # withdraws the alarm
        revs = get_revisions(c2).json()
        assert [r["revision"] for r in revs] == [1]
        rev = revs[0]
        assert rev["reading_changes"] == [
            {"sampled_at": "2026-01-01T00:01:00Z", "old_temperature": 31.0, "new_temperature": 10.0}
        ]
        assert rev["withdrawn_alarms"] == [
            {"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}
        ]


# ------------------------------------------------------------ concurrency

def test_concurrent_reading_writes_keep_revisions_consistent(client):
    set_thresholds(client)

    def post(m):
        resp = post_readings(client, [(ts(m), 31.0)])
        assert resp.status_code == 201

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(post, range(16)))

    revs = get_revisions(client, limit=100).json()
    # 1 creation + 16 effective batches: consecutive numbers, no gaps/dupes.
    assert [r["revision"] for r in revs] == list(range(1, 18))
    # Each batch added exactly one new sample point, and together the
    # recorded changes reconstruct exactly the stored readings.
    changed = {}
    for rev in revs[1:]:
        assert rev["op"] == "readings"
        assert len(rev["reading_changes"]) == 1
        ch = rev["reading_changes"][0]
        assert ch["old_temperature"] is None
        changed[ch["sampled_at"]] = ch["new_temperature"]
    stored = {
        r["sampled_at"]: r["temperature"]
        for r in client.get("/devices/dev-1/readings").json()
    }
    assert changed == stored


def test_concurrent_threshold_and_reading_writes_keep_revisions_consistent(client):
    set_thresholds(client, h=30.0, l=20.0)

    def put(i):
        resp = set_thresholds(client, h=31.0 + i, l=20.0)  # always a real change
        assert resp.status_code == 200

    def post(i):
        resp = post_readings(client, [(ts(i), 31.0)])
        assert resp.status_code == 201

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(put, i) for i in range(8)]
        futures += [pool.submit(post, i) for i in range(8)]
        for f in futures:
            f.result()

    revs = get_revisions(client, limit=100).json()
    assert [r["revision"] for r in revs] == list(range(1, 18))  # 1 + 8 + 8
    # The threshold before/after chain is consistent in revision order, and
    # the device's final thresholds are the last revision's new values.
    threshold_revs = [r for r in revs if r["op"] == "thresholds"]
    assert len(threshold_revs) == 9
    assert threshold_revs[0]["threshold_change"]["old_h"] is None
    for prev, cur in zip(threshold_revs, threshold_revs[1:]):
        assert cur["threshold_change"]["old_h"] == prev["threshold_change"]["new_h"]
        assert cur["threshold_change"]["old_l"] == prev["threshold_change"]["new_l"]
    final = client.get("/devices/dev-1").json()
    assert final["threshold_h"] == threshold_revs[-1]["threshold_change"]["new_h"]
    assert final["threshold_l"] == threshold_revs[-1]["threshold_change"]["new_l"]


# ===================================================== alarm acknowledgements

def open_dev1_alarm(client, readings=None):
    """Create dev-1 with one open alarm (opened at 00:02 by default)."""
    set_thresholds(client)
    post_readings(client, readings or [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])


def ack(client, opened_at="2026-01-01T00:02:00+00:00", operator="alice",
        expected_revision=2, device="dev-1"):
    return client.post(
        f"/devices/{device}/alarm-acknowledgements",
        json={
            "opened_at": opened_at,
            "operator": operator,
            "expected_revision": expected_revision,
        },
    )


# ------------------------------------------------------------ reading

def test_alarms_carry_ack_fields_and_revision_header(client):
    open_dev1_alarm(client)
    resp = get_alarms(client)
    assert resp.headers["x-alarm-revision"] == "2"
    alarm = resp.json()[0]
    assert alarm["acknowledged_by"] is None
    assert alarm["acknowledged_at"] is None


def test_revision_header_tracks_alarm_revisions(client):
    set_thresholds(client)
    assert get_alarms(client).headers["x-alarm-revision"] == "1"
    post_readings(client, [(ts(0), 31.0)])
    assert get_alarms(client).headers["x-alarm-revision"] == "2"
    post_readings(client, [(ts(0), 31.0)])  # same-value retry: no new revision
    assert get_alarms(client).headers["x-alarm-revision"] == "2"


# ------------------------------------------------------------ acknowledging

def test_acknowledge_open_alarm(client):
    open_dev1_alarm(client)
    # The same instant named in +08:00 matches the alarm opened at 00:02 UTC.
    resp = ack(client, opened_at="2026-01-01T08:02:00+08:00")
    assert resp.status_code == 201
    body = resp.json()
    assert body["device_id"] == "dev-1"
    assert body["opened_at"] == "2026-01-01T00:02:00Z"
    assert body["operator"] == "alice"
    datetime.fromisoformat(body["acknowledged_at"])  # parseable timestamp

    resp = get_alarms(client)
    # Acknowledging does not create an alarm revision.
    assert resp.headers["x-alarm-revision"] == "2"
    alarm = resp.json()[0]
    assert alarm["acknowledged_by"] == "alice"
    assert alarm["acknowledged_at"] == body["acknowledged_at"]


def test_acknowledge_unknown_device_is_404(client):
    assert ack(client, device="ghost").status_code == 404


def test_acknowledge_without_matching_open_alarm_is_404(client):
    open_dev1_alarm(client)
    # No alarm opened at this instant.
    assert ack(client, opened_at="2026-01-01T00:03:00+00:00").status_code == 404
    # Close the alarm: its interval exists but is no longer open.
    post_readings(client, [(ts(3), 10.0), (ts(4), 10.0)])  # revision 3
    assert get_alarms(client).json()[0]["status"] == "closed"
    assert ack(client, expected_revision=3).status_code == 404


def test_acknowledge_revision_mismatch_is_409(client):
    open_dev1_alarm(client)  # current revision is 2
    assert ack(client, expected_revision=1).status_code == 409
    assert ack(client, expected_revision=3).status_code == 409
    # A write bumps the revision: the stale expectation is rejected even
    # though the alarm itself is still open.
    post_readings(client, [(ts(3), 25.0)])  # revision 3
    assert ack(client, expected_revision=2).status_code == 409
    assert ack(client, expected_revision=3).status_code == 201


def test_acknowledge_validates_body(client):
    open_dev1_alarm(client)
    assert ack(client, opened_at="2026-01-01T00:02:00").status_code == 422  # naive
    assert ack(client, operator="").status_code == 422
    assert ack(client, operator="   ").status_code == 422
    assert ack(client, expected_revision=-1).status_code == 422
    resp = client.post(
        "/devices/dev-1/alarm-acknowledgements",
        json={"opened_at": "2026-01-01T00:02:00Z", "operator": "alice"},
    )
    assert resp.status_code == 422  # missing expected_revision


def test_repeat_acknowledge_keeps_first_operator_and_time(client):
    open_dev1_alarm(client)
    first = ack(client, operator="alice")
    assert first.status_code == 201
    again = ack(client, operator="bob")
    assert again.status_code == 200
    assert again.json()["operator"] == "alice"
    assert again.json()["acknowledged_at"] == first.json()["acknowledged_at"]
    alarm = get_alarms(client).json()[0]
    assert alarm["acknowledged_by"] == "alice"
    assert alarm["acknowledged_at"] == first.json()["acknowledged_at"]


# ------------------------------------------------------------ recompute retention

def test_ack_survives_alarm_close(client):
    open_dev1_alarm(client)
    assert ack(client).status_code == 201
    post_readings(client, [(ts(3), 10.0), (ts(4), 10.0)])  # closes at 00:04
    resp = get_alarms(client)
    assert resp.headers["x-alarm-revision"] == "3"
    alarm = resp.json()[0]
    assert alarm["status"] == "closed"
    assert alarm["acknowledged_by"] == "alice"
    assert alarm["acknowledged_at"] is not None


def test_ack_kept_when_only_close_time_changes(client):
    set_thresholds(client)
    post_readings(
        client,
        [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0),  # open @00:02
         (ts(3), 10.0), (ts(4), 25.0)],                 # close streak broken: still open
    )
    assert ack(client).status_code == 201
    first_ack_at = get_alarms(client).json()[0]["acknowledged_at"]
    # Correcting 00:04 to a low value closes the alarm at 00:04; the open
    # instant is unchanged, so the acknowledgement is kept.
    post_readings(client, [(ts(4), 10.0)])
    alarm = get_alarms(client).json()[0]
    assert alarm["status"] == "closed"
    assert alarm["closed_at"].startswith("2026-01-01T00:04:00")
    assert alarm["acknowledged_by"] == "alice"
    assert alarm["acknowledged_at"] == first_ack_at


def test_ack_dropped_when_open_instant_moves(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(2), 31.0), (ts(3), 31.0)])  # opens 00:03
    assert ack(client, opened_at="2026-01-01T00:03:00+00:00").status_code == 201
    # A late reading fills the gap: the alarm moves to 00:02 and the
    # acknowledgement tied to 00:03 is dropped in the same transaction.
    post_readings(client, [(ts(1), 31.0)])
    alarm = get_alarms(client).json()[0]
    assert alarm["opened_at"].startswith("2026-01-01T00:02:00")
    assert alarm["acknowledged_by"] is None
    assert alarm["acknowledged_at"] is None


def test_ack_dropped_when_alarm_withdrawn_by_correction(client):
    open_dev1_alarm(client)
    assert ack(client).status_code == 201
    post_readings(client, [(ts(1), 10.0)])  # breaks the >= H streak
    assert get_alarms(client).json() == []


def test_ack_dropped_when_alarm_withdrawn_by_threshold_change(client):
    open_dev1_alarm(client)
    assert ack(client).status_code == 201
    set_thresholds(client, h=35.0, l=20.0)  # alarm withdrawn
    assert get_alarms(client).json() == []
    # Lowering H brings back an alarm at the same instant, but the
    # acknowledgement is gone: it is not resurrected.
    set_thresholds(client, h=31.0, l=20.0)
    alarm = get_alarms(client).json()[0]
    assert alarm["opened_at"].startswith("2026-01-01T00:02:00")
    assert alarm["acknowledged_by"] is None
    assert alarm["acknowledged_at"] is None


# ------------------------------------------------------------ persistence

def test_ack_persists_across_restarts(tmp_path, monkeypatch):
    db = tmp_path / "persist-ack.db"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    with TestClient(app) as c1:
        open_dev1_alarm(c1)
        assert ack(c1, operator="carol").status_code == 201
    with TestClient(app) as c2:
        resp = get_alarms(c2)
        assert resp.headers["x-alarm-revision"] == "2"
        alarm = resp.json()[0]
        assert alarm["acknowledged_by"] == "carol"
        assert alarm["acknowledged_at"] is not None
        # A repeat acknowledgement after the restart is still idempotent.
        again = ack(c2, operator="dave")
        assert again.status_code == 200
        assert again.json()["operator"] == "carol"


# ================================================================ alarm preview

def preview(client, payload, device="dev-1"):
    return client.post(f"/devices/{device}/alarm-preview", json=payload)


def preview_readings(readings):
    return {"readings": [{"sampled_at": t, "temperature": v} for t, v in readings]}


# ------------------------------------------------------------ validation

def test_preview_requires_at_least_one_input(client):
    set_thresholds(client)
    assert preview(client, {}).status_code == 422
    assert preview(client, {"readings": []}).status_code == 422
    assert preview(client, {"thresholds": None, "readings": []}).status_code == 422
    # Either input alone is enough.
    assert preview(client, {"thresholds": {"h": 30.0, "l": 20.0}}).status_code == 200
    assert preview(client, preview_readings([(ts(0), 31.0)])).status_code == 200


def test_preview_validates_thresholds(client):
    set_thresholds(client)
    assert preview(client, {"thresholds": {"h": 20.0, "l": 30.0}}).status_code == 422
    assert preview(client, {"thresholds": {"h": 30.0, "l": 30.0}}).status_code == 422
    assert preview(client, {"thresholds": {"h": "inf", "l": 20.0}}).status_code == 422
    assert preview(client, {"thresholds": {"h": 30.0, "l": "NaN"}}).status_code == 422


def test_preview_validates_candidate_readings(client):
    set_thresholds(client)
    assert preview(client, preview_readings([(ts(0), "NaN")])).status_code == 422
    assert preview(client, preview_readings([(ts(0), "inf")])).status_code == 422
    # Naive (timezone-less) sample times are rejected.
    assert preview(client, preview_readings([("2026-01-01T00:00:00", 31.0)])).status_code == 422


def test_preview_for_unknown_device_returns_404(client):
    assert preview(client, {"thresholds": {"h": 30.0, "l": 20.0}}, device="ghost").status_code == 404
    assert preview(client, preview_readings([(ts(0), 31.0)]), device="ghost").status_code == 404


# ------------------------------------------------------------ preview semantics

def test_preview_with_candidate_thresholds_only(client):
    set_thresholds(client, h=30.0, l=20.0)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])  # open @00:02
    body = preview(client, {"thresholds": {"h": 35.0, "l": 20.0}}).json()
    assert body["device_id"] == "dev-1"
    assert body["base_revision"] == 2
    # Raising H above all readings withdraws the current alarm.
    assert body["alarms"] == []
    assert body["added_alarms"] == []
    assert body["withdrawn_alarms"] == [
        {"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}
    ]


def test_preview_with_candidate_readings_only(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 25.0)])  # stored: no alarm
    body = preview(
        client, preview_readings([(ts(1), 31.0), (ts(2), 31.0), (ts(3), 31.0)])
    ).json()
    assert body["base_revision"] == 2
    assert body["alarms"] == [{"opened_at": "2026-01-01T00:03:00Z", "closed_at": None}]
    assert body["added_alarms"] == [{"opened_at": "2026-01-01T00:03:00Z", "closed_at": None}]
    assert body["withdrawn_alarms"] == []


def test_preview_with_thresholds_and_readings_together(client):
    set_thresholds(client, h=30.0, l=20.0)
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0)])
    # Candidate readings alone stay below the current H=30; with the
    # candidate H=26 the same readings open an alarm at 00:04.
    payload = preview_readings([(ts(2), 27.0), (ts(3), 27.0), (ts(4), 27.0)])
    assert preview(client, payload).json()["alarms"] == []
    payload["thresholds"] = {"h": 26.0, "l": 20.0}
    body = preview(client, payload).json()
    assert body["alarms"] == [{"opened_at": "2026-01-01T00:04:00Z", "closed_at": None}]
    assert body["added_alarms"] == body["alarms"]
    assert body["withdrawn_alarms"] == []


def test_preview_candidate_readings_overlay_stored_by_utc_instant(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])  # open @00:02
    # A candidate at the same instant as a stored reading (named in +08:00)
    # replaces it: the >= H streak breaks and the alarm is withdrawn.
    body = preview(client, preview_readings([("2026-01-01T08:01:00+08:00", 10.0)])).json()
    assert body["alarms"] == []
    assert body["withdrawn_alarms"] == [
        {"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}
    ]
    assert body["added_alarms"] == []


def test_preview_batch_same_instant_last_entry_wins(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])  # open @00:02
    # 00:01 appears twice: the last value (10.0) wins and breaks the streak.
    payload = preview_readings([(ts(1), 31.0), (ts(1), 10.0)])
    assert preview(client, payload).json()["alarms"] == []
    # Ending on a high value keeps the alarm instead.
    payload = preview_readings([(ts(1), 10.0), (ts(1), 31.0)])
    body = preview(client, payload).json()
    assert body["alarms"] == [{"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}]
    assert body["added_alarms"] == [] and body["withdrawn_alarms"] == []


def test_preview_moved_alarm_is_withdrawn_plus_added(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(2), 31.0), (ts(3), 31.0)])  # open @00:03
    # A candidate late reading at 00:01 moves the open instant to 00:02.
    body = preview(client, preview_readings([(ts(1), 31.0)])).json()
    assert body["alarms"] == [{"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}]
    assert body["withdrawn_alarms"] == [
        {"opened_at": "2026-01-01T00:03:00Z", "closed_at": None}
    ]
    assert body["added_alarms"] == [
        {"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}
    ]


def test_preview_reports_multiple_intervals_in_order(client):
    set_thresholds(client)
    post_readings(
        client,
        [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0),   # open  @00:02
         (ts(3), 10.0), (ts(4), 10.0)],                   # close @00:04
    )
    # Candidate readings open a second alarm at 00:08.
    body = preview(
        client, preview_readings([(ts(6), 31.0), (ts(7), 31.0), (ts(8), 31.0)])
    ).json()
    assert body["alarms"] == [
        {"opened_at": "2026-01-01T00:02:00Z", "closed_at": "2026-01-01T00:04:00Z"},
        {"opened_at": "2026-01-01T00:08:00Z", "closed_at": None},
    ]
    assert body["added_alarms"] == [
        {"opened_at": "2026-01-01T00:08:00Z", "closed_at": None}
    ]
    assert body["withdrawn_alarms"] == []


def test_preview_base_revision_tracks_alarm_revisions(client):
    set_thresholds(client)
    assert preview(client, {"thresholds": {"h": 31.0, "l": 20.0}}).json()["base_revision"] == 1
    post_readings(client, [(ts(0), 31.0)])
    assert preview(client, {"thresholds": {"h": 31.0, "l": 20.0}}).json()["base_revision"] == 2


# ------------------------------------------------------------ read-only guarantee

def test_preview_persists_nothing(client):
    open_dev1_alarm(client)  # revisions 1 (thresholds) + 2 (readings), open @00:02
    assert ack(client).status_code == 201
    before = {
        "device": client.get("/devices/dev-1").json(),
        "readings": client.get("/devices/dev-1/readings").json(),
        "alarms": get_alarms(client).json(),
        "revision_header": get_alarms(client).headers["x-alarm-revision"],
        "revisions": get_revisions(client).json(),
    }
    # A preview that would change everything: withdraw the current alarm via
    # thresholds, move/extend readings, add a brand-new alarm.
    payload = preview_readings(
        [(ts(1), 10.0), (ts(5), 40.0), (ts(6), 40.0), (ts(7), 40.0)]
    )
    payload["thresholds"] = {"h": 35.0, "l": 15.0}
    resp = preview(client, payload)
    assert resp.status_code == 200
    assert resp.json()["alarms"] != before["alarms"]

    assert client.get("/devices/dev-1").json() == before["device"]
    assert client.get("/devices/dev-1/readings").json() == before["readings"]
    resp = get_alarms(client)
    assert resp.json() == before["alarms"]
    assert resp.headers["x-alarm-revision"] == before["revision_header"]
    assert resp.json()[0]["acknowledged_by"] == "alice"  # acknowledgement kept
    assert get_revisions(client).json() == before["revisions"]


def test_preview_repeatable_and_independent_across_devices(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b", h=50.0, l=40.0)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)], device="a")
    # Repeating the same preview gives the same result...
    payload = preview_readings([(ts(3), 10.0), (ts(4), 10.0)])
    first = preview(client, payload, device="a").json()
    assert preview(client, payload, device="a").json() == first
    assert first["alarms"] == [
        {"opened_at": "2026-01-01T00:02:00Z", "closed_at": "2026-01-01T00:04:00Z"}
    ]
    # ...and a's preview neither touches nor reads b's state.
    assert preview(client, payload, device="b").json()["alarms"] == []
    assert get_alarms(client, device="a").json()[0]["closed_at"] is None


# ==================================================== alarm revision snapshot

def get_snapshot(client, revision, device="dev-1"):
    return client.get(f"/devices/{device}/alarm-revisions/{revision}/snapshot")


# ------------------------------------------------------------ validation

def test_snapshot_for_unknown_device_returns_404(client):
    assert get_snapshot(client, 1, device="ghost").status_code == 404


def test_snapshot_revision_must_be_a_positive_integer(client):
    set_thresholds(client)
    assert get_snapshot(client, 0).status_code == 422
    assert get_snapshot(client, -1).status_code == 422
    assert get_snapshot(client, 1.5).status_code == 422
    assert get_snapshot(client, "x").status_code == 422
    assert get_snapshot(client, 1).status_code == 200


def test_snapshot_revision_beyond_current_is_404(client):
    set_thresholds(client)                      # revision 1
    post_readings(client, [(ts(0), 31.0)])      # revision 2
    assert get_snapshot(client, 3).status_code == 404
    assert get_snapshot(client, 2).status_code == 200
    assert get_snapshot(client, 1).status_code == 200


# ------------------------------------------------------------ reconstruction

def test_snapshot_at_current_revision_matches_live_state(client):
    set_thresholds(client, h=30.0, l=20.0)
    post_readings(
        client,
        [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0), (ts(3), 10.0), (ts(4), 10.0)],
    )
    body = get_snapshot(client, 2).json()
    assert body["device_id"] == "dev-1"
    assert body["revision"] == 2
    assert body["threshold_h"] == 30.0 and body["threshold_l"] == 20.0
    assert body["readings"] == [
        {"sampled_at": f"2026-01-01T00:0{m}:00Z", "temperature": t}
        for m, t in enumerate([31.0, 31.0, 31.0, 10.0, 10.0])
    ]
    assert body["alarms"] == [
        {"opened_at": "2026-01-01T00:02:00Z", "closed_at": "2026-01-01T00:04:00Z"}
    ]


def test_snapshot_rolls_back_reading_additions(client):
    set_thresholds(client, h=30.0, l=20.0)                            # revision 1
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])  # revision 2
    body = get_snapshot(client, 1).json()
    # Right after the device was created: thresholds in place, no readings yet
    # (the points added by revision 2 had null old values and are removed).
    assert body["revision"] == 1
    assert body["threshold_h"] == 30.0 and body["threshold_l"] == 20.0
    assert body["readings"] == []
    assert body["alarms"] == []


def test_snapshot_rolls_back_correction_and_restores_old_temperature(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])  # rev 2: alarm opens
    post_readings(client, [(ts(1), 10.0)])  # rev 3: correction withdraws the alarm
    assert get_alarms(client).json() == []

    body = get_snapshot(client, 2).json()
    assert [r["temperature"] for r in body["readings"]] == [31.0, 31.0, 31.0]
    assert body["alarms"] == [{"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}]

    body = get_snapshot(client, 3).json()
    assert [r["temperature"] for r in body["readings"]] == [31.0, 10.0, 31.0]
    assert body["alarms"] == []


def test_snapshot_removes_late_arrived_point(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(2), 31.0), (ts(3), 31.0)])  # rev 2: open @00:03
    post_readings(client, [(ts(1), 31.0)])  # rev 3: late arrival moves the alarm to 00:02
    body = get_snapshot(client, 2).json()
    # The late point did not exist at revision 2: it is removed and the
    # alarm is back at its then-current position.
    assert [r["sampled_at"] for r in body["readings"]] == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:02:00Z",
        "2026-01-01T00:03:00Z",
    ]
    assert body["alarms"] == [{"opened_at": "2026-01-01T00:03:00Z", "closed_at": None}]


def test_snapshot_rolls_back_threshold_change(client):
    set_thresholds(client, h=30.0, l=20.0)                            # rev 1
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])  # rev 2: alarm opens
    set_thresholds(client, h=35.0, l=25.0)  # rev 3: alarm withdrawn
    assert get_alarms(client).json() == []

    body = get_snapshot(client, 2).json()
    assert body["threshold_h"] == 30.0 and body["threshold_l"] == 20.0
    assert body["alarms"] == [{"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}]

    body = get_snapshot(client, 3).json()
    assert body["threshold_h"] == 35.0 and body["threshold_l"] == 25.0
    assert body["alarms"] == []


def test_snapshot_rolls_back_a_chain_of_revisions_in_order(client):
    set_thresholds(client, h=30.0, l=20.0)                                    # rev 1
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])      # rev 2: open @00:02
    post_readings(client, [(ts(3), 10.0), (ts(4), 10.0)])                     # rev 3: close @00:04
    set_thresholds(client, h=32.0, l=21.0)                                    # rev 4: alarm withdrawn
    post_readings(client, [(ts(5), 33.0), (ts(6), 33.0), (ts(7), 33.0)])      # rev 5: open @00:07

    body = get_snapshot(client, 5).json()  # current state
    assert body["threshold_h"] == 32.0 and body["threshold_l"] == 21.0
    assert body["alarms"] == [{"opened_at": "2026-01-01T00:07:00Z", "closed_at": None}]

    body = get_snapshot(client, 3).json()
    assert body["threshold_h"] == 30.0 and body["threshold_l"] == 20.0
    assert [r["sampled_at"] for r in body["readings"]] == [
        f"2026-01-01T00:0{m}:00Z" for m in range(5)
    ]
    assert body["alarms"] == [
        {"opened_at": "2026-01-01T00:02:00Z", "closed_at": "2026-01-01T00:04:00Z"}
    ]

    body = get_snapshot(client, 1).json()
    assert body["readings"] == [] and body["alarms"] == []


def test_snapshot_with_repeated_corrections_of_the_same_point(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])  # rev 2
    post_readings(client, [(ts(1), 10.0)])  # rev 3: 00:01 -> 10.0
    post_readings(client, [(ts(1), 20.0)])  # rev 4: 00:01 -> 20.0
    # Rolling back revisions 4 then 3 restores the oldest recorded value.
    body = get_snapshot(client, 2).json()
    assert [r["temperature"] for r in body["readings"]] == [31.0, 31.0, 31.0]
    body = get_snapshot(client, 3).json()
    assert [r["temperature"] for r in body["readings"]] == [31.0, 10.0, 31.0]
    body = get_snapshot(client, 4).json()
    assert [r["temperature"] for r in body["readings"]] == [31.0, 20.0, 31.0]


def test_snapshot_readings_sorted_by_utc_instant(client):
    set_thresholds(client)
    post_readings(
        client,
        [
            ("2026-01-01T10:02:00+08:00", 31.0),  # 02:02 UTC
            ("2026-01-01T01:00:00+00:00", 31.0),  # 01:00 UTC (earliest)
            ("2026-01-01T02:01:00+00:00", 31.0),  # 02:01 UTC
        ],
    )
    body = get_snapshot(client, 2).json()
    assert [r["sampled_at"] for r in body["readings"]] == [
        "2026-01-01T01:00:00Z",
        "2026-01-01T02:01:00Z",
        "2026-01-01T02:02:00Z",
    ]
    assert body["alarms"] == [{"opened_at": "2026-01-01T02:02:00Z", "closed_at": None}]


def test_snapshot_is_per_device(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b", h=50.0, l=40.0)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)], device="a")
    body = get_snapshot(client, 2, device="a").json()
    assert body["threshold_h"] == 30.0
    assert body["alarms"] == [{"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}]
    # b's revisions and state are independent of a's.
    body = get_snapshot(client, 1, device="b").json()
    assert body["threshold_h"] == 50.0 and body["threshold_l"] == 40.0
    assert body["readings"] == [] and body["alarms"] == []
    assert get_snapshot(client, 2, device="b").status_code == 404


# ------------------------------------------------------------ acknowledgements

def test_snapshot_does_not_carry_current_acknowledgements(client):
    open_dev1_alarm(client)  # rev 2, open alarm @00:02
    assert ack(client).status_code == 201
    body = get_snapshot(client, 2).json()
    # The snapshot interval carries no acknowledgement fields at all, even
    # though the live alarm is acknowledged.
    assert body["alarms"] == [{"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}]
    assert get_alarms(client).json()[0]["acknowledged_by"] == "alice"


# ------------------------------------------------------------ read-only guarantee

def test_snapshot_persists_nothing(client):
    open_dev1_alarm(client)               # rev 1 (thresholds) + 2 (readings)
    assert ack(client).status_code == 201
    post_readings(client, [(ts(1), 10.0)])  # rev 3: correction withdraws the alarm
    before = {
        "device": client.get("/devices/dev-1").json(),
        "readings": client.get("/devices/dev-1/readings").json(),
        "alarms": get_alarms(client).json(),
        "revision_header": get_alarms(client).headers["x-alarm-revision"],
        "revisions": get_revisions(client).json(),
    }
    for rev in (1, 2, 3):
        resp = get_snapshot(client, rev)
        assert resp.status_code == 200
        assert resp.json()["revision"] == rev

    assert client.get("/devices/dev-1").json() == before["device"]
    assert client.get("/devices/dev-1/readings").json() == before["readings"]
    resp = get_alarms(client)
    assert resp.json() == before["alarms"]
    assert resp.headers["x-alarm-revision"] == before["revision_header"]
    assert get_revisions(client).json() == before["revisions"]


def test_snapshot_works_across_restarts(tmp_path, monkeypatch):
    db = tmp_path / "persist-snap.db"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    with TestClient(app) as c1:
        set_thresholds(c1)
        post_readings(c1, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
        post_readings(c1, [(ts(1), 10.0)])  # rev 3 withdraws the alarm
    with TestClient(app) as c2:
        body = get_snapshot(c2, 2).json()
        assert [r["temperature"] for r in body["readings"]] == [31.0, 31.0, 31.0]
        assert body["alarms"] == [{"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}]


# ============================================================ peer deviations

def get_peer_deviations(client, group="grp-1", **params):
    return client.get(f"/groups/{group}/peer-deviations", params=params)


def peer_params(device="a", start="2026-01-01T00:00:00Z", end="2026-01-01T01:00:00Z",
                delta=2.0, min_samples=2):
    return {
        "device_id": device,
        "start": start,
        "end": end,
        "delta": delta,
        "min_samples": min_samples,
    }


def setup_peer_group(client, members=("a", "b", "c"), max_interval=120, group="grp-1"):
    """Create the member devices, give each a sampling config and group them."""
    for d in members:
        set_thresholds(client, device=d)
        set_sampling(client, max_interval, 0, device=d)
    assert set_group(client, list(members), 1, group=group).status_code == 200


def steady_readings(minutes, temperature=20.0):
    return [(ts(m), temperature) for m in minutes]


# ------------------------------------------------------------ validation

def test_peer_deviations_unknown_group_is_404(client):
    assert get_peer_deviations(client, **peer_params()).status_code == 404


def test_peer_deviations_target_must_be_a_member(client):
    setup_peer_group(client)
    # Unknown device and existing non-member alike: not a group member -> 404.
    assert get_peer_deviations(client, **peer_params(device="ghost")).status_code == 404
    set_thresholds(client, device="outsider")
    set_sampling(client, 120, 0, device="outsider")
    assert get_peer_deviations(client, **peer_params(device="outsider")).status_code == 404


def test_peer_deviations_target_without_sampling_config_is_409(client):
    setup_peer_group(client)
    # Remove the target's config by re-creating it without one is not possible,
    # so use a fresh member that never got a sampling config.
    set_thresholds(client, device="d")
    assert set_group(client, ["a", "b", "c", "d"], 1).status_code == 200
    resp = get_peer_deviations(client, **peer_params(device="d"))
    assert resp.status_code == 409


def test_peer_deviations_validates_params(client):
    setup_peer_group(client)
    base = peer_params()

    def get(**overrides):
        params = {**base, **overrides}
        return get_peer_deviations(client, **params)

    # missing parameters -> 422
    for key in base:
        assert get_peer_deviations(
            client, **{k: v for k, v in base.items() if k != key}
        ).status_code == 422
    # naive start / end -> 422
    assert get(start="2026-01-01T00:00:00").status_code == 422
    assert get(end="2026-01-01T01:00:00").status_code == 422
    # start == end and start > end -> 422
    assert get(start=ts(1), end=ts(1)).status_code == 422
    assert get(start=ts(2), end=ts(0)).status_code == 422
    # delta must be a finite positive number -> 422 otherwise
    assert get(delta=0).status_code == 422
    assert get(delta=-1.5).status_code == 422
    assert get(delta="NaN").status_code == 422
    assert get(delta="inf").status_code == 422
    # min_samples must be an integer >= 2 -> 422 otherwise
    assert get(min_samples=0).status_code == 422
    assert get(min_samples=1).status_code == 422
    assert get(min_samples=2.5).status_code == 422
    assert get(min_samples="two").status_code == 422


# ------------------------------------------------------------ deviation rules

def test_peer_deviations_reports_consecutive_segments(client):
    setup_peer_group(client)
    post_readings(client, steady_readings(range(10)), device="b")
    post_readings(client, steady_readings(range(10)), device="c")
    # Deviating at 00:00-00:01 and 00:03-00:04, in line at 00:02.
    post_readings(
        client,
        [(ts(0), 25.0), (ts(1), 25.0), (ts(2), 20.0), (ts(3), 25.0), (ts(4), 25.0)],
        device="a",
    )
    assert get_peer_deviations(client, **peer_params()).json() == [
        {
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:01:00Z",
            "samples": 2,
            "max_abs_deviation": 5.0,
        },
        {
            "started_at": "2026-01-01T00:03:00Z",
            "ended_at": "2026-01-01T00:04:00Z",
            "samples": 2,
            "max_abs_deviation": 5.0,
        },
    ]


def test_peer_deviations_exact_peer_value_is_used_as_is(client):
    setup_peer_group(client, max_interval=60)
    # b has an exact reading at 00:01 but a 240s gap (00:01 -> 00:05) beyond M
    # around 00:02: the exact value is used at 00:01, no interpolation at 00:02.
    post_readings(client, [(ts(0), 20.0), (ts(1), 20.0), (ts(5), 20.0)], device="b")
    post_readings(client, steady_readings(range(6)), device="c")
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0), (ts(2), 25.0)], device="a")
    assert get_peer_deviations(client, **peer_params()).json() == [
        {
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:01:00Z",
            "samples": 2,
            "max_abs_deviation": 5.0,
        }
    ]


def test_peer_deviations_interpolates_within_peer_m(client):
    setup_peer_group(client, max_interval=120)
    # b: 20 at 00:00, 24 at 00:02 -> interpolated 22 at 00:01 (gap == M is fine).
    post_readings(client, [(ts(0), 20.0), (ts(2), 24.0)], device="b")
    post_readings(client, steady_readings(range(3)), device="c")
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0), (ts(2), 25.0)], device="a")
    assert get_peer_deviations(client, **peer_params(min_samples=3)).json() == [
        {
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:02:00Z",
            "samples": 3,
            "max_abs_deviation": 5.0,  # medians: 20.0, 21.0, 22.0
        }
    ]


def test_peer_deviations_interpolation_uses_readings_outside_window(client):
    setup_peer_group(client, max_interval=120)
    # b's bracketing readings sit just outside/inside the window edges.
    post_readings(client, [("2025-12-31T23:59:00+00:00", 20.0), (ts(1), 22.0)], device="b")
    post_readings(client, steady_readings(range(3)), device="c")
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0)], device="a")
    assert get_peer_deviations(
        client, **peer_params(end="2026-01-01T00:02:00Z")
    ).json() == [
        {
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:01:00Z",
            "samples": 2,
            "max_abs_deviation": 4.5,  # b interpolates to 21 at 00:00 -> median 20.5
        }
    ]


def test_peer_deviations_never_extrapolates(client):
    setup_peer_group(client, max_interval=120)
    post_readings(client, steady_readings([0, 1]), device="b")    # ends before the window
    post_readings(client, steady_readings([10, 11]), device="c")  # starts after it
    post_readings(client, [(ts(5), 25.0), (ts(6), 25.0)], device="a")
    assert get_peer_deviations(client, **peer_params()).json() == []


def test_peer_deviations_peer_without_m_does_not_participate(client):
    set_thresholds(client, device="a")
    set_thresholds(client, device="b")
    set_thresholds(client, device="c")  # c never gets a sampling config
    set_sampling(client, 120, 0, device="a")
    set_sampling(client, 120, 0, device="b")
    set_group(client, ["a", "b", "c"], 1)
    post_readings(client, steady_readings(range(3)), device="b")
    post_readings(client, steady_readings(range(3)), device="c")
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0)], device="a")
    # Only b is comparable -> fewer than two peers -> no deviation.
    assert get_peer_deviations(client, **peer_params()).json() == []


def test_peer_deviations_median_of_even_peer_count(client):
    setup_peer_group(client, members=("a", "b", "c", "d", "e"))
    post_readings(client, steady_readings(range(3), 20.0), device="b")
    post_readings(client, steady_readings(range(3), 20.0), device="c")
    post_readings(client, steady_readings(range(3), 22.0), device="d")
    post_readings(client, steady_readings(range(3), 100.0), device="e")  # outlier ignored by median
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0)], device="a")
    # Median of [20, 20, 22, 100] is 21 -> deviation 4.
    assert get_peer_deviations(client, **peer_params()).json() == [
        {
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:01:00Z",
            "samples": 2,
            "max_abs_deviation": 4.0,
        }
    ]


def test_peer_deviations_difference_must_exceed_delta(client):
    setup_peer_group(client)
    post_readings(client, steady_readings(range(3)), device="b")
    post_readings(client, steady_readings(range(3)), device="c")
    # Exactly delta away at 00:00 (not a deviation), beyond delta at 00:01/00:02.
    post_readings(client, [(ts(0), 22.0), (ts(1), 22.5), (ts(2), 22.5)], device="a")
    assert get_peer_deviations(client, **peer_params()).json() == [
        {
            "started_at": "2026-01-01T00:01:00Z",
            "ended_at": "2026-01-01T00:02:00Z",
            "samples": 2,
            "max_abs_deviation": 2.5,
        }
    ]


def test_peer_deviations_min_samples_not_reached(client):
    setup_peer_group(client)
    post_readings(client, steady_readings(range(3)), device="b")
    post_readings(client, steady_readings(range(3)), device="c")
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0)], device="a")
    assert get_peer_deviations(client, **peer_params(min_samples=3)).json() == []


def test_peer_deviations_target_gap_beyond_own_m_breaks_run(client):
    setup_peer_group(client, max_interval=60)
    post_readings(client, steady_readings(range(5)), device="b")
    post_readings(client, steady_readings(range(5)), device="c")
    # All deviating, but 00:01 -> 00:03 is 120s > M=60: two separate segments.
    post_readings(
        client,
        [(ts(0), 25.0), (ts(1), 25.0), (ts(3), 25.0), (ts(4), 25.0)],
        device="a",
    )
    assert get_peer_deviations(client, **peer_params()).json() == [
        {
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:01:00Z",
            "samples": 2,
            "max_abs_deviation": 5.0,
        },
        {
            "started_at": "2026-01-01T00:03:00Z",
            "ended_at": "2026-01-01T00:04:00Z",
            "samples": 2,
            "max_abs_deviation": 5.0,
        },
    ]


def test_peer_deviations_target_gap_of_exactly_m_does_not_break(client):
    setup_peer_group(client, max_interval=120)
    post_readings(client, steady_readings(range(5)), device="b")
    post_readings(client, steady_readings(range(5)), device="c")
    post_readings(client, [(ts(0), 25.0), (ts(2), 25.0)], device="a")  # 120s == M
    assert get_peer_deviations(client, **peer_params()).json() == [
        {
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:02:00Z",
            "samples": 2,
            "max_abs_deviation": 5.0,
        }
    ]


def test_peer_deviations_only_target_points_inside_window(client):
    setup_peer_group(client)
    post_readings(client, steady_readings(range(5)), device="b")
    post_readings(client, steady_readings(range(5)), device="c")
    post_readings(
        client,
        [(ts(0), 25.0), (ts(1), 25.0), (ts(2), 25.0), (ts(3), 25.0)],
        device="a",
    )
    # [00:01, 00:03): only the 00:01 and 00:02 points are processed.
    assert get_peer_deviations(
        client,
        **peer_params(start="2026-01-01T00:01:00Z", end="2026-01-01T00:03:00Z"),
    ).json() == [
        {
            "started_at": "2026-01-01T00:01:00Z",
            "ended_at": "2026-01-01T00:02:00Z",
            "samples": 2,
            "max_abs_deviation": 5.0,
        }
    ]
    # A window holding a single deviating point yields no segment.
    assert get_peer_deviations(
        client,
        **peer_params(start="2026-01-01T00:01:00Z", end="2026-01-01T00:02:00Z"),
    ).json() == []


def test_peer_deviations_timezone_offsets_are_normalized(client):
    setup_peer_group(client)
    # 08:00/08:01/08:02 at +08:00 are the same instants as 00:00/00:01/00:02 UTC.
    post_readings(client, [(ts(m, hour=8, tz="+08:00"), 20.0) for m in range(3)], device="b")
    post_readings(client, steady_readings(range(3)), device="c")
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0)], device="a")
    assert get_peer_deviations(
        client,
        **peer_params(
            start="2026-01-01T08:00:00+08:00", end="2026-01-01T09:00:00+08:00"
        ),
    ).json() == [
        {
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:01:00Z",
            "samples": 2,
            "max_abs_deviation": 5.0,
        }
    ]


def test_peer_deviations_single_member_group_has_no_peers(client):
    set_thresholds(client, device="a")
    set_sampling(client, 120, 0, device="a")
    set_group(client, ["a"], 1)
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0)], device="a")
    assert get_peer_deviations(client, **peer_params()).json() == []


# ------------------------------------------------------------ recompute on query

def test_peer_deviations_recomputed_after_correction(client):
    setup_peer_group(client)
    post_readings(client, steady_readings(range(3)), device="b")
    post_readings(client, steady_readings(range(3)), device="c")
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0), (ts(2), 25.0)], device="a")
    assert len(get_peer_deviations(client, **peer_params()).json()) == 1
    # Correcting the middle point back in line splits the run below min_samples.
    post_readings(client, [(ts(1), 20.0)], device="a")
    assert get_peer_deviations(client, **peer_params()).json() == []


def test_peer_deviations_recomputed_after_late_arrival(client):
    setup_peer_group(client, max_interval=120)
    post_readings(client, [(ts(0), 20.0), (ts(10), 20.0)], device="b")  # 600s gap > M
    post_readings(client, steady_readings(range(3)), device="c")
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0), (ts(2), 25.0)], device="a")
    # b cannot be interpolated across the gap -> too few peers at 00:01/00:02.
    assert get_peer_deviations(client, **peer_params()).json() == []
    # A late b reading at 00:02 makes the 00:00 -> 00:02 gap exactly M again.
    post_readings(client, [(ts(2), 20.0)], device="b")
    assert get_peer_deviations(client, **peer_params()).json() == [
        {
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:02:00Z",
            "samples": 3,
            "max_abs_deviation": 5.0,
        }
    ]


def test_peer_deviations_recomputed_after_membership_change(client):
    setup_peer_group(client)
    post_readings(client, steady_readings(range(3)), device="b")
    post_readings(client, steady_readings(range(3)), device="c")
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0)], device="a")
    assert len(get_peer_deviations(client, **peer_params()).json()) == 1
    # Removing c leaves a single comparable peer -> no deviation segments.
    set_group(client, ["a", "b"], 1)
    assert get_peer_deviations(client, **peer_params()).json() == []


def test_peer_deviations_persists_nothing(client):
    setup_peer_group(client)
    post_readings(client, steady_readings(range(3)), device="b")
    post_readings(client, steady_readings(range(3)), device="c")
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0)], device="a")
    before = {
        "group": get_group(client).json(),
        "readings": {
            d: client.get(f"/devices/{d}/readings").json() for d in ("a", "b", "c")
        },
        "alarms": {d: get_alarms(client, device=d).json() for d in ("a", "b", "c")},
    }
    resp = get_peer_deviations(client, **peer_params())
    assert resp.status_code == 200
    assert get_group(client).json() == before["group"]
    for d in ("a", "b", "c"):
        assert client.get(f"/devices/{d}/readings").json() == before["readings"][d]
        assert get_alarms(client, device=d).json() == before["alarms"][d]


# ================================================================= forecast

def get_forecast(client, device="dev-1", **params):
    return client.get(f"/devices/{device}/forecast", params=params)


def forecast_params(**overrides):
    params = {"as_of": ts(10), "lookback": 600, "horizon": 400}
    params.update(overrides)
    return params


def setup_forecast_device(client):
    """Device with H=30/L=20, M=60 and a +1 deg/min ramp 22->23->24 at 00:08~00:10."""
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(8), 22.0), (ts(9), 23.0), (ts(10), 24.0)])


# ------------------------------------------------------------ validation

def test_forecast_unknown_device_is_404(client):
    assert get_forecast(client, device="ghost", **forecast_params()).status_code == 404


def test_forecast_requires_sampling_config(client):
    set_thresholds(client)
    post_readings(client, [(ts(8), 22.0), (ts(9), 23.0), (ts(10), 24.0)])
    assert get_forecast(client, **forecast_params()).status_code == 409


def test_forecast_validates_params(client):
    setup_forecast_device(client)
    base = forecast_params()

    # missing parameters -> 422
    for key in base:
        assert get_forecast(
            client, **{k: v for k, v in base.items() if k != key}
        ).status_code == 422
    # naive as_of -> 422
    assert get_forecast(client, **forecast_params(as_of="2026-01-01T00:10:00")).status_code == 422
    # lookback / horizon must be positive integers -> 422 otherwise
    for bad in (0, -1, 1.5, "abc"):
        assert get_forecast(client, **forecast_params(lookback=bad)).status_code == 422
        assert get_forecast(client, **forecast_params(horizon=bad)).status_code == 422


# ------------------------------------------------------------ prediction rules

def test_forecast_predicts_crossing_of_linear_ramp(client):
    # Ramp 22 -> 23 -> 24 at 00:08~00:10 (+1 deg/min): the fit is exact,
    # intercept 24 at as_of, slope 1/60 deg/s, H=30 reached 360 s after as_of.
    setup_forecast_device(client)
    body = get_forecast(client, **forecast_params()).json()
    assert body["status"] == "predicted"
    assert body["predicted_at"] == "2026-01-01T00:16:00Z"
    assert body["samples"] == 3
    assert body["slope"] == pytest.approx(1.0 / 60.0)
    assert body["intercept"] == pytest.approx(24.0)
    # echoes the request and the current threshold, normalized to UTC
    assert body["device_id"] == "dev-1"
    assert body["as_of"] == "2026-01-01T00:10:00Z"
    assert body["lookback"] == 600
    assert body["horizon"] == 400
    assert body["threshold_h"] == 30.0


def test_forecast_crossing_exactly_at_horizon_is_predicted(client):
    setup_forecast_device(client)
    body = get_forecast(client, **forecast_params(horizon=360)).json()
    assert body["status"] == "predicted"
    assert body["predicted_at"] == "2026-01-01T00:16:00Z"


def test_forecast_crossing_beyond_horizon_is_no_crossing(client):
    setup_forecast_device(client)
    body = get_forecast(client, **forecast_params(horizon=359)).json()
    assert body["status"] == "no_crossing"
    assert body["predicted_at"] is None


def test_forecast_flat_trend_is_no_crossing(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(8), 25.0), (ts(9), 25.0), (ts(10), 25.0)])
    body = get_forecast(client, **forecast_params()).json()
    assert body["status"] == "no_crossing"
    assert body["slope"] == pytest.approx(0.0)
    assert body["predicted_at"] is None


def test_forecast_falling_trend_is_no_crossing(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(8), 26.0), (ts(9), 25.0), (ts(10), 24.0)])
    body = get_forecast(client, **forecast_params(horizon=100000)).json()
    assert body["status"] == "no_crossing"
    assert body["slope"] == pytest.approx(-1.0 / 60.0)
    assert body["predicted_at"] is None


def test_forecast_already_above_threshold_is_no_crossing(client):
    # Rising, but the fitted level at as_of is already >= H: no future crossing.
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(8), 35.0), (ts(9), 36.0), (ts(10), 37.0)])
    body = get_forecast(client, **forecast_params(horizon=100000)).json()
    assert body["status"] == "no_crossing"
    assert body["intercept"] == pytest.approx(37.0)
    assert body["predicted_at"] is None


def test_forecast_slope_is_median_of_all_point_pairs(client):
    # Points (-180 s, 0), (-120 s, 1), (-60 s, 2), (0, 10): the six pairwise
    # slopes are 1/60 (x3), 1/18, 3/40, 2/15 -> median 13/360, not the
    # endpoint slope 1/18; intercepts 6.5, 16/3, 25/6, 10 -> median 71/12.
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(7), 0.0), (ts(8), 1.0), (ts(9), 2.0), (ts(10), 10.0)])
    body = get_forecast(client, **forecast_params(horizon=700)).json()
    assert body["slope"] == pytest.approx(13.0 / 360.0)
    assert body["intercept"] == pytest.approx(71.0 / 12.0)
    assert body["status"] == "predicted"
    # crossing in (30 - 71/12) / (13/360) = 8670/13 seconds after as_of
    predicted = datetime.fromisoformat(body["predicted_at"])
    as_of = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)
    assert (predicted - as_of).total_seconds() == pytest.approx(8670.0 / 13.0)


def test_forecast_as_of_timezone_offset_is_normalized(client):
    setup_forecast_device(client)
    body = get_forecast(
        client, **forecast_params(as_of="2026-01-01T08:10:00+08:00")
    ).json()
    assert body["as_of"] == "2026-01-01T00:10:00Z"
    assert body["status"] == "predicted"
    assert body["predicted_at"] == "2026-01-01T00:16:00Z"


# ------------------------------------------------------------ segment selection

def test_forecast_no_readings_is_insufficient_data(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    body = get_forecast(client, **forecast_params()).json()
    assert body["status"] == "insufficient_data"
    assert body["samples"] == 0
    assert body["slope"] is None
    assert body["intercept"] is None
    assert body["predicted_at"] is None


def test_forecast_fewer_than_three_points_is_insufficient_data(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(9), 23.0), (ts(10), 24.0)])
    body = get_forecast(client, **forecast_params()).json()
    assert body["status"] == "insufficient_data"
    assert body["samples"] == 2
    assert body["predicted_at"] is None


def test_forecast_stale_newest_point_is_insufficient_data(client):
    # Newest sample (00:02) is 8 minutes before as_of, well beyond M=60.
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 22.0), (ts(1), 23.0), (ts(2), 24.0)])
    body = get_forecast(client, **forecast_params()).json()
    assert body["status"] == "insufficient_data"
    assert body["samples"] == 3
    assert body["predicted_at"] is None


def test_forecast_newest_point_exactly_m_old_is_usable(client):
    # Newest sample (00:09) is exactly M=60 s before as_of: still fresh enough.
    # The ramp gives intercept 25 at as_of, so H=30 is reached 300 s later.
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(7), 22.0), (ts(8), 23.0), (ts(9), 24.0)])
    body = get_forecast(client, **forecast_params()).json()
    assert body["status"] == "predicted"
    assert body["samples"] == 3
    assert body["predicted_at"] == "2026-01-01T00:15:00Z"


def test_forecast_gap_beyond_m_breaks_contiguity(client):
    # Old points at 00:00/00:01 are cut off by the 7-minute gap before
    # 00:08: only the newest contiguous segment 00:08~00:10 is fitted.
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(
        client,
        [(ts(0), -50.0), (ts(1), -50.0), (ts(8), 22.0), (ts(9), 23.0), (ts(10), 24.0)],
    )
    body = get_forecast(client, **forecast_params()).json()
    assert body["samples"] == 3
    assert body["slope"] == pytest.approx(1.0 / 60.0)
    assert body["intercept"] == pytest.approx(24.0)
    assert body["status"] == "predicted"


def test_forecast_gap_of_exactly_m_keeps_contiguity(client):
    # 2-minute spacing with M=120: exactly-M gaps stay contiguous; the ramp
    # rises 1 deg per 120 s, so H=30 is reached 720 s after as_of.
    set_thresholds(client)
    set_sampling(client, max_interval=120, grace=0)
    post_readings(client, [(ts(6), 22.0), (ts(8), 23.0), (ts(10), 24.0)])
    body = get_forecast(client, **forecast_params(horizon=720)).json()
    assert body["samples"] == 3
    assert body["status"] == "predicted"
    assert body["predicted_at"] == "2026-01-01T00:22:00Z"


def test_forecast_segment_is_capped_at_30_points(client):
    # 35 minutely points; the oldest 5 are wild outliers. Only the newest
    # 30 (00:05~00:34, a clean +1 deg/min ramp to 24 at as_of) are fitted.
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    readings = [(ts(m), -100.0) for m in range(5)]
    readings += [(ts(m), float(m - 10)) for m in range(5, 35)]
    post_readings(client, readings)
    body = get_forecast(client, **forecast_params(as_of=ts(34), lookback=3600)).json()
    assert body["samples"] == 30
    assert body["slope"] == pytest.approx(1.0 / 60.0)
    assert body["intercept"] == pytest.approx(24.0)
    assert body["status"] == "predicted"
    assert body["predicted_at"] == "2026-01-01T00:40:00Z"


def test_forecast_lookback_window_bounds_points(client):
    # Points at 00:00, 00:09, 00:10 with M=600: the 00:00 point sits exactly
    # at as_of - lookback and is included; shortening lookback to 599 drops
    # it and only 2 points remain.
    set_thresholds(client)
    set_sampling(client, max_interval=600, grace=0)
    post_readings(client, [(ts(0), 22.0), (ts(9), 23.0), (ts(10), 24.0)])
    body = get_forecast(client, **forecast_params()).json()
    assert body["samples"] == 3
    body = get_forecast(client, **forecast_params(lookback=599)).json()
    assert body["status"] == "insufficient_data"
    assert body["samples"] == 2


def test_forecast_readings_after_as_of_are_excluded(client):
    setup_forecast_device(client)
    post_readings(client, [(ts(11), 99.0)])
    body = get_forecast(client, **forecast_params()).json()
    assert body["samples"] == 3
    assert body["intercept"] == pytest.approx(24.0)
    assert body["predicted_at"] == "2026-01-01T00:16:00Z"


# ------------------------------------------------------------ recompute on query

def test_forecast_recomputed_after_late_arrival(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(9), 23.0), (ts(10), 24.0)])
    assert get_forecast(client, **forecast_params()).json()["status"] == "insufficient_data"
    # Late-arriving older point completes the segment.
    post_readings(client, [(ts(8), 22.0)])
    body = get_forecast(client, **forecast_params()).json()
    assert body["status"] == "predicted"
    assert body["predicted_at"] == "2026-01-01T00:16:00Z"


def test_forecast_recomputed_after_correction(client):
    setup_forecast_device(client)
    assert get_forecast(client, **forecast_params()).json()["status"] == "predicted"
    # Correcting the last point flattens the trend to a constant 24.
    post_readings(client, [(ts(8), 24.0), (ts(9), 24.0)])
    body = get_forecast(client, **forecast_params()).json()
    assert body["status"] == "no_crossing"
    assert body["slope"] == pytest.approx(0.0)
    assert body["predicted_at"] is None


def test_forecast_recomputed_after_sampling_config_change(client):
    # Points 5 minutes apart: contiguous only once M is raised to 300; the
    # ramp then rises 1 deg per 300 s, so H=30 is reached 1800 s after as_of.
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 22.0), (ts(5), 23.0), (ts(10), 24.0)])
    assert get_forecast(client, **forecast_params()).json()["status"] == "insufficient_data"
    set_sampling(client, max_interval=300, grace=0)
    body = get_forecast(client, **forecast_params(horizon=1800)).json()
    assert body["status"] == "predicted"
    assert body["samples"] == 3
    assert body["predicted_at"] == "2026-01-01T00:40:00Z"


def test_forecast_recomputed_after_threshold_change(client):
    setup_forecast_device(client)
    assert get_forecast(client, **forecast_params()).json()["status"] == "predicted"
    # Raising H pushes the crossing beyond the horizon.
    set_thresholds(client, h=100.0, l=20.0)
    body = get_forecast(client, **forecast_params()).json()
    assert body["status"] == "no_crossing"
    assert body["threshold_h"] == 100.0
    assert body["predicted_at"] is None


def test_forecast_persists_nothing(client):
    setup_forecast_device(client)
    before = {
        "device": client.get("/devices/dev-1").json(),
        "readings": client.get("/devices/dev-1/readings").json(),
        "alarms": get_alarms(client).json(),
        "revisions": get_revisions(client).json(),
        "sampling": get_sampling(client).json(),
    }
    resp = get_forecast(client, **forecast_params())
    assert resp.status_code == 200
    assert client.get("/devices/dev-1").json() == before["device"]
    assert client.get("/devices/dev-1/readings").json() == before["readings"]
    assert get_alarms(client).json() == before["alarms"]
    assert get_revisions(client).json() == before["revisions"]
    assert get_sampling(client).json() == before["sampling"]


# ============================================== alarm response deadline

def set_policy(client, deadline, device="dev-1"):
    return client.put(
        f"/devices/{device}/response-policy", json={"deadline": deadline}
    )


def get_policy(client, device="dev-1"):
    return client.get(f"/devices/{device}/response-policy")


def get_status(client, as_of, device="dev-1"):
    return client.get(f"/devices/{device}/response-status", params={"as_of": as_of})


def ack_open_alarm(client, opened_at="2026-01-01T00:02:00+00:00", operator="alice",
                   device="dev-1", revision=None):
    if revision is None:
        revision = get_alarms(client, device=device).headers["X-Alarm-Revision"]
    return client.post(
        f"/devices/{device}/alarm-acknowledgements",
        json={"opened_at": opened_at, "operator": operator, "expected_revision": int(revision)},
    )


# ------------------------------------------------------------ policy

def test_response_policy_validation(client):
    set_thresholds(client)
    assert set_policy(client, 0).status_code == 422
    assert set_policy(client, -1).status_code == 422
    assert set_policy(client, 1.5).status_code == 422
    assert set_policy(client, "sixty").status_code == 422


def test_response_policy_requires_existing_device(client):
    assert set_policy(client, 60, device="ghost").status_code == 404
    assert get_status(client, ts(10), device="ghost").status_code == 404


def test_response_policy_persisted_and_updatable(client):
    set_thresholds(client)
    resp = set_policy(client, 300)
    assert resp.status_code == 200
    body = resp.json()
    assert body["device_id"] == "dev-1"
    assert body["deadline"] == 300
    assert "created_at" in body and "updated_at" in body

    assert get_policy(client).json()["deadline"] == 300

    assert set_policy(client, 120).status_code == 200
    assert get_policy(client).json()["deadline"] == 120

    # Whole-number floats are accepted as integers.
    assert set_policy(client, 90.0).status_code == 200
    assert get_policy(client).json()["deadline"] == 90


def test_missing_response_policy_conflicts(client):
    set_thresholds(client)
    assert get_policy(client).status_code == 409
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    set_sampling(client, max_interval=60, grace=0)
    # M configured but D missing -> 409
    assert get_status(client, ts(10)).status_code == 409


def test_missing_sampling_config_conflicts_for_status(client):
    set_thresholds(client)
    set_policy(client, 300)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    # D configured but M missing -> 409
    assert get_status(client, ts(10)).status_code == 409


def test_response_policy_persists_across_restarts(tmp_path, monkeypatch):
    db = tmp_path / "persist-policy.db"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    with TestClient(app) as c1:
        set_thresholds(c1)
        set_policy(c1, 300)
        set_sampling(c1, max_interval=60, grace=0)
        post_readings(c1, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    with TestClient(app) as c2:
        assert get_policy(c2).json()["deadline"] == 300
        body = get_status(c2, ts(10)).json()
        assert len(body) == 1
        assert body[0]["deadline"] == 300


# ------------------------------------------------------------ listing / params

def test_response_status_requires_timezone_aware_as_of(client):
    set_thresholds(client)
    set_policy(client, 300)
    set_sampling(client, max_interval=60, grace=0)
    # timezone-naive as_of -> 422; missing as_of -> 422
    assert get_status(client, "2026-01-01T00:10:00").status_code == 422
    assert client.get("/devices/dev-1/response-status").status_code == 422


def test_response_status_lists_only_open_unacknowledged(client):
    set_thresholds(client)
    set_policy(client, 300)
    set_sampling(client, max_interval=60, grace=0)
    # One closed cycle (open 00:02, close 00:04) and a still-open one (open 00:07).
    post_readings(
        client,
        [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0),
         (ts(3), 19.0), (ts(4), 19.0),
         (ts(5), 31.0), (ts(6), 31.0), (ts(7), 31.0)],
    )
    body = get_status(client, ts(10)).json()
    assert [item["opened_at"][11:19] for item in body] == ["00:07:00"]

    ack_open_alarm(client, opened_at="2026-01-01T00:07:00+00:00")
    assert get_status(client, ts(10)).json() == []


def test_response_status_filters_by_as_of(client):
    set_thresholds(client)
    set_policy(client, 300)
    set_sampling(client, max_interval=60, grace=0)
    # Alarm opens at 00:07; as_of 00:06 must not list it yet.
    post_readings(
        client,
        [(ts(5), 31.0), (ts(6), 31.0), (ts(7), 31.0)],
    )
    assert get_status(client, ts(6)).json() == []
    body = get_status(client, ts(7)).json()
    assert [item["opened_at"][11:19] for item in body] == ["00:07:00"]


def test_response_status_echoes_fields(client):
    set_thresholds(client)
    set_policy(client, 300)
    set_sampling(client, max_interval=60, grace=0)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    body = get_status(client, "2026-01-01T08:10:00+08:00").json()
    item = body[0]
    assert item["device_id"] == "dev-1"
    assert item["opened_at"] == "2026-01-01T00:02:00Z"
    assert item["as_of"] == "2026-01-01T00:10:00Z"  # +08:00 normalized to UTC
    assert item["deadline"] == 300
    # Last reading 00:02, M=60, as_of 00:10: tail deducts [00:03,00:10] = 420 s
    # out of the 480 s window -> 60 s observable, not reached.
    assert item["observable_seconds"] == 60.0
    assert item["deadline_reached"] is False
    assert item["deadline_reached_at"] is None


# ------------------------------------------------------------ timing rules

def setup_open_alarm(client, minutes=(0, 1, 2), m=60, d=300):
    set_thresholds(client)
    set_sampling(client, max_interval=m, grace=0)
    set_policy(client, d)
    post_readings(
        client,
        [(f"2026-01-01T00:{minute:02d}:00+00:00", 31.0) for minute in minutes],
    )


def test_response_status_continuous_sampling_tail_gap(client):
    # Minute readings 00:00..00:02 (opens 00:02), M=60, as_of 00:10:
    # only [00:02,00:03] is observable (60 s), tail [00:03,00:10] deducted.
    setup_open_alarm(client)
    item = get_status(client, ts(10)).json()[0]
    assert item["observable_seconds"] == 60.0
    assert item["deadline_reached"] is False
    assert item["deadline_reached_at"] is None


def test_response_status_deadline_reached_with_first_instant(client):
    # Minute readings 00:00..00:07 (opens 00:02), M=60, D=300, as_of 00:10:
    # observable [00:02,00:08) = 360 s (tail [00:07+M,00:10] deducts the
    # final 120 s), so 300 s was first accumulated exactly at 00:07.
    setup_open_alarm(client, minutes=range(8))
    item = get_status(client, ts(10)).json()[0]
    assert item["observable_seconds"] == 360.0
    assert item["deadline_reached"] is True
    assert item["deadline_reached_at"] == "2026-01-01T00:07:00Z"


def test_response_status_internal_gap_deduction(client):
    # Readings at 00,01,02 (open),05,06; M=60; gap [00:03,00:05) deducts
    # 120 s; tail [00:07,00:10] deducts 180 s; window 480 s -> 180 s.
    setup_open_alarm(client, minutes=(0, 1, 2, 5, 6), d=180)
    item = get_status(client, ts(10)).json()[0]
    assert item["observable_seconds"] == 180.0
    assert item["deadline_reached"] is True
    # 60 s [00:02,00:03) + 120 s [00:05,00:07) = 180 s -> reached at 00:07.
    assert item["deadline_reached_at"] == "2026-01-01T00:07:00Z"

    # 30 seconds earlier: 60 + 90 = 150 s -> not reached.
    item = get_status(client, "2026-01-01T00:06:30Z").json()[0]
    assert item["observable_seconds"] == 150.0
    assert item["deadline_reached"] is False
    assert item["deadline_reached_at"] is None


def test_response_status_interval_exactly_m_is_observable(client):
    # Readings 00:00,00:01,00:02 (opens 00:02) then 00:04 with M=120: every
    # interval is <= M (00:02->00:04 is exactly M) and the last reading 00:04
    # is exactly M before as_of 00:06 -> no internal gap, no tail deduction.
    setup_open_alarm(client, minutes=(0, 1, 2, 4), m=120, d=240)
    item = get_status(client, ts(6)).json()[0]
    assert item["observable_seconds"] == 240.0
    assert item["deadline_reached"] is True
    assert item["deadline_reached_at"] == "2026-01-01T00:06:00Z"


def test_response_status_gap_before_opened_at_is_not_deducted(client):
    # A wide gap before the alarm's opened_at is clipped away: the readings
    # 23:00 -> 00:01 (M=60) are irrelevant to an alarm that opens at 00:02.
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    set_policy(client, 60)
    post_readings(
        client,
        [("2025-12-31T23:00:00Z", 31.0), (ts(1), 31.0), (ts(2), 31.0), (ts(3), 31.0)],
    )
    # Opened 00:02 (three highs 00:01..00:03 after the pre-history low/high:
    # 23:00 >= H also counts, so the streak is 23:00,00:01,00:02 -> open 00:02).
    item = get_status(client, ts(3)).json()[0]
    assert item["opened_at"] == "2026-01-01T00:02:00Z"
    assert item["observable_seconds"] == 60.0
    assert item["deadline_reached"] is True
    assert item["deadline_reached_at"] == "2026-01-01T00:03:00Z"


def test_response_status_reached_inside_observable_chunk(client):
    setup_open_alarm(client, minutes=range(8), d=90)
    item = get_status(client, ts(10)).json()[0]
    assert item["deadline_reached"] is True
    assert item["deadline_reached_at"] == "2026-01-01T00:03:30Z"


def test_response_status_microsecond_boundary_exact_deadline(client):
    # M=1s, D=8s. Microsecond-jittered samples across three recovered gaps:
    # the observable chunks are 2.279285 + 3.227535 + 1.863487 + 0.629693 s,
    # exactly 8 s in total. The cumulative measure, the reached check and the
    # first-reached instant share one exact (microsecond) precision, so the
    # exact-deadline case reports reached=true with the instant the 8th
    # observable second completes — never reached=true with a null instant.
    set_thresholds(client)
    set_sampling(client, max_interval=1, grace=0)
    set_policy(client, 8)
    post_readings(
        client,
        [("2026-01-01T00:00:08.000000Z", 31.0),
         ("2026-01-01T00:00:09.000000Z", 31.0),
         ("2026-01-01T00:00:10.000000Z", 31.0),   # alarm opens here
         ("2026-01-01T00:00:11.000000Z", 31.0),
         ("2026-01-01T00:00:11.279285Z", 31.0),   # gap [11.279285+1s, 14.279285)
         ("2026-01-01T00:00:14.279285Z", 31.0),
         ("2026-01-01T00:00:15.279285Z", 31.0),
         ("2026-01-01T00:00:16.279285Z", 31.0),
         ("2026-01-01T00:00:16.506820Z", 31.0),   # gap [17.506820, 19.506820)
         ("2026-01-01T00:00:19.506820Z", 31.0),
         ("2026-01-01T00:00:20.370307Z", 31.0),   # gap [21.370307, 23.370307)
         ("2026-01-01T00:00:23.370307Z", 31.0)],  # last sample, 0.629693 s before as_of
    )
    item = get_status(client, "2026-01-01T00:00:24.000000Z").json()[0]
    assert item["observable_seconds"] == 8.0
    assert item["deadline_reached"] is True
    assert item["deadline_reached_at"] == "2026-01-01T00:00:24Z"

    # One microsecond earlier the 8th observable second has not completed.
    item = get_status(client, "2026-01-01T00:00:23.999999Z").json()[0]
    assert item["observable_seconds"] == 7.999999
    assert item["deadline_reached"] is False
    assert item["deadline_reached_at"] is None


def test_response_status_multiple_alarms_are_independent(client):
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    set_policy(client, 60)
    post_readings(
        client,
        [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0),   # open 00:02
         (ts(3), 19.0), (ts(4), 19.0),                  # close 00:04
         (ts(6), 31.0), (ts(7), 31.0), (ts(8), 31.0)],  # open 00:08
    )
    body = get_status(client, ts(9)).json()
    # Only the second alarm is still open: observable [00:08,00:09] = 60 s.
    assert len(body) == 1
    assert body[0]["opened_at"] == "2026-01-01T00:08:00Z"
    assert body[0]["observable_seconds"] == 60.0
    assert body[0]["deadline_reached"] is True
    assert body[0]["deadline_reached_at"] == "2026-01-01T00:09:00Z"


# ------------------------------------------------------------ recompute on query

def test_response_status_recomputed_after_late_arrival(client):
    # Open alarm 00:03 with readings 00:01..00:03; gap 00:03->as_of 00:10
    # (M=60) leaves only 60 observable seconds -> D=180 not reached.
    setup_open_alarm(client, minutes=(1, 2, 3), d=180)
    item = get_status(client, ts(10)).json()[0]
    assert item["observable_seconds"] == 60.0
    assert item["deadline_reached"] is False

    # Late arrivals 00:04..00:09 extend the observable lifetime:
    # [00:03,00:10) minus nothing = 420 s (last reading exactly M before as_of).
    post_readings(client, [(ts(m), 31.0) for m in (4, 5, 6, 7, 8, 9)])
    item = get_status(client, ts(10)).json()[0]
    assert item["observable_seconds"] == 420.0
    assert item["deadline_reached"] is True
    assert item["deadline_reached_at"] == "2026-01-01T00:06:00Z"


def test_response_status_recomputed_after_correction(client):
    setup_open_alarm(client, minutes=range(8), d=300)
    assert get_status(client, ts(10)).json()[0]["deadline_reached"] is True
    # Correct 00:01 below H: the high streak restarts at 00:02 and the alarm
    # moves from 00:02 to 00:04; observable [00:04,00:08) is 240 s < D.
    post_readings(client, [(ts(1), 10.0)])
    body = get_status(client, ts(10)).json()
    assert len(body) == 1
    item = body[0]
    assert item["opened_at"] == "2026-01-01T00:04:00Z"
    assert item["observable_seconds"] == 240.0
    assert item["deadline_reached"] is False
    assert item["deadline_reached_at"] is None

    # Two more low readings close the alarm at 00:07 -> it leaves the listing.
    post_readings(client, [(ts(6), 10.0), (ts(7), 10.0)])
    assert get_status(client, ts(10)).json() == []


def test_response_status_recomputed_after_m_change(client):
    # Readings every 2 minutes 00:00..00:06 (opens 00:04), as_of 00:10:
    # with M=60 every interval is a gap; observable pieces are just the
    # M-wide heads [t,t+60) clipped to the window.
    set_thresholds(client)
    set_sampling(client, max_interval=60, grace=0)
    set_policy(client, 360)
    post_readings(client, [(ts(m), 31.0) for m in (0, 2, 4, 6)])
    item = get_status(client, ts(10)).json()[0]
    # [00:04,00:05) + [00:06,00:07) + tail deducts from 00:07 -> 120 s
    assert item["observable_seconds"] == 120.0
    assert item["deadline_reached"] is False

    # Raising M to 120 makes every interval observable: the window
    # [00:04,00:10] is 360 s with only the tail [00:08,00:10] (120 s)
    # deducted -> 240 s, still short of D=360.
    set_sampling(client, max_interval=120, grace=0)
    item = get_status(client, ts(10)).json()[0]
    assert item["observable_seconds"] == 240.0
    assert item["deadline_reached"] is False

    # With D = 240 the deadline is met exactly at the start of the tail gap.
    set_policy(client, 240)
    item = get_status(client, ts(10)).json()[0]
    assert item["deadline_reached"] is True
    assert item["deadline_reached_at"] == "2026-01-01T00:08:00Z"


def test_response_status_recomputed_after_d_change(client):
    setup_open_alarm(client, minutes=range(8), d=300)
    item = get_status(client, ts(10)).json()[0]
    assert item["deadline_reached"] is True
    set_policy(client, 361)
    item = get_status(client, ts(10)).json()[0]
    assert item["deadline"] == 361
    assert item["observable_seconds"] == 360.0
    assert item["deadline_reached"] is False
    assert item["deadline_reached_at"] is None


def test_response_status_recomputed_after_acknowledgement(client):
    setup_open_alarm(client, minutes=range(8), d=240)
    assert len(get_status(client, ts(10)).json()) == 1
    ack_open_alarm(client)
    assert get_status(client, ts(10)).json() == []
    # Correcting the first two highs below H restarts the streak at 00:02,
    # moving the alarm's open to 00:04; the acknowledgement is dropped with
    # the old open instant, so the alarm reappears unacknowledged.
    post_readings(client, [(ts(0), 10.0), (ts(1), 10.0)])
    body = get_status(client, ts(10)).json()
    assert len(body) == 1
    assert body[0]["opened_at"] == "2026-01-01T00:04:00Z"
    assert body[0]["deadline_reached"] is True
    # Re-acknowledging the new open instant hides it again.
    ack_open_alarm(client, opened_at="2026-01-01T00:04:00+00:00")
    assert get_status(client, ts(10)).json() == []


def test_response_status_persists_nothing(client):
    setup_open_alarm(client, minutes=range(8), d=300)
    before = {
        "device": client.get("/devices/dev-1").json(),
        "readings": client.get("/devices/dev-1/readings").json(),
        "alarms": get_alarms(client).json(),
        "revisions": get_revisions(client).json(),
        "sampling": get_sampling(client).json(),
        "policy": get_policy(client).json(),
    }
    resp = get_status(client, ts(10))
    assert resp.status_code == 200
    assert client.get("/devices/dev-1").json() == before["device"]
    assert client.get("/devices/dev-1/readings").json() == before["readings"]
    assert get_alarms(client).json() == before["alarms"]
    assert get_revisions(client).json() == before["revisions"]
    assert get_sampling(client).json() == before["sampling"]
    assert get_policy(client).json() == before["policy"]


# ============================================================ temperature lag

def get_lag(client, group="grp-1", **params):
    return client.get(f"/groups/{group}/lag", params=params)


def lag_params(target="a", reference="b", start="2026-01-01T00:00:00Z",
               end="2026-01-01T01:00:00Z", max_lag=120, min_pairs=3):
    return {
        "target_id": target,
        "reference_id": reference,
        "start": start,
        "end": end,
        "max_lag": max_lag,
        "min_pairs": min_pairs,
    }


def tsec(second, minute=0, tz="+00:00"):
    """UTC timestamp at minute*60 + second total seconds (supports second >= 60)."""
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute, seconds=second)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + tz


def setup_lag_group(client, members=("a", "b"), max_interval=120, group="grp-1"):
    """Create the member devices (thresholds + sampling config) and group them."""
    for d in members:
        set_thresholds(client, device=d)
        set_sampling(client, max_interval, 0, device=d)
    assert set_group(client, list(members), 1, group=group).status_code == 200


def ramp_readings(seconds, temp_at_second):
    return [(tsec(s), temp_at_second(s)) for s in seconds]


# ------------------------------------------------------------ validation

def test_lag_unknown_group_is_404(client):
    assert get_lag(client, **lag_params()).status_code == 404


def test_lag_target_and_reference_must_be_distinct_members(client):
    setup_lag_group(client)
    set_thresholds(client, device="outsider")
    set_sampling(client, 120, 0, device="outsider")
    assert get_lag(client, **lag_params(target="ghost", reference="b")).status_code == 404
    assert get_lag(client, **lag_params(target="a", reference="outsider")).status_code == 404
    # Same device as target and reference -> 422.
    assert get_lag(client, **lag_params(target="a", reference="a")).status_code == 422


def test_lag_reference_without_sampling_config_is_409(client):
    for d in ("a", "b"):
        set_thresholds(client, device=d)
        set_sampling(client, 120, 0, device=d)
    set_thresholds(client, device="c")  # device c has thresholds but no sampling config
    assert set_group(client, ["a", "b", "c"], 1).status_code == 200
    resp = get_lag(client, **lag_params(target="a", reference="c"))
    assert resp.status_code == 409
    # The target itself needs no sampling config.
    resp = get_lag(client, **lag_params(target="c", reference="b"))
    assert resp.status_code == 200


def test_lag_validates_params(client):
    setup_lag_group(client)
    base = lag_params()

    def get(**overrides):
        return get_lag(client, **{**base, **overrides})

    for key in base:
        assert get_lag(
            client, **{k: v for k, v in base.items() if k != key}
        ).status_code == 422
    # naive start / end -> 422
    assert get(start="2026-01-01T00:00:00").status_code == 422
    assert get(end="2026-01-01T01:00:00").status_code == 422
    # start == end and start > end -> 422
    assert get(start=ts(1), end=ts(1)).status_code == 422
    assert get(start=ts(2), end=ts(0)).status_code == 422
    # max_lag must be an integer 1..300 -> 422 otherwise
    for bad in (0, -1, 301, 1.5, "x"):
        assert get(max_lag=bad).status_code == 422
    # min_pairs must be an integer >= 3 -> 422 otherwise
    for bad in (0, 1, 2, 2.5, "x"):
        assert get(min_pairs=bad).status_code == 422


# ------------------------------------------------------------ estimation rules

def test_lag_finds_shift_between_second_grids_via_interpolation(client):
    # Both devices ramp +1 deg/min; the reference series is shifted +45s, with
    # no reading on the target grid, so the win comes from interpolation.
    setup_lag_group(client, max_interval=60)
    post_readings(client, ramp_readings(range(0, 300, 60), lambda s: 20 + s // 60), device="a")
    post_readings(client, ramp_readings(range(45, 360, 60), lambda s: 20 + (s - 45) // 60), device="b")
    resp = get_lag(client, **lag_params(end="2026-01-01T00:05:00Z"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["lag"] == 45
    assert body["pairs"] == 5
    assert body["median_abs_diff"] == 0.0


def test_lag_exact_reference_reading_is_used_as_is(client):
    # Reference lags exactly one minute behind; every shifted target instant
    # is an exact reference reading (no interpolation needed).
    setup_lag_group(client, max_interval=120)
    post_readings(client, [(ts(m), 20.0 + m) for m in range(5)], device="a")
    post_readings(client, [(ts(m), 20.0 + (m - 1)) for m in range(1, 6)], device="b")
    resp = get_lag(client, **lag_params(end="2026-01-01T00:05:00Z"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["lag"] == 60
    assert body["pairs"] == 5
    assert body["median_abs_diff"] == 0.0


def test_lag_negative_shift_when_reference_runs_ahead(client):
    # Same as the previous case with the roles' timing reversed: matching the
    # reference requires shifting target instants backwards.
    setup_lag_group(client, max_interval=120)
    post_readings(client, [(ts(m), 20.0 + (m - 1)) for m in range(1, 6)], device="a")
    post_readings(client, [(ts(m), 20.0 + m) for m in range(5)], device="b")
    resp = get_lag(
        client,
        **lag_params(start="2026-01-01T00:01:00Z", end="2026-01-01T00:06:00Z"),
    )
    body = resp.json()
    assert body["lag"] == -60
    assert body["pairs"] == 5
    assert body["median_abs_diff"] == 0.0


def test_lag_never_extrapolates(client):
    # The matching reference readings start five minutes after the window;
    # no shift within max_lag reaches them without extrapolating.
    setup_lag_group(client, max_interval=60)
    post_readings(client, [(ts(m), 20.0) for m in range(3)], device="a")
    post_readings(client, [(ts(m), 20.0) for m in range(5, 8)], device="b")
    resp = get_lag(client, **lag_params(end="2026-01-01T00:03:00Z", max_lag=120))
    assert resp.status_code == 200
    assert resp.json() == {
        "group_id": "grp-1",
        "target_id": "a",
        "reference_id": "b",
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T00:03:00Z",
        "max_lag": 120,
        "min_pairs": 3,
        "status": "insufficient_data",
        "lag": None,
        "pairs": 0,
        "median_abs_diff": None,
    }


def test_lag_reference_gap_beyond_m_blocks_interpolation(client):
    # Reference readings 5 minutes apart with M=60: only exact matches count,
    # and every shift has at most one of them.
    setup_lag_group(client, max_interval=60)
    post_readings(client, [(ts(m), 20.0) for m in range(3)], device="a")
    post_readings(client, [(ts(0), 20.0), (ts(5), 20.0)], device="b")
    resp = get_lag(client, **lag_params(end="2026-01-01T00:03:00Z"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "insufficient_data"
    assert body["lag"] is None
    assert body["pairs"] == 0
    assert body["median_abs_diff"] is None


def test_lag_reference_interval_equal_to_m_interpolates(client):
    # Reference readings 2 minutes apart with M=120: the gap equals M, so the
    # midpoint is interpolated and all three target points pair at lag 0.
    setup_lag_group(client, max_interval=120)
    post_readings(client, [(ts(0), 20.0), (ts(1), 21.0), (ts(2), 22.0)], device="a")
    post_readings(client, [(ts(0), 20.0), (ts(2), 22.0)], device="b")
    resp = get_lag(client, **lag_params(end="2026-01-01T00:03:00Z"))
    body = resp.json()
    assert body["status"] == "ok"
    assert body["lag"] == 0
    assert body["pairs"] == 3
    assert body["median_abs_diff"] == 0.0


def test_lag_insufficient_data_when_no_shift_reaches_min_pairs(client):
    setup_lag_group(client, max_interval=60)
    post_readings(client, [(ts(m), 20.0) for m in range(5)], device="a")
    post_readings(client, [(ts(0), 20.0)], device="b")  # one exact match at most
    resp = get_lag(client, **lag_params(end="2026-01-01T00:05:00Z"))
    assert resp.json()["status"] == "insufficient_data"


def test_lag_only_target_points_inside_half_open_window(client):
    # Reference one minute behind; end excludes the target's 00:03 point, so
    # lag 60 gets exactly three pairs instead of four.
    setup_lag_group(client, max_interval=120)
    post_readings(client, [(ts(m), 20.0 + m) for m in range(4)], device="a")
    post_readings(client, [(ts(m), 20.0 + (m - 1)) for m in range(1, 5)], device="b")
    resp = get_lag(client, **lag_params(end="2026-01-01T00:03:00Z"))
    body = resp.json()
    assert body["lag"] == 60
    assert body["pairs"] == 3


def test_lag_tie_broken_by_pair_count(client):
    # All values are 20. Reference cluster at minutes 1..4 (M=60, so the
    # minute gaps interpolate but wider stretches do not). At lag 0 the
    # target gets 3 exact pairs; at lag +60 it gets 4; the pair count beats
    # the smaller |lag| despite the equal (zero) median.
    setup_lag_group(client, max_interval=60)
    post_readings(client, [(ts(m), 20.0) for m in (0, 1, 2, 3, 5)], device="a")
    post_readings(client, [(ts(m), 20.0) for m in range(1, 5)], device="b")
    resp = get_lag(client, **lag_params(end="2026-01-01T00:06:00Z", max_lag=120))
    body = resp.json()
    assert body["lag"] == 60
    assert body["pairs"] == 4
    assert body["median_abs_diff"] == 0.0


def test_lag_tie_broken_by_smallest_absolute_lag(client):
    # Constant 20 everywhere; the reference spans both sides of the target's
    # three points, so every shift in [-60, 60] pairs all three points (exact
    # or interpolated) with zero median; |lag| picks 0.
    setup_lag_group(client, max_interval=120)
    post_readings(client, [(ts(m), 20.0) for m in (2, 3, 4)], device="a")
    post_readings(client, [(ts(m), 20.0) for m in range(0, 6)], device="b")
    resp = get_lag(client, **lag_params(end="2026-01-01T00:06:00Z", max_lag=120))
    body = resp.json()
    assert body["lag"] == 0
    assert body["pairs"] == 3
    assert body["median_abs_diff"] == 0.0


def test_lag_tie_broken_by_smallest_lag(client):
    # The target at minutes 0/10/20 matches the reference exactly at both
    # lag -60 and lag +60 (3 pairs, zero median); lag 0 and every other shift
    # have no comparable value because reference readings sit two minutes
    # apart with M=60. The final tie-break is the smallest lag: -60.
    setup_lag_group(client, max_interval=60)
    post_readings(
        client,
        [(ts(0), 20.0), (ts(10), 22.0), (ts(20), 20.0)],
        device="a",
    )
    post_readings(
        client,
        [("2025-12-31T23:59:00Z", 20.0), (ts(1), 20.0), (ts(9), 22.0),
         (ts(11), 22.0), (ts(19), 20.0), (ts(21), 20.0)],
        device="b",
    )
    resp = get_lag(client, **lag_params(end="2026-01-01T00:21:00Z", max_lag=120))
    body = resp.json()
    assert body["lag"] == -60
    assert body["pairs"] == 3
    assert body["median_abs_diff"] == 0.0


def test_lag_does_not_search_beyond_max_lag(client):
    # Same +45s setup as the interpolation test, but max_lag=30: the true
    # shift is out of range, so lag 30 wins with a non-zero median and the
    # first target point cannot be paired (no extrapolation).
    setup_lag_group(client, max_interval=60)
    post_readings(client, ramp_readings(range(0, 300, 60), lambda s: 20 + s // 60), device="a")
    post_readings(client, ramp_readings(range(45, 360, 60), lambda s: 20 + (s - 45) // 60), device="b")
    body = get_lag(client, **lag_params(end="2026-01-01T00:05:00Z", max_lag=30)).json()
    assert body["lag"] == 30
    assert body["pairs"] == 4
    assert body["median_abs_diff"] == 0.25


def test_lag_timezone_offsets_are_normalized(client):
    # The window given in +08:00 is the same UTC interval as elsewhere.
    setup_lag_group(client, max_interval=60)
    post_readings(client, ramp_readings(range(0, 300, 60), lambda s: 20 + s // 60), device="a")
    post_readings(client, ramp_readings(range(45, 360, 60), lambda s: 20 + (s - 45) // 60), device="b")
    resp = get_lag(
        client,
        **lag_params(
            start="2026-01-01T08:00:00+08:00",
            end="2026-01-01T08:05:00+08:00",
        ),
    )
    body = resp.json()
    assert body["lag"] == 45
    assert body["pairs"] == 5
    assert body["start"] == "2026-01-01T00:00:00Z"
    assert body["end"] == "2026-01-01T00:05:00Z"


def test_lag_recomputed_after_reference_correction(client):
    setup_lag_group(client, max_interval=60)
    post_readings(client, ramp_readings(range(0, 300, 60), lambda s: 20 + s // 60), device="a")
    post_readings(client, ramp_readings(range(45, 360, 60), lambda s: 20 + (s - 45) // 60), device="b")
    assert get_lag(client, **lag_params(end="2026-01-01T00:05:00Z")).json()["lag"] == 45
    # Correcting the reference values at the same instants removes the 45s
    # offset: its continuous ramp now matches the target's (lag 0 best).
    post_readings(client, ramp_readings(range(45, 360, 60), lambda s: 20 + s / 60), device="b")
    body = get_lag(client, **lag_params(end="2026-01-01T00:05:00Z")).json()
    assert body["lag"] == 0
    assert body["pairs"] == 4
    assert body["median_abs_diff"] == 0.0


def test_lag_recomputed_after_membership_change(client):
    setup_lag_group(client)
    post_readings(client, steady_readings(range(5)), device="a")
    post_readings(client, steady_readings(range(5)), device="b")
    resp = get_lag(client, **lag_params(end="2026-01-01T00:05:00Z"))
    assert resp.status_code == 200
    # Removing the reference device makes the next query 404.
    assert set_group(client, ["a"], 1).status_code == 200
    assert get_lag(client, **lag_params(end="2026-01-01T00:05:00Z")).status_code == 404


def test_lag_persists_nothing(client):
    setup_lag_group(client)
    post_readings(client, steady_readings(range(4)), device="a")
    post_readings(client, steady_readings(range(4)), device="b")
    before = {
        "group": get_group(client).json(),
        "readings": {
            d: client.get(f"/devices/{d}/readings").json() for d in ("a", "b")
        },
        "alarms": {d: get_alarms(client, device=d).json() for d in ("a", "b")},
        "sampling": {d: get_sampling(client, device=d).json() for d in ("a", "b")},
    }
    resp = get_lag(client, **lag_params(end="2026-01-01T00:04:00Z"))
    assert resp.status_code == 200
    assert get_group(client).json() == before["group"]
    for d in ("a", "b"):
        assert client.get(f"/devices/{d}/readings").json() == before["readings"][d]
        assert get_alarms(client, device=d).json() == before["alarms"][d]
        assert get_sampling(client, device=d).json() == before["sampling"][d]


# ===================================================== alarm consistency / reconcile

OPEN_TS = "2026-01-01T00:02:00.000000+00:00"
CLOSE_TS = "2026-01-01T00:04:00.000000+00:00"
OPEN_INTERVAL = {"opened_at": "2026-01-01T00:02:00Z", "closed_at": "2026-01-01T00:04:00Z"}


def consistency(client, device="dev-1"):
    return client.get(f"/devices/{device}/alarm-consistency")


def reconcile(client, device="dev-1"):
    return client.post(f"/devices/{device}/alarm-reconcile")


def open_and_closed_dev1(client):
    """dev-1 with exactly one closed alarm (open 00:02, close 00:04)."""
    set_thresholds(client)
    post_readings(
        client,
        [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0),
         (ts(3), 10.0), (ts(4), 10.0)],
    )


def raw_db():
    """Open the test database directly (the app only holds per-request connections)."""
    conn = sqlite3.connect(os.environ["DATABASE_PATH"])
    conn.execute("PRAGMA foreign_keys=OFF")
    return conn


def insert_alarm(conn, device="dev-1", opened_at=OPEN_TS, closed_at=CLOSE_TS):
    conn.execute(
        "INSERT INTO alarms (device_id, opened_at, closed_at) VALUES (?, ?, ?)",
        (device, opened_at, closed_at),
    )


def delete_alarms(conn, device="dev-1"):
    conn.execute("DELETE FROM alarms WHERE device_id = ?", (device,))


# ------------------------------------------------------------ GET: consistent

def test_consistency_reports_clean_device(client):
    open_and_closed_dev1(client)
    body = consistency(client).json()
    assert body["device_id"] == "dev-1"
    assert body["consistent"] is True
    assert body["stored_alarm_count"] == 1
    assert body["recomputed_alarm_count"] == 1
    assert body["missing_alarms"] == []
    assert body["extra_alarms"] == []
    assert body["duplicate_alarms"] == []
    assert body["orphan_acknowledgements"] == []


def test_consistency_clean_device_with_multiple_cycles(client):
    set_thresholds(client)
    post_readings(
        client,
        [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0),   # open  @00:02
         (ts(3), 10.0), (ts(4), 10.0),                  # close @00:04
         (ts(5), 31.0), (ts(6), 31.0), (ts(7), 31.0),   # open  @00:07
         (ts(8), 10.0), (ts(9), 10.0)],                 # close @00:09
    )
    body = consistency(client).json()
    assert body["consistent"] is True
    assert body["stored_alarm_count"] == 2
    assert body["recomputed_alarm_count"] == 2


def test_consistency_clean_device_with_no_alarms(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 25.0)])  # below H: no alarm
    body = consistency(client).json()
    assert body["consistent"] is True
    assert body["stored_alarm_count"] == 0
    assert body["recomputed_alarm_count"] == 0


def test_consistency_unknown_device_is_404(client):
    assert consistency(client, device="ghost").status_code == 404
    assert reconcile(client, device="ghost").status_code == 404


def test_consistency_is_read_only(client):
    open_and_closed_dev1(client)
    before = {
        "alarms": get_alarms(client).json(),
        "readings": client.get("/devices/dev-1/readings").json(),
        "revisions": [r["revision"] for r in get_revisions(client).json()],
    }
    assert consistency(client).status_code == 200
    assert consistency(client).status_code == 200  # repeatable
    assert get_alarms(client).json() == before["alarms"]
    assert client.get("/devices/dev-1/readings").json() == before["readings"]
    assert [r["revision"] for r in get_revisions(client).json()] == before["revisions"]


# ------------------------------------------------------------ GET: differences

def test_consistency_reports_missing_alarm(client):
    open_and_closed_dev1(client)
    conn = raw_db()
    delete_alarms(conn)
    conn.commit()
    conn.close()

    body = consistency(client).json()
    assert body["consistent"] is False
    assert body["stored_alarm_count"] == 0
    assert body["recomputed_alarm_count"] == 1
    assert body["missing_alarms"] == [OPEN_INTERVAL]
    assert body["extra_alarms"] == []
    assert body["duplicate_alarms"] == []


def test_consistency_reports_extra_alarm(client):
    open_and_closed_dev1(client)
    conn = raw_db()
    insert_alarm(conn, opened_at="2026-01-01T09:00:00.000000+00:00",
                 closed_at="2026-01-01T09:05:00.000000+00:00")
    conn.commit()
    conn.close()

    body = consistency(client).json()
    assert body["consistent"] is False
    assert body["stored_alarm_count"] == 2
    assert body["missing_alarms"] == []
    assert body["extra_alarms"] == [
        {"opened_at": "2026-01-01T09:00:00Z", "closed_at": "2026-01-01T09:05:00Z"}
    ]
    assert body["duplicate_alarms"] == []  # stored once: extra, not a duplicate


def test_consistency_reports_missing_and_extra_for_moved_alarm(client):
    open_and_closed_dev1(client)
    conn = raw_db()
    # The stored open instant moved away from the recomputed 00:02: with
    # item-wise multiset comparison this is one missing + one extra.
    conn.execute(
        "UPDATE alarms SET opened_at = ? WHERE device_id = ?",
        ("2026-01-01T00:03:00.000000+00:00", "dev-1"),
    )
    conn.commit()
    conn.close()

    body = consistency(client).json()
    assert body["consistent"] is False
    assert body["missing_alarms"] == [OPEN_INTERVAL]
    assert body["extra_alarms"] == [
        {"opened_at": "2026-01-01T00:03:00Z", "closed_at": "2026-01-01T00:04:00Z"}
    ]


def test_consistency_reports_duplicate_alarm_rows(client):
    open_and_closed_dev1(client)
    conn = raw_db()
    # Two extra copies of the real interval: stored 3 times, recomputed once.
    insert_alarm(conn)
    insert_alarm(conn)
    conn.commit()
    conn.close()

    body = consistency(client).json()
    assert body["consistent"] is False
    assert body["stored_alarm_count"] == 3
    assert body["recomputed_alarm_count"] == 1
    assert body["missing_alarms"] == []
    # All surplus stored copies are extras...
    assert body["extra_alarms"] == [OPEN_INTERVAL, OPEN_INTERVAL]
    # ...and the copies past the first are reported as duplicates.
    assert body["duplicate_alarms"] == [OPEN_INTERVAL, OPEN_INTERVAL]


def test_consistency_open_interval_duplicate_keeps_null_close(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])  # open @00:02
    conn = raw_db()
    insert_alarm(conn, closed_at=None)
    conn.commit()
    conn.close()

    body = consistency(client).json()
    assert body["missing_alarms"] == []
    assert body["extra_alarms"] == [
        {"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}
    ]
    assert body["duplicate_alarms"] == [
        {"opened_at": "2026-01-01T00:02:00Z", "closed_at": None}
    ]


def test_consistency_reports_orphan_acknowledgement(client):
    open_dev1_alarm(client)  # one open alarm at 00:02, acknowledged
    assert ack(client, expected_revision=2).status_code == 201
    conn = raw_db()
    conn.execute(
        "INSERT INTO alarm_acknowledgements (device_id, opened_at, operator, acknowledged_at)"
        " VALUES (?, ?, ?, ?)",
        ("dev-1", "2026-01-01T08:00:00.000000+00:00", "mallory",
         "2026-09-26T01:00:00.000000+00:00"),
    )
    conn.commit()
    conn.close()

    body = consistency(client).json()
    assert body["consistent"] is False
    # The real alarm at 00:02 matches the recompute: its acknowledgement is
    # not an orphan; only the 08:00 one is.
    assert len(body["orphan_acknowledgements"]) == 1
    orphan = body["orphan_acknowledgements"][0]
    assert orphan["opened_at"] == "2026-01-01T08:00:00Z"
    assert orphan["operator"] == "mallory"
    assert body["missing_alarms"] == []
    assert body["extra_alarms"] == []
    assert body["duplicate_alarms"] == []


def test_consistency_does_not_report_ack_as_orphan_when_only_close_changes(client):
    # An acknowledgement follows the open instant: a different closed_at does
    # not make it an orphan (the recompute still has an alarm at 00:02).
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    assert ack(client, expected_revision=2).status_code == 201
    conn = raw_db()
    conn.execute(
        "UPDATE alarms SET closed_at = ? WHERE device_id = ?",
        (CLOSE_TS, "dev-1"),  # recompute leaves it open; stored row says closed
    )
    conn.commit()
    conn.close()

    body = consistency(client).json()
    assert body["consistent"] is False  # intervals differ (close time)...
    assert body["orphan_acknowledgements"] == []  # ...but the ack is not orphaned


# ------------------------------------------------------------ POST: no diff

def test_reconcile_no_diff_touches_nothing(client):
    open_dev1_alarm(client)  # one open alarm at 00:02, acknowledged
    assert ack(client, expected_revision=2).status_code == 201
    alarms_before = get_alarms(client).json()
    ack_at_before = alarms_before[0]["acknowledged_at"]

    resp = reconcile(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["fixed"] is False
    assert body["before"] == body["after"]
    assert body["after"]["consistent"] is True

    alarms_after = get_alarms(client).json()
    # Alarm row ids are untouched when there is nothing to fix.
    assert [a["id"] for a in alarms_after] == [a["id"] for a in alarms_before]
    assert alarms_after[0]["acknowledged_by"] == "alice"
    assert alarms_after[0]["acknowledged_at"] == ack_at_before
    # No new revision is created by a reconcile.
    assert get_alarms(client).headers["x-alarm-revision"] == "2"


def test_reconcile_no_diff_on_empty_alarms(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 25.0)])  # no alarm
    body = reconcile(client).json()
    assert body["fixed"] is False
    assert body["after"]["stored_alarm_count"] == 0
    assert body["after"]["recomputed_alarm_count"] == 0


# ------------------------------------------------------------ POST: repairs

def test_reconcile_rebuilds_missing_and_extra_alarms(client):
    open_and_closed_dev1(client)
    conn = raw_db()
    delete_alarms(conn)
    insert_alarm(conn, opened_at="2026-01-01T09:00:00.000000+00:00",
                 closed_at="2026-01-01T09:05:00.000000+00:00")
    insert_alarm(conn, opened_at="2026-01-01T09:00:00.000000+00:00",
                 closed_at="2026-01-01T09:05:00.000000+00:00")
    conn.commit()
    conn.close()

    body = reconcile(client).json()
    assert body["fixed"] is True
    before, after = body["before"], body["after"]
    assert before["consistent"] is False
    assert before["missing_alarms"] == [OPEN_INTERVAL]
    assert len(before["extra_alarms"]) == 2
    assert len(before["duplicate_alarms"]) == 1  # second copy of the 09:00 interval
    assert after["consistent"] is True
    assert after["missing_alarms"] == []
    assert after["extra_alarms"] == []
    assert after["duplicate_alarms"] == []
    assert after["stored_alarm_count"] == 1
    assert get_alarms(client).json() == [
        {
            "id": get_alarms(client).json()[0]["id"],
            "device_id": "dev-1",
            "opened_at": "2026-01-01T00:02:00Z",
            "closed_at": "2026-01-01T00:04:00Z",
            "status": "closed",
            "acknowledged_by": None,
            "acknowledged_at": None,
        }
    ]


def test_reconcile_removes_duplicate_rows(client):
    open_and_closed_dev1(client)
    conn = raw_db()
    insert_alarm(conn)
    insert_alarm(conn)
    conn.commit()
    conn.close()

    body = reconcile(client).json()
    assert body["fixed"] is True
    assert len(body["before"]["duplicate_alarms"]) == 2
    assert body["after"]["consistent"] is True
    assert body["after"]["stored_alarm_count"] == 1
    assert len(get_alarms(client).json()) == 1


def test_reconcile_keeps_first_ack_of_still_matching_open_instant(client):
    # Open alarm at 00:02, acknowledged by alice; an extra bogus alarm with
    # an orphan acknowledgement is injected alongside.
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    assert ack(client, expected_revision=2).status_code == 201
    first_ack = get_alarms(client).json()[0]["acknowledged_at"]

    conn = raw_db()
    insert_alarm(conn, opened_at="2026-01-01T09:00:00.000000+00:00", closed_at=None)
    conn.execute(
        "INSERT INTO alarm_acknowledgements (device_id, opened_at, operator, acknowledged_at)"
        " VALUES (?, ?, ?, ?)",
        ("dev-1", "2026-01-01T09:00:00.000000+00:00", "mallory",
         "2026-09-26T01:00:00.000000+00:00"),
    )
    conn.commit()
    conn.close()

    body = reconcile(client).json()
    assert body["fixed"] is True
    assert [o["operator"] for o in body["before"]["orphan_acknowledgements"]] == ["mallory"]
    assert body["after"]["orphan_acknowledgements"] == []

    alarms = get_alarms(client).json()
    assert len(alarms) == 1
    assert alarms[0]["opened_at"] == "2026-01-01T00:02:00Z"
    assert alarms[0]["acknowledged_by"] == "alice"  # first ack kept verbatim
    assert alarms[0]["acknowledged_at"] == first_ack


def test_reconcile_keeps_ack_when_only_close_time_differs(client):
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    assert ack(client, expected_revision=2).status_code == 201
    first_ack = get_alarms(client).json()[0]["acknowledged_at"]
    conn = raw_db()
    conn.execute(
        "UPDATE alarms SET closed_at = ? WHERE device_id = ?", (CLOSE_TS, "dev-1")
    )
    conn.commit()
    conn.close()

    body = reconcile(client).json()
    assert body["fixed"] is True
    alarm = get_alarms(client).json()[0]
    assert alarm["closed_at"] is None  # recompute leaves it open
    assert alarm["acknowledged_by"] == "alice"
    assert alarm["acknowledged_at"] == first_ack


def test_reconcile_clears_orphan_ack_without_rebuilding_consistent_alarms(client):
    open_and_closed_dev1(client)
    conn = raw_db()
    real_row_id = conn.execute(
        "SELECT id FROM alarms WHERE device_id = 'dev-1'"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO alarm_acknowledgements (device_id, opened_at, operator, acknowledged_at)"
        " VALUES (?, ?, ?, ?)",
        ("dev-1", "2026-01-01T08:00:00.000000+00:00", "ghost",
         "2026-09-26T01:00:00.000000+00:00"),
    )
    conn.commit()
    conn.close()

    body = reconcile(client).json()
    assert body["fixed"] is True
    assert body["before"]["missing_alarms"] == []
    assert body["before"]["extra_alarms"] == []
    assert len(body["before"]["orphan_acknowledgements"]) == 1
    assert body["after"]["consistent"] is True
    # The alarm rows were already correct: they are not rebuilt, so the row
    # id survives even though the orphan acknowledgement was deleted.
    conn = raw_db()
    assert conn.execute(
        "SELECT id FROM alarms WHERE device_id = 'dev-1'"
    ).fetchone()[0] == real_row_id
    assert conn.execute(
        "SELECT COUNT(*) FROM alarm_acknowledgements WHERE device_id = 'dev-1'"
    ).fetchone()[0] == 0
    conn.close()


def test_reconcile_deletes_all_acks_when_recompute_has_no_alarms(client):
    open_dev1_alarm(client)  # open alarm at 00:02, acknowledged by alice
    assert ack(client, expected_revision=2).status_code == 201
    conn = raw_db()
    # Stored state claims the alarm is already closed; the recompute still
    # produces an alarm at the same open instant (left open), so the ack
    # keyed at 00:02 must survive the repair.
    conn.execute(
        "UPDATE alarms SET closed_at = ? WHERE device_id = ?", (CLOSE_TS, "dev-1")
    )
    conn.commit()
    conn.close()

    body = reconcile(client).json()
    assert body["fixed"] is True
    alarm = get_alarms(client).json()[0]
    assert alarm["opened_at"] == "2026-01-01T00:02:00Z"
    assert alarm["acknowledged_by"] == "alice"

    # Now make the recompute produce no alarm at all: every ack must go.
    post_readings(client, [(ts(0), 25.0), (ts(1), 25.0), (ts(2), 25.0)])
    assert consistency(client).json()["consistent"] is True
    conn = raw_db()
    # Directly attach an orphan ack the normal API would reject.
    conn.execute(
        "INSERT INTO alarm_acknowledgements (device_id, opened_at, operator, acknowledged_at)"
        " VALUES (?, ?, ?, ?)",
        ("dev-1", OPEN_TS, "bob", "2026-09-26T02:00:00.000000+00:00"),
    )
    conn.commit()
    conn.close()
    body = reconcile(client).json()
    assert body["fixed"] is True
    assert body["after"]["orphan_acknowledgements"] == []
    conn = raw_db()
    assert conn.execute(
        "SELECT COUNT(*) FROM alarm_acknowledgements WHERE device_id = 'dev-1'"
    ).fetchone()[0] == 0
    conn.close()


def test_reconcile_is_idempotent(client):
    open_and_closed_dev1(client)
    conn = raw_db()
    delete_alarms(conn)
    insert_alarm(conn, opened_at="2026-01-01T09:00:00.000000+00:00",
                 closed_at="2026-01-01T09:05:00.000000+00:00")
    conn.commit()
    conn.close()

    first = reconcile(client).json()
    assert first["fixed"] is True
    second = reconcile(client).json()
    assert second["fixed"] is False
    assert second["before"] == second["after"]
    assert second["after"] == first["after"]


def test_reconcile_does_not_touch_readings_thresholds_or_revisions(client):
    open_and_closed_dev1(client)
    revisions_before = get_revisions(client).json()
    readings_before = client.get("/devices/dev-1/readings").json()
    device_before = client.get("/devices/dev-1").json()

    conn = raw_db()
    delete_alarms(conn)
    conn.commit()
    conn.close()

    assert reconcile(client).json()["fixed"] is True

    assert client.get("/devices/dev-1/readings").json() == readings_before
    assert client.get("/devices/dev-1").json() == device_before
    revisions_after = get_revisions(client).json()
    assert revisions_after == revisions_before  # no new revision recorded
    assert get_alarms(client).headers["x-alarm-revision"] == "2"


def test_consistency_uses_utc_open_instants_for_ack_matching(client):
    # An acknowledgement stored under the UTC instant matches even though the
    # recompute internally normalizes other timezone spellings.
    set_thresholds(client)
    post_readings(client, [(ts(0), 31.0), (ts(1), 31.0), (ts(2), 31.0)])
    assert ack(client, opened_at="2026-01-01T08:02:00+08:00").status_code == 201
    body = consistency(client).json()
    assert body["consistent"] is True
    assert body["orphan_acknowledgements"] == []
