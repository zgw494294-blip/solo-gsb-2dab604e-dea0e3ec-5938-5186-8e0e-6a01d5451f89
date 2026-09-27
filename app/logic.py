"""Alarm state machine.

Rules (readings processed in ascending sample-time order):
  * No active alarm: 3 consecutive readings with temperature >= H open an
    alarm at the 3rd reading's sample time. Any reading < H resets the
    consecutive count.
  * Active alarm: 2 consecutive readings with temperature <= L close the
    alarm at the 2nd reading's sample time. Any reading > L resets the
    consecutive count.

Because H > L, the reading that opens/closes an alarm can never count
toward the opposite streak, so resetting the opposite counter on a
transition is safe.
"""

import math
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Iterable, List, Optional, Tuple


@dataclass
class Alarm:
    opened_at: str
    closed_at: Optional[str]  # None while the alarm is still open


def compute_alarms(
    readings: Iterable[Tuple[str, float]], threshold_h: float, threshold_l: float
) -> list[Alarm]:
    """Derive alarms from (sampled_at, temperature) pairs in time order."""
    alarms: list[Alarm] = []
    high_streak = 0  # consecutive readings >= H (while no alarm is active)
    low_streak = 0   # consecutive readings <= L (while an alarm is active)
    current_open: Optional[str] = None

    for sampled_at, temperature in readings:
        if current_open is None:
            if temperature >= threshold_h:
                high_streak += 1
                if high_streak == 3:
                    current_open = sampled_at
                    low_streak = 0
            else:
                high_streak = 0
        else:
            if temperature <= threshold_l:
                low_streak += 1
                if low_streak == 2:
                    alarms.append(Alarm(opened_at=current_open, closed_at=sampled_at))
                    current_open = None
                    high_streak = 0
            else:
                low_streak = 0

    if current_open is not None:
        alarms.append(Alarm(opened_at=current_open, closed_at=None))
    return alarms


# ------------------------------------------------------- alarm revision diff

def diff_alarm_intervals(
    old: Iterable[Tuple[str, Optional[str]]],
    new: Iterable[Tuple[str, Optional[str]]],
) -> Tuple[List[Tuple[str, Optional[str]]], List[Tuple[str, Optional[str]]]]:
    """Compare alarm intervals by (opened_at, closed_at), ignoring alarm row ids.

    Returns (added, withdrawn), each sorted by opened_at: an interval present
    only in ``new`` is an addition, one present only in ``old`` is a
    withdrawal, so a moved alarm shows up as a withdrawal of the old interval
    plus an addition of the new one.
    """
    before, after = set(old), set(new)
    key = lambda iv: (iv[0], iv[1] or "")  # noqa: E731
    return sorted(after - before, key=key), sorted(before - after, key=key)


# ------------------------------------------------- alarm consistency (reconcile)

@dataclass
class AlarmConsistency:
    """Multiset comparison of stored alarm rows against recomputed intervals.

    Alarms are derived data (a pure function of readings + thresholds), so
    stored rows should always equal what the state machine recomputes. The
    comparison is item-wise by (opened_at, closed_at) multiplicity, so
    duplicated stored rows show up instead of being hidden by a set compare:

      missing   -- occurrences the recompute has but the stored rows lack
      extra     -- occurrences stored beyond the recomputed multiplicity
      duplicate -- the surplus occurrences of one interval past its first
                   stored copy (an interval stored 3 times contributes 2
                   entries); empty when every interval is stored once
    """
    missing: List[Tuple[str, Optional[str]]]
    extra: List[Tuple[str, Optional[str]]]
    duplicate: List[Tuple[str, Optional[str]]]


def compare_alarm_consistency(
    stored: Iterable[Tuple[str, Optional[str]]],
    recomputed: Iterable[Tuple[str, Optional[str]]],
) -> AlarmConsistency:
    """Compare stored alarm intervals against recomputed ones as multisets.

    Both inputs are (opened_at, closed_at) pairs compared item-wise, ignoring
    alarm row ids. Intervals the recompute has and storage lacks become
    ``missing``; storage occurrences beyond the recomputed multiplicity become
    ``extra``; among the surplus stored occurrences, those of an interval
    stored more than once are additionally listed in ``duplicate`` (every
    copy past the first). All three lists are sorted by opened_at.
    """
    stored_counts = Counter(stored)
    recomputed_counts = Counter(recomputed)
    key = lambda iv: (iv[0], iv[1] or "")  # noqa: E731

    missing: List[Tuple[str, Optional[str]]] = []
    extra: List[Tuple[str, Optional[str]]] = []
    duplicate: List[Tuple[str, Optional[str]]] = []
    for interval in sorted(set(stored_counts) | set(recomputed_counts), key=key):
        stored_n = stored_counts.get(interval, 0)
        recomputed_n = recomputed_counts.get(interval, 0)
        if recomputed_n > stored_n:
            missing.extend([interval] * (recomputed_n - stored_n))
        elif stored_n > recomputed_n:
            surplus = stored_n - recomputed_n
            extra.extend([interval] * surplus)
            # Copies past the first are duplicates; when the recompute also
            # has the interval, that many surplus copies are extras instead.
            duplicate.extend([interval] * min(stored_n - 1, surplus))
    return AlarmConsistency(missing=missing, extra=extra, duplicate=duplicate)


# ------------------------------------------------------------- sampling gaps

@dataclass
class Gap:
    started_at: str          # UTC ISO-8601: prev sampled_at + M
    ended_at: Optional[str]  # UTC ISO-8601: next sampled_at; None while ongoing
    status: str              # "recovered" | "ongoing"


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def compute_gaps(
    readings: Iterable[Tuple[str, float]],
    max_interval: int,
    grace: int,
    as_of: datetime,
    *,
    utc_format,
) -> List[Gap]:
    """Derive missing-sample intervals from readings with sampled_at <= as_of.

    Inputs:
      readings      -- (sampled_at_iso, temperature) pairs (any order); only
                       sample instants not later than ``as_of`` are considered.
      max_interval  -- M seconds: a gap strictly larger than M between adjacent
                       readings is a missing interval (exactly M is fine).
      grace         -- G seconds: the last reading only opens an *ongoing* gap
                       at as_of once as_of - last_sampled_at > M + G.
      as_of         -- aware query instant; comparison happens in UTC.
      utc_format(s) -- callable mapping an aware datetime to the stored/returned
                       UTC ISO string (caller owns the serialization format).

    No readings at all yields no gaps (nothing can be inferred).
    Pure function of current data: corrections and late arrivals simply change
    the ``readings`` input, so results update on the next query.
    """
    cutoff = as_of.astimezone(timezone.utc)
    instants = (_parse_utc(sampled_at) for sampled_at, _ in readings)
    times = sorted({t for t in instants if t <= cutoff})
    if not times:
        return []

    m = timedelta(seconds=max_interval)
    gaps: List[Gap] = []

    # Adjacent readings: a delta strictly greater than M is a recovered gap
    # from (prev + M) up to the next reading's sample time.
    for prev, nxt in zip(times, times[1:]):
        if nxt - prev > m:
            gaps.append(
                Gap(
                    started_at=utc_format(prev + m),
                    ended_at=utc_format(nxt),
                    status="recovered",
                )
            )

    # Last reading: beyond M + G from as_of, sampling is currently interrupted.
    last = times[-1]
    if cutoff - last > m + timedelta(seconds=grace):
        gaps.append(Gap(started_at=utc_format(last + m), ended_at=None, status="ongoing"))

    return gaps


# ------------------------------------------------------- alarm response deadline

@dataclass
class ResponseStatus:
    observable_seconds: float        # observable seconds in [opened_at, as_of]
    deadline_reached: bool
    deadline_reached_at: Optional[str]  # UTC ISO-8601 first instant D is reached; None otherwise


def compute_response_status(
    opened_at: str,
    readings: Iterable[Tuple[str, float]],
    max_interval: int,
    deadline: int,
    as_of: datetime,
    *,
    utc_format,
) -> ResponseStatus:
    """Measure the observable lifetime of one open alarm against deadline D.

    Inputs:
      opened_at     -- UTC ISO-8601 open instant of the alarm; observable time
                       is measured only within [opened_at, as_of].
      readings      -- (sampled_at_iso, temperature) pairs (any order); only
                       sample instants not later than ``as_of`` are considered.
      max_interval  -- M seconds: an interval strictly larger than M between
                       adjacent readings is unobservable (exactly M is fine).
      deadline      -- D seconds: the response deadline to reach.
      as_of         -- aware query instant; comparison happens in UTC.
      utc_format(s) -- callable mapping an aware datetime to the returned
                       UTC ISO string (caller owns the serialization format).

    Observable time starts at ``opened_at``. For two adjacent readings whose
    interval is strictly greater than M, the half-open stretch
    [prev + M, next) is deducted; if the last reading is more than M before
    ``as_of``, [last + M, as_of] is deducted. Every deducted stretch is
    clipped to [opened_at, as_of]. The remaining measure is the cumulative
    observable time; ``deadline_reached`` is true once it has accumulated D
    seconds, and ``deadline_reached_at`` is the first UTC instant at which
    that happens (null while it has not).

    The cumulative measure, the reached check and the first-reached instant
    are all computed in exact timedelta arithmetic (microsecond resolution,
    the precision of the stored timestamps), so a cumulative measure of
    exactly D seconds can never split the three apart through float
    rounding; only the returned ``observable_seconds`` is a float.

    Pure function of current data: corrections, late arrivals, config
    changes and alarm recomputes simply change the inputs, so the next query
    sees them.
    """
    cutoff = as_of.astimezone(timezone.utc)
    start = _parse_utc(opened_at)
    times = sorted(
        {
            _parse_utc(sampled_at)
            for sampled_at, _ in readings
            if _parse_utc(sampled_at) <= cutoff
        }
    )
    if not times:
        # An alarm's opened_at is itself a reading time, so a still-open alarm
        # opened at or before as_of always has at least one reading; keep the
        # degenerate case well-defined regardless.
        return ResponseStatus(0.0, False, None)

    m = timedelta(seconds=max_interval)
    d = timedelta(seconds=deadline)

    # Unobservable half-open stretches, each clipped to [opened_at, as_of]:
    # [prev + M, next) for adjacent readings more than M apart, plus
    # [last + M, as_of] when sampling has been interrupted past as_of.
    gaps: List[Tuple[datetime, datetime]] = []
    for prev, nxt in zip(times, times[1:]):
        if nxt - prev > m:
            gaps.append((max(prev + m, start), min(nxt, cutoff)))
    last = times[-1]
    if cutoff - last > m:
        gaps.append((max(last + m, start), cutoff))

    # Accumulate in exact timedelta arithmetic (microsecond resolution):
    # the cumulative measure, the reached check and the first-reached
    # instant share one precision, so they cannot disagree through float
    # rounding when the true cumulative measure is D exactly.
    observable = timedelta(0)
    cursor = start  # observable stretches run between consecutive gap starts
    reached_at: Optional[datetime] = None
    for gap_start, gap_end in gaps:
        if gap_start >= gap_end:
            continue  # the raw stretch lies entirely outside [opened_at, as_of]
        chunk = gap_start - cursor
        if reached_at is None and observable + chunk >= d:
            reached_at = cursor + (d - observable)
        observable += chunk
        cursor = gap_end
    if cursor < cutoff:
        chunk = cutoff - cursor
        if reached_at is None and observable + chunk >= d:
            reached_at = cursor + (d - observable)
        observable += chunk

    return ResponseStatus(
        observable_seconds=observable.total_seconds(),
        deadline_reached=observable >= d,
        deadline_reached_at=utc_format(reached_at) if reached_at is not None else None,
    )


# ----------------------------------------------------------- heat exposure

@dataclass
class HeatExposure:
    covered_seconds: float        # window time covered by valid interpolation
    above_seconds: float          # covered time with temperature strictly above T
    excess_degree_seconds: float  # integral of max(temperature - T, 0) dt over the window


def _segment_above(
    v0: float,
    v1: float,
    baseline: float,
    off_start_us: int,
    off_end_us: int,
    span_us: int,
    duration: float,
) -> tuple[float, float]:
    """Above-baseline time and degree-seconds of one clipped linear segment.

    Temperature changes linearly from v0 to v1 across the segment's
    ``span_us`` microseconds; the clip window is the microsecond offset
    range [off_start_us, off_end_us] within that span and lasts
    ``duration`` seconds. When the temperature crosses ``baseline`` inside
    the window it is split at the crossing instant: only the part strictly
    above T counts, and the excess integral of that part is the
    trapezoidal area of (temperature - T).

    All temperature arithmetic runs on one common power-of-two scale
    derived from the largest magnitude involved. The scale is the largest
    power of two not exceeding that magnitude (taken from frexp's binary
    exponent, which never overflows even for sys.float_info.max itself),
    division by it is exact and every scaled value lands in [-2, 2), which
    fixes both ends of the finite-input range at once:

      * the scale never overflows while being chosen, so legitimate extreme
        readings (e.g. -1 -> sys.float_info.max around T = 0) compute
        instead of raising OverflowError from a scale as large as 2**1024;
      * the scaled difference can never overflow, so legitimate extreme
        readings (e.g. -1e308 -> 1e308 around a baseline near +1e308)
        keep a finite crossing ratio and area instead of collapsing to
        zero through 1 - tiny_ratio cancellation;
      * the scaled crossing denominator can never underflow to zero, so
        the smallest positive readings (e.g. 0 -> 5e-324, the minimum
        subnormal, at T = 0) interpolate instead of dividing by zero.

    The clip-boundary heights are affine interpolations of the scaled
    endpoint excess heights a0/a1, evaluated from the *nearer* endpoint:
    a boundary one microsecond before t1 over a 1e11-second span sits at
    fraction 1 - 1e-17 from t0, which rounds to exactly 1.0 in float64,
    so a "from t0" evaluation (or any 1 - f form) loses the offset and
    misreads the boundary temperature by a whole endpoint ulp. Taking the
    fraction from the nearer endpoint keeps the microsecond offset as a
    tiny but nonzero fraction of the span, so the boundary height keeps
    its proper value between the baseline-adjacent floats (e.g. T one ulp
    below sys.float_info.max) and the area of a microsecond-wide window
    at the segment's end stays the true trapezoid instead of a
    constant-endpoint rectangle.

    The scale is folded back in by multiplication *last*, after the tiny
    crossing ratio and the bounded scaled heights have already been
    multiplied together: multiplying it earlier (before the ~1e-16
    crossing ratio) overflows intermediate products for a perfectly
    finite area (e.g. a ten-second -1e308 -> 1e308 ramp crossing
    T = 9.999999999999998e307, whose true area is ~9.958e276). The
    result can only become infinite when the true degree-second value
    itself exceeds the float range.
    """
    # Largest of |v0|, |v1|, |T| may be zero (all three zero): then nothing
    # is strictly above T. Otherwise frexp gives the binary exponent
    # directly: the largest power of two not exceeding ``largest`` is
    # 2**(exponent - 1). Unlike ldexp(1, floor(log2(largest))) this never
    # asks for 2**1024, so it stays finite when largest is
    # sys.float_info.max (log2 of which is exactly 1024.0).
    largest = max(abs(v0), abs(v1), abs(baseline))
    if largest == 0.0:
        return 0.0, 0.0
    scale = math.ldexp(1.0, math.frexp(largest)[1] - 1)
    # Scaled excess heights (v - T) at the segment endpoints: the
    # power-of-two divisions are exact and the subtractions of values in
    # [-2, 2) can neither overflow nor underflow to a zero denominator.
    a0 = v0 / scale - baseline / scale
    a1 = v1 / scale - baseline / scale
    slope = a1 - a0

    def height_at(off_us: int) -> float:
        # Affine interpolation of the scaled excess height at a clip
        # boundary, evaluated from the nearer endpoint so the boundary's
        # microsecond offset survives as a tiny fraction of the span
        # (a "from the far endpoint" form rounds 1 - 1e-17 to 1.0 and
        # loses it). Integer microsecond offsets keep the ratio exact
        # relative to the whole span; an offset of 0 yields exactly a0
        # and an offset of span_us exactly a1.
        if off_us * 2 <= span_us:
            return a0 + slope * (off_us / span_us)
        return a1 - slope * ((span_us - off_us) / span_us)

    h_start = height_at(off_start_us)
    h_end = height_at(off_end_us)
    above0, above1 = h_start > 0.0, h_end > 0.0
    if not above0 and not above1:
        return 0.0, 0.0
    if above0 and above1:
        # Trapezoid 0.5 * (h_start + h_end) * duration on the common
        # scale; the scaled heights stay finite for any finite endpoints,
        # and scale is multiplied back in last so a finite trapezoid
        # cannot overflow on an intermediate product.
        return duration, 0.5 * duration * (h_start + h_end) * scale

    # The temperature crosses T inside the clip window: split at the
    # (unique) crossing point. above_w is the scaled excess of the
    # above-side boundary, below_w the scaled depth of the other; the
    # crossing fraction of the window is measured from the above side.
    # The denominator is a sum of positive magnitudes, so it can neither
    # cancel to zero nor overflow. The above part is a triangle in
    # (temperature - T): area 0.5 * excess * above_time, all evaluated on
    # the common scale with scale folded back in last.
    if above1:
        above_w, below_w = h_end, -h_start
    else:
        above_w, below_w = h_start, -h_end
    above_ratio = above_w / (above_w + below_w)
    above_time = duration * above_ratio
    excess = 0.5 * above_time * above_w * scale
    return above_time, excess


def compute_heat_exposure(
    readings: Iterable[Tuple[str, float]],
    max_interval: int,
    start: datetime,
    end: datetime,
    baseline: float,
) -> HeatExposure:
    """Integrate temperature exposure over [start, end] from interpolated readings.

    Inputs:
      readings      -- (sampled_at_iso, temperature) pairs (any order).
      max_interval  -- M seconds: an interval between adjacent readings is
                       interpolated only when it does not exceed M.
      start, end    -- aware window bounds (start < end), compared in UTC.
      baseline      -- T: temperature is "above" only when strictly greater.

    For each adjacent pair of readings whose interval is <= M, temperature
    is treated as a linear function of time. The segment is clipped to the
    query window (a boundary may therefore sit between two readings, using
    readings outside the window for the interpolation). Long gaps (> M)
    and time before the first / after the last reading are never covered
    and are never extrapolated. Where the clipped segment crosses T it is
    split at the crossing so above-time and the excess degree-second
    integral (trapezoid of temperature - T) count only the part above T.

    Clip boundaries are passed to the segment math as exact integer
    microsecond offsets within the segment's span (the resolution of the
    stored timestamps), and the boundary temperatures are evaluated
    inside the segment math as affine interpolations of the scaled
    endpoint excess heights, taken from the *nearer* endpoint. A plain
    fraction-of-span from t0 (or any ``1 - f`` form) rounds to exactly
    1.0 in float64 for a boundary one microsecond before t1 of a
    1e11-second span, silently snapping the boundary temperature to the
    far endpoint; a slope form ``v0 + (v1 - v0) * fraction`` overflows to
    +-inf whenever the endpoint difference exceeds the float range
    (legitimate extremes such as -1e308 -> 1e308). The near-endpoint
    affine form keeps the microsecond offset as a tiny nonzero fraction
    of the span and stays finite for any finite endpoint temperatures,
    so a microsecond-wide window at either end of a huge-span segment
    keeps its true interpolated boundary temperature even when the
    endpoints are the extreme finite floats and T sits one ulp away.
    The crossing ratio and the (temperature - T) area are likewise taken
    on one common power-of-two scale — the largest power of two not
    exceeding the largest magnitude of the two endpoints and T, taken
    from frexp's binary exponent so that choosing the scale cannot
    overflow even for sys.float_info.max — rather than as raw
    differences or ``1 - ratio``: this keeps them finite for endpoint
    differences near the float range (no cancellation of a tiny
    crossing fraction) and nonzero for the smallest subnormal readings
    (no underflow to a zero denominator). The scale is multiplied back
    into the area only *after* the bounded scaled height and the
    crossing fraction, so a narrow crossing above a huge baseline keeps
    its finite area over any window length instead of overflowing an
    intermediate product; the area is infinite only when the true
    integral itself exceeds the float range.

    Pure function of current data: corrections, late arrivals and M
    changes simply change the inputs, so the next query sees them.
    """
    w_start = start.astimezone(timezone.utc)
    w_end = end.astimezone(timezone.utc)
    points = sorted(
        {
            (_parse_utc(sampled_at), float(temperature))
            for sampled_at, temperature in readings
        }
    )

    covered = above = excess = 0.0
    m = timedelta(seconds=max_interval)
    microsecond = timedelta(microseconds=1)
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t1 - t0 > m:
            continue  # long missing-sample stretch: no interpolation
        c_start, c_end = max(t0, w_start), min(t1, w_end)
        if c_start >= c_end:
            continue  # no overlap with the window (endpoints touching add zero)

        duration = (c_end - c_start).total_seconds()
        covered += duration
        # Exact integer microsecond offsets of the clip window within the
        # segment's span (timedelta arithmetic is exact at microsecond
        # resolution): the boundary temperatures are interpolated from
        # these inside _segment_above, where the fraction is taken from
        # the nearer endpoint so a microsecond offset from a segment end
        # survives instead of rounding to exactly 0 or 1.
        span_us = (t1 - t0) // microsecond
        off_start_us = (c_start - t0) // microsecond
        off_end_us = (c_end - t0) // microsecond
        seg_above, seg_excess = _segment_above(
            v0, v1, baseline, off_start_us, off_end_us, span_us, duration
        )
        above += seg_above
        excess += seg_excess

    return HeatExposure(
        covered_seconds=covered,
        above_seconds=above,
        excess_degree_seconds=excess,
    )


# -------------------------------------------------------------- group alarms

@dataclass
class GroupAlarm:
    opened_at: str
    closed_at: Optional[str]  # None while the group alarm is still open


def compute_group_alarms(
    intervals: Iterable[Tuple[str, Optional[str]]],
    k: int,
    *,
    utc_format,
) -> List[GroupAlarm]:
    """Derive group alarm intervals from member devices' alarm intervals.

    Inputs:
      intervals -- (opened_at_iso, closed_at_iso) alarm intervals of the
                   group's current members, each [opened_at, closed_at);
                   closed_at None means the device alarm is still open.
      k         -- minimum concurrent member alarms: a group alarm is open
                   while at least k members are in alarm at the same time.
      utc_format(s) -- callable mapping an aware datetime to the returned
                       UTC ISO string (caller owns the serialization format).

    Member opens and closes at the same instant are netted before the count
    is evaluated, so simultaneous transitions that never let the count drop
    below k do not split an interval, and no zero-length interval can be
    produced. Pure function of current data: member/K changes and member
    alarm recomputes simply change the inputs, so the next query sees them.
    """
    deltas: dict[datetime, int] = {}
    for opened_at, closed_at in intervals:
        opened = _parse_utc(opened_at)
        deltas[opened] = deltas.get(opened, 0) + 1
        if closed_at is not None:
            closed = _parse_utc(closed_at)
            deltas[closed] = deltas.get(closed, 0) - 1

    alarms: List[GroupAlarm] = []
    concurrent = 0  # members in alarm; >= k exactly while a group alarm is open
    current_open: Optional[datetime] = None
    for t in sorted(deltas):
        new_count = concurrent + deltas[t]
        if current_open is None and new_count >= k:
            current_open = t
        elif current_open is not None and new_count < k:
            alarms.append(
                GroupAlarm(opened_at=utc_format(current_open), closed_at=utc_format(t))
            )
            current_open = None
        concurrent = new_count

    if current_open is not None:
        alarms.append(GroupAlarm(opened_at=utc_format(current_open), closed_at=None))
    return alarms


# ---------------------------------------------------------- peer deviations

@dataclass
class PeerDeviation:
    started_at: str          # UTC ISO-8601: first deviation point of the segment
    ended_at: str            # UTC ISO-8601: last deviation point of the segment
    samples: int             # consecutive deviation points in the segment
    max_abs_deviation: float  # largest |target - peer median| across the segment


def _sorted_points(readings: Iterable[Tuple[str, float]]) -> List[Tuple[datetime, float]]:
    """(instant, temperature) list sorted by instant, deduplicated by instant."""
    by_instant = {}
    for sampled_at, temperature in readings:
        by_instant[_parse_utc(sampled_at)] = float(temperature)
    return sorted(by_instant.items())


def _peer_value_at(
    times: List[datetime],
    values: List[float],
    instant: datetime,
    max_interval: int,
) -> Optional[float]:
    """One peer's temperature at an instant: exact reading or bounded interpolation.

    An exact reading at ``instant`` is used as-is. Otherwise the value is
    linearly interpolated from the peer's adjacent readings bracketing
    ``instant``, but only when their sampling interval does not exceed
    ``max_interval`` (M) seconds; a wider gap yields no value. Time before
    the first / after the last reading is never extrapolated (None).
    """
    i = bisect_left(times, instant)
    if i < len(times) and times[i] == instant:
        return values[i]  # exact reading at the same instant: take the stored value
    if i == 0 or i == len(times):
        return None  # outside the first/last reading: no extrapolation
    t0, t1 = times[i - 1], times[i]
    if t1 - t0 > timedelta(seconds=max_interval):
        return None  # bracketing readings too far apart: no interpolation
    v0, v1 = values[i - 1], values[i]
    fraction = (instant - t0).total_seconds() / (t1 - t0).total_seconds()
    return v0 + (v1 - v0) * fraction


def compute_peer_deviations(
    target_readings: Iterable[Tuple[str, float]],
    target_max_interval: int,
    peers: Iterable[Tuple[Iterable[Tuple[str, float]], int]],
    start: datetime,
    end: datetime,
    delta: float,
    min_samples: int,
    *,
    utc_format,
) -> List[PeerDeviation]:
    """Find segments where a device persistently deviates from its group peers.

    Inputs:
      target_readings     -- (sampled_at_iso, temperature) pairs of the target
                             device (any order); only sample instants inside
                             [start, end) are processed.
      target_max_interval -- the target's M seconds: a sampling gap strictly
                             larger than M between adjacent target points
                             breaks a consecutive deviation segment.
      peers               -- one (readings, max_interval) pair per participating
                             peer (members without a configured M are excluded
                             by the caller and never reach this function).
      start, end          -- aware window bounds (start < end), compared in UTC.
      delta               -- a point deviates when |target - peer median| is
                             strictly greater than delta.
      min_samples         -- a run of consecutive deviation points is reported
                             once it holds at least this many points.
      utc_format(s)       -- callable mapping an aware datetime to the returned
                             UTC ISO string (caller owns the serialization format).

    Each target sample point is checked in ascending time order: a peer
    contributes its exact reading at that instant, or a linear interpolation
    between its adjacent readings at most M apart (never extrapolated). With
    at least two peer values their median is the reference. Too few
    comparable peers, a non-deviating point or a target sampling gap beyond
    the target's M all break the consecutive run. Each surviving run of at
    least ``min_samples`` deviation points is returned with its first/last
    instants, point count and maximum absolute deviation, in UTC order.

    Pure function of current data: late arrivals, corrections, config and
    membership changes simply change the inputs, so the next query sees them.
    """
    w_start = start.astimezone(timezone.utc)
    w_end = end.astimezone(timezone.utc)
    points = [p for p in _sorted_points(target_readings) if w_start <= p[0] < w_end]

    # Pre-sort each participating peer's series once, then bisect per instant.
    peer_series = []
    for readings, max_interval in peers:
        series = _sorted_points(readings)
        peer_series.append(
            ([t for t, _ in series], [v for _, v in series], max_interval)
        )

    m_target = timedelta(seconds=target_max_interval)
    segments: List[PeerDeviation] = []
    run: List[Tuple[datetime, float]] = []  # current run: (instant, abs deviation)
    prev_instant: Optional[datetime] = None

    def close_run() -> None:
        if len(run) >= min_samples:
            segments.append(
                PeerDeviation(
                    started_at=utc_format(run[0][0]),
                    ended_at=utc_format(run[-1][0]),
                    samples=len(run),
                    max_abs_deviation=max(d for _, d in run),
                )
            )
        run.clear()

    for instant, temperature in points:
        # A sampling gap beyond the target's own M breaks the consecutive run.
        if prev_instant is not None and instant - prev_instant > m_target:
            close_run()
        prev_instant = instant

        peer_values = []
        for times, values, peer_m in peer_series:
            value = _peer_value_at(times, values, instant, peer_m)
            if value is not None:
                peer_values.append(value)
        if len(peer_values) < 2:
            close_run()  # too few comparable peers at this instant
            continue
        deviation = abs(temperature - median(peer_values))
        if deviation > delta:
            run.append((instant, deviation))
        else:
            close_run()  # not a deviation point

    close_run()
    return segments


# ---------------------------------------------------------- temperature lag

@dataclass
class LagEstimate:
    status: str                   # "ok" | "insufficient_data"
    lag: Optional[int]            # chosen whole-second shift; None when insufficient_data
    pairs: int                    # comparable pairs backing the chosen lag (0 when insufficient)
    median_abs_diff: Optional[float]  # median |target(t) - reference(t + lag)|; None when insufficient


def compute_temperature_lag(
    target_readings: Iterable[Tuple[str, float]],
    reference_readings: Iterable[Tuple[str, float]],
    reference_max_interval: int,
    start: datetime,
    end: datetime,
    max_lag: int,
    min_pairs: int,
) -> LagEstimate:
    """Estimate the integer-second time offset between two devices' temperatures.

    Inputs:
      target_readings      -- (sampled_at_iso, temperature) pairs of the target
                              device (any order); only sample instants inside
                              [start, end) are traversed.
      reference_readings   -- (sampled_at_iso, temperature) pairs of the
                              reference device (any order).
      reference_max_interval -- the reference's M seconds: its temperature at a
                              shifted instant is an exact reading when one
                              exists, otherwise a linear interpolation between
                              its adjacent readings only when their interval
                              does not exceed M (never extrapolated).
      start, end           -- aware window bounds (start < end), compared in UTC.
      max_lag              -- whole-second shifts -max_lag..max_lag are tried.
      min_pairs            -- a shift is a candidate only when at least this
                              many target points have a comparable reference
                              value.

    For every target sample point t in [start, end) and every integer lag in
    [-max_lag, max_lag], the reference temperature at t + lag is taken as the
    stored reading at that instant, or linearly interpolated between the
    reference readings bracketing it when their interval is at most M; time
    outside the reference's first/last reading is never extrapolated and that
    pair is skipped. Shifts backed by at least ``min_pairs`` pairs are ranked
    by: smallest median absolute temperature difference, then most pairs,
    then smallest |lag|, then smallest lag. When no shift reaches
    ``min_pairs`` the result is insufficient_data with a null lag.

    Pure function of current data: late arrivals, corrections, membership and
    config changes simply change the inputs, so the next query sees them.
    """
    w_start = start.astimezone(timezone.utc)
    w_end = end.astimezone(timezone.utc)
    points = [p for p in _sorted_points(target_readings) if w_start <= p[0] < w_end]
    ref_series = _sorted_points(reference_readings)
    ref_times = [t for t, _ in ref_series]
    ref_values = [v for _, v in ref_series]

    # (median diff, -pair count, |lag|, lag): ascending lexicographic order is
    # exactly the required ranking — smallest median, most pairs, smallest
    # |lag|, smallest lag.
    best_key = None
    best: Optional[Tuple[float, int, int]] = None
    for lag in range(-max_lag, max_lag + 1):
        shift = timedelta(seconds=lag)
        diffs = []
        for instant, temperature in points:
            value = _peer_value_at(
                ref_times, ref_values, instant + shift, reference_max_interval
            )
            if value is not None:
                diffs.append(abs(temperature - value))
        if len(diffs) >= min_pairs:
            med = median(diffs)
            key = (med, -len(diffs), abs(lag), lag)
            if best_key is None or key < best_key:
                best_key = key
                best = (med, len(diffs), lag)

    if best is None:
        return LagEstimate(status="insufficient_data", lag=None, pairs=0, median_abs_diff=None)
    med, pairs, lag = best
    return LagEstimate(status="ok", lag=lag, pairs=pairs, median_abs_diff=med)


# ------------------------------------------------------------------ forecast

@dataclass
class Forecast:
    status: str                  # "predicted" | "no_crossing" | "insufficient_data"
    samples: int                 # points of the latest contiguous segment (<= max_points)
    slope: Optional[float]       # fitted slope b (temperature/second); None when insufficient_data
    intercept: Optional[float]   # fitted intercept a at as_of; None when insufficient_data
    predicted_at: Optional[str]  # UTC ISO-8601 crossing instant; None unless predicted


def compute_forecast(
    readings: Iterable[Tuple[str, float]],
    max_interval: int,
    threshold_h: float,
    as_of: datetime,
    lookback: int,
    horizon: int,
    *,
    utc_format,
    max_points: int = 30,
    min_points: int = 3,
) -> Forecast:
    """Forecast when the temperature trend crosses the threshold H.

    Inputs:
      readings      -- (sampled_at_iso, temperature) pairs (any order).
      max_interval  -- M seconds: adjacent samples of the contiguous segment
                       may be at most this far apart, and the newest sample
                       may be at most this old relative to ``as_of``.
      threshold_h   -- H: the threshold the fitted line is projected against.
      as_of         -- aware query instant; the zero point of the fit.
      lookback      -- seconds: only samples inside [as_of - lookback, as_of]
                       are considered.
      horizon       -- seconds: the crossing must happen within this far
                       after as_of to be predicted.
      utc_format(s) -- callable mapping an aware datetime to the returned
                       UTC ISO string (caller owns the serialization format).
      max_points    -- the segment is capped at this many newest points.
      min_points    -- fewer points than this yields insufficient_data.

    The latest contiguous segment is collected by walking back from the
    newest in-window sample while adjacent intervals stay <= M, keeping at
    most ``max_points`` points. The segment is unusable (insufficient_data)
    when its newest point is more than M before as_of or it holds fewer than
    ``min_points`` points. Otherwise a Theil-Sen style line is fitted with
    as_of as the zero instant: the slope b is the median of every point
    pair's temperature difference divided by its time difference, and the
    intercept a is the median of (temperature - b * relative seconds) over
    all points. The crossing is predicted only when b > 0, a < H and
    (H - a) / b <= horizon; every other fitted case is no_crossing.

    Pure function of current data: late arrivals, corrections and config
    changes simply change the inputs, so the next query sees them.
    """
    cutoff = as_of.astimezone(timezone.utc)
    window_start = cutoff - timedelta(seconds=lookback)
    points = [
        point for point in _sorted_points(readings) if window_start <= point[0] <= cutoff
    ]

    # Latest contiguous segment: walk back from the newest in-window point
    # while adjacent sampling intervals stay within M, newest points first.
    m = timedelta(seconds=max_interval)
    newest_first: List[Tuple[datetime, float]] = []
    for point in reversed(points):
        if newest_first and newest_first[-1][0] - point[0] > m:
            break  # sampling gap beyond M: older points are not contiguous
        newest_first.append(point)
        if len(newest_first) == max_points:
            break
    segment = list(reversed(newest_first))

    if (
        len(segment) < min_points
        or cutoff - segment[-1][0] > m  # newest sample too old: no current trend
    ):
        return Forecast(
            status="insufficient_data",
            samples=len(segment),
            slope=None,
            intercept=None,
            predicted_at=None,
        )

    # Relative seconds from as_of (<= 0); instants are unique, so every pair
    # has a non-zero time difference.
    xs = [(t - cutoff).total_seconds() for t, _ in segment]
    vs = [v for _, v in segment]
    slope = median(
        (vs[j] - vs[i]) / (xs[j] - xs[i])
        for i in range(len(segment))
        for j in range(i + 1, len(segment))
    )
    intercept = median(v - slope * x for x, v in zip(xs, vs))

    status = "no_crossing"
    predicted_at: Optional[str] = None
    if slope > 0 and intercept < threshold_h:
        seconds_to_cross = (threshold_h - intercept) / slope
        if seconds_to_cross <= horizon:
            status = "predicted"
            predicted_at = utc_format(cutoff + timedelta(seconds=seconds_to_cross))
    return Forecast(
        status=status,
        samples=len(segment),
        slope=slope,
        intercept=intercept,
        predicted_at=predicted_at,
    )
