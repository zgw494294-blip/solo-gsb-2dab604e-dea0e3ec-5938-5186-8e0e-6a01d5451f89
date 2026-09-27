"""Pydantic request/response models."""

import math
from datetime import datetime
from typing import Annotated, List, Literal, Optional

from pydantic import (
    AwareDatetime,
    BaseModel,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


class ThresholdsIn(BaseModel):
    h: float = Field(description="Trigger threshold: alarm opens after 3 consecutive readings >= h")
    l: float = Field(description="Recovery threshold: alarm closes after 2 consecutive readings <= l")

    @field_validator("h", "l")
    @classmethod
    def must_be_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("threshold must be a finite number")
        return v

    @model_validator(mode="after")
    def h_must_exceed_l(self):
        if not self.h > self.l:
            raise ValueError("h must be strictly greater than l")
        return self


class ReadingIn(BaseModel):
    sampled_at: AwareDatetime = Field(description="Sample time; must include a timezone offset")
    temperature: float

    @field_validator("temperature")
    @classmethod
    def must_be_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("temperature must be a finite number")
        return v


class ReadingsBatchIn(BaseModel):
    readings: List[ReadingIn] = Field(min_length=1)


class DeviceOut(BaseModel):
    device_id: str
    threshold_h: float
    threshold_l: float
    created_at: datetime
    updated_at: datetime


class ReadingOut(BaseModel):
    device_id: str
    sampled_at: datetime
    temperature: float
    received_at: datetime


class AlarmOut(BaseModel):
    id: int
    device_id: str
    opened_at: datetime
    closed_at: Optional[datetime]
    status: str  # "open" | "closed"
    acknowledged_by: Optional[str] = Field(
        default=None, description="Operator who acknowledged the alarm; null while unacknowledged"
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None, description="UTC time of the (first) acknowledgement; null while unacknowledged"
    )


class AlarmAcknowledgementIn(BaseModel):
    opened_at: AwareDatetime = Field(
        description="Open instant of the alarm to acknowledge; must include a timezone offset"
    )
    operator: str = Field(min_length=1, description="Who acknowledges the alarm; must be non-empty")
    expected_revision: int = Field(
        ge=0,
        description="Alarm revision the caller saw in X-Alarm-Revision; a mismatch yields 409",
    )

    @field_validator("operator")
    @classmethod
    def operator_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("operator must be non-empty")
        return v


class AlarmAcknowledgementOut(BaseModel):
    device_id: str
    opened_at: datetime = Field(description="UTC open instant of the acknowledged alarm")
    operator: str = Field(description="Operator of the first (and kept) acknowledgement")
    acknowledged_at: datetime = Field(description="UTC time the first acknowledgement was recorded")


class WriteResult(BaseModel):
    device_id: str
    written: int


# ------------------------------------------------------------ alarm revisions

class ThresholdChangeOut(BaseModel):
    old_h: Optional[float] = Field(description="Trigger threshold before the change; null for a newly created device")
    old_l: Optional[float] = Field(description="Recovery threshold before the change; null for a newly created device")
    new_h: float = Field(description="Trigger threshold after the change")
    new_l: float = Field(description="Recovery threshold after the change")


class ReadingChangeOut(BaseModel):
    sampled_at: datetime = Field(description="UTC sample time of the changed point")
    old_temperature: Optional[float] = Field(
        description="Stored temperature before the batch; null for a first-time sample point"
    )
    new_temperature: float = Field(description="Temperature after the batch")


class AlarmIntervalOut(BaseModel):
    opened_at: datetime = Field(description="UTC open time of the alarm interval")
    closed_at: Optional[datetime] = Field(
        default=None, description="UTC close time; null while the alarm is open"
    )


class AlarmRevisionOut(BaseModel):
    device_id: str
    revision: int = Field(description="Per-device revision number, consecutive from 1")
    op: Literal["thresholds", "readings"] = Field(description="Which write caused this revision")
    created_at: datetime = Field(description="UTC time the change was applied")
    threshold_change: Optional[ThresholdChangeOut] = Field(
        default=None, description="Threshold before/after values; present iff op = 'thresholds'"
    )
    reading_changes: List[ReadingChangeOut] = Field(
        description="Sample points whose temperature actually changed (op = 'readings'), by sample time"
    )
    added_alarms: List[AlarmIntervalOut] = Field(
        description="Alarm intervals present after the recompute but not before"
    )
    withdrawn_alarms: List[AlarmIntervalOut] = Field(
        description="Alarm intervals present before the recompute but not after"
    )


# ---------------------------------------------------- alarm revision snapshot

class SnapshotReadingOut(BaseModel):
    sampled_at: datetime = Field(description="UTC sample time of the reading")
    temperature: float = Field(description="Stored temperature as of the snapshot revision")


class AlarmRevisionSnapshotOut(BaseModel):
    device_id: str
    revision: int = Field(description="Target revision number this snapshot reconstructs")
    threshold_h: float = Field(
        description="Trigger threshold in effect right after this revision completed"
    )
    threshold_l: float = Field(
        description="Recovery threshold in effect right after this revision completed"
    )
    readings: List[SnapshotReadingOut] = Field(
        description="Readings as of this revision, ordered by UTC sample instant"
    )
    alarms: List[AlarmIntervalOut] = Field(
        description="Alarm intervals recomputed from the snapshot readings and thresholds;"
        " current acknowledgements are not part of the snapshot"
    )


# -------------------------------------------------------------- alarm preview

class AlarmPreviewIn(BaseModel):
    thresholds: Optional[ThresholdsIn] = Field(
        default=None,
        description="Candidate thresholds (finite, h > l); omitted: the device's current thresholds are used",
    )
    readings: List[ReadingIn] = Field(
        default_factory=list,
        description="Candidate readings (timezone-aware) overlaid on the stored readings by UTC sample"
        " instant; within the batch the last entry for an instant wins",
    )

    @model_validator(mode="after")
    def at_least_one_input(self):
        if self.thresholds is None and not self.readings:
            raise ValueError("provide candidate thresholds and/or candidate readings")
        return self


class AlarmPreviewOut(BaseModel):
    device_id: str
    base_revision: int = Field(
        description="The device's current alarm revision the preview was computed against"
    )
    alarms: List[AlarmIntervalOut] = Field(
        description="Alarm intervals the candidate thresholds/readings would produce, by open time"
    )
    added_alarms: List[AlarmIntervalOut] = Field(
        description="Intervals present in the preview but not in the device's current alarms"
    )
    withdrawn_alarms: List[AlarmIntervalOut] = Field(
        description="Intervals present in the device's current alarms but not in the preview"
    )


# ------------------------------------------------------- alarm consistency

class OrphanAcknowledgementOut(BaseModel):
    opened_at: datetime = Field(
        description="UTC open instant of an acknowledgement with no recomputed alarm open instant"
    )
    operator: str = Field(description="Operator of the orphan acknowledgement")
    acknowledged_at: datetime = Field(
        description="UTC time the orphan acknowledgement was recorded"
    )


class AlarmConsistencyOut(BaseModel):
    device_id: str
    consistent: bool = Field(
        description="True exactly when stored alarms match the recomputed alarms item-wise and no"
        " acknowledgement lacks a recomputed alarm open instant"
    )
    stored_alarm_count: int = Field(description="Number of stored alarm rows (duplicates counted)")
    recomputed_alarm_count: int = Field(
        description="Number of alarm intervals recomputed from the current thresholds and readings"
    )
    missing_alarms: List[AlarmIntervalOut] = Field(
        description="Recomputed alarm intervals absent from the stored alarm rows (item-wise)"
    )
    extra_alarms: List[AlarmIntervalOut] = Field(
        description="Stored alarm intervals beyond the recomputed multiplicity (item-wise)"
    )
    duplicate_alarms: List[AlarmIntervalOut] = Field(
        description="Surplus copies of an alarm interval stored more than once (every copy past the"
        " first), item-wise and sorted by open time"
    )
    orphan_acknowledgements: List[OrphanAcknowledgementOut] = Field(
        description="Stored acknowledgements whose opened_at matches no recomputed alarm open instant"
    )


class AlarmReconcileOut(BaseModel):
    device_id: str
    fixed: bool = Field(
        description="Whether the reconcile changed any alarm rows or acknowledgements"
    )
    before: AlarmConsistencyOut = Field(
        description="Item-wise consistency comparison before the repair"
    )
    after: AlarmConsistencyOut = Field(
        description="Item-wise consistency comparison after the repair (identical when nothing was fixed)"
    )


class SamplingConfigIn(BaseModel):
    max_interval: int = Field(
        ge=1, description="M: maximum allowed interval between adjacent samples, in whole seconds (>= 1)"
    )
    grace: int = Field(
        ge=0, description="G: confirmation grace after M before an ongoing gap is reported, in whole seconds (>= 0)"
    )


class SamplingConfigOut(BaseModel):
    device_id: str
    max_interval: int
    grace: int
    created_at: datetime
    updated_at: datetime


class GapOut(BaseModel):
    started_at: datetime = Field(description="UTC: previous sample time + M")
    ended_at: Optional[datetime] = Field(
        default=None, description="UTC: next sample time; null while the interruption is ongoing"
    )
    status: Literal["recovered", "ongoing"]


# ----------------------------------------------------- alarm response deadline

class ResponsePolicyIn(BaseModel):
    deadline: int = Field(
        ge=1, description="D: alarm response deadline in whole positive seconds (>= 1)"
    )


class ResponsePolicyOut(BaseModel):
    device_id: str
    deadline: int = Field(description="Response deadline D in whole seconds")
    created_at: datetime
    updated_at: datetime


class ResponseStatusOut(BaseModel):
    device_id: str
    opened_at: datetime = Field(description="UTC open instant of the still-open, unacknowledged alarm")
    as_of: datetime = Field(description="Query instant, normalized to UTC")
    deadline: int = Field(description="Configured response deadline D in whole seconds")
    observable_seconds: float = Field(
        description="Observable seconds accumulated within [opened_at, as_of] after deducting"
        " sampling gaps longer than M"
    )
    deadline_reached: bool = Field(
        description="Whether the cumulative observable time has reached D by as_of"
    )
    deadline_reached_at: Optional[datetime] = Field(
        default=None,
        description="First UTC instant at which the observable time accumulated to D;"
        " null while deadline_reached is false",
    )


class HeatExposureOut(BaseModel):
    device_id: str
    start: datetime = Field(description="Query window start, normalized to UTC")
    end: datetime = Field(description="Query window end, normalized to UTC")
    baseline: float = Field(description="Temperature baseline T")
    covered_seconds: float = Field(
        description="Seconds of the window covered by linear interpolation between"
        " adjacent readings whose interval is at most M"
    )
    above_seconds: float = Field(
        description="Covered seconds during which the interpolated temperature is strictly above T"
    )
    excess_degree_seconds: float = Field(
        description="Integral of max(interpolated temperature - T, 0) over the window, in degree-seconds"
    )


# -------------------------------------------------------------- device groups

MemberDeviceId = Annotated[str, StringConstraints(min_length=1, max_length=200)]


class GroupIn(BaseModel):
    device_ids: List[MemberDeviceId] = Field(
        min_length=1,
        description="Member device IDs; non-empty, no duplicates, and every device must already exist",
    )
    k: int = Field(
        ge=1,
        description="Minimum number of members simultaneously in alarm that opens a group alarm",
    )

    @field_validator("device_ids")
    @classmethod
    def no_duplicate_members(cls, v: List[str]) -> List[str]:
        if len(set(v)) != len(v):
            raise ValueError("device_ids must not contain duplicates")
        return v

    @model_validator(mode="after")
    def k_within_member_count(self):
        if self.k > len(self.device_ids):
            raise ValueError("k must not exceed the number of members")
        return self


class GroupOut(BaseModel):
    group_id: str
    device_ids: List[str]
    k: int
    created_at: datetime
    updated_at: datetime


class GroupAlarmOut(BaseModel):
    opened_at: datetime
    closed_at: Optional[datetime] = Field(
        default=None, description="UTC close time; null while the group alarm is ongoing"
    )
    status: Literal["open", "closed"]


class PeerDeviationOut(BaseModel):
    started_at: datetime = Field(description="UTC instant of the segment's first deviation point")
    ended_at: datetime = Field(description="UTC instant of the segment's last deviation point")
    samples: int = Field(description="Number of consecutive deviation points in the segment")
    max_abs_deviation: float = Field(
        description="Largest |target temperature - peer median| across the segment's points"
    )


class LagEstimateOut(BaseModel):
    group_id: str
    target_id: str
    reference_id: str
    start: datetime = Field(description="Query window start, normalized to UTC")
    end: datetime = Field(description="Query window end, normalized to UTC")
    max_lag: int = Field(description="Whole-second shifts -max_lag..max_lag that were tried")
    min_pairs: int = Field(description="Minimum comparable pairs a shift needed to qualify")
    status: Literal["ok", "insufficient_data"] = Field(
        description="ok: a shift with at least min_pairs comparisons was found;"
        " insufficient_data: no shift reached min_pairs"
    )
    lag: Optional[int] = Field(
        default=None,
        description="Chosen whole-second shift: reference value at t + lag matches target at t;"
        " null when status is insufficient_data",
    )
    pairs: int = Field(
        description="Number of comparable target/reference pairs backing the chosen lag"
        " (0 when insufficient_data)"
    )
    median_abs_diff: Optional[float] = Field(
        default=None,
        description="Median absolute temperature difference across the chosen lag's pairs;"
        " null when status is insufficient_data",
    )


# ------------------------------------------------------------------ forecast

class ForecastOut(BaseModel):
    device_id: str
    as_of: datetime = Field(description="Query instant and zero point of the fit, normalized to UTC")
    lookback: int = Field(description="Lookback window in seconds: only samples in [as_of - lookback, as_of] are used")
    horizon: int = Field(description="Forecast horizon in seconds")
    threshold_h: float = Field(description="The device's current trigger threshold H")
    samples: int = Field(
        description="Sample points of the latest contiguous segment (adjacent intervals <= M, at most 30)"
    )
    slope: Optional[float] = Field(
        default=None,
        description="Fitted slope b in temperature/second (median of all point-pair slopes);"
        " null when insufficient_data",
    )
    intercept: Optional[float] = Field(
        default=None,
        description="Fitted intercept a: trend temperature at as_of"
        " (median of temperature - b * relative seconds); null when insufficient_data",
    )
    status: Literal["predicted", "no_crossing", "insufficient_data"] = Field(
        description="predicted: the fitted line reaches H within horizon;"
        " no_crossing: fitted but no upward crossing within horizon;"
        " insufficient_data: too few/stale samples to fit"
    )
    predicted_at: Optional[datetime] = Field(
        default=None,
        description="UTC instant the fitted line reaches H; null unless status is predicted",
    )
