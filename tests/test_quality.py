"""Trip semantics and additive aggregation, including awkward real-data cases."""

import numpy as np
import pandas as pd
import pytest

from mobility.aggregate import OD_KEYS, ZONE_KEYS, aggregate_cpu, merge_aggregates
from mobility.data.quality import QualityFlag, hourly_dst_mask, normalize_yellow


def source(month="2024-01"):
    return {"sha256": "a" * 64, "source_id": "test-source", "nominal_month": month}


def raw_trips(count=1):
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2024-01-01 10:10"] * count),
            "tpep_dropoff_datetime": pd.to_datetime(["2024-01-01 10:20"] * count),
            "PULocationID": np.ones(count, dtype=float),
            "DOLocationID": np.full(count, 2.0),
            "trip_distance": np.full(count, 2.0),
        }
    )


def has_flag(value, flag):
    return bool(int(value) & int(flag))


def test_missing_pickup_preserves_independent_dropoff():
    raw = raw_trips()
    raw.loc[0, "tpep_pickup_datetime"] = pd.NaT
    trips, _ = normalize_yellow(raw, source(), {1, 2})
    row = trips.iloc[0]
    assert not row.eligible_pickup
    assert row.eligible_dropoff
    assert not row.eligible_od
    assert not row.eligible_duration
    assert has_flag(row.quality_flags, QualityFlag.MISSING_PICKUP)
    zone, od = aggregate_cpu(trips)
    assert zone.pickup_count.sum() == 0
    assert zone.dropoff_count.sum() == 1
    assert od.trip_count.sum() == 0


def test_missing_dropoff_timestamp_preserves_pickup_and_known_od_pair():
    raw = raw_trips()
    raw.loc[0, "tpep_dropoff_datetime"] = pd.NaT
    trips, _ = normalize_yellow(raw, source(), {1, 2})
    row = trips.iloc[0]
    assert row.eligible_pickup
    assert not row.eligible_dropoff
    assert row.eligible_od  # Destination zone is known; OD is pickup-anchored.
    assert not row.eligible_duration
    zone, od = aggregate_cpu(trips)
    assert zone.pickup_count.sum() == 1
    assert zone.dropoff_count.sum() == 0
    assert od.trip_count.sum() == 1
    assert od.duration_count.sum() == 0


def test_invalid_destination_zone_does_not_erase_pickup():
    raw = raw_trips()
    raw.loc[0, "DOLocationID"] = 999
    trips, _ = normalize_yellow(raw, source(), {1, 2})
    assert trips.eligible_pickup.iloc[0]
    assert not trips.eligible_dropoff.iloc[0]
    assert not trips.eligible_od.iloc[0]
    assert has_flag(trips.quality_flags.iloc[0], QualityFlag.UNMAPPED_DROPOFF)


def test_out_of_month_pickup_stays_canonical_but_excluded_from_pickup_study():
    raw = raw_trips()
    raw.loc[0, "tpep_pickup_datetime"] = pd.Timestamp("2023-12-31 23:50")
    raw.loc[0, "tpep_dropoff_datetime"] = pd.Timestamp("2024-01-01 00:10")
    trips, quality = normalize_yellow(raw, source(), {1, 2})
    assert len(trips) == 1
    assert not trips.eligible_pickup.iloc[0]
    assert trips.eligible_dropoff.iloc[0]
    assert not trips.eligible_od.iloc[0]
    assert not trips.eligible_ml.iloc[0]
    assert quality.set_index("rule").loc["outside_source_month", "affected_rows"] == 1


def test_valid_cross_month_trip_keeps_both_endpoints_at_their_own_hours():
    raw = raw_trips()
    raw.loc[0, "tpep_pickup_datetime"] = pd.Timestamp("2024-01-31 23:50")
    raw.loc[0, "tpep_dropoff_datetime"] = pd.Timestamp("2024-02-01 00:10")
    trips, _ = normalize_yellow(raw, source(), {1, 2})
    assert trips.eligible_pickup.iloc[0] and trips.eligible_dropoff.iloc[0]
    assert not has_flag(trips.quality_flags.iloc[0], QualityFlag.OUTSIDE_SOURCE_MONTH)
    zone, od = aggregate_cpu(trips)
    assert zone.loc[zone.pickup_count.eq(1), "hour"].iloc[0] == pd.Timestamp("2024-01-31 23:00")
    assert zone.loc[zone.dropoff_count.eq(1), "hour"].iloc[0] == pd.Timestamp("2024-02-01 00:00")
    assert od.hour.iloc[0] == pd.Timestamp("2024-01-31 23:00")
    assert trips.duration_seconds.iloc[0] == 1200


@pytest.mark.parametrize(
    "pickup,dropoff",
    [
        ("2024-03-10 01:55", "2024-03-10 03:05"),
        ("2024-03-10 02:10", "2024-03-10 03:10"),
        ("2024-11-03 00:55", "2024-11-03 02:05"),
        ("2024-11-03 01:15", "2024-11-03 01:45"),
    ],
)
def test_dst_uncertainty_excludes_duration_without_guessing_offsets(pickup, dropoff):
    raw = raw_trips()
    raw.loc[0, "tpep_pickup_datetime"] = pd.Timestamp(pickup)
    raw.loc[0, "tpep_dropoff_datetime"] = pd.Timestamp(dropoff)
    trips, _ = normalize_yellow(raw, source(pickup[:7]), {1, 2})
    assert has_flag(trips.quality_flags.iloc[0], QualityFlag.DST_UNSAFE_DURATION)
    assert pd.isna(trips.duration_seconds.iloc[0])
    assert not trips.eligible_ml.iloc[0]
    assert trips.eligible_pickup.iloc[0] and trips.eligible_dropoff.iloc[0]


def test_dst_calendar_marks_only_discontinuous_hours():
    hours = pd.DatetimeIndex(
        [
            "2024-03-10 01:00",
            "2024-03-10 02:00",
            "2024-03-10 03:00",
            "2024-11-03 00:00",
            "2024-11-03 01:00",
            "2024-11-03 02:00",
        ]
    )
    assert hourly_dst_mask(hours).tolist() == [False, True, False, False, True, False]


@pytest.mark.parametrize("distance", [np.nan, np.inf, -np.inf, -2.0])
def test_bad_distance_excludes_distance_statistics_only(distance):
    raw = raw_trips()
    raw.loc[0, "trip_distance"] = distance
    trips, _ = normalize_yellow(raw, source(), {1, 2})
    assert trips.eligible_pickup.iloc[0] and trips.eligible_od.iloc[0]
    assert trips.eligible_duration.iloc[0]
    assert not trips.eligible_ml.iloc[0]
    assert has_flag(trips.quality_flags.iloc[0], QualityFlag.INVALID_DISTANCE)
    zone, od = aggregate_cpu(trips)
    assert zone.pickup_count.sum() == od.trip_count.sum() == 1
    assert zone.distance_count.sum() == od.distance_count.sum() == 0
    assert zone.duration_count.sum() == od.duration_count.sum() == 1


def test_zero_distance_is_a_flag_not_an_erased_trip():
    raw = raw_trips()
    raw.loc[0, "trip_distance"] = 0
    trips, _ = normalize_yellow(raw, source(), {1, 2})
    assert trips.eligible_ml.iloc[0]
    assert trips.mean_speed_mph.iloc[0] == 0
    assert has_flag(trips.quality_flags.iloc[0], QualityFlag.ZERO_DISTANCE)
    assert not has_flag(trips.quality_flags.iloc[0], QualityFlag.INVALID_DISTANCE)


@pytest.mark.parametrize("destination", [1.2, np.nan, np.inf, 2**40])
def test_invalid_integer_zone_values_are_not_truncated(destination):
    raw = raw_trips()
    raw.loc[0, "DOLocationID"] = destination
    trips, _ = normalize_yellow(raw, source(), {1, 2})
    assert pd.isna(trips.dropoff_zone_id.iloc[0])
    assert not trips.eligible_od.iloc[0]


def test_zero_and_negative_durations_preserve_activity_counts():
    raw = raw_trips(2)
    raw["tpep_dropoff_datetime"] = pd.to_datetime(["2024-01-01 10:10", "2024-01-01 10:00"])
    trips, _ = normalize_yellow(raw, source(), {1, 2})
    assert not trips.eligible_duration.any()
    assert trips.eligible_pickup.all() and trips.eligible_dropoff.all() and trips.eligible_od.all()
    zone, od = aggregate_cpu(trips)
    assert zone.pickup_count.sum() == zone.dropoff_count.sum() == od.trip_count.sum() == 2
    assert zone.duration_count.sum() == 0


def test_optional_fares_and_passenger_counts_remain_missing():
    trips, _ = normalize_yellow(raw_trips(), source(), {1, 2})
    assert trips[["fare_amount", "total_amount", "passenger_count"]].isna().all().all()
    raw = raw_trips()
    raw["fare_amount"] = -5
    trips, _ = normalize_yellow(raw, source(), {1, 2})
    assert has_flag(trips.quality_flags.iloc[0], QualityFlag.NEGATIVE_FARE)
    assert trips.eligible_pickup.iloc[0]


def test_record_identity_and_duplicate_looking_trips_are_preserved():
    raw = raw_trips(2)
    whole, _ = normalize_yellow(raw, source(), {1, 2}, row_offset=8)
    first, _ = normalize_yellow(raw.iloc[:1], source(), {1, 2}, row_offset=8)
    second, _ = normalize_yellow(raw.iloc[1:], source(), {1, 2}, row_offset=9)
    assert list(whole.record_id) == list(first.record_id) + list(second.record_id)
    assert whole.record_id.nunique() == 2
    zone, _ = aggregate_cpu(whole)
    assert zone.pickup_count.sum() == 2


def test_chunked_aggregation_conserves_counts_and_weighted_means():
    raw = raw_trips(10)
    raw["tpep_dropoff_datetime"] = raw.tpep_pickup_datetime + pd.to_timedelta(
        [10] * 2 + [20] * 8, unit="min"
    )
    trips, _ = normalize_yellow(raw, source(), {1, 2})
    complete_zone, complete_od = aggregate_cpu(trips)
    parts = [aggregate_cpu(trips.iloc[:2]), aggregate_cpu(trips.iloc[2:])]
    for merged, complete, keys in [
        (merge_aggregates([item[0] for item in parts], ZONE_KEYS), complete_zone, ZONE_KEYS),
        (merge_aggregates([item[1] for item in parts], OD_KEYS), complete_od, OD_KEYS),
    ]:
        pd.testing.assert_frame_equal(
            merged.sort_values(keys).reset_index(drop=True),
            complete.sort_values(keys).reset_index(drop=True),
            check_dtype=False,
        )
    pickup = complete_zone.loc[complete_zone.pickup_count.gt(0)].iloc[0]
    assert pickup.duration_sum_seconds / pickup.duration_count / 60 == 18


def test_timezone_aware_source_requires_explicit_adapter():
    raw = raw_trips()
    raw["tpep_pickup_datetime"] = raw.tpep_pickup_datetime.dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_yellow(raw, source(), {1, 2})
