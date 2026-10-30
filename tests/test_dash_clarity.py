"""Regression checks for charts that previously hid ordinary trip distances."""

import json

import numpy as np
import pandas as pd
from plotly.utils import PlotlyJSONEncoder

from mobility.dashboard import charts
from mobility.dashboard.app import performance, table


def test_typical_scatter_preserves_all_samples_and_exposes_full_extent():
    samples = pd.DataFrame(
        {
            "trip_distance_miles": [0.0, 2.0, 100125.14, 5.0],
            "duration_seconds": [60.0, 600.0, 780.0, 86193.0],
            "cluster": [0, 0, 1, 1],
        }
    )
    figure = charts.scatter_sample(samples, 5000)
    assert figure.layout.xaxis.range == (0, 50)
    assert figure.layout.yaxis.range == (0, 120)
    plotted = [float(value) for trace in figure.data for value in trace.x]
    assert sorted(plotted) == sorted(samples.trip_distance_miles)
    assert "2 of 4" in figure.layout.annotations[0].text
    full = figure.layout.updatemenus[0].buttons[1]
    assert full.label == "Full extent"
    assert full.args[0] == {"xaxis.autorange": True, "yaxis.autorange": True}


def test_sample_histogram_keeps_zero_boundaries_and_extreme_overflow():
    samples = pd.DataFrame({"trip_distance_miles": [0, 1, 2, 50, 50.001, 100125.14, -1, np.nan]})
    figure = charts.sample_distance_histogram(samples)
    counts = dict(zip(figure.data[0].x, figure.data[0].y))
    assert counts["0–<1"] == 1
    assert counts["1–<2"] == 1
    assert counts["2–<3"] == 1
    assert counts["20–50"] == 1
    assert counts[">50"] == 2
    assert sum(counts.values()) == 6
    assert "2 invalid values excluded" in figure.layout.annotations[0].text


def test_od_map_uses_highest_counts_even_when_input_is_unsorted():
    pairs = pd.DataFrame(
        {
            "pickup_zone_id": [1, 2, 1],
            "dropoff_zone_id": [2, 2, 1],
            "trip_count": [1, 100, 50],
        }
    )
    zones = pd.DataFrame(
        {
            "zone_id": [1, 2],
            "zone": ["North", "South"],
            "longitude": [-74, -73.9],
            "latitude": [40.7, 40.8],
        }
    )
    figure = charts.flows(pairs, zones, 1)
    assert len(figure.data) == 2  # One selected connection and its zone anchor.
    assert "South → South" in figure.data[0].text[0]
    assert "100 recorded trips" in figure.data[0].text[0]
    assert figure.data[0].mode == "markers"


def test_performance_reports_cpu_win_and_preserves_small_times():
    class Repository:
        manifest = {}

        def read(self, name):
            return pd.DataFrame(
                {
                    "experiment_id": ["same", "same"],
                    "stage": ["features", "features"],
                    "timing_scope": ["host_to_host_features"] * 2,
                    "input_rows": [100000] * 2,
                    "backend": ["cpu", "gpu_library"],
                    "elapsed_s": [0.004186, 0.004763],
                    "validation": ["passed"] * 2,
                }
            )

    serialized = json.dumps(performance(Repository(), "all", "all"), cls=PlotlyJSONEncoder)
    assert "CPU 1.138" in serialized
    assert "0.004186" in serialized
    assert "0.004763" in serialized
    assert "0.878858" in serialized


def test_review_display_does_not_change_stored_status():
    incidents = pd.DataFrame({"label": ["unresolved"]})
    serialized = json.dumps(table(incidents), cls=PlotlyJSONEncoder)
    assert "Unreviewed" in serialized
    assert incidents.label.iloc[0] == "unresolved"
