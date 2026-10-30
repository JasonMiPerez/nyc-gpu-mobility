"""Six read-only research views backed by one validated snapshot."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px

from mobility.serving.repository import Filters, QueryError, SnapshotRepository
from . import charts

LOGGER = logging.getLogger(__name__)
PAGES = {
    "overview": ("Overview", "A measured view of recorded movement."),
    "patterns": ("Mobility patterns", "Explore when trips happen and how they differ."),
    "zones": ("Zones & connections", "Follow recorded connections between taxi zones."),
    "anomalies": ("Unusual activity", "Compare recorded pickups with their seasonal baseline."),
    "performance": (
        "CPU & GPU performance",
        "Compare measured work, with its conditions attached.",
    ),
    "methods": ("Data & methods", "Trace every finding to its sources and assumptions."),
}


def graph(figure, identifier=None):
    properties = dict(
        figure=figure,
        config={"displaylogo": False, "responsive": True, "scrollZoom": False},
        className="chart",
        style={"height": f"{figure.layout.height or 350}px", "width": "100%"},
    )
    if identifier:
        properties["id"] = identifier
    return dcc.Graph(**properties)


def panel(title, children, description=None, class_name=""):
    return html.Section(
        [
            html.Div(
                [html.H2(title), html.P(description, className="muted") if description else None],
                className="panel-heading",
            ),
            children,
        ],
        className=f"panel {class_name}",
    )


def notice(message, kind="info"):
    return html.Div(message, className=f"notice {kind}", role="status")


def table(frame: pd.DataFrame, columns=None, limit=100):
    if frame.empty:
        return notice("No rows are available for this view.")
    names = [name for name in (columns or frame.columns.tolist()) if name in frame]
    rows = frame[names].head(limit).copy()

    def display(value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, default=str)
        if value is None or pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:,.3f}".rstrip("0").rstrip(".")
        return str(value)

    return html.Div(
        [
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [html.Th(name.replace("_", " "), scope="col") for name in names]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr([html.Td(display(value)) for value in row])
                                for row in rows.itertuples(index=False, name=None)
                            ]
                        ),
                    ]
                ),
                className="table-scroll",
                tabIndex=0,
                role="region",
                **{"aria-label": "Scrollable data table"},
            ),
            html.P(
                f"Showing {min(len(frame), limit):,} of {len(frame):,} rows.",
                className="table-caption",
            ),
        ]
    )


def _number(value, precision=0):
    return "—" if pd.isna(value) else f"{value:,.{precision}f}"


def _coverage(frame: pd.DataFrame):
    if frame.empty:
        return notice(
            "No zone-hours match these filters. Try a wider date range or reset the filters."
        )
    counts = (
        frame.pickup_coverage.value_counts() if "pickup_coverage" in frame else pd.Series(dtype=int)
    )
    observed = int(counts.get("observed", 0))
    others = [
        f"{int(count):,} {str(status).replace('_', ' ')}"
        for status, count in counts.items()
        if status != "observed"
    ]
    message = f"Pickup coverage: {observed:,} of {len(frame):,} selected zone-hours observed."
    if others:
        message += " Other statuses: " + "; ".join(others) + "."
    message += " Missing coverage is not zero activity."
    if "dropoff_coverage" in frame and frame.dropoff_coverage.eq("partial_source_context").any():
        message += " Drop-off totals have partial source context: earlier pickup files can contain additional arrivals."
    else:
        message += " Drop-off coverage is tracked separately."
    return notice(message, "warning" if others else "info")


def overview(repo: SnapshotRepository, filters: Filters, aggregation, tiles):
    data = repo.read("zone_hour", filters)
    if data.empty:
        return _coverage(data)
    pickups = data.pickup_count.sum(min_count=1)
    dropoffs = data.dropoff_count.sum(min_count=1)
    duration_count = data.duration_count.sum() if "duration_count" in data else 0
    mean_duration = (
        data.duration_sum_seconds.sum() / duration_count / 60 if duration_count else float("nan")
    )
    active_zones = data.loc[data.pickup_count > 0, "zone_id"].nunique()
    cards = [
        ("Mapped pickups", _number(pickups), "Eligible recorded origins"),
        ("Mapped drop-offs", _number(dropoffs), "Eligible recorded destinations"),
        (
            "Mean valid duration",
            _number(mean_duration, 1) + " min",
            f"{duration_count:,.0f} valid pickup-anchored trips",
        ),
        ("Zones with pickups", str(active_zones), "Within the applied selection"),
    ]
    totals = charts.zone_totals(data, repo.zones()).sort_values("pickup_count", ascending=False)
    return [
        html.Div(
            [
                html.Div(
                    [
                        html.P(label, className="kpi-label"),
                        html.Strong(value, className="kpi-value"),
                        html.P(detail, className="kpi-detail"),
                    ],
                    className="kpi",
                )
                for label, value, detail in cards
            ],
            className="kpi-grid",
        ),
        _coverage(data),
        panel(
            "Activity over time",
            graph(charts.activity(data, aggregation)),
            "Pickups and drop-offs use their respective endpoint times. Counts include only mapped, eligible records.",
        ),
        html.Div(
            [
                panel(
                    "Where trips begin",
                    graph(
                        charts.zone_map(
                            data,
                            repo.zones(),
                            repo.geojson(),
                            revision=repo.snapshot_id,
                            tiles=tiles,
                        ),
                        "zone-map",
                    ),
                    "Click a zone to select it, then Apply filters. Geometry: NYC TLC. Optional background tiles: CARTO / OpenStreetMap.",
                ),
                panel(
                    "Most active zones",
                    table(totals, ["zone", "borough", "pickup_count", "dropoff_count"], 15),
                    "A map-free view of the same applied selection.",
                ),
            ],
            className="two-columns",
        ),
    ]


def patterns(repo: SnapshotRepository, filters: Filters, selected_cluster):
    hours = repo.read("zone_hour", filters)
    observed = hours[hours.pickup_coverage.eq("observed")] if "pickup_coverage" in hours else hours
    summary = repo.read("cluster_summary")
    samples = (
        repo.read("trip_plot_sample", filters) if repo.has("trip_plot_sample") else pd.DataFrame()
    )
    if selected_cluster not in (None, "all") and "cluster" in samples:
        samples = samples[samples.cluster.astype(str).eq(str(selected_cluster))]
    samples = samples.head(repo.max_points)
    model_status = repo.manifest.get("models", {}).get("clustering", {}).get("status", "")
    model_notice = (
        notice(
            "These clusters are exploratory: this snapshot has no separate calibration period. The selected cluster count has not been validated on a calibration period.",
            "warning",
        )
        if model_status == "exploratory_no_calibration"
        else None
    )
    if summary.empty:
        summary_content = notice(
            "Clustering results have not been published for this snapshot. Activity patterns remain available."
        )
    else:
        summary_content = table(
            summary,
            [
                "cluster",
                "trip_count",
                "share",
                "mean_distance_miles",
                "mean_duration_minutes",
                "scoring_start",
                "scoring_end",
            ],
            20,
        )
    exact_distributions = None
    if repo.has("trip_histogram"):
        histogram_filters = Filters(filters.start, filters.end, filters.dataset, ())
        histograms = repo.read("trip_histogram", histogram_filters)
        histogram_charts = []
        for feature, label in [
            ("distance_miles", "Distance (miles)"),
            ("duration_minutes", "Duration (minutes)"),
        ]:
            subset = (
                histograms[histograms.feature.eq(feature)]
                if "feature" in histograms
                else pd.DataFrame()
            )
            if subset.empty:
                continue
            bins = (
                subset.groupby(["bin_index", "bin_label"], as_index=False, dropna=False)["count"]
                .sum()
                .sort_values("bin_index")
            )
            figure = charts.style(
                px.bar(
                    bins,
                    x="bin_label",
                    y="count",
                    labels={"bin_label": label, "count": "Eligible recorded trips"},
                    color_discrete_sequence=[charts.BLUE],
                )
            )
            histogram_charts.append(graph(figure))
        if histogram_charts:
            exact_distributions = panel(
                "Exact trip distributions",
                html.Div(histogram_charts, className="two-columns"),
                "Exact counts for the applied source and date range, across all zones and clusters. Zone and cluster filters do not apply to these saved histogram bins.",
            )
    return [
        _coverage(hours),
        model_notice,
        panel(
            "A typical week",
            graph(charts.heatmap(observed)),
            "Mean pickups per observed hour across the selected zones. Missing and DST-ambiguous hours are excluded.",
        ),
        panel(
            "Trip groups",
            summary_content,
            "Fixed model/scoring-period summary: date, zone, and selected-cluster controls do not change this table. Group numbers are labels, not ranked quality or known passenger types.",
        ),
        exact_distributions,
        panel(
            "Distance and duration",
            graph(charts.scatter_sample(samples, repo.max_points)),
            f"Sampled trips only, at most {repo.max_points:,} points. The applied date, origin-zone, and cluster selections affect this plot; sample counts are not population totals.",
        ),
        panel(
            "Sample distribution",
            graph(
                charts.style(
                    px.histogram(
                        samples,
                        x="trip_distance_miles",
                        nbins=40,
                        labels={"trip_distance_miles": "Trip distance (miles)"},
                        color_discrete_sequence=[charts.BLUE],
                    )
                )
            )
            if not samples.empty
            else notice("No sampled trips match the current filters."),
            "Distribution of the displayed sample, not an exact full-dataset histogram.",
        ),
    ]


def zone_explorer(repo: SnapshotRepository, filters: Filters, destinations, metric, tiles):
    hours = repo.read("zone_hour", filters)
    connections = repo.od_totals(filters, destinations or ())
    zones = repo.zones()
    names = zones.set_index("zone_id").zone.to_dict() if not zones.empty else {}
    connections["origin"] = connections.pickup_zone_id.map(names).fillna(
        connections.pickup_zone_id.astype(str)
    )
    connections["destination"] = connections.dropoff_zone_id.map(names).fillna(
        connections.dropoff_zone_id.astype(str)
    )
    return [
        notice(
            "The main zone selection filters endpoint zones on the activity map and origins in the OD views. The destination control applies only to OD connections. OD counts use pickup time."
        ),
        _coverage(hours),
        panel(
            "Endpoint activity",
            graph(
                charts.zone_map(
                    hours,
                    zones,
                    repo.geojson(),
                    metric=metric,
                    revision=repo.snapshot_id,
                    tiles=tiles,
                )
            ),
            "The pickup/drop-off selector affects this map. Destination selection does not apply to endpoint totals.",
        ),
        panel(
            "Recorded zone connections",
            graph(charts.flows(connections, zones, repo.max_flows, tiles)),
            f"Top {min(len(connections), repo.max_flows):,} of {len(connections):,} zone pairs. Lines connect representative zone anchors; they are not actual routes. Line width indicates trip volume. Self-zone trips remain in the table.",
        ),
        panel(
            "Origin–destination table",
            table(
                connections,
                ["origin", "destination", "trip_count", "mean_duration_minutes"],
                repo.max_flows,
            ),
            "Mean duration uses only eligible duration records; trip counts also retain records without a usable duration.",
        ),
    ]


def anomalies(repo: SnapshotRepository, filters: Filters):
    if not repo.has("anomaly_scores"):
        return notice(
            "Anomaly results have not been published. A seasonal baseline needs multiple months of observed history."
        )
    data = repo.read("anomaly_scores", filters)
    if data.empty:
        return notice(
            "No calibration or test scores are available in this date range. Training-period hours are not held-out evaluation results."
        )
    statuses = data.status.value_counts().rename_axis("status").reset_index(name="zone_hours")
    alerts = data[data.is_alert.fillna(False)].copy() if "is_alert" in data else pd.DataFrame()
    if not alerts.empty:
        alerts["absolute_score"] = alerts.score.abs()
        alerts = alerts.sort_values(["absolute_score", "hour"], ascending=[False, True]).head(20)
    choices = [
        {
            "label": f"Zone {int(row.zone_id)} · {pd.Timestamp(row.hour):%b %d, %H:%M} · score {row.score:.2f} ({getattr(row, 'split', 'scored')})",
            "value": f"{int(row.zone_id)}|{pd.Timestamp(row.hour).isoformat()}",
        }
        for row in alerts.itertuples()
    ]
    initial = choices[0]["value"] if choices else None
    context = (
        charts.anomaly_context(data, int(initial.split("|", 1)[0]), initial.split("|", 1)[1])
        if initial
        else charts.empty(
            "No threshold alerts in this selection. This does not establish that all hours were normal."
        )
    )
    return [
        notice(
            "Alerts are statistical deviations in recorded pickup activity. They do not establish an incident, fraud, congestion, or a cause. Synthetic test scores are not real-world detection accuracy."
        ),
        html.Div(
            [
                panel(
                    "Scoring coverage",
                    table(statuses, limit=20),
                    "Insufficient history and missing coverage are explicit statuses, not zero scores.",
                ),
                panel(
                    "How to read an alert",
                    html.Div(
                        [
                            html.P(
                                "Each zone-hour is compared with the median of matching hours of the week in the training period."
                            ),
                            html.P(
                                "A scaled median absolute deviation provides a robust reference. Both the score threshold and an absolute trip-count threshold must be exceeded."
                            ),
                            html.P(
                                "Calibration and test rows are labeled separately. A score is not a probability or a p-value."
                            ),
                        ]
                    ),
                ),
            ],
            className="two-columns",
        ),
        panel(
            "Observed versus expected",
            html.Div(
                [
                    html.Label("Inspect an alert", htmlFor="incident-choice"),
                    dcc.Dropdown(
                        id="incident-choice",
                        options=choices,
                        value=initial,
                        clearable=False,
                        disabled=not choices,
                        placeholder="No alerts in this selection",
                    ),
                    graph(context, "anomaly-context"),
                ]
            ),
            "Context is limited to the selected date range and up to 48 hours on each side of the selected alert.",
        ),
        panel(
            "Largest threshold alerts",
            table(
                alerts,
                [
                    "zone_id",
                    "hour",
                    "split",
                    "direction",
                    "observed",
                    "baseline",
                    "score",
                    "threshold",
                    "absolute_threshold",
                    "history_count",
                    "status",
                ],
                20,
            ),
            "Top 20 selected hours ranked by absolute score. The display limit does not change detection thresholds.",
        ),
    ]


def performance(repo: SnapshotRepository, timing_scope, backend):
    data = repo.read("benchmark_runs")
    if data.empty:
        return notice(
            "No benchmark measurements have been published. CPU/GPU speedups will appear only after matching workloads pass correctness checks and complete timed runs."
        )
    if timing_scope != "all":
        data = data[data.timing_scope.eq(timing_scope)]
    if backend != "all":
        data = data[data.backend.eq(backend)]
    if data.empty:
        return notice("No benchmark trials match this timing scope and backend.")
    valid = data[pd.to_numeric(data.elapsed_s, errors="coerce").gt(0)].copy()
    if "validation" in valid:
        valid = valid[
            valid.validation.astype(str).str.lower().isin(["passed", "pass", "true", "ok"])
        ]
    keys = [
        c for c in ["experiment_id", "stage", "timing_scope", "input_rows", "backend"] if c in valid
    ]
    medians = (
        valid.groupby(keys, as_index=False).elapsed_s.median()
        if not valid.empty
        else pd.DataFrame()
    )
    if not medians.empty:
        medians["workload"] = medians.apply(
            lambda row: f"{row['stage']} · {row['timing_scope']} · {int(row['input_rows']):,} rows",
            axis=1,
        )
        bars = charts.style(
            px.bar(
                medians,
                x="workload",
                y="elapsed_s",
                color="backend",
                barmode="group",
                color_discrete_map=charts.BACKEND_COLORS,
                labels={"elapsed_s": "Median elapsed seconds", "workload": "Measured workload"},
            )
        )
        trials = charts.style(
            px.strip(
                valid,
                x="backend",
                y="elapsed_s",
                color="backend",
                hover_data=[
                    c for c in ["stage", "timing_scope", "input_rows", "repeat"] if c in valid
                ],
                color_discrete_map=charts.BACKEND_COLORS,
                labels={"elapsed_s": "Elapsed seconds"},
            )
        )
    else:
        bars = charts.empty("No correctness-validated timing results in this selection.")
        trials = bars
    compare_keys = [
        c
        for c in ["experiment_id", "stage", "timing_scope", "input_rows", "config_hash"]
        if c in valid
    ]
    speedups = pd.DataFrame()
    if compare_keys and not valid.empty:
        paired = (
            valid.groupby(compare_keys + ["backend"], dropna=False)
            .elapsed_s.median()
            .unstack("backend")
        )
        if "cpu" in paired:
            comparisons = []
            for gpu_backend in ["gpu", "gpu_library", "gpu_custom", "cuda", "custom_cuda"]:
                if gpu_backend not in paired:
                    continue
                comparison = (
                    paired[["cpu", gpu_backend]]
                    .dropna()
                    .reset_index()
                    .rename(columns={gpu_backend: "gpu_seconds", "cpu": "cpu_seconds"})
                )
                comparison["gpu_backend"] = gpu_backend
                comparison["cpu_over_gpu_speedup"] = comparison.cpu_seconds / comparison.gpu_seconds
                comparisons.append(comparison)
            if comparisons:
                speedups = pd.concat(comparisons, ignore_index=True)
    return [
        notice(
            "Date and zone filters do not apply to benchmark trials. Compare matching experiment, stage, input size, and timing scope. Kernel-only and end-to-end times answer different questions."
        ),
        panel(
            "Measured elapsed time",
            graph(bars),
            "Median of successful, correctness-validated trials. Hardware, cache conditions, transfers, and warmups belong to the experiment record.",
        ),
        panel(
            "Repeat distribution",
            graph(trials),
            "Every displayed point is a recorded trial; failed or invalid trials remain visible in the raw table.",
        ),
        panel(
            "Matched CPU/GPU comparison",
            table(speedups)
            if not speedups.empty
            else notice("No matching validated CPU and GPU measurements are available yet."),
            "Speedup = median CPU seconds ÷ median GPU seconds. Values below 1 mean the CPU was faster.",
        ),
        panel("Raw trials", table(data, limit=200)),
        panel(
            "Benchmark environment",
            html.Pre(
                json.dumps(repo.manifest.get("benchmark_environment", {}), indent=2),
                className="code-block",
            ),
            "Hardware and software for the measured trials. An empty record means benchmark environment metadata has not been published; the snapshot-build machine is not a substitute.",
        ),
    ]


def methods(repo: SnapshotRepository):
    quality = repo.read("quality_summary")
    sources = repo.manifest.get("sources", [])
    if isinstance(sources, dict):
        sources = list(sources.values())
    source_rows = (
        pd.DataFrame(sources)
        if sources and isinstance(sources[0], dict)
        else pd.DataFrame({"source": sources})
    )
    columns = [
        c
        for c in [
            "source_id",
            "dataset_type",
            "nominal_month",
            "source_month",
            "byte_count",
            "parquet_row_count",
            "sha256",
            "url",
        ]
        if c in source_rows
    ]
    return [
        notice(
            "This snapshot is fixed. Date and zone controls do not change provenance, quality totals, model definitions, or benchmark configuration."
        ),
        panel(
            "What this data can tell us",
            html.Div(
                [
                    html.P(
                        "TLC records describe reported taxi and ride-hailing trips. The analysis measures recorded activity, not all travel demand, road-level traffic, or individual travelers."
                    ),
                    html.P(
                        "Pickup and drop-off counts are independently timed. Origin–destination counts are anchored to pickup time. A connection between zone anchors is not a reconstructed route."
                    ),
                    html.P(
                        "Times follow an explicit America/New_York source-local convention. Repeated or nonexistent daylight-saving hours cannot be treated as ordinary hourly observations. Missing source coverage remains distinct from a recorded zero."
                    ),
                    html.P(
                        "KMeans groups similar feature values; cluster names are interpretations. Anomaly scores compare observed counts with a historical seasonal reference and are not calibrated probabilities."
                    ),
                ]
            ),
        ),
        panel(
            "Frozen sources",
            table(source_rows, columns or None, 100),
            "Official TLC source files and zone geometry are retained with provenance. Publication of a monthly file does not establish complete real-world reporting.",
        ),
        panel(
            "Data quality",
            table(quality, limit=100),
            "Rules can overlap. Do not add their affected-row counts as though each row belongs to only one rule.",
        ),
        panel(
            "Models and evaluation",
            html.Pre(
                json.dumps(repo.manifest.get("models", {}), indent=2, default=str),
                className="code-block",
            ),
        ),
        panel(
            "Snapshot configuration",
            html.Pre(
                json.dumps(
                    {
                        key: repo.manifest.get(key)
                        for key in [
                            "snapshot_id",
                            "created_at",
                            "data_kind",
                            "study_start",
                            "study_end",
                            "backend",
                            "validation",
                            "config",
                        ]
                    },
                    indent=2,
                    default=str,
                ),
                className="code-block",
            ),
        ),
    ]


def render_page(
    repo,
    page,
    filters,
    aggregation="day",
    destinations=(),
    metric="pickup_count",
    tiles=False,
    selected_cluster="all",
    timing_scope="all",
    backend="all",
):
    if page == "overview":
        return overview(repo, filters, aggregation, tiles)
    if page == "patterns":
        return patterns(repo, filters, selected_cluster)
    if page == "zones":
        return zone_explorer(repo, filters, destinations, metric, tiles)
    if page == "anomalies":
        return anomalies(repo, filters)
    if page == "performance":
        return performance(repo, timing_scope, backend)
    return methods(repo)


def create_app(snapshot_dir: Path) -> Dash:
    repo = SnapshotRepository(snapshot_dir)
    default = repo.default_filters()
    zones = repo.zones()
    zone_options = (
        [
            {"label": f"{row.zone} · {getattr(row, 'borough', '')}", "value": int(row.zone_id)}
            for row in zones.itertuples()
        ]
        if not zones.empty
        else []
    )
    summary = repo.read("cluster_summary")
    trials = repo.read("benchmark_runs")
    timing_scopes = (
        sorted(trials.timing_scope.dropna().unique()) if "timing_scope" in trials else []
    )
    benchmark_backends = sorted(trials.backend.dropna().unique()) if "backend" in trials else []
    cluster_options = (
        [{"label": "All clusters", "value": "all"}]
        + [
            {"label": f"Cluster {value}", "value": str(value)}
            for value in sorted(summary.cluster.unique())
        ]
        if "cluster" in summary
        else [{"label": "No published clusters", "value": "all"}]
    )
    end_date = (pd.Timestamp(default.end) - pd.Timedelta(days=1)).date().isoformat()
    start_date = pd.Timestamp(default.start).date().isoformat()
    kind = repo.manifest.get("data_kind", "unknown")
    app = Dash(
        __name__,
        title="NYC Mobility Observatory",
        assets_folder=str(Path(__file__).parent / "assets"),
        suppress_callback_exceptions=True,
        update_title="Loading…",
    )
    app.server.config.update(MAX_CONTENT_LENGTH=1_000_000)

    @app.server.get("/healthz")
    def health():
        return {"status": "ok", "snapshot_id": repo.snapshot_id, "data_kind": kind}

    app.layout = html.Div(
        [
            html.A("Skip to results", href="#main-content", className="skip-link"),
            dcc.Store(id="applied-filters", data=default.to_dict()),
            dcc.Download(id="download"),
            html.Aside(
                [
                    html.Div(
                        [
                            html.Div("NYC", className="brand-mark"),
                            html.Div([html.Strong("Mobility"), html.Span("OBSERVATORY")]),
                        ],
                        className="brand",
                    ),
                    html.P("Recorded trips. Measured compute.", className="brand-caption"),
                    html.Nav(
                        dcc.RadioItems(
                            id="page",
                            options=[
                                {
                                    "label": html.Span(
                                        [html.Span(f"0{i + 1}", className="nav-number"), name]
                                    ),
                                    "value": key,
                                }
                                for i, (key, (name, _)) in enumerate(PAGES.items())
                            ],
                            value="overview",
                            className="navigation",
                        ),
                        **{"aria-label": "Dashboard sections"},
                    ),
                    html.Div(
                        [
                            html.Span("SOURCE SNAPSHOT", className="eyebrow"),
                            html.P(repo.snapshot_id, className="snapshot-id"),
                            html.Span(
                                "Synthetic demonstration"
                                if kind == "synthetic"
                                else "Recorded TLC data"
                                if kind == "real"
                                else "Unverified source type",
                                className=f"badge {'warning' if kind != 'real' else ''}",
                            ),
                            html.P("Read-only published results", className="muted"),
                        ],
                        className="sidebar-footer",
                    ),
                ],
                className="sidebar",
            ),
            html.Main(
                [
                    html.Header(
                        [
                            html.Div(
                                [
                                    html.P(
                                        "NEW YORK CITY / HISTORICAL ANALYTICS", className="eyebrow"
                                    ),
                                    html.H1("Overview", id="page-title"),
                                    html.P(
                                        PAGES["overview"][1],
                                        id="page-subtitle",
                                        className="subtitle",
                                    ),
                                ]
                            ),
                            html.Span(f"{start_date} – {end_date}", className="period-pill"),
                        ],
                        className="page-header",
                    ),
                    notice(
                        "Synthetic demonstration data — useful for checking the software and interactions. These are not actual NYC findings.",
                        "warning",
                    )
                    if kind == "synthetic"
                    else None,
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Label("Source family", htmlFor="dataset"),
                                    dcc.Dropdown(
                                        id="dataset",
                                        options=[
                                            {
                                                "label": "Yellow taxis"
                                                if default.dataset == "yellow"
                                                else "High-volume ride-hailing",
                                                "value": default.dataset,
                                            }
                                        ],
                                        value=default.dataset,
                                        clearable=False,
                                    ),
                                ],
                                className="filter-source",
                            ),
                            html.Div(
                                [
                                    html.Label("Date range · inclusive days", htmlFor="date-range"),
                                    dcc.DatePickerRange(
                                        id="date-range",
                                        start_date=start_date,
                                        end_date=end_date,
                                        min_date_allowed=start_date,
                                        max_date_allowed=end_date,
                                        display_format="MMM D, YYYY",
                                        minimum_nights=0,
                                    ),
                                ],
                                className="filter-dates",
                            ),
                            html.Div(
                                [
                                    html.Label("Zones · all by default", htmlFor="zones"),
                                    dcc.Dropdown(
                                        id="zones",
                                        options=zone_options,
                                        value=[],
                                        multi=True,
                                        placeholder="Search taxi zones…",
                                    ),
                                ],
                                className="filter-zones",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Apply filters",
                                        id="apply",
                                        n_clicks=0,
                                        className="button primary",
                                    ),
                                    html.Button(
                                        "Reset",
                                        id="reset",
                                        n_clicks=0,
                                        className="button secondary",
                                    ),
                                ],
                                className="filter-actions",
                            ),
                            html.Div(
                                id="filter-message",
                                children="All covered dates and zones selected.",
                                className="filter-message",
                                role="status",
                            ),
                        ],
                        className="filters",
                        id="main-filters",
                        **{"aria-label": "Applied data filters"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Timeline interval", htmlFor="aggregation"),
                                    dcc.Dropdown(
                                        id="aggregation",
                                        options=[
                                            {"label": "Daily", "value": "day"},
                                            {"label": "Hourly", "value": "hour"},
                                        ],
                                        value="day",
                                        clearable=False,
                                    ),
                                ],
                                id="aggregation-control",
                                className="compact-control",
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Destination zones · OD only", htmlFor="destinations"
                                    ),
                                    dcc.Dropdown(
                                        id="destinations",
                                        options=zone_options,
                                        value=[],
                                        multi=True,
                                        placeholder="All destinations",
                                    ),
                                ],
                                id="destination-control",
                                className="wide-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Map activity", htmlFor="map-metric"),
                                    dcc.Dropdown(
                                        id="map-metric",
                                        options=[
                                            {"label": "Pickups", "value": "pickup_count"},
                                            {"label": "Drop-offs", "value": "dropoff_count"},
                                        ],
                                        value="pickup_count",
                                        clearable=False,
                                    ),
                                ],
                                id="metric-control",
                                className="compact-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Sample cluster", htmlFor="selected-cluster"),
                                    dcc.Dropdown(
                                        id="selected-cluster",
                                        options=cluster_options,
                                        value="all",
                                        clearable=False,
                                    ),
                                ],
                                id="cluster-control",
                                className="compact-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Timing scope", htmlFor="timing-scope"),
                                    dcc.Dropdown(
                                        id="timing-scope",
                                        options=[
                                            {
                                                "label": value.replace("_", " ").capitalize(),
                                                "value": value,
                                            }
                                            for value in ["all", *timing_scopes]
                                        ],
                                        value="all",
                                        clearable=False,
                                    ),
                                ],
                                id="timing-control",
                                className="compact-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Backend", htmlFor="benchmark-backend"),
                                    dcc.Dropdown(
                                        id="benchmark-backend",
                                        options=[
                                            {"label": value.upper(), "value": value}
                                            for value in ["all", *benchmark_backends]
                                        ],
                                        value="all",
                                        clearable=False,
                                    ),
                                ],
                                id="backend-control",
                                className="compact-control",
                            ),
                            html.Div(
                                dcc.Checklist(
                                    id="map-tiles",
                                    options=[
                                        {
                                            "label": " Background streets (requires internet)",
                                            "value": "on",
                                        }
                                    ],
                                    value=[],
                                ),
                                id="tiles-control",
                            ),
                            html.Button(
                                "Export zone-hour CSV",
                                id="export",
                                n_clicks=0,
                                className="button secondary export-button",
                                title=f"Export up to {repo.max_export:,} rows for the applied date and zone filters.",
                            ),
                        ],
                        className="view-controls",
                    ),
                    html.Div(id="export-status", role="status", className="muted"),
                    dcc.Loading(
                        html.Div(
                            id="page-content", children=render_page(repo, "overview", default)
                        ),
                        type="circle",
                        color=charts.BLUE,
                    ),
                    html.Footer(
                        [
                            html.Span("NYC TLC source data · America/New_York source local time"),
                            html.Span("Recorded activity does not represent all citywide travel."),
                        ],
                        className="page-footer",
                    ),
                ],
                id="main-content",
                className="main-content",
                tabIndex=-1,
            ),
        ],
        className="app-shell",
    )

    @app.callback(
        Output("applied-filters", "data"),
        Output("filter-message", "children"),
        Input("apply", "n_clicks"),
        Input("reset", "n_clicks"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
        State("dataset", "value"),
        State("zones", "value"),
        prevent_initial_call=True,
    )
    def apply_filters(_apply, _reset, start, end, dataset, selected_zones):
        if ctx.triggered_id == "reset":
            return default.to_dict(), "All covered dates and zones selected."
        try:
            filters = Filters.from_dict(
                {
                    "start": start,
                    "end": (pd.Timestamp(end) + pd.Timedelta(days=1)).isoformat(),
                    "dataset": dataset,
                    "zones": selected_zones or [],
                }
            )
            if pd.Timestamp(filters.start) < pd.Timestamp(default.start) or pd.Timestamp(
                filters.end
            ) > pd.Timestamp(default.end):
                raise QueryError("Choose dates inside the published study period.")
            return (
                filters.to_dict(),
                f"Applied {start} through {end} · {len(filters.zones)} selected zones"
                if filters.zones
                else f"Applied {start} through {end} · all zones",
            )
        except (ValueError, TypeError, KeyError) as error:
            return no_update, f"Filters not applied: {error}"

    @app.callback(
        Output("date-range", "start_date"),
        Output("date-range", "end_date"),
        Output("zones", "value"),
        Output("destinations", "value"),
        Input("reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_controls(_clicks):
        return start_date, end_date, [], []

    @app.callback(
        Output("zones", "value", allow_duplicate=True),
        Input("zone-map", "clickData"),
        prevent_initial_call=True,
    )
    def select_zone(click):
        if not click or not click.get("points"):
            raise PreventUpdate
        value = click["points"][0].get("customdata")
        if not value:
            raise PreventUpdate
        return [int(value[0])]

    @app.callback(
        Output("page-content", "children"),
        Output("page-title", "children"),
        Output("page-subtitle", "children"),
        Input("page", "value"),
        Input("applied-filters", "data"),
        Input("aggregation", "value"),
        Input("destinations", "value"),
        Input("map-metric", "value"),
        Input("map-tiles", "value"),
        Input("selected-cluster", "value"),
        Input("timing-scope", "value"),
        Input("benchmark-backend", "value"),
    )
    def update_page(
        page, filter_data, aggregation, destinations, metric, tiles, cluster, timing_scope, backend
    ):
        title, subtitle = PAGES.get(page, PAGES["overview"])
        try:
            filters = Filters.from_dict(filter_data)
            content = render_page(
                repo,
                page,
                filters,
                aggregation,
                destinations,
                metric,
                bool(tiles),
                cluster,
                timing_scope,
                backend,
            )
        except QueryError as error:
            content = notice(str(error), "warning")
        except Exception:
            error_id = uuid4().hex[:10]
            LOGGER.exception(
                "Dashboard query failed snapshot=%s error=%s", repo.snapshot_id, error_id
            )
            content = notice(
                f"This view could not be loaded. Reset the filters and try again. Error {error_id}; snapshot {repo.snapshot_id}.",
                "error",
            )
        return content, title, subtitle

    @app.callback(
        *[
            Output(identifier, "style")
            for identifier in [
                "main-filters",
                "aggregation-control",
                "destination-control",
                "metric-control",
                "cluster-control",
                "timing-control",
                "backend-control",
                "tiles-control",
                "export",
            ]
        ],
        Input("page", "value"),
    )
    def show_controls(page):
        visible = [
            page not in {"methods", "performance"},
            page == "overview",
            page == "zones",
            page == "zones",
            page == "patterns",
            page == "performance",
            page == "performance",
            page in {"overview", "zones"},
            page not in {"methods", "performance"},
        ]
        return [{} if item else {"display": "none"} for item in visible]

    @app.callback(
        Output("anomaly-context", "figure"),
        Input("incident-choice", "value"),
        State("applied-filters", "data"),
        prevent_initial_call=True,
    )
    def update_incident(value, filter_data):
        if not value:
            raise PreventUpdate
        try:
            zone, hour = value.split("|", 1)
            data = repo.read("anomaly_scores", Filters.from_dict(filter_data))
            return charts.anomaly_context(data, int(zone), hour)
        except (QueryError, ValueError):
            return charts.empty("This alert selection is unavailable. Choose another alert.")

    @app.callback(
        Output("download", "data"),
        Output("export-status", "children"),
        Input("export", "n_clicks"),
        State("applied-filters", "data"),
        prevent_initial_call=True,
    )
    def export_data(_clicks, filter_data):
        try:
            csv, count, truncated = repo.export_zone_hours(Filters.from_dict(filter_data))
            suffix = "-truncated" if truncated else ""
            message = f"Exported {count:,} zone-hour rows for the applied filters." + (
                f" The export is limited to {repo.max_export:,} rows; shorten the date range for a complete selection."
                if truncated
                else ""
            )
            return {
                "content": csv,
                "filename": f"zone-hours-{repo.snapshot_id}{suffix}.csv",
                "type": "text/csv",
            }, message
        except QueryError as error:
            return no_update, str(error)

    # Exposed for a deployment health check or an integration test, not user writes.
    app.mobility_repository = repo
    return app
