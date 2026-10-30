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
from .static_export import clustering_quality_rows, cpu_execution_evidence, processing_metadata

LOGGER = logging.getLogger(__name__)
PAGES = {
    "overview": ("Overview", "A measured view of recorded movement."),
    "patterns": ("Mobility patterns", "Explore when trips happen and how they differ."),
    "zones": ("Zones & connections", "Follow recorded connections between taxi zones."),
    "anomalies": (
        "Unusual activity",
        "Compare pickup activity and trip duration with seasonal history.",
    ),
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


def table(frame: pd.DataFrame, columns=None, limit=100, precision=3):
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
            formatted = f"{value:,.{precision}f}"
            return formatted if precision != 3 else formatted.rstrip("0").rstrip(".")
        if value == "unresolved":
            return "Unreviewed"
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


def _cluster_inclusion(model):
    eligibility = model.get("eligibility", {})
    thresholds = eligibility.get("thresholds", {})
    names = ("max_distance_miles", "max_duration_minutes", "max_mean_speed_mph")
    if not all(name in thresholds for name in names):
        return (
            "This older snapshot does not record the newer clustering inclusion limits. "
            "Finite, nonnegative distance and positive duration alone can retain implausible trips."
        )
    distance, duration, speed = (thresholds[name] for name in names)
    text = (
        f"This clustering study includes trips of at most {distance:g} miles, "
        f"{duration:g} minutes, and {speed:g} mph mean speed. These fixed study limits "
        "restrict clustering and its sample plots; they do not remove valid endpoint counts "
        "or change raw source files. Zero reported distances can still be included."
    )
    excluded = eligibility.get("excluded_from_canonical_eligible_rows")
    if excluded is not None:
        text += f" {excluded:,} otherwise eligible trips were excluded by these limits."
    return text


def _model_preparation(manifest):
    model = manifest.get("models", {}).get("clustering", {})
    native = processing_metadata(manifest)["execution_mode"] == "native"
    explanation = (
        "Both native backends use the same numeric SplitMix64 source-row priority for "
        "bounded samples and seeded initial centers. Features and training-only scaler "
        "statistics use float64; standardized KMeans matrices use float32. "
        if native
        else "This historical model uses bounded deterministic samples and its saved "
        "initialization policy, preceding the native numeric sampling policy. "
    )
    sampling = model.get("sampling_policy") or model.get("samples", {}).get("train", {}).get(
        "method"
    )
    return (
        explanation
        + f"Sampling policy: {sampling or 'not recorded'}; "
        + f"initialization policy: {model.get('initialization_policy', 'not recorded')}."
    )


def _processing_notes(manifest):
    processing = processing_metadata(manifest)
    names = {
        "aggregation_backend": "aggregations",
        "clustering_backend": "clustering",
        "anomaly_backend": "pickup-volume anomalies",
        "duration_anomaly_backend": "duration anomalies",
        "graph_backend": "graph analysis",
    }
    backends = "; ".join(f"{label}: {processing[key]}" for key, label in names.items())
    native = processing["execution_mode"] == "native"
    explanation = (
        "Native execution uses the recorded backend for bulk normalization, eligibility, "
        "features, sampling/scaling, aggregation, clustering, both anomaly detectors and "
        "graph algorithms. The CPU path uses CPU libraries throughout; the GPU path uses "
        "cuDF/CuPy, cuML and cuGraph. Host services handle checksums, metadata, small calendar "
        "and zone inputs, job control and reduced output finalization. Independent CPU "
        "reference checks run after timed GPU processing."
        if native
        else "Historical execution uses a hybrid GPU path. Source preparation, model "
        "sampling/scaling and anomaly scoring use CPU work, and the GPU graph's CPU "
        "reference remains inside its pipeline time. These results predate native execution."
    )
    return [
        f"Saved result backends — {backends}.",
        explanation,
        "Viewing charts displays saved results and does not start a GPU job.",
    ]


def _pipeline_execution_note(data, environment):
    modes = {
        row.get("execution_mode")
        or (
            "native"
            if row.get("stage") == "native_production_pipeline"
            else environment.get("execution_mode", "legacy")
        )
        for row in data.where(pd.notna(data), None).to_dict("records")
    }
    notes = []
    if "native" in modes:
        notes.append(
            "Native trials compare entirely CPU analysis with GPU bulk processing, including "
            "normalization, sampling/scaling and anomaly scoring. Independent CPU reference "
            "checks, including graph validation, occur after timing."
        )
    if "legacy" in modes:
        notes.append(
            "Historical hybrid trials retain CPU preparation, sampling/scaling and anomaly "
            "work; the GPU graph's CPU reference remains timed. Their speedups do not "
            "describe the newer native implementation."
        )
    return " ".join(notes)


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
        notice(
            "A pickup is one recorded trip beginning; a drop-off is one recorded trip ending. "
            "Mapped totals count endpoints with a usable time and a recognized taxi zone "
            "inside the applied selection. They count trips, not individual passengers. "
            "The totals can differ because endpoint validity and selected dates or zones differ."
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
    model = repo.manifest.get("models", {}).get("clustering", {})
    model_status = model.get("status", "")
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
            html.Div(
                [
                    html.P(
                        "KMeans groups trips using log(1 + distance), log(1 + duration), and "
                        "pickup hour represented by sine and cosine so midnight stays close to "
                        "11 p.m. Each feature is standardized using training data only. "
                        "Zone IDs, fares and passenger identities are not clustering features."
                    ),
                    html.P(
                        "The model learns from a fixed training sample; a separate calibration "
                        "sample selects the number of clusters. Similar distances can belong to "
                        "different groups because their duration or pickup time differs."
                        if model_status != "exploratory_no_calibration"
                        else "The model learns from a fixed training sample. This exploratory run "
                        "does not have a separate calibration sample to select the cluster count."
                    ),
                    html.P(_cluster_inclusion(model)),
                    html.P(_model_preparation(repo.manifest)),
                    summary_content,
                ]
            ),
            "Fixed model/scoring-period summary: date, zone, and selected-cluster controls do not change this table. Group numbers are labels, not ranked quality or known passenger types.",
        ),
        exact_distributions,
        panel(
            "Distance and duration",
            graph(charts.scatter_sample(samples, repo.max_points)),
            f"Sampled trips only, at most {repo.max_points:,} points. Typical range opens at 0–50 miles and 0–120 minutes; Full extent reveals all sampled points. The displayed outside-window count is not a data deletion. Date, origin-zone and cluster selections affect this plot; sample counts are not population totals.",
        ),
        panel(
            "Sample distribution",
            graph(charts.sample_distance_histogram(samples))
            if not samples.empty
            else notice("No sampled trips match the current filters."),
            "Fixed bins for the same selected sample, including points outside the scatter's typical window. Bins include their lower edge and exclude their upper edge, except that 20–50 also includes 50 miles. The >50-mile bin contains every longer distance. Bar heights are sample counts, not density or full-dataset totals.",
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
            "With no origin or destination selected, all matching zone pairs compete for the "
            "top positions by trip count. The lines show the busiest recorded connections, "
            "not a random sample; yellow-taxi activity is concentrated around Manhattan. "
            "The main zone selection filters endpoint zones on the activity map and origins "
            "in the OD views. The destination control applies only to OD connections. "
            "OD counts use pickup time."
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
            f"The {min(len(connections), repo.max_flows):,} largest of {len(connections):,} matching zone pairs, ranked by recorded trip count. Lines connect representative zone anchors, not streets or actual routes. Thicker lines mean more trips. Opposite directions can overlap; within-zone trips appear at one anchor and remain in the table. Hover a connection or use the table to identify its direction.",
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
        *_graph_panels(repo, tiles),
    ]


def _graph_panels(repo: SnapshotRepository, tiles=False):
    if not repo.has("graph_nodes"):
        return [
            panel(
                "Zone network analysis",
                notice("PageRank and community results have not been published in this snapshot."),
            )
        ]
    nodes = repo.read("graph_nodes")
    if nodes.empty:
        return [panel("Zone network analysis", notice("No graph nodes are available."))]
    model = repo.manifest.get("models", {}).get("graph", {})
    rank = nodes.sort_values("pagerank", ascending=False).copy() if "pagerank" in nodes else nodes
    start, end = (
        repo.manifest.get("study_start", "unrecorded"),
        repo.manifest.get("study_end", "unrecorded"),
    )
    threshold = model.get("parameters", {}).get("min_trips_per_edge")
    backend = model.get("backend", "not recorded")
    return [
        panel(
            "Zone network analysis",
            html.Div(
                [
                    notice(
                        f"Fixed graph for {str(start)[:10]} through {str(end)[:10]} (end exclusive). "
                        "Date, origin, destination and endpoint-metric controls do not recompute "
                        f"these graph results. Recorded graph backend: {backend}."
                    ),
                    html.P(
                        "Graph validation: "
                        + model.get("validation", {})
                        .get("status", "not recorded")
                        .replace("_", " ")
                        + ". "
                        + model.get("validation", {}).get("reference", "")
                    ),
                    html.P(
                        model.get(
                            "graph_definition",
                            "Zones are connected by recorded origin–destination trips.",
                        )
                    ),
                    html.P(
                        model.get(
                            "pagerank_meaning",
                            "PageRank measures weighted directed connectivity, not passenger counts.",
                        )
                    ),
                    html.P(
                        model.get(
                            "community_meaning",
                            "Louvain community IDs are arbitrary labels, not ranks.",
                        )
                    ),
                    html.P(
                        model.get(
                            "degree_meaning",
                            "Degree counts retained neighbors; trip strength sums their edge weights.",
                        )
                    ),
                    html.P(
                        model.get(
                            "self_trip_policy",
                            "See the saved model definition for treatment of within-zone trips.",
                        )
                    ),
                    html.P(
                        f"Minimum retained directed-edge weight: {threshold:,} trips."
                        if threshold is not None
                        else "No edge threshold was recorded."
                    ),
                ]
            ),
        ),
        html.Div(
            [
                panel(
                    "PageRank across zones",
                    graph(
                        charts.graph_zone_map(
                            nodes, repo.zones(), repo.geojson(), "pagerank", repo.snapshot_id, tiles
                        )
                    ),
                    "A continuous connectivity score; larger values do not imply more passengers or a better zone.",
                ),
                panel(
                    "Louvain communities",
                    graph(
                        charts.graph_zone_map(
                            nodes,
                            repo.zones(),
                            repo.geojson(),
                            "community",
                            repo.snapshot_id,
                            tiles,
                        )
                    ),
                    "Categorical colors identify groups. Community numbers have no ranking. Use group labels in the table when colors repeat.",
                ),
            ],
            className="two-columns",
        ),
        panel(
            "Zone connectivity ranking",
            table(
                rank,
                [
                    "zone",
                    "borough",
                    "zone_id",
                    "pagerank",
                    "community",
                    "in_degree",
                    "out_degree",
                    "degree",
                    "in_trip_strength",
                    "out_trip_strength",
                    "self_trip_count",
                    "is_isolated",
                ],
                300,
                precision=6,
            ),
            "Sorted by PageRank. Degree and trip strength use the retained network; within-zone trips are shown separately.",
        ),
    ]


def _period_text(model: dict, config: dict, period: str) -> str:
    bounds = model.get("splits", config.get("splits", {}))
    start, end = bounds.get(f"{period}_start"), bounds.get(f"{period}_end")
    return (
        f"{str(start)[:10]} to {str(end)[:10]} (end exclusive)"
        if start and end
        else "dates not recorded"
    )


def _duration_panels(repo: SnapshotRepository, filters: Filters):
    if not repo.has("duration_scores"):
        return [
            panel(
                "Unusual trip durations",
                notice("Duration anomaly results have not been published in this snapshot."),
            )
        ]
    model = repo.manifest.get("models", {}).get("duration_anomalies", {})
    config = repo.manifest.get("config", {})
    params = model.get("duration_anomalies_config", config.get("duration_anomalies", {}))
    values = {
        name: params.get(name, "not recorded")
        for name in [
            "min_history",
            "min_trips_per_hour",
            "scale_floor_minutes",
            "absolute_threshold_minutes",
        ]
    }
    threshold = model.get("calibration", {}).get(
        "selected_threshold", model.get("threshold", params.get("score_threshold", "not recorded"))
    )
    data = repo.read("duration_scores", filters)
    explanation = panel(
        "Unusual trip durations",
        html.Div(
            [
                html.P(
                    "This separate detector compares the mean duration of eligible trips "
                    "starting in each zone and hour, measured in minutes. It does not count trips "
                    "or estimate travel time along a fixed route."
                ),
                html.P(
                    f"An hour needs at least {values['min_trips_per_hour']} eligible trips. "
                    f"The baseline needs at least {values['min_history']} comparable training "
                    "hours for the same pickup zone, weekday and hour. Baseline is the median "
                    "of those hourly means; history counts hours, not trips."
                ),
                html.P(
                    "Score = (observed mean minutes − baseline minutes) / "
                    f"max(1.4826 × training MAD, {values['scale_floor_minutes']} minutes). "
                    f"An alert needs an absolute score of at least {threshold} and an absolute "
                    f"change of at least {values['absolute_threshold_minutes']} minutes. "
                    "Positive scores mean longer trips; negative scores mean shorter trips."
                ),
                html.P(
                    f"Training: {_period_text(model, config, 'train')}. "
                    f"Calibration: {_period_text(model, config, 'calibration')}. "
                    f"Test: {_period_text(model, config, 'test')}. Calibration selects the "
                    "threshold; the test period uses it frozen. An alert budget is not an accuracy or false-positive rate."
                ),
                html.P(
                    model.get(
                        "limitations",
                        "Route, distance and destination composition can change mean duration. These scores do not establish congestion or a cause.",
                    )
                ),
                html.P(
                    "Missing hours are not zero minutes. Low-trip and unsupported hours receive no score. Unreviewed means a person has not yet checked an alert's explanation."
                ),
                html.P(
                    f"Recorded computation backend: {model.get('computation_backend', 'not recorded')}."
                ),
            ]
        ),
        "Date and origin-zone filters apply to the following duration scores and incidents.",
    )
    if data.empty:
        return [
            explanation,
            notice("No duration calibration or test hours match the current filters."),
        ]
    alerts = data.loc[data["is_alert"].fillna(False)].copy()
    alerts["absolute_score"] = alerts["score"].abs()
    alerts = alerts.sort_values(["absolute_score", "hour"], ascending=[False, True])
    choices = alerts if not alerts.empty else data.loc[data["status"].eq("scored")]
    context = charts.empty("No supported duration hours are available in this selection.")
    context_description = "Context is limited to this selection and 48 hours either side; missing hours break the line."
    if not choices.empty:
        chosen = choices.iloc[0]
        context = charts.anomaly_context(
            data, int(chosen["zone_id"]), str(chosen["hour"]), "mean_duration_minutes"
        )
        context_description = (
            f"Zone {int(chosen['zone_id'])}, {chosen['hour']}: "
            + (
                "largest absolute-score duration alert. "
                if not alerts.empty
                else "first supported duration hour; no alerts match. "
            )
            + context_description
        )
    renamed = alerts.rename(
        columns={
            "observed": "observed_mean_minutes",
            "baseline": "baseline_mean_minutes",
            "deviation": "change_in_mean_minutes",
            "absolute_threshold": "minimum_change_minutes",
        }
    )
    incidents = (
        repo.read("duration_incidents", filters)
        if repo.has("duration_incidents")
        else pd.DataFrame()
    )
    if not incidents.empty:
        incidents = incidents.sort_values("peak_score", ascending=False).rename(
            columns={
                "total_deviation": "sum_of_hourly_mean_changes_minutes",
                "label": "review_label",
            }
        )
    return [
        explanation,
        panel(
            "Duration scoring coverage",
            table(
                data["status"].value_counts().rename_axis("status").reset_index(name="zone_hours")
            ),
        ),
        panel("Mean trip duration versus baseline", graph(context), context_description),
        panel(
            "Largest duration alerts",
            table(
                renamed,
                [
                    "zone_id",
                    "hour",
                    "split",
                    "eligible_trip_count",
                    "observed_mean_minutes",
                    "baseline_mean_minutes",
                    "change_in_mean_minutes",
                    "score",
                    "threshold",
                    "minimum_change_minutes",
                    "history_count",
                ],
                20,
            ),
            "Top 20 selected hours by absolute duration score. Observations, baselines and changes are in minutes.",
        ),
        panel(
            "Duration incidents and review",
            table(
                incidents,
                [
                    "zone_id",
                    "split",
                    "start_hour",
                    "end_hour",
                    "affected_hours",
                    "peak_score",
                    "sum_of_hourly_mean_changes_minutes",
                    "review_label",
                ],
                20,
            ),
            model.get(
                "incident_total_deviation_meaning",
                "Sum of hourly mean-minute deviations, not total passenger delay. Unreviewed alerts are not confirmed incidents.",
            ),
        ),
    ]


def anomalies(repo: SnapshotRepository, filters: Filters):
    pickup_panels = _pickup_anomalies(repo, filters)
    return [
        *(pickup_panels if isinstance(pickup_panels, list) else [pickup_panels]),
        *_duration_panels(repo, filters),
    ]


def _pickup_anomalies(repo: SnapshotRepository, filters: Filters):
    if not repo.has("anomaly_scores"):
        return notice(
            "Pickup-count anomaly results have not been published. A seasonal baseline needs multiple months of observed history."
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
    parameters = repo.manifest.get("config", {}).get("anomalies", {})
    model = repo.manifest.get("models", {}).get("anomalies", {})
    scale_floor = parameters.get("scale_floor", 5)
    minimum_history = parameters.get("min_history", 8)
    absolute_threshold = parameters.get("absolute_threshold", 20)
    score_threshold = model.get("calibration", {}).get(
        "selected_threshold", parameters.get("score_threshold", 4)
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
                                "Baseline means the median pickup count for this zone at the "
                                "same weekday and hour during January–March 2024 training. "
                                "History means the number of comparable observed training "
                                "hours, not the number of trips."
                            ),
                            html.P(
                                f"Score = (observed − baseline) / max(1.4826 × MAD, {scale_floor:g}). "
                                "MAD is the median absolute difference between training counts "
                                "and their baseline. Positive scores mean more pickups; negative "
                                "scores mean fewer. A score is not a percentage or probability."
                            ),
                            html.P(
                                f"A flag requires usable coverage, at least {minimum_history:g} "
                                f"history hours, an absolute score of at least {score_threshold:g} "
                                f"and a change of at least {absolute_threshold:g} trips."
                            ),
                            html.P(
                                "Calibration uses April 2024 to choose the threshold. Test uses "
                                "May–June with that choice frozen. The calibration alert budget "
                                "controls how many hours are flagged; it is not a false-positive "
                                "rate. Seasonal change can also produce large scores."
                            ),
                            html.P(
                                "Unreviewed (stored as unresolved) means a person has not yet "
                                "checked the alert's real-world explanation. The calculation "
                                "has finished; an alert is not a confirmed incident. This "
                                "pickup-count detector is separate from the duration detector below."
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


def _pipeline_benchmark_panels(repo: SnapshotRepository):
    if not repo.has("pipeline_benchmark_runs"):
        return [
            panel(
                "Full raw-file pipeline benchmark",
                notice("No full raw-file pipeline trials have been published in this snapshot."),
            )
        ]
    data = repo.read("pipeline_benchmark_runs")
    environment = repo.manifest.get("pipeline_benchmark_environment", {})
    if data.empty:
        return [
            panel(
                "Full raw-file pipeline benchmark",
                notice("No completed pipeline trial rows are available."),
            )
        ]
    valid = data.loc[
        data["validation"].astype(str).str.lower().eq("passed")
        & pd.to_numeric(data["elapsed_s"], errors="coerce").between(
            0, float("inf"), inclusive="neither"
        )
    ].copy()
    if "phase" in valid:
        valid = valid.loc[valid["phase"].eq("measured")]
    pair_keys = [
        name
        for name in [
            "experiment_id",
            "protocol_hash",
            "stage",
            "timing_scope",
            "input_rows",
            "months",
        ]
        if name in valid
    ]
    summary, comparisons = pd.DataFrame(), pd.DataFrame()
    if not valid.empty and pair_keys:
        grouped = valid.groupby(pair_keys + ["backend"], dropna=False)
        summary = grouped.agg(
            measured_trials=("elapsed_s", "size"),
            median_seconds=("elapsed_s", "median"),
            minimum_seconds=("elapsed_s", "min"),
            maximum_seconds=("elapsed_s", "max"),
        ).reset_index()
        for column, label in [
            ("peak_host_rss_sampled_bytes", "largest_sampled_host_RSS_GiB"),
            ("peak_gpu_memory_sampled_bytes", "largest_sampled_GPU_memory_GiB"),
        ]:
            if column in valid:
                memory = grouped[column].max().reset_index(name=label)
                memory[label] = memory[label] / 1024**3
                summary = summary.merge(memory, on=pair_keys + ["backend"], validate="one_to_one")
        paired = (
            valid.groupby(pair_keys + ["backend"], dropna=False)["elapsed_s"]
            .median()
            .unstack("backend")
        )
        if {"cpu", "gpu"}.issubset(paired.columns):
            comparisons = (
                paired[["cpu", "gpu"]]
                .dropna()
                .reset_index()
                .rename(columns={"cpu": "cpu_seconds", "gpu": "gpu_seconds"})
            )
            comparisons["cpu_over_gpu_ratio"] = (
                comparisons["cpu_seconds"] / comparisons["gpu_seconds"]
            )
            comparisons["result"] = comparisons["cpu_over_gpu_ratio"].map(
                lambda value: (
                    f"GPU {value:.3f}× faster"
                    if value > 1
                    else f"CPU {1 / value:.3f}× faster"
                    if value < 1
                    else "Equal median time"
                )
            )
    if not summary.empty:
        plotted = summary.copy()
        plotted["workload"] = plotted["input_rows"].map(lambda rows: f"{int(rows):,} raw rows")
        figure = charts.style(
            px.bar(
                plotted,
                x="workload",
                y="median_seconds",
                color="backend",
                barmode="group",
                color_discrete_map=charts.BACKEND_COLORS,
                labels={
                    "median_seconds": "Median full-pipeline seconds",
                    "workload": "Recorded source size",
                },
            )
        )
    else:
        figure = charts.empty("No complete, correctness-validated measured pipeline trials.")
    scope = environment.get(
        "scope_notes",
        "Raw-source checksums, file reads, normalization, features, aggregation, model fitting/evaluation/prediction, graph/duration analysis, artifact writes and built-in reconciliation. Network acquisition and publication are excluded.",
    )
    quality = pd.DataFrame(clustering_quality_rows(data))
    evidence = cpu_execution_evidence(data)
    cpu_note = (
        f"Native CPU isolation: {evidence['guard_and_no_cuda_imports_confirmed']} of "
        f"{evidence['measured_native_cpu_trials']} measured CPU trials recorded an active CUDA "
        "import guard and no imported CUDA libraries. GPU process memory was sampled as zero "
        f"in {evidence['zero_sampled_gpu_memory_trials']} trials, nonzero in "
        f"{evidence['nonzero_sampled_gpu_memory_trials']}, and unavailable in "
        f"{evidence['gpu_memory_unavailable_trials']}. Sampled zero describes recorded "
        "observations, not continuous memory monitoring."
        if evidence["measured_native_cpu_trials"]
        else ""
    )
    return [
        panel(
            "Full raw-file pipeline benchmark",
            html.Div(
                [
                    notice(
                        "This separate experiment retains its own full-pipeline scope and both backends. Date, zone and stage timing/backend controls do not filter these panels."
                    ),
                    html.P(scope),
                    html.P(
                        "Worker startup/imports, GPU preflight, cross-trial comparisons and final artifact hashing occur outside the timed interval. Lazy imports and initialization reached inside the pipeline and built-in reconciliation remain timed. "
                        + _pipeline_execution_note(data, environment)
                    ),
                    html.P(
                        "January and January–March have no April calibration or May–June test data. The six-month workload adds held-out evaluation as well as more records, so scaling across these prefixes does not isolate row-count growth. This full-pipeline scope is distinct from the stage benchmarks above."
                    ),
                    html.P(
                        f"Recorded benchmark host: {environment.get('hostname', 'not recorded')}. CPU/GPU pairs must share the same experiment, input size and protocol. Only measured trials marked passed contribute; warmups, failed trials and incomplete pairs remain in the raw table."
                    ),
                    graph(figure),
                ]
            ),
        ),
        panel(
            "Full-pipeline CPU/GPU comparison",
            table(comparisons, precision=6)
            if not comparisons.empty
            else notice("No matched validated full-pipeline CPU/GPU pair is available."),
            "Ratio = median CPU seconds / median GPU seconds. Above 1 favors GPU, below 1 favors CPU, and 1 is equal. A positive ratio alone does not mean the GPU won. "
            + cpu_note,
        ),
        panel(
            "Clustering quality by measured trial",
            table(
                quality,
                [
                    "months",
                    "backend",
                    "repeat",
                    "selected_k",
                    "training_rows",
                    "training_inertia_per_row",
                    "calibration_rows",
                    "calibration_silhouette",
                    "test_rows",
                    "test_silhouette",
                    "test_mean_squared_center_distance",
                    "selection_status",
                    "test_status",
                    "validation",
                    "diagnostics_status",
                ],
                limit=200,
                precision=6,
            ),
            "All measured trials, including failures; warmups excluded. January and January–March "
            "have no April calibration or May–June test data. Inertia and center distance are squared "
            "distances in standardized feature space, not miles or minutes. Lower is more compact "
            "for comparable inputs and group counts; higher silhouette indicates greater geometric "
            "separation, not verified passenger categories. Matched inputs and passed output checks "
            "do not guarantee identical fitted clusters or equal model quality. Missing values are unavailable, not zero.",
        ),
        panel(
            "Pipeline time and sampled memory",
            table(
                summary,
                [
                    "input_rows",
                    "months",
                    "backend",
                    "measured_trials",
                    "median_seconds",
                    "minimum_seconds",
                    "maximum_seconds",
                    "largest_sampled_host_RSS_GiB",
                    "largest_sampled_GPU_memory_GiB",
                ],
                precision=6,
            ),
            "Memory values are the largest sampled process-tree RSS and process GPU memory across measured trials, not exact peaks. Short spikes can be missed; process-tree RSS can count shared pages more than once. Missing GPU memory is unavailable, not zero. Sampling intervals and sample counts are recorded below.",
        ),
        panel(
            "Full-pipeline raw trials",
            table(
                data,
                [
                    "trial_id",
                    "backend",
                    "phase",
                    "months",
                    "input_rows",
                    "elapsed_s",
                    "validation",
                    "validation_elapsed_s",
                    "execution_mode",
                    "cpu_import_guard_enabled",
                    "cpu_cuda_context_check",
                    "rss_sample_interval_s",
                    "gpu_sample_interval_s",
                    "rss_samples",
                    "gpu_samples",
                    "background_gpu_processes",
                    "selected_k",
                    "cluster_eligible_rows",
                    "error",
                ],
                250,
                precision=6,
            ),
        ),
        panel(
            "Full-pipeline environment and protocol",
            html.Pre(json.dumps(environment, indent=2, default=str), className="code-block"),
        ),
    ]


def performance(repo: SnapshotRepository, timing_scope, backend):
    stage_panels = _stage_performance(repo, timing_scope, backend)
    return [
        panel(
            "Where the GPU was used", [html.P(note) for note in _processing_notes(repo.manifest)]
        ),
        *(stage_panels if isinstance(stage_panels, list) else [stage_panels]),
        *_pipeline_benchmark_panels(repo),
    ]


def _stage_performance(repo: SnapshotRepository, timing_scope, backend):
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
                comparison["cpu_over_gpu_ratio"] = comparison.cpu_seconds / comparison.gpu_seconds
                comparison["result"] = comparison.cpu_over_gpu_ratio.map(
                    lambda ratio: (
                        f"GPU {ratio:.3f}× faster"
                        if ratio > 1
                        else f"CPU {1 / ratio:.3f}× faster"
                        if ratio < 1
                        else "Equal median time"
                    )
                )
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
            table(speedups, precision=6)
            if not speedups.empty
            else notice("No matching validated CPU and GPU measurements are available yet."),
            "CPU/GPU ratio = median CPU seconds ÷ median GPU seconds. It stays positive because both times are positive: above 1 means GPU faster, below 1 means CPU faster, and 1 means equal time. A 0.879 ratio means CPU about 1.14× faster. Times and ratios are shown to six decimal places; trailing zeros are display precision, not additional speedup.",
        ),
        panel("Raw trials", table(data, limit=200, precision=6)),
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
    cluster_model = repo.manifest.get("models", {}).get("clustering", {})
    graph_model = repo.manifest.get("models", {}).get("graph", {})
    duration_model = repo.manifest.get("models", {}).get("duration_anomalies", {})
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
                        "KMeans groups similar feature values; cluster names are interpretations. Anomaly scores compare a recorded metric with its historical seasonal reference and are not calibrated probabilities."
                    ),
                ]
            ),
        ),
        panel(
            "Executed analysis and its limits",
            html.Div(
                [
                    *[html.P(note) for note in _processing_notes(repo.manifest)],
                    html.Details(
                        [
                            html.Summary("Recorded stage backends and row counts"),
                            html.Pre(
                                json.dumps(processing_metadata(repo.manifest), indent=2),
                                className="code-block",
                            ),
                        ]
                    ),
                    html.P(
                        f"Graph analysis backend: {graph_model.get('backend', 'not recorded')}. "
                        + (
                            "This snapshot publishes weighted PageRank and Louvain communities. "
                            + graph_model.get("preparation_backend", "")
                            + ". "
                            + graph_model.get(
                                "limitations",
                                "A taxi-zone graph is small; GPU acceleration is not assumed beneficial.",
                            )
                            if repo.has("graph_nodes")
                            else "Graph algorithm results have not been published in this snapshot; OD lines alone are not a PageRank or community analysis."
                        )
                    ),
                    html.P(
                        (
                            "Separate seasonal detectors cover pickup counts and hourly mean "
                            "trip duration. Duration computation: "
                            + duration_model.get("computation_backend", "not recorded")
                            + ". "
                            + duration_model.get(
                                "limitations",
                                "Route and distance composition can change mean duration; an alert does not establish congestion.",
                            )
                            if repo.has("duration_scores")
                            else "The published seasonal detector covers pickup counts. Duration anomaly results have not been published in this snapshot."
                        )
                        + " Neither detector measures unserved travel demand."
                    ),
                    html.P(
                        "Stage benchmarks describe their recorded operation and transfer scope. "
                        + (
                            "A separate full raw-file pipeline experiment is also published; it "
                            "includes model analysis and artifact writes but excludes acquisition "
                            "and publication. Its own trial rows record the measured source sizes. "
                            "Process RSS and GPU memory are sampled observations, not exact peaks."
                            if repo.has("pipeline_benchmark_runs")
                            else "No full raw-file pipeline benchmark is published in this snapshot. "
                            "A canonical-file aggregation benchmark excludes raw normalization "
                            "and models, so its speedup cannot describe the whole system."
                        )
                    ),
                    html.P(_cluster_inclusion(cluster_model)),
                    html.P(repo.manifest.get("refit", {}).get("policy_revision_context", "")),
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
