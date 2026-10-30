"""Chart construction kept separate from callbacks and data access."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLUE = "#2854c5"
TEAL = "#0f766e"
COLORS = [BLUE, TEAL, "#9a3412", "#7e22ce", "#b91c1c", "#0369a1", "#4d7c0f", "#a21caf"]
BACKEND_COLORS = {
    "cpu": BLUE,
    "gpu": TEAL,
    "gpu_library": TEAL,
    "gpu_custom": "#9a3412",
    "cuda": "#9a3412",
    "custom_cuda": "#9a3412",
}


def style(figure: go.Figure, *, height=350, revision="mobility") -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=20, t=35, b=45),
        font=dict(family="system-ui, sans-serif", color="#172033", size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=COLORS,
        legend=dict(orientation="h", y=1.12, x=0),
        uirevision=revision,
        hoverlabel=dict(bgcolor="white"),
    )
    figure.update_xaxes(gridcolor="#edf0f5", zeroline=False)
    figure.update_yaxes(gridcolor="#edf0f5", zeroline=False, rangemode="tozero")
    return figure


def empty(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return style(figure)


def activity(frame: pd.DataFrame, aggregation="day") -> go.Figure:
    if frame.empty:
        return empty("No zone-hours match this selection.")
    data = frame.copy()
    data["time"] = pd.to_datetime(data.hour).dt.floor("D" if aggregation == "day" else "h")
    values = data.groupby("time")[["pickup_count", "dropoff_count"]].sum(min_count=1)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=values.index,
            y=values.pickup_count,
            name="Mapped pickups",
            mode="lines",
            line=dict(color=BLUE, width=2.5),
            connectgaps=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=values.index,
            y=values.dropoff_count,
            name="Mapped drop-offs",
            mode="lines",
            line=dict(color=TEAL, width=2, dash="dot"),
            connectgaps=False,
        )
    )
    figure.update_layout(
        yaxis_title="Recorded endpoint count",
        xaxis_title="NYC source local date/time",
        hovermode="x unified",
    )
    return style(figure)


def zone_totals(frame: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    columns = [
        c
        for c in ["pickup_count", "dropoff_count", "duration_sum_seconds", "duration_count"]
        if c in frame
    ]
    result = frame.groupby("zone_id", as_index=False)[columns].sum(min_count=1)
    if not zones.empty:
        result = result.merge(zones, on="zone_id", how="left", validate="one_to_one")
    if "zone" not in result:
        result["zone"] = result.zone_id.astype(str)
    return result


def zone_map(
    frame: pd.DataFrame,
    zones: pd.DataFrame,
    geojson: dict | None,
    metric="pickup_count",
    revision="mobility",
    tiles=False,
) -> go.Figure:
    data = zone_totals(frame, zones)
    if data.empty or metric not in data or not data[metric].notna().any():
        return empty("No observed activity is available for this map.")
    label = "Mapped pickups" if metric == "pickup_count" else "Mapped drop-offs"
    arguments = dict(
        color=metric,
        hover_name="zone",
        color_continuous_scale=["#eef3ff", "#9bb8ed", "#2854c5", "#142b6f"],
        labels={metric: label},
        center=dict(lat=40.74, lon=-73.96),
        zoom=9.5,
    )
    if geojson:
        figure = px.choropleth_map(
            data,
            geojson=geojson,
            locations="zone_id",
            featureidkey="properties.zone_id",
            opacity=0.83,
            custom_data=["zone_id"],
            **arguments,
        )
    elif {"latitude", "longitude"}.issubset(data.columns):
        data = data.dropna(subset=["latitude", "longitude"])
        if data.empty:
            return empty("Zone geometry is unavailable. Use the zone table below.")
        figure = px.scatter_map(
            data,
            lat="latitude",
            lon="longitude",
            size=metric,
            size_max=36,
            custom_data=["zone_id"],
            **arguments,
        )
    else:
        return empty("Zone geometry is unavailable. Use the zone table below.")
    figure.update_layout(
        map_style="carto-positron" if tiles else "white-bg",
        coloraxis_colorbar=dict(title="Trips", thickness=12, len=0.65),
    )
    return style(figure, height=470, revision=revision)


def heatmap(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return empty("No observed pickup hours in this selection.")
    # Average after summing selected zones per hour, avoiding a per-zone denominator.
    data = frame.groupby("hour").pickup_count.sum(min_count=1).reset_index()
    data["weekday"] = pd.to_datetime(data.hour).dt.day_of_week
    data["hour_of_day"] = pd.to_datetime(data.hour).dt.hour
    values = data.pivot_table(
        index="weekday", columns="hour_of_day", values="pickup_count", aggfunc="mean"
    ).reindex(index=range(7), columns=range(24))
    figure = go.Figure(
        go.Heatmap(
            x=list(range(24)),
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            z=values.to_numpy(),
            colorscale=[[0, "#eef3ff"], [1, BLUE]],
            colorbar=dict(title="Pickups"),
            hovertemplate="%{y} %{x}:00<br>Mean recorded pickups: %{z:,.1f}<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis_title="Pickup hour — NYC source local time", yaxis_title="Day of week"
    )
    return style(figure)


def graph_zone_map(
    nodes: pd.DataFrame,
    zones: pd.DataFrame,
    geojson: dict | None,
    metric="pagerank",
    revision="mobility",
    tiles=False,
) -> go.Figure:
    """Show continuous PageRank or categorical communities without ranking labels."""
    if nodes.empty or metric not in nodes:
        return empty("Graph results are unavailable for this snapshot.")
    data = nodes.copy()
    missing = [column for column in zones if column not in data and column != "zone_id"]
    if missing:
        data = data.merge(zones[["zone_id", *missing]], on="zone_id", how="left")
    if "zone" not in data:
        data["zone"] = data["zone_id"].astype(str)
    arguments = {
        "hover_name": "zone",
        "custom_data": ["zone_id"],
        "hover_data": [name for name in ["pagerank", "degree", "community"] if name in data],
        "center": {"lat": 40.74, "lon": -73.96},
        "zoom": 9.4,
    }
    if metric == "community":
        data["community_label"] = data["community"].map(
            lambda value: "Unavailable" if pd.isna(value) else f"Group {int(value)}"
        )
        arguments.update(
            color="community_label",
            color_discrete_sequence=px.colors.qualitative.Alphabet,
            labels={"community_label": "Community label"},
        )
    else:
        arguments.update(
            color="pagerank",
            color_continuous_scale=["#eef3ff", "#9bb8ed", BLUE, "#142b6f"],
            labels={"pagerank": "PageRank score"},
        )
    if geojson:
        figure = px.choropleth_map(
            data,
            geojson=geojson,
            locations="zone_id",
            featureidkey="properties.zone_id",
            opacity=0.85,
            **arguments,
        )
    elif {"latitude", "longitude"}.issubset(data.columns):
        data = data.dropna(subset=["latitude", "longitude"])
        if data.empty:
            return empty("Zone geometry is unavailable; use the graph table.")
        figure = px.scatter_map(data, lat="latitude", lon="longitude", **arguments)
        figure.update_traces(marker_size=11)
    else:
        return empty("Zone geometry is unavailable; use the graph table.")
    figure.update_layout(map_style="carto-positron" if tiles else "white-bg")
    if metric == "pagerank":
        figure.update_layout(coloraxis_colorbar=dict(title="PageRank", thickness=12, len=0.65))
    return style(figure, height=470, revision=revision)


def flows(data: pd.DataFrame, zones: pd.DataFrame, limit: int, tiles=False) -> go.Figure:
    if data.empty or not {"longitude", "latitude"}.issubset(zones.columns):
        return empty("No mapped OD connections match these filters.")
    coordinates = zones.set_index("zone_id")[["longitude", "latitude"]].dropna()
    figure = go.Figure()
    subset = data.sort_values(
        ["trip_count", "pickup_zone_id", "dropoff_zone_id"], ascending=[False, True, True]
    ).head(limit)
    names = zones.set_index("zone_id").zone.to_dict() if "zone" in zones else {}
    for row in subset.itertuples():
        if (
            row.pickup_zone_id not in coordinates.index
            or row.dropoff_zone_id not in coordinates.index
        ):
            continue
        origin, destination = (
            coordinates.loc[row.pickup_zone_id],
            coordinates.loc[row.dropoff_zone_id],
        )
        width = 0.8 + 3.5 * np.sqrt(row.trip_count / max(float(subset.trip_count.max()), 1))
        description = (
            f"{names.get(row.pickup_zone_id, row.pickup_zone_id)} → "
            f"{names.get(row.dropoff_zone_id, row.dropoff_zone_id)}"
            f"<br>{row.trip_count:,} recorded trips"
        )
        figure.add_trace(
            go.Scattermap(
                lon=[origin.longitude, destination.longitude],
                lat=[origin.latitude, destination.latitude],
                mode="markers" if row.pickup_zone_id == row.dropoff_zone_id else "lines",
                line=dict(width=width, color=BLUE),
                marker=dict(size=10, color=BLUE),
                opacity=0.45,
                showlegend=False,
                text=[description, description],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    ids = sorted(
        set(subset.pickup_zone_id).union(subset.dropoff_zone_id).intersection(coordinates.index)
    )
    anchors = coordinates.loc[ids]
    figure.add_trace(
        go.Scattermap(
            lon=anchors.longitude,
            lat=anchors.latitude,
            text=[f"{names.get(i, i)} (zone {i})" for i in ids],
            mode="markers",
            marker=dict(size=6, color=TEAL),
            name="Zone anchors",
            hovertemplate="%{text}<extra></extra>",
        )
    )
    figure.update_layout(
        map=dict(
            style="carto-positron" if tiles else "white-bg",
            center=dict(lat=40.74, lon=-73.96),
            zoom=9.4,
        )
    )
    return style(figure, height=470)


def scatter_sample(data: pd.DataFrame, maximum: int) -> go.Figure:
    if data.empty:
        return empty("No sampled trips match these filters.")
    data = data.head(maximum).copy()
    data["duration_minutes"] = data.duration_seconds / 60
    if "cluster" in data:
        data["cluster"] = data.cluster.astype(str)
    # Keep hover metadata textual: nullable zone integers plus datetime arrays
    # otherwise require an unsupported common dtype in the Plotly adapter.
    hover_columns = [
        column for column in ["pickup_local", "pickup_zone_id", "dropoff_zone_id"] if column in data
    ]
    for column in hover_columns:
        data[column] = data[column].astype(str)
    figure = px.scatter(
        data,
        x="trip_distance_miles",
        y="duration_minutes",
        color="cluster" if "cluster" in data else None,
        opacity=0.5,
        color_discrete_sequence=COLORS,
        labels={
            "trip_distance_miles": "Trip distance (miles)",
            "duration_minutes": "Trip duration (minutes)",
            "cluster": "Cluster",
        },
        hover_data=hover_columns,
        render_mode="webgl",
    )
    outside = int(
        (~data.trip_distance_miles.between(0, 50) | ~data.duration_minutes.between(0, 120)).sum()
    )
    style(figure, height=470)
    figure.update_layout(
        xaxis=dict(range=[0, 50], autorange=False),
        yaxis=dict(range=[0, 120], autorange=False),
        margin=dict(t=110, b=75),
        legend=dict(y=-0.23),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0,
                y=1.18,
                buttons=[
                    dict(
                        label="Typical range",
                        method="relayout",
                        args=[
                            {
                                "xaxis.range": [0, 50],
                                "xaxis.autorange": False,
                                "yaxis.range": [0, 120],
                                "yaxis.autorange": False,
                            }
                        ],
                    ),
                    dict(
                        label="Full extent",
                        method="relayout",
                        args=[{"xaxis.autorange": True, "yaxis.autorange": True}],
                    ),
                ],
            )
        ],
    )
    figure.add_annotation(
        text=(
            f"Typical window: 0–50 miles, 0–120 minutes.<br>{outside:,} of {len(data):,} "
            "sample points outside; Full extent shows all."
        ),
        x=0,
        y=1.35,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font=dict(size=11),
    )
    return figure


def sample_distance_histogram(data: pd.DataFrame) -> go.Figure:
    """Keep a huge source value from stretching every ordinary sample bin."""
    if data.empty:
        return empty("No sampled trips match these filters.")
    distance = pd.to_numeric(data.trip_distance_miles, errors="coerce")
    valid = np.isfinite(distance) & distance.ge(0)
    labels = ["0–<1", "1–<2", "2–<3", "3–<5", "5–<10", "10–<20", "20–50", ">50"]
    # Match the public chart: left-inclusive bands, with 50 in the final finite band.
    indices = np.searchsorted([1, 2, 3, 5, 10, 20, 50], distance[valid], side="right")
    indices = np.where(distance[valid].eq(50), 6, indices)
    counts = np.bincount(indices, minlength=len(labels))
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            marker_color=BLUE,
            hovertemplate="%{x} miles<br>%{y:,} sampled trips<extra></extra>",
        )
    )
    figure.update_layout(xaxis_title="Trip distance (miles)", yaxis_title="Sampled trips")
    style(figure, height=390)
    figure.update_layout(margin=dict(t=70))
    figure.add_annotation(
        text=(
            f"{int(valid.sum()):,} sample distances; {int((distance[valid] > 50).sum()):,} "
            f"in >50-mile overflow.<br>{int((~valid).sum()):,} invalid values excluded."
        ),
        x=0,
        y=1.20,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font=dict(size=11),
    )
    return figure


def anomaly_context(
    data: pd.DataFrame, zone: int, selected_hour: str, metric="pickup_count"
) -> go.Figure:
    hour = pd.Timestamp(selected_hour)
    subset = data[
        (data.zone_id == zone)
        & (pd.to_datetime(data.hour) >= hour - pd.Timedelta(hours=48))
        & (pd.to_datetime(data.hour) <= hour + pd.Timedelta(hours=48))
    ].sort_values("hour")
    if subset.empty:
        return empty("No scored context is available for this selection.")
    duration = metric == "mean_duration_minutes"
    if duration:
        # Sparse duration artifacts omit missing hours. Explicit gaps prevent a
        # line from implying observations between two nonadjacent recorded hours.
        subset = subset.set_index("hour").reindex(
            pd.date_range(subset.hour.min(), subset.hour.max(), freq="h")
        )
        subset.index.name = "hour"
        subset = subset.reset_index()
    figure = go.Figure()
    observation_options = {}
    if duration and "eligible_trip_count" in subset:
        observation_options = {
            "customdata": subset[["eligible_trip_count", "status"]].to_numpy(dtype=object),
            "hovertemplate": (
                "%{x}<br>Mean duration: %{y:.2f} min<br>Eligible trips: %{customdata[0]}"
                "<br>%{customdata[1]}<extra></extra>"
            ),
        }
    figure.add_trace(
        go.Scatter(
            x=subset.hour,
            y=subset.observed,
            name="Mean eligible trip duration" if duration else "Observed pickups",
            mode="lines",
            line=dict(color=BLUE, width=2.5),
            connectgaps=False,
            **observation_options,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=subset.hour,
            y=subset.baseline,
            name="Seasonal median",
            mode="lines",
            line=dict(color=TEAL, dash="dash"),
            connectgaps=False,
        )
    )
    alerts = subset[subset.get("is_alert", pd.Series(False, index=subset.index)).fillna(False)]
    figure.add_trace(
        go.Scatter(
            x=alerts.hour,
            y=alerts.observed,
            name="Statistical alert",
            mode="markers",
            marker=dict(symbol="diamond", color="#b91c1c", size=9),
        )
    )
    figure.update_layout(
        yaxis_title="Mean trip duration (minutes)" if duration else "Recorded pickups",
        xaxis_title="NYC source local time",
        hovermode="x unified",
    )
    return style(figure, height=380)
