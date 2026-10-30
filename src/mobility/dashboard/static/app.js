"use strict";

/* All inputs below are published aggregates. No remote computation is started. */
const state = {
  page: "overview",
  request: 0,
  filters: null,
  summary: null,
  cache: new Map(),
};
const pages = {
  overview: ["Overview", "A measured view of recorded movement."],
  patterns: [
    "Mobility patterns",
    "Explore when trips happen and how they differ.",
  ],
  zones: [
    "Zones & connections",
    "Follow recorded connections between taxi zones.",
  ],
  anomalies: [
    "Unusual activity",
    "Compare recorded pickups with their seasonal baseline.",
  ],
  performance: [
    "CPU & GPU performance",
    "Compare measured work, with its conditions attached.",
  ],
  methods: [
    "Data & methods",
    "Trace every finding to its sources and assumptions.",
  ],
};
const blue = "#2854c5",
  teal = "#0f766e";
const colors = [
  blue,
  teal,
  "#9a3412",
  "#7e22ce",
  "#b91c1c",
  "#0369a1",
  "#4d7c0f",
  "#a21caf",
];
const byId = (id) => document.getElementById(id);
const number = (value, precision = 0) =>
  value == null || !Number.isFinite(Number(value))
    ? "—"
    : Number(value).toLocaleString(undefined, {
        maximumFractionDigits: precision,
        minimumFractionDigits: precision,
      });
const escapeHtml = (value) =>
  String(value ?? "—").replace(
    /[&<>"']/g,
    (character) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        character
      ],
  );
const dateOnly = (value) => String(value).slice(0, 10);
const notice = (text, kind = "info") =>
  `<div class="notice ${kind}" role="status">${escapeHtml(text)}</div>`;
const panel = (title, content, description = "") =>
  `<section class="panel"><div class="panel-heading"><h2>${escapeHtml(title)}</h2>${description ? `<p class="muted">${escapeHtml(description)}</p>` : ""}</div>${content}</section>`;
const chartDiv = (id) => `<div id="${id}" class="chart" role="img"></div>`;
const sum = (rows, key) =>
  rows.reduce(
    (total, row) => total + (Number.isFinite(row[key]) ? row[key] : 0),
    0,
  );
const observedSum = (rows, key) =>
  rows.some((row) => row[key] != null) ? sum(rows, key) : null;

function table(rows, columns, limit = 100) {
  if (!rows.length) return notice("No rows are available for this selection.");
  const cells = (value) =>
    typeof value === "number"
      ? number(value, Number.isInteger(value) ? 0 : 3)
      : escapeHtml(
          typeof value === "object" && value !== null
            ? JSON.stringify(value)
            : value,
        );
  return `<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable data table"><table><thead><tr>${columns.map(([key, label]) => `<th scope="col">${escapeHtml(label || key.replaceAll("_", " "))}</th>`).join("")}</tr></thead><tbody>${rows
    .slice(0, limit)
    .map(
      (row) =>
        `<tr>${columns.map(([key]) => `<td>${cells(row[key])}</td>`).join("")}</tr>`,
    )
    .join(
      "",
    )}</tbody></table></div><p class="table-caption">Showing ${number(Math.min(rows.length, limit))} of ${number(rows.length)} available rows.</p>`;
}

function plot(id, traces, options = {}) {
  const element = byId(id);
  if (!element) return;
  const layout = {
    height: 350,
    margin: { l: 45, r: 24, t: 35, b: 50 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    font: { family: "system-ui, sans-serif", size: 12, color: "#172033" },
    colorway: colors,
    legend: { orientation: "h", y: 1.13, x: 0 },
    xaxis: { gridcolor: "#edf0f5", zeroline: false },
    yaxis: { gridcolor: "#edf0f5", zeroline: false, rangemode: "tozero" },
    uirevision: state.summary.meta.snapshot_id,
    ...options,
  };
  element.style.height = `${layout.height}px`;
  return Plotly.newPlot(element, traces, layout, {
    responsive: true,
    displaylogo: false,
    scrollZoom: false,
  });
}

function emptyPlot(id, message) {
  return plot(id, [], {
    annotations: [
      {
        text: message,
        showarrow: false,
        x: 0.5,
        y: 0.5,
        xref: "paper",
        yref: "paper",
      },
    ],
    xaxis: { visible: false },
    yaxis: { visible: false },
  });
}

async function load(name, optional = false) {
  if (!state.cache.has(name)) {
    const promise = fetch(`data/${name}`)
      .then(async (response) => {
        if (!response.ok) {
          if (optional && response.status === 404) return null;
          throw new Error(`The ${name} snapshot could not be loaded.`);
        }
        return response.json();
      })
      .catch((error) => {
        state.cache.delete(name);
        throw error;
      });
    state.cache.set(name, promise);
  }
  return state.cache.get(name);
}

function zoneName(id) {
  return (
    state.summary.zones.find((zone) => Number(zone.zone_id) === Number(id))
      ?.zone ?? `Zone ${id}`
  );
}
function zoneOptions(all = "All zones") {
  return (
    `<option value="all">${escapeHtml(all)}</option>` +
    state.summary.zones
      .map(
        (zone) =>
          `<option value="${zone.zone_id}">${escapeHtml(zone.zone)} · ${escapeHtml(zone.borough)}</option>`,
      )
      .join("")
  );
}
function selectedDays() {
  return state.summary.zone_days.filter(
    (row) =>
      row.date >= state.filters.start &&
      row.date <= state.filters.end &&
      (state.filters.zone === "all" ||
        Number(row.zone_id) === Number(state.filters.zone)),
  );
}
function groupRows(rows, key) {
  const groups = new Map();
  for (const row of rows) {
    const value = row[key];
    if (!groups.has(value)) groups.set(value, []);
    groups.get(value).push(row);
  }
  return groups;
}

async function showZoneMap(id, totals, request) {
  const geometry = await load("zones.geojson", true);
  if (!byId(id) || request !== state.request) return;
  const data = totals.filter((row) => row.pickups != null);
  if (!data.length)
    return emptyPlot(id, "No observed activity is available for this map.");
  let traces;
  if (geometry) {
    traces = [
      {
        type: "choroplethmap",
        geojson: geometry,
        locations: data.map((row) => row.zone_id),
        featureidkey: "properties.zone_id",
        z: data.map((row) => row.pickups),
        customdata: data.map((row) => row.zone_id),
        text: data.map((row) => zoneName(row.zone_id)),
        colorscale: [
          [0, "#edf2fd"],
          [0.5, "#8eaee8"],
          [1, "#2149ae"],
        ],
        marker: { opacity: 0.85, line: { width: 0.25, color: "#fff" } },
        colorbar: { title: { text: "Pickups" }, thickness: 12, len: 0.65 },
        hovertemplate: "%{text}<br>Mapped pickups: %{z:,.0f}<extra></extra>",
      },
    ];
  } else {
    const anchors = data
      .map((row) => ({
        ...row,
        ...state.summary.zones.find((zone) => zone.zone_id === row.zone_id),
      }))
      .filter((row) => row.latitude != null && row.longitude != null);
    if (!anchors.length)
      return emptyPlot(
        id,
        "Zone geometry is unavailable. Use the table beside this map.",
      );
    const maximum = Math.max(1, ...anchors.map((row) => row.pickups));
    traces = [
      {
        type: "scattermap",
        lat: anchors.map((row) => row.latitude),
        lon: anchors.map((row) => row.longitude),
        customdata: anchors.map((row) => row.zone_id),
        text: anchors.map(
          (row) => `${row.zone}: ${number(row.pickups)} pickups`,
        ),
        mode: "markers",
        marker: {
          color: blue,
          size: anchors.map((row) => 5 + 25 * Math.sqrt(row.pickups / maximum)),
          opacity: 0.7,
        },
        hovertemplate: "%{text}<extra></extra>",
      },
    ];
  }
  await plot(id, traces, {
    height: 470,
    map: { style: "white-bg", center: { lat: 40.74, lon: -73.96 }, zoom: 9.5 },
    margin: { l: 0, r: 0, t: 0, b: 0 },
  });
  const element = byId(id);
  if (element && request === state.request && state.page === "overview")
    element.on("plotly_click", (event) => {
      const zone = event.points?.[0]?.customdata;
      if (zone != null) {
        byId("zone-select").value = String(zone);
        byId("filter-message").textContent =
          `Selected ${zoneName(zone)}. Choose Apply filters to update the view.`;
      }
    });
}

async function overview(request) {
  const data = selectedDays();
  const pickups = observedSum(data, "pickups"),
    dropoffs = observedSum(data, "dropoffs"),
    durationN = sum(data, "duration_n");
  const totals = [...groupRows(data, "zone_id")]
    .map(([zone_id, rows]) => ({
      zone_id,
      zone: zoneName(zone_id),
      pickups: observedSum(rows, "pickups"),
      dropoffs: observedSum(rows, "dropoffs"),
    }))
    .sort((a, b) => (b.pickups ?? -1) - (a.pickups ?? -1));
  const active = totals.filter((row) => row.pickups > 0).length;
  const cards = [
    ["Mapped pickups", number(pickups), "Eligible recorded origins"],
    ["Mapped drop-offs", number(dropoffs), "Eligible recorded destinations"],
    [
      "Mean valid duration",
      durationN
        ? `${number(sum(data, "duration_sum") / durationN / 60, 1)} min`
        : "—",
      `${number(durationN)} valid pickup-anchored trips`,
    ],
    ["Zones with pickups", number(active), "Within the applied selection"],
  ];
  const observed = sum(data, "observed_hours"),
    possible = sum(data, "total_hours");
  byId("view-controls").innerHTML =
    `<span class="muted">Daily aggregates · NYC source local time</span><button id="export" class="button secondary export-button">Export selected daily data</button>`;
  byId("page-content").innerHTML =
    `<div class="kpi-grid">${cards.map(([label, value, detail]) => `<div class="kpi"><p class="kpi-label">${label}</p><strong class="kpi-value">${value}</strong><p class="kpi-detail">${detail}</p></div>`).join("")}</div>` +
    notice(
      `Pickup coverage: ${number(observed)} of ${number(possible)} selected zone-hours observed. Missing and DST-excluded hours are not zero activity. Drop-off totals have partial source context: earlier pickup files can contain additional arrivals.`,
      observed < possible ? "warning" : "info",
    ) +
    panel(
      "Activity over time",
      chartDiv("timeline"),
      "Pickups and drop-offs use their respective endpoint times. Daily totals sum eligible recorded observations; incomplete coverage remains visible above.",
    ) +
    `<div class="two-columns">${panel("Where trips begin", chartDiv("zone-map"), "Click a zone to select it, then Apply filters. TLC boundaries; no internet-dependent street tiles.")}${panel(
      "Most active zones",
      table(
        totals,
        [
          ["zone", "Zone"],
          ["pickups", "Pickups"],
          ["dropoffs", "Drop-offs"],
        ],
        20,
      ),
      "The same applied selection, in a map-free table.",
    )}</div>`;
  const daily = [...groupRows(data, "date")]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, rows]) => ({
      date,
      pickups: observedSum(rows, "pickups"),
      dropoffs: observedSum(rows, "dropoffs"),
    }));
  plot(
    "timeline",
    [
      {
        x: daily.map((row) => row.date),
        y: daily.map((row) => row.pickups),
        name: "Mapped pickups",
        type: "scatter",
        mode: "lines",
        line: { color: blue, width: 2.5 },
        connectgaps: false,
      },
      {
        x: daily.map((row) => row.date),
        y: daily.map((row) => row.dropoffs),
        name: "Mapped drop-offs",
        type: "scatter",
        mode: "lines",
        line: { color: teal, width: 2, dash: "dot" },
        connectgaps: false,
      },
    ],
    {
      yaxis: {
        title: { text: "Recorded endpoint count" },
        rangemode: "tozero",
      },
      xaxis: { title: { text: "NYC source local date" } },
      hovermode: "x unified",
    },
  );
  byId("export").onclick = () => downloadDaily(data);
  await showZoneMap("zone-map", totals, request);
}

function downloadDaily(data) {
  const limit = state.summary.meta.max_export_rows,
    truncated = data.length > limit;
  const columns = [
    "dataset_type",
    "zone_id",
    "date",
    "pickups",
    "dropoffs",
    "duration_sum",
    "duration_n",
    "observed_hours",
    "dropoff_observed_hours",
    "total_hours",
  ];
  const lines = [
    ["snapshot_id", "export_truncated", ...columns].join(","),
    ...data
      .slice(0, limit)
      .map((row) =>
        [
          state.summary.meta.snapshot_id,
          truncated,
          ...columns.map((key) => row[key] ?? ""),
        ]
          .map((value) => `"${String(value).replaceAll('"', '""')}"`)
          .join(","),
      ),
  ];
  const url = URL.createObjectURL(
    new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `zone-days-${state.summary.meta.snapshot_id}${truncated ? "-truncated" : ""}.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
  byId("filter-message").textContent =
    `Exported ${number(Math.min(data.length, limit))} daily rows.${truncated ? ` Export limited to ${number(limit)} rows; use a shorter date range for a complete selection.` : ""}`;
}

async function patterns(request) {
  const data = await load("patterns.json");
  if (request !== state.request) return;
  const clusters = [
    ...new Set(data.samples.map((row) => String(row.cluster))),
  ].sort();
  byId("view-controls").innerHTML =
    `<div class="compact-control"><label for="sample-cluster">Sample cluster</label><select id="sample-cluster"><option value="all">All clusters</option>${clusters.map((value) => `<option value="${escapeHtml(value)}">Cluster ${escapeHtml(value)}</option>`).join("")}</select></div><span class="muted">All dates and zones in this fixed snapshot</span>`;
  const status = state.summary.meta.models?.clustering?.status;
  byId("page-content").innerHTML =
    notice(
      "Patterns and cluster summaries describe the full published period. The cluster selector filters only the sampled plots; it does not change the full-population cluster table.",
    ) +
    (status === "exploratory_no_calibration"
      ? notice(
          "Exploratory clustering: this snapshot has no separate calibration period, so the selected cluster count has not been validated on a calibration period.",
          "warning",
        )
      : "") +
    panel(
      "A typical week",
      chartDiv("heatmap"),
      "Mean citywide mapped pickups per observed hour, by weekday and pickup hour. Missing and DST-ambiguous hours are excluded.",
    ) +
    panel(
      "Trip groups",
      table(
        data.clusters,
        [
          ["cluster", "Cluster"],
          ["trip_count", "Trips"],
          ["share", "Share"],
          ["mean_distance_miles", "Mean miles"],
          ["mean_duration_minutes", "Mean minutes"],
          ["scoring_start", "Scoring start"],
          ["scoring_end", "Scoring end"],
        ],
        20,
      ),
      "Cluster numbers identify feature-based groups, not known passenger types. Counts and raw-unit means refer to the declared scoring period.",
    ) +
    panel(
      "Distance and duration",
      chartDiv("sample-scatter"),
      `At most ${number(state.summary.meta.max_points)} sampled feature pairs. Sample counts are not population totals; no traveler or source-record identifiers are exported.`,
    ) +
    panel(
      "Sample distance distribution",
      chartDiv("sample-histogram"),
      "A distribution of the displayed sample, not an exact full-population histogram.",
    );
  if (data.histograms?.length) {
    byId("page-content").insertAdjacentHTML(
      "beforeend",
      panel(
        "Exact trip distributions",
        `<div class="two-columns">${chartDiv("exact-distance")}${chartDiv("exact-duration")}</div>`,
        "Exact eligible counts across all dates, zones, and clusters in the published snapshot. The sample-cluster selector does not apply to these bins.",
      ),
    );
    for (const [feature, id, label] of [
      ["distance_miles", "exact-distance", "Distance (miles)"],
      ["duration_minutes", "exact-duration", "Duration (minutes)"],
    ]) {
      const bins = data.histograms
        .filter((row) => row.feature === feature)
        .sort((a, b) => a.bin_index - b.bin_index);
      plot(
        id,
        [
          {
            type: "bar",
            x: bins.map((row) => row.bin_label),
            y: bins.map((row) => row.count),
            marker: { color: blue },
          },
        ],
        {
          xaxis: { title: { text: label }, automargin: true },
          yaxis: {
            title: { text: "Eligible recorded trips" },
            rangemode: "tozero",
          },
        },
      );
    }
  }
  const z = Array.from({ length: 7 }, () => Array(24).fill(null));
  data.hour_profile.forEach((row) => {
    z[row.weekday][row.hour_of_day] = row.mean_pickups;
  });
  plot(
    "heatmap",
    [
      {
        type: "heatmap",
        x: Array.from({ length: 24 }, (_, i) => i),
        y: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        z,
        colorscale: [
          [0, "#eef3ff"],
          [1, blue],
        ],
        colorbar: { title: { text: "Pickups" } },
        hovertemplate: "%{y} %{x}:00<br>Mean pickups: %{z:,.1f}<extra></extra>",
      },
    ],
    {
      xaxis: { title: { text: "Pickup hour · NYC source local time" } },
      yaxis: { title: { text: "Day of week" } },
    },
  );
  function drawSample() {
    const selected = byId("sample-cluster").value,
      rows = data.samples.filter(
        (row) => selected === "all" || String(row.cluster) === selected,
      );
    if (!rows.length) {
      emptyPlot("sample-scatter", "No sampled trips for this selection.");
      emptyPlot("sample-histogram", "No sampled trips for this selection.");
      return;
    }
    const traces = [...groupRows(rows, "cluster")].map(
      ([cluster, values], i) => ({
        type: "scattergl",
        mode: "markers",
        x: values.map((row) => row.trip_distance_miles),
        y: values.map((row) => row.duration_seconds / 60),
        name: `Cluster ${cluster}`,
        marker: {
          color:
            colors[Number(cluster) % colors.length] ||
            colors[i % colors.length],
          size: 5,
          opacity: 0.5,
        },
        hovertemplate:
          "Distance %{x:.2f} mi<br>Duration %{y:.1f} min<extra>%{fullData.name}</extra>",
      }),
    );
    plot("sample-scatter", traces, {
      height: 400,
      xaxis: { title: { text: "Trip distance (miles)" } },
      yaxis: {
        title: { text: "Trip duration (minutes)" },
        rangemode: "tozero",
      },
    });
    plot(
      "sample-histogram",
      [
        {
          type: "histogram",
          x: rows.map((row) => row.trip_distance_miles),
          nbinsx: 40,
          marker: { color: blue },
        },
      ],
      {
        xaxis: { title: { text: "Trip distance (miles)" } },
        yaxis: {
          title: { text: "Displayed sample count" },
          rangemode: "tozero",
        },
      },
    );
  }
  byId("sample-cluster").onchange = drawSample;
  drawSample();
}

async function zones(request) {
  const data = await load("od.json");
  if (request !== state.request) return;
  byId("view-controls").innerHTML =
    `<div class="wide-control"><label for="od-origin">Origin zone</label><select id="od-origin">${zoneOptions("All origins")}</select></div><div class="wide-control"><label for="od-destination">Destination zone</label><select id="od-destination">${zoneOptions("All destinations")}</select></div>`;
  byId("page-content").innerHTML =
    notice(
      "OD connections are aggregated over the entire published period. Date filtering is unavailable in this static view; the full Dash application supports hourly/date-specific OD queries. Origin and destination filters below are exact for this saved period.",
    ) +
    panel(
      "Recorded zone connections",
      chartDiv("flow-map"),
      "Lines connect representative zone anchors, not actual routes. Self-zone trips remain in the table but cannot appear as a visible line.",
    ) +
    panel(
      "Origin–destination table",
      `<div id="od-table"></div>`,
      "Trip counts retain records with valid endpoints even when duration is unavailable. Mean duration uses only eligible duration records.",
    );
  let drawRequest = 0;
  async function draw() {
    const localRequest = ++drawRequest;
    const origin = byId("od-origin").value,
      destination = byId("od-destination").value;
    const rows = data.connections
      .filter(
        (row) =>
          (origin === "all" || Number(row.pickup_zone_id) === Number(origin)) &&
          (destination === "all" ||
            Number(row.dropoff_zone_id) === Number(destination)),
      )
      .sort((a, b) => b.trip_count - a.trip_count);
    const limit = state.summary.meta.max_flows,
      shown = rows.slice(0, limit);
    byId("od-table").innerHTML =
      notice(
        `Showing the top ${number(shown.length)} of ${number(rows.length)} matching zone pairs; the map and table have the same ${number(limit)}-pair limit.`,
      ) +
      table(
        shown.map((row) => ({
          ...row,
          origin: zoneName(row.pickup_zone_id),
          destination: zoneName(row.dropoff_zone_id),
        })),
        [
          ["origin", "Origin"],
          ["destination", "Destination"],
          ["trip_count", "Trips"],
          ["mean_duration_minutes", "Mean minutes"],
        ],
        limit,
      );
    if (!shown.length) {
      emptyPlot("flow-map", "No recorded zone pairs match these filters.");
      return;
    }
    const geometry = await load("zones.geojson", true);
    if (
      !byId("flow-map") ||
      request !== state.request ||
      localRequest !== drawRequest
    )
      return;
    const traces = [];
    if (geometry)
      traces.push({
        type: "choroplethmap",
        geojson: geometry,
        locations: state.summary.zones.map((zone) => zone.zone_id),
        featureidkey: "properties.zone_id",
        z: state.summary.zones.map(() => 1),
        colorscale: [
          [0, "#e6ecf6"],
          [1, "#e6ecf6"],
        ],
        showscale: false,
        marker: { opacity: 0.6, line: { width: 0.3, color: "#fff" } },
        hoverinfo: "skip",
      });
    const largest = Math.max(1, ...shown.map((row) => row.trip_count));
    for (const row of shown) {
      const a = state.summary.zones.find(
          (zone) => zone.zone_id === row.pickup_zone_id,
        ),
        b = state.summary.zones.find(
          (zone) => zone.zone_id === row.dropoff_zone_id,
        );
      if (!a || !b || a.latitude == null || b.latitude == null) continue;
      traces.push({
        type: "scattermap",
        lat: [a.latitude, b.latitude],
        lon: [a.longitude, b.longitude],
        mode: "lines",
        line: {
          width: 0.8 + 3.5 * Math.sqrt(row.trip_count / largest),
          color: blue,
        },
        opacity: 0.45,
        hoverinfo: "skip",
        showlegend: false,
      });
    }
    plot("flow-map", traces, {
      height: 470,
      map: {
        style: "white-bg",
        center: { lat: 40.74, lon: -73.96 },
        zoom: 9.5,
      },
      margin: { l: 0, r: 0, t: 0, b: 0 },
    });
  }
  byId("od-origin").onchange = draw;
  byId("od-destination").onchange = draw;
  await draw();
}

async function anomalies(request) {
  const data = await load("anomalies.json");
  if (request !== state.request) return;
  byId("view-controls").innerHTML =
    `<div class="wide-control"><label for="alert-zone">Zone · exported alerts only</label><select id="alert-zone">${zoneOptions()}</select></div>`;
  byId("page-content").innerHTML =
    notice(
      "Alerts are statistical deviations, not verified incidents, fraud, congestion, or causal explanations. Synthetic evaluation is not real-world detection accuracy.",
    ) +
    notice(
      `This public snapshot exports the ${number(data.alert_limit)} largest alert hours and at most ${number(data.incident_limit)} merged incidents from the full period. Zone filtering selects within that bounded export; it does not rerank all underlying records.`,
    ) +
    panel(
      "Scoring coverage",
      table(
        data.status_counts,
        [
          ["status", "Status"],
          ["zone_hours", "Zone-hours"],
        ],
        20,
      ),
      `Total threshold alert hours in the source result: ${number(data.total_alert_hours)}. Total merged incidents: ${number(data.total_incidents)}.`,
    ) +
    panel(
      "Observed versus expected",
      `<label for="alert-choice">Inspect an exported alert</label><select id="alert-choice"></select>${chartDiv("alert-context")}`,
      "Observed pickups and the seasonal median, up to 48 hours on either side. The score is not a probability or a p-value.",
    ) +
    panel("Largest exported alert hours", `<div id="alert-table"></div>`) +
    panel(
      "Merged incidents",
      `<div id="incident-table"></div>`,
      "Adjacent alerts are merged using the saved detector policy. Labels remain unresolved unless independently reviewed.",
    );
  function drawContext() {
    const id = byId("alert-choice").value,
      context = data.contexts[id];
    if (!context?.length) {
      emptyPlot(
        "alert-context",
        "No exported alert context. Insufficient history is not evidence of normal activity.",
      );
      return;
    }
    plot(
      "alert-context",
      [
        {
          type: "scatter",
          mode: "lines",
          x: context.map((row) => row.hour),
          y: context.map((row) => row.observed),
          name: "Observed pickups",
          line: { color: blue, width: 2.5 },
          connectgaps: false,
        },
        {
          type: "scatter",
          mode: "lines",
          x: context.map((row) => row.hour),
          y: context.map((row) => row.baseline),
          name: "Seasonal median",
          line: { color: teal, width: 2, dash: "dash" },
          connectgaps: false,
        },
        {
          type: "scatter",
          mode: "markers",
          x: context.filter((row) => row.is_alert).map((row) => row.hour),
          y: context.filter((row) => row.is_alert).map((row) => row.observed),
          name: "Statistical alert",
          marker: { symbol: "diamond", size: 8, color: "#b91c1c" },
        },
      ],
      {
        hovermode: "x unified",
        xaxis: { title: { text: "NYC source local time" } },
        yaxis: { title: { text: "Recorded pickups" }, rangemode: "tozero" },
      },
    );
  }
  function draw() {
    const zone = byId("alert-zone").value,
      alerts = data.alerts.filter(
        (row) => zone === "all" || Number(row.zone_id) === Number(zone),
      );
    byId("alert-choice").innerHTML = alerts.length
      ? alerts
          .map(
            (row) =>
              `<option value="${row.context_id}">${escapeHtml(zoneName(row.zone_id))} · ${escapeHtml(row.hour.slice(0, 16).replace("T", " "))} · score ${number(row.score, 2)} · ${escapeHtml(row.split)}</option>`,
          )
          .join("")
      : `<option value="">No exported alerts in this selection</option>`;
    byId("alert-choice").disabled = !alerts.length;
    byId("alert-table").innerHTML = table(
      alerts.map((row) => ({ ...row, zone: zoneName(row.zone_id) })),
      [
        ["zone", "Zone"],
        ["hour", "Local hour"],
        ["split", "Split"],
        ["observed", "Observed"],
        ["baseline", "Baseline"],
        ["score", "Score"],
        ["threshold", "Threshold"],
        ["history_count", "History"],
      ],
      data.alert_limit,
    );
    const incidents = data.incidents
      .filter((row) => zone === "all" || Number(row.zone_id) === Number(zone))
      .map((row) => ({ ...row, zone: zoneName(row.zone_id) }));
    byId("incident-table").innerHTML = table(
      incidents,
      [
        ["zone", "Zone"],
        ["start_hour", "Start"],
        ["end_hour", "End"],
        ["direction", "Direction"],
        ["affected_hours", "Hours"],
        ["peak_score", "Peak score"],
        ["label", "Review label"],
      ],
      data.incident_limit,
    );
    drawContext();
  }
  byId("alert-zone").onchange = draw;
  byId("alert-choice").onchange = drawContext;
  draw();
}

async function renderPerformance(request) {
  const data = await load("performance.json");
  if (request !== state.request) return;
  if (!data.trials.length) {
    byId("view-controls").innerHTML = "";
    byId("page-content").innerHTML =
      notice(
        "No benchmark measurements have been published. CPU/GPU comparisons will appear only after matching workloads pass correctness checks and complete timed runs.",
      ) +
      panel(
        "What will be measured",
        `<p>Loaded computation, custom CUDA kernel execution, and complete local-file-to-result processing are separate timing scopes.</p><p>Comparisons use matching rows, precision, features, and outputs on the same host. Repetitions, failures, software versions, and hardware conditions remain part of the result.</p><p>A CPU win is a valid result. No speedup is assumed.</p>`,
      );
    return;
  }
  const scopes = [...new Set(data.trials.map((row) => row.timing_scope))],
    backends = [...new Set(data.trials.map((row) => row.backend))];
  byId("view-controls").innerHTML =
    `<div class="compact-control"><label for="trial-scope">Timing scope</label><select id="trial-scope"><option value="all">All scopes</option>${scopes.map((scope) => `<option>${escapeHtml(scope)}</option>`).join("")}</select></div><div class="compact-control"><label for="trial-backend">Backend</label><select id="trial-backend"><option value="all">All backends</option>${backends.map((backend) => `<option>${escapeHtml(backend)}</option>`).join("")}</select></div>`;
  byId("page-content").innerHTML =
    notice(
      `Published trials: ${number(data.total_trials)}. This static export includes at most ${number(data.display_limit)} raw trials. Date and zone filters do not apply; compare matching experiment, input size, stage, and timing scope.`,
    ) +
    panel(
      "Measured elapsed time",
      chartDiv("trial-chart"),
      "Every point is a recorded correctness-validated trial. Different workloads and timing scopes should not be interpreted as a single speedup.",
    ) +
    panel(
      "Matched CPU/GPU comparisons",
      `<div id="speedup-table"></div>`,
      "Speedup uses medians over all published validated trials, not just the bounded raw-trial display. Matching experiment, workload, and timing scope are required. A value below 1 means the CPU was faster.",
    ) +
    panel("Raw trials", `<div id="trial-table"></div>`) +
    panel(
      "Benchmark environment",
      `<pre class="code-block">${escapeHtml(JSON.stringify(data.environment || {}, null, 2))}</pre>`,
      "The hardware and software used for these measurements. An empty object means no benchmark environment record was published; the snapshot-build machine is not a substitute.",
    );
  function draw() {
    const scope = byId("trial-scope").value,
      backend = byId("trial-backend").value,
      rows = data.trials.filter(
        (row) =>
          (scope === "all" || row.timing_scope === scope) &&
          (backend === "all" || row.backend === backend),
      );
    const valid = rows.filter(
      (row) =>
        Number.isFinite(row.elapsed_s) &&
        row.elapsed_s > 0 &&
        ["passed", "pass", "true", "ok"].includes(
          String(row.validation).toLowerCase(),
        ),
    );
    const traces = [...groupRows(valid, "backend")].map(([key, values]) => ({
      type: "scatter",
      mode: "markers",
      x: values.map(
        (row) =>
          `${row.stage} · ${row.timing_scope} · ${number(row.input_rows)} rows`,
      ),
      y: values.map((row) => row.elapsed_s),
      name: key.toUpperCase(),
      text: values.map((row) => `Repeat ${row.repeat} · ${row.experiment_id}`),
      marker: {
        color:
          key === "cpu"
            ? blue
            : ["gpu", "gpu_library"].includes(key)
              ? teal
              : "#9a3412",
        size: 8,
      },
      hovertemplate:
        "%{x}<br>%{y:.5f} seconds<br>%{text}<extra>%{fullData.name}</extra>",
    }));
    if (traces.length)
      plot("trial-chart", traces, {
        height: 400,
        yaxis: { title: { text: "Elapsed seconds" }, rangemode: "tozero" },
        xaxis: { automargin: true },
      });
    else
      emptyPlot(
        "trial-chart",
        "No correctness-validated timings in this selection.",
      );
    const comparisons = (data.comparisons || []).filter(
      (row) =>
        (scope === "all" || row.timing_scope === scope) &&
        (backend === "all" || backend === "cpu" || row.gpu_backend === backend),
    );
    byId("speedup-table").innerHTML = comparisons.length
      ? table(comparisons, [
          ["stage", "Stage"],
          ["timing_scope", "Scope"],
          ["input_rows", "Rows"],
          ["gpu_backend", "GPU backend"],
          ["cpu_seconds", "CPU seconds"],
          ["gpu_seconds", "GPU seconds"],
          ["speedup", "CPU/GPU speedup"],
        ])
      : notice(
          "No matching validated CPU and GPU measurements are available in this selection.",
        );
    byId("trial-table").innerHTML = table(
      rows,
      [
        ["experiment_id", "Experiment"],
        ["repeat", "Repeat"],
        ["backend", "Backend"],
        ["stage", "Stage"],
        ["timing_scope", "Scope"],
        ["input_rows", "Rows"],
        ["elapsed_s", "Seconds"],
        ["validation", "Validation"],
      ],
      200,
    );
  }
  byId("trial-scope").onchange = draw;
  byId("trial-backend").onchange = draw;
  draw();
}

async function methods(request) {
  const data = await load("methods.json");
  if (request !== state.request) return;
  byId("view-controls").innerHTML = "";
  byId("page-content").innerHTML =
    panel(
      "About this public snapshot",
      `<p>This is a static, read-only publication of validated analytical results. Daily Overview filters run in your browser. The full Python/Dash application supports richer hourly queries and is separate from this public export.</p><p>GPU processing, model fitting, source acquisition, and benchmarking happen offline. Viewing this website does not start a training job or expose the processing server.</p><p>The export includes daily zone aggregates, the full-period zone-pair matrix, at most ${number(state.summary.meta.max_points)} sampled feature pairs, bounded anomaly summaries, measured benchmark trials, and provenance. It does not include raw source files or individual record identifiers.</p>`,
    ) +
    panel(
      "Interpretation and limitations",
      `<p>TLC data describes recorded taxi and ride-hailing activity, not all citywide travel demand. Zone connections do not reveal streets taken or intermediate passenger positions.</p><p>Pickups and drop-offs are independently timed. OD connections are anchored to pickup time. Missing coverage is not zero activity, and boundary drop-offs may have partial source coverage.</p><p>Timestamps use the declared New York source-local convention. Repeated and nonexistent daylight-saving hours cannot be treated as ordinary hourly observations.</p><p>KMeans labels describe groups in the selected feature space. Statistical alerts do not establish real-world incidents or causality. Synthetic tests measure response to constructed perturbations.</p><p>Map boundaries and source records: NYC Taxi and Limousine Commission. This project is not an official NYC service.</p>`,
    ) +
    panel(
      "Frozen source records",
      table(
        data.sources,
        [
          ["source_id", "Source"],
          ["nominal_month", "Month"],
          ["byte_count", "Bytes"],
          ["parquet_row_count", "Rows"],
          ["sha256", "SHA-256"],
          ["url", "Official URL"],
        ],
        100,
      ),
      "Checksums identify the actual source files used. Monthly publication does not establish complete real-world reporting.",
    ) +
    panel(
      "Data quality",
      table(
        data.quality,
        [
          ["source_id", "Source"],
          ["rule", "Rule"],
          ["affected_rows", "Affected rows"],
          ["total_rows", "Source rows"],
          ["action", "Action"],
        ],
        200,
      ),
      "Quality rules overlap. Do not sum their counts as though they partition all records.",
    ) +
    panel(
      "Model status",
      `<pre class="code-block">${escapeHtml(JSON.stringify(data.models, null, 2))}</pre>`,
    ) +
    panel(
      "Validation and export",
      `<pre class="code-block">${escapeHtml(JSON.stringify({ snapshot_id: state.summary.meta.snapshot_id, data_kind: state.summary.meta.data_kind, study_start: state.summary.meta.study_start, study_end_exclusive: state.summary.meta.study_end, validation: data.validation, capabilities: state.summary.meta.capabilities }, null, 2))}</pre><p><a href="export-manifest.json">Download the export manifest and data checksums</a></p>`,
    );
}

async function navigate(page) {
  state.page = page;
  const request = ++state.request;
  document.querySelectorAll("[data-page]").forEach((button) => {
    button.setAttribute(
      "aria-current",
      button.dataset.page === page ? "page" : "false",
    );
  });
  byId("page-title").textContent = pages[page][0];
  byId("page-subtitle").textContent = pages[page][1];
  byId("filters").hidden = page !== "overview";
  byId("view-controls").innerHTML = "";
  byId("page-content").innerHTML = notice("Loading the published view…");
  try {
    if (page === "overview") await overview(request);
    else
      await { patterns, zones, anomalies, performance: renderPerformance, methods }[page](request);
  } catch (error) {
    if (request === state.request) {
      byId("page-content").innerHTML = notice(
        `${error.message} Select another view or reload this page. Snapshot ${state.summary.meta.snapshot_id}.`,
        "error",
      );
      console.error(error);
    }
  }
}

async function start() {
  try {
    state.summary = await load("summary.json");
    const meta = state.summary.meta;
    const last = new Date(`${dateOnly(meta.study_end)}T00:00:00Z`);
    last.setUTCDate(last.getUTCDate() - 1);
    state.filters = {
      start: dateOnly(meta.study_start),
      end: last.toISOString().slice(0, 10),
      zone: "all",
    };
    const defaults = { ...state.filters };
    byId("snapshot-id").textContent = meta.snapshot_id;
    byId("data-kind").textContent =
      meta.data_kind === "real"
        ? "Recorded TLC data"
        : "Synthetic demonstration";
    if (meta.data_kind !== "real") {
      byId("data-kind").classList.add("warning");
      byId("source-notice").innerHTML = notice(
        "Synthetic demonstration data — these are software test results, not actual NYC findings.",
        "warning",
      );
    }
    byId("period-pill").textContent = `${defaults.start} – ${defaults.end}`;
    for (const id of ["start-date", "end-date"]) {
      byId(id).min = defaults.start;
      byId(id).max = defaults.end;
    }
    byId("start-date").value = defaults.start;
    byId("end-date").value = defaults.end;
    byId("zone-select").innerHTML = zoneOptions();
    byId("apply").onclick = () => {
      const start = byId("start-date").value,
        end = byId("end-date").value;
      if (
        !start ||
        !end ||
        start > end ||
        start < defaults.start ||
        end > defaults.end
      ) {
        byId("filter-message").textContent =
          "Choose a valid date range within the published period.";
        return;
      }
      state.filters = { start, end, zone: byId("zone-select").value };
      byId("filter-message").textContent =
        `Applied ${start} through ${end} · ${state.filters.zone === "all" ? "all zones" : zoneName(state.filters.zone)}.`;
      navigate("overview");
    };
    byId("reset").onclick = () => {
      state.filters = { ...defaults };
      byId("start-date").value = defaults.start;
      byId("end-date").value = defaults.end;
      byId("zone-select").value = "all";
      byId("filter-message").textContent =
        "All published dates and zones selected.";
      navigate("overview");
    };
    document
      .querySelectorAll("[data-page]")
      .forEach(
        (button) => (button.onclick = () => navigate(button.dataset.page)),
      );
    await navigate("overview");
  } catch (error) {
    byId("page-content").innerHTML = notice(
      `The published snapshot could not be loaded. ${error.message} Serve this folder over HTTP rather than opening index.html directly.`,
      "error",
    );
    console.error(error);
  }
}
start();
