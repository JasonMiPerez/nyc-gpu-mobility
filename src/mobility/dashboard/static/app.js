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

// View limits change the picture only; the published records remain available.
function sampleSelection(rows, mode) {
  const valid = rows.filter(
    (row) =>
      Number.isFinite(row.trip_distance_miles) &&
      row.trip_distance_miles >= 0 &&
      Number.isFinite(row.duration_seconds) &&
      row.duration_seconds >= 0,
  );
  const shown = mode === "typical"
    ? valid.filter((row) => row.trip_distance_miles <= 50 && row.duration_seconds <= 7200)
    : valid;
  return { valid, shown, outside: valid.length - shown.length, invalid: rows.length - valid.length };
}

function sampleDistanceBins(rows) {
  const edges = [1, 2, 3, 5, 10, 20, 50];
  const labels = ["0–<1", "1–<2", "2–<3", "3–<5", "5–<10", "10–<20", "20–50", ">50"];
  const counts = labels.map(() => 0);
  for (const row of rows) {
    const distance = row.trip_distance_miles;
    if (!Number.isFinite(distance) || distance < 0) continue;
    const index = edges.findIndex((edge, i) => i === edges.length - 1 ? distance <= edge : distance < edge);
    counts[index === -1 ? edges.length : index] += 1;
  }
  return { labels, counts };
}

function logAxis(values, title) {
  const largest = Math.max(1, ...values);
  const ticks = [0];
  for (let power = 1; power <= largest * 10; power *= 10) {
    for (const factor of [1, 2, 5]) {
      const tick = power * factor;
      if (tick <= largest) ticks.push(tick);
    }
  }
  return {
    title: { text: `${title} · log(1 + value) spacing` },
    tickvals: ticks.map(Math.log1p),
    ticktext: ticks.map((value) => number(value)),
    rangemode: "tozero",
  };
}

function comparisonDisplay(row) {
  const ratio = row.cpu_seconds > 0 && row.gpu_seconds > 0
    ? row.cpu_seconds / row.gpu_seconds
    : null;
  return {
    ...row,
    cpu_seconds: number(row.cpu_seconds, 6),
    gpu_seconds: number(row.gpu_seconds, 6),
    time_ratio: ratio == null ? "—" : `${number(ratio, 3)}×`,
    faster: ratio == null ? "Unavailable" : ratio === 1 ? "Equal measured times" :
      `${ratio > 1 ? "GPU" : "CPU"} faster · ${number(ratio > 1 ? ratio : 1 / ratio, 2)}×`,
  };
}

function periodLabel(splits, prefix) {
  const start = new Date(`${dateOnly(splits[`${prefix}_start`])}T00:00:00Z`);
  const end = new Date(`${dateOnly(splits[`${prefix}_end`])}T00:00:00Z`);
  if (!Number.isFinite(start.getTime()) || !Number.isFinite(end.getTime())) return "See model metadata";
  end.setUTCDate(end.getUTCDate() - 1);
  const format = { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" };
  return `${start.toLocaleDateString("en-US", format)} – ${end.toLocaleDateString("en-US", format)}`;
}

function modelSplits() {
  return state.summary.meta.splits || state.summary.meta.models?.anomalies?.splits || {
    train_start: "2024-01-01", train_end: "2024-04-01",
    calibration_start: "2024-04-01", calibration_end: "2024-05-01",
    test_start: "2024-05-01", test_end: "2024-07-01",
  };
}

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
    ["Mapped pickups", number(pickups), "Recorded trip starts in known zones"],
    ["Mapped drop-offs", number(dropoffs), "Recorded trip ends in known zones"],
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
      "One pickup is the start of one recorded taxi trip; one drop-off is its end. These count trips, not passengers. ‘Mapped’ means the endpoint has a valid time and a known taxi zone. Each total uses its own endpoint time and the selected dates and zones, so the two totals need not match.",
    ) +
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
    `<div class="compact-control"><label for="sample-cluster">Sample cluster</label><select id="sample-cluster"><option value="all">All clusters</option>${clusters.map((value) => `<option value="${escapeHtml(value)}">Cluster ${escapeHtml(value)}</option>`).join("")}</select></div><div class="wide-control"><label for="sample-view">Scatterplot view</label><select id="sample-view"><option value="typical">Typical trip view · 0–50 mi, 0–120 min</option><option value="full">Full range · linear axes</option><option value="log">Full range · log(1 + value) axes</option></select></div><span class="muted">All dates and zones in this fixed snapshot</span>`;
  const status = state.summary.meta.models?.clustering?.status;
  const clusterCount = state.summary.meta.models?.clustering?.selected_k;
  const eligibility = state.summary.meta.models?.clustering?.eligibility;
  byId("page-content").innerHTML =
    notice(
      "Patterns and cluster summaries describe the full published period. The cluster selector filters only the sampled plots; it does not change the full-population cluster table.",
    ) +
    (eligibility
      ? notice(
          `Clustering uses ${number(eligibility.model_eligible_rows)} eligible trips. Model-only limits: at most ${number(eligibility.thresholds.max_distance_miles)} miles, ${number(eligibility.thresholds.max_duration_minutes)} minutes, and ${number(eligibility.thresholds.max_mean_speed_mph)} mph mean speed. ${number(eligibility.excluded_from_canonical_eligible_rows)} otherwise eligible trips are excluded from clustering under these study limits. Their valid pickup/drop-off counts remain in the activity totals. These limits do not certify that every retained trip is correct.`,
        )
      : "") +
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
      `KMeans groups trips by log(1 + distance), log(1 + duration), and pickup hour encoded around a 24-hour circle, so 23:00 and 00:00 are close. Features are standardized using training data only. ${status === "calibration_selected" ? `${clusterCount ? `${clusterCount} groups were` : "The group count was"} selected on calibration data (${periodLabel(modelSplits(), "calibration")}) after fitting on training data (${periodLabel(modelSplits(), "train")}).` : "See model status for the fitting and selection periods."} Zone IDs are not clustering inputs. Group numbers are arbitrary labels, not neighborhoods or known passenger purposes. Counts and means describe eligible trips in the scoring period.`,
    ) +
    panel(
      "Distance and duration",
      `<div id="sample-view-summary"></div>${chartDiv("sample-scatter")}`,
      `Each point is one sampled trip, colored by its cluster. At most ${number(state.summary.meta.max_points)} feature pairs are published. The default view makes ordinary distances readable; choose a full-range view to inspect outliers. Log spacing retains zero values and shows original units in hover labels.`,
    ) +
    panel(
      "Sample distance distribution",
      chartDiv("sample-histogram"),
      "All valid trips in the selected sample cluster are counted here, including trips outside the scatterplot view. Fixed distance bands and an overflow bar prevent a single extreme distance from stretching the chart. These are sample counts, not population totals.",
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
      ),
      mode = byId("sample-view").value,
      selection = sampleSelection(rows, mode),
      useLog = mode === "log";
    byId("sample-view-summary").innerHTML = notice(
      `${mode === "typical" ? "Typical trip view: 0–50 miles and 0–120 minutes" : useLog ? "Full range with log(1 + value) spacing" : "Full range with linear axes"}. Showing ${number(selection.shown.length)} of ${number(selection.valid.length)} valid selected sample trips; ${number(selection.outside)} are outside this view. ${number(selection.invalid)} selected rows have invalid coordinates. The distance histogram includes all ${number(selection.valid.length)} valid selected trips.`,
    );
    if (!selection.valid.length) {
      emptyPlot("sample-scatter", "No sampled trips for this selection.");
      emptyPlot("sample-histogram", "No sampled trips for this selection.");
      return;
    }
    const traces = [...groupRows(selection.shown, "cluster")].map(
      ([cluster, values], i) => ({
        type: "scattergl",
        mode: "markers",
        x: values.map((row) => useLog ? Math.log1p(row.trip_distance_miles) : row.trip_distance_miles),
        y: values.map((row) => useLog ? Math.log1p(row.duration_seconds / 60) : row.duration_seconds / 60),
        customdata: values.map((row) => [row.trip_distance_miles, row.duration_seconds / 60]),
        name: `Cluster ${cluster}`,
        marker: {
          color:
            colors[Number(cluster) % colors.length] ||
            colors[i % colors.length],
          size: 5,
          opacity: 0.5,
        },
        hovertemplate:
          "Distance %{customdata[0]:,.2f} mi<br>Duration %{customdata[1]:,.1f} min<extra>%{fullData.name}</extra>",
      }),
    );
    plot("sample-scatter", traces, {
      height: 400,
      uirevision: `${state.summary.meta.snapshot_id}-${mode}-${selected}`,
      xaxis: useLog ? logAxis(selection.shown.map((row) => row.trip_distance_miles), "Trip distance (miles)") : {
        title: { text: "Trip distance (miles)" },
        rangemode: "tozero",
        ...(mode === "typical" ? { range: [0, 50] } : {}),
      },
      yaxis: useLog ? logAxis(selection.shown.map((row) => row.duration_seconds / 60), "Trip duration (minutes)") : {
        title: { text: "Trip duration (minutes)" },
        rangemode: "tozero",
        ...(mode === "typical" ? { range: [0, 120] } : {}),
      },
    });
    const bins = sampleDistanceBins(selection.valid);
    plot(
      "sample-histogram",
      [
        {
          type: "bar",
          x: bins.labels,
          y: bins.counts,
          marker: { color: bins.labels.map((label, index) => index === bins.labels.length - 1 ? "#9a3412" : blue) },
          hovertemplate: "%{x} miles<br>%{y:,} sampled trips<extra></extra>",
        },
      ],
      {
        xaxis: { title: { text: "Trip distance bands (miles) · >50 is overflow" }, type: "category" },
        yaxis: {
          title: { text: "Selected sample trips" },
          rangemode: "tozero",
        },
      },
    );
  }
  byId("sample-cluster").onchange = drawSample;
  byId("sample-view").onchange = drawSample;
  drawSample();
}

async function zones(request) {
  const data = await load("od.json");
  if (request !== state.request) return;
  const pairCap = state.summary.meta.max_flows || 200;
  const pairLimits = [25, 50, 100, 200].filter((limit) => limit <= pairCap);
  if (!pairLimits.length) pairLimits.push(pairCap);
  byId("view-controls").innerHTML =
    `<div class="wide-control"><label for="od-origin">Origin zone</label><select id="od-origin">${zoneOptions("All origins")}</select></div><div class="wide-control"><label for="od-destination">Destination zone</label><select id="od-destination">${zoneOptions("All destinations")}</select></div><div class="compact-control"><label for="od-limit">Busiest pairs to show</label><select id="od-limit">${pairLimits.map((limit) => `<option value="${limit}"${limit === Math.min(50, pairCap) ? " selected" : ""}>Top ${limit}</option>`).join("")}</select></div>`;
  byId("page-content").innerHTML =
    notice(
      "With all origins and destinations selected, this view ranks ordered zone pairs by recorded trip count over the entire published period. A → B and B → A are separate pairs. The busiest 50 are selected by default. Date filtering is unavailable in this public snapshot; the full Dash application supports date-specific queries.",
    ) +
    panel(
      "Recorded zone connections",
      `<div id="od-summary"></div><p class="map-legend"><span><i class="origin-dot"></i> Pickup endpoint</span><span><i class="destination-dot"></i> Drop-off endpoint</span><span>Thicker line = more trips</span></p>${chartDiv("flow-map")}`,
      "Straight lines connect representative points inside each zone; they are not street routes. Hover an endpoint for the direction and trip count. Within-zone pairs remain in the table but do not form a visible line.",
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
    const limit = Math.min(pairCap, Number(byId("od-limit").value)),
      shown = rows.slice(0, limit);
    const allTrips = sum(data.connections, "trip_count"),
      matchingTrips = sum(rows, "trip_count"),
      shownTrips = sum(shown, "trip_count"),
      withinZone = shown.filter((row) => Number(row.pickup_zone_id) === Number(row.dropoff_zone_id)),
      zonesById = new Map(state.summary.zones.map((zone) => [Number(zone.zone_id), zone])),
      manhattanTrips = sum(shown.filter((row) =>
        zonesById.get(Number(row.pickup_zone_id))?.borough === "Manhattan" &&
        zonesById.get(Number(row.dropoff_zone_id))?.borough === "Manhattan"), "trip_count");
    byId("od-summary").innerHTML = notice(
      `Showing ${number(shown.length)} of ${number(rows.length)} matching ordered pairs, covering ${number(shownTrips)} trips: ${number(allTrips ? 100 * shownTrips / allTrips : 0, 1)}% of all eligible OD trips and ${number(matchingTrips ? 100 * shownTrips / matchingTrips : 0, 1)}% of trips matching these filters. ${number(shownTrips ? 100 * manhattanTrips / shownTrips : 0, 1)}% of shown trips have both endpoints in Manhattan, which explains the concentration when that share is high. ${number(withinZone.length)} shown pairs begin and end in the same zone (${number(sum(withinZone, "trip_count"))} trips); they have no visible connecting line.`,
    );
    byId("od-table").innerHTML =
      notice(
        `The map and table use the same top ${number(shown.length)} zone pairs, ranked by trip count.`,
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
      const a = zonesById.get(Number(row.pickup_zone_id)),
        b = zonesById.get(Number(row.dropoff_zone_id));
      if (!a || !b || a.latitude == null || b.latitude == null) continue;
      const connection = `${escapeHtml(zoneName(row.pickup_zone_id))} → ${escapeHtml(zoneName(row.dropoff_zone_id))}<br>${number(row.trip_count)} recorded trips`;
      traces.push({
        type: "scattermap",
        lat: [a.latitude, b.latitude],
        lon: [a.longitude, b.longitude],
        mode: "lines+markers",
        marker: { size: [8, 6], color: [blue, teal], opacity: 0.9 },
        text: [`Pickup endpoint<br>${connection}`, `Drop-off endpoint<br>${connection}`],
        line: {
          width: 0.8 + 3.5 * Math.sqrt(row.trip_count / largest),
          color: blue,
        },
        opacity: 0.6,
        hovertemplate: "%{text}<extra></extra>",
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
  byId("od-limit").onchange = draw;
  await draw();
}

async function anomalies(request) {
  const data = await load("anomalies.json");
  if (request !== state.request) return;
  const anomalyModel = state.summary.meta.models?.anomalies || {};
  const parameters = anomalyModel.anomalies_config || state.summary.meta.anomaly_parameters || anomalyModel.parameters || {};
  const scoreThreshold = anomalyModel.threshold ?? data.alerts[0]?.threshold ?? parameters.score_threshold ?? 4;
  const countThreshold = parameters.absolute_threshold ?? data.alerts[0]?.absolute_threshold ?? 20;
  const scaleFloor = parameters.scale_floor ?? 5;
  const minHistory = parameters.min_history ?? 8;
  const splits = modelSplits();
  const splitLabel = (split) => ({ calibration: "Calibration", test: "Test", train: "Training" })[split] || split;
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
      "How to read an alert",
      `<p><strong>Training (${escapeHtml(periodLabel(splits, "train"))})</strong> establishes the historical baseline. <strong>Calibration (${escapeHtml(periodLabel(splits, "calibration"))})</strong> is the separate period used to check detector choices. <strong>Test (${escapeHtml(periodLabel(splits, "test"))})</strong> evaluates the frozen detector on later data. A test alert is not automatically a verified event.</p><p><strong>Observed</strong> is the number of recorded pickups in one zone during one hour. <strong>Baseline</strong> is that zone’s median pickup count for the same weekday and hour in the training period. <strong>History</strong> counts the matching training observations used, not the number of past alerts.</p><p><code>score = (observed − baseline) / max(1.4826 × MAD, ${number(scaleFloor)})</code></p><p>MAD is the median absolute deviation of those historical counts from their median. The minimum scale of ${number(scaleFloor)} prevents small baselines from producing unstable scores. A positive score means more pickups than usual; a negative score means fewer. An alert requires both <strong>|score| ≥ ${number(scoreThreshold)}</strong> and an absolute difference of at least <strong>${number(countThreshold)} pickups</strong>, with at least <strong>${number(minHistory)} matching historical observations</strong>. The score is not a percentage, probability, or explanation of the cause.</p>`,
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
      "Adjacent alert hours are grouped into candidate incidents using the saved detector policy. Unreviewed means no independent evidence has confirmed or explained the event; it is the display name for the saved ‘unresolved’ label.",
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
              `<option value="${row.context_id}">${escapeHtml(zoneName(row.zone_id))} · ${escapeHtml(row.hour.slice(0, 16).replace("T", " "))} · score ${number(row.score, 2)} · ${escapeHtml(splitLabel(row.split))}</option>`,
          )
          .join("")
      : `<option value="">No exported alerts in this selection</option>`;
    byId("alert-choice").disabled = !alerts.length;
    byId("alert-table").innerHTML = table(
      alerts.map((row) => ({ ...row, zone: zoneName(row.zone_id), split: splitLabel(row.split) })),
      [
        ["zone", "Zone"],
        ["hour", "Local hour"],
        ["split", "Split"],
        ["observed", "Observed pickups"],
        ["baseline", "Baseline pickups"],
        ["score", "Score"],
        ["threshold", "Threshold"],
        ["history_count", "Matching historical hours"],
      ],
      data.alert_limit,
    );
    const incidents = data.incidents
      .filter((row) => zone === "all" || Number(row.zone_id) === Number(zone))
      .map((row) => ({ ...row, zone: zoneName(row.zone_id), label: row.label === "unresolved" ? "Unreviewed" : row.label }));
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
  const processing = state.summary.meta.processing || {
    aggregation_backend: "cpu", clustering_backend: "cpu", anomaly_backend: "cpu",
  };
  byId("view-controls").innerHTML =
    `<div class="compact-control"><label for="trial-scope">Timing scope</label><select id="trial-scope"><option value="all">All scopes</option>${scopes.map((scope) => `<option>${escapeHtml(scope)}</option>`).join("")}</select></div><div class="compact-control"><label for="trial-backend">Backend</label><select id="trial-backend"><option value="all">All backends</option>${backends.map((backend) => `<option>${escapeHtml(backend)}</option>`).join("")}</select></div>`;
  byId("page-content").innerHTML =
    notice(
      `Published trials: ${number(data.total_trials)}. This static export includes at most ${number(data.display_limit)} raw trials. Date and zone filters do not apply; compare matching experiment, input size, stage, and timing scope.`,
    ) +
    panel(
      "Where the GPU was used",
      `<p>Published snapshot backends: <strong>aggregations ${escapeHtml(processing.aggregation_backend.toUpperCase())}</strong>, <strong>clustering ${escapeHtml(processing.clustering_backend.toUpperCase())}</strong>, and <strong>pickup-volume anomalies ${escapeHtml(processing.anomaly_backend.toUpperCase())}</strong>. These are the six-month January–June results shown on this site.</p><p>A separate January run of about 2.96 million source records used the NVIDIA TITAN RTX server for CUDA trip-feature calculations, cuDF filtering and aggregations, and cuML clustering. That January clustering was exploratory: it had no April calibration data. The matched CPU/GPU timings below were measured on that server.</p><p>Source loading and schema/time validation still use CPU work. The seasonal pickup-volume anomaly detector runs on the CPU. The zone-connection map displays saved origin–destination counts; no cuGraph graph algorithm has been run. Opening this site displays saved results and does not start a GPU job.</p><p>Benchmarks retain the configurations used when they were measured. They have not been rerun with the revised cluster plausibility limits and do not measure a matched CPU/GPU run of the full six-month dataset.</p>`,
    ) +
    panel(
      "Measured elapsed time",
      chartDiv("trial-chart"),
      "Every point is a recorded correctness-validated trial. Different workloads and timing scopes should not be interpreted as a single speedup.",
    ) +
    panel(
      "Matched CPU/GPU comparisons",
      `<div id="speedup-table"></div>`,
      "CPU time / GPU time is a ratio of median seconds across all validated trials. It is always positive: above 1 means the GPU was faster; below 1 means the CPU was faster; 1 means equal times. The faster-backend column expresses the advantage for the winner (for example, a ratio of 0.5 means CPU faster by 2×). Seconds have six decimal places; original measurements remain in data/performance.json.",
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
        "%{x}<br>%{y:.6f} seconds<br>%{text}<extra>%{fullData.name}</extra>",
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
      ? table(comparisons.map(comparisonDisplay), [
          ["experiment_id", "Experiment"],
          ["stage", "Stage"],
          ["timing_scope", "Scope"],
          ["input_rows", "Rows"],
          ["gpu_backend", "GPU backend"],
          ["cpu_seconds", "CPU seconds"],
          ["gpu_seconds", "GPU seconds"],
          ["time_ratio", "CPU time / GPU time"],
          ["faster", "Faster backend"],
        ])
      : notice(
          "No matching validated CPU and GPU measurements are available in this selection.",
        );
    byId("trial-table").innerHTML = table(
      rows.map((row) => ({ ...row, elapsed_s: number(row.elapsed_s, 6) })),
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
      "Implemented scope and remaining project work",
      `<p>The system acquires TLC records, cleans and aggregates them, fits trip clusters, detects unusual pickup counts, and publishes maps, charts, data quality checks, and matched CPU/GPU timings. CUDA, cuDF, and cuML were exercised in a separate January GPU run; the six-month public results were built on the CPU.</p><p>The advanced cuGraph component is not implemented. Origin–destination lines are a visualization, not a graph algorithm. Anomalies currently cover pickup volume only; travel-duration anomalies and separate demand analysis remain to be built. Completed taxi trips are an indicator of recorded activity, not a direct measurement of unmet travel demand.</p><p>Performance experiments compare selected input sizes and operations. A matched CPU/GPU benchmark of all six months, broader scalability testing, and fuller resource measurements remain project work. The historical measurements are not evidence that every pipeline stage is GPU accelerated or faster.</p>${state.summary.meta.models?.clustering?.eligibility ? "<p>The revised clustering plausibility limits were chosen after inspecting full-period chart outliers, rather than optimized for a cluster-quality score. Revised May–June clustering results are a retrospective evaluation; an additional untouched period is needed for independent confirmation.</p>" : ""}`,
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
