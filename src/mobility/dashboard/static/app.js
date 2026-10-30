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
    "Compare pickup volume and trip duration with their historical baselines.",
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

function alertDefinition(metric, meta, data) {
  const duration = metric === "duration";
  const model = meta.models?.[duration ? "duration_anomalies" : "anomalies"] || {};
  const parameters = duration ? model.duration_anomalies_config || {} :
    model.anomalies_config || meta.anomaly_parameters || model.parameters || {};
  return {
    duration,
    model,
    unit: duration ? "minutes" : "pickups",
    observedLabel: duration ? "Mean duration (minutes)" : "Observed pickups",
    baselineLabel: duration ? "Baseline minutes" : "Baseline pickups",
    scoreThreshold: model.threshold ?? data.alerts?.[0]?.threshold ?? parameters.score_threshold ?? 4,
    absoluteThreshold: duration ? parameters.absolute_threshold_minutes ?? 5 : parameters.absolute_threshold ?? 20,
    scaleFloor: duration ? parameters.scale_floor_minutes ?? 2 : parameters.scale_floor ?? 5,
    minHistory: parameters.min_history ?? 8,
    minTrips: duration ? parameters.min_trips_per_hour ?? 20 : null,
    splits: model.splits || meta.splits || modelSplits(),
  };
}

function backendLabel(value) {
  return typeof value === "string" && value && value !== "unknown" ? value.toUpperCase() : "Not published";
}

function processingDescription(meta) {
  const processing = meta.processing || {};
  const native = (meta.execution_mode || processing.execution_mode) === "native";
  const plan = meta.execution_plan || processing.execution_plan || {};
  const explanation = native ?
    "Native execution uses the recorded backend for bulk normalization, eligibility, features, sampling, scaling, aggregation, clustering, both anomaly detectors and graph algorithms. The CPU implementation uses CPU libraries throughout. The GPU implementation uses cuDF/CuPy, cuML and cuGraph for those stages; host work handles file checksums and metadata, small calendar/zone inputs, job control and reduced result finalization. Independent CPU reference checks occur after timed GPU processing." :
    "Historical execution uses a hybrid GPU path: CUDA/CuPy features, cuDF aggregations, cuML clustering and cuGraph algorithms, while source preparation, model sampling/scaling and anomaly calculations use CPU work. Its GPU graph validation includes a CPU reference inside the pipeline. This scope differs from the newer native implementation.";
  return `<p>This snapshot contains <strong>${number(meta.raw_rows)} raw source records</strong>. Saved result backends: <strong>aggregations ${escapeHtml(backendLabel(processing.aggregation_backend))}</strong>, <strong>trip clustering ${escapeHtml(backendLabel(processing.clustering_backend))}</strong>, <strong>pickup-volume anomalies ${escapeHtml(backendLabel(processing.anomaly_backend))}</strong>, <strong>duration anomalies ${escapeHtml(backendLabel(processing.duration_anomaly_backend))}</strong>, and <strong>graph analysis ${escapeHtml(backendLabel(processing.graph_backend))}</strong>.</p><p>${explanation} Viewing this website displays saved results and starts no GPU jobs.</p>${Object.keys(plan).length ? `<details class="technical-details"><summary>Recorded stage backends and row counts</summary><pre class="code-block">${escapeHtml(JSON.stringify(plan, null, 2))}</pre></details>` : ""}`;
}

function modelPreparationDescription(meta) {
  const model = meta.models?.clustering || {};
  const native = (meta.execution_mode || meta.processing?.execution_mode) === "native";
  const method = native ?
    "Both native backends use the same numeric SplitMix64 source-row priority to select bounded samples and seeded initial centers. Features and training-only scaler statistics use float64; the standardized KMeans matrix uses float32." :
    "This historical model uses bounded deterministic samples and its saved initialization policy; it predates the native numeric sampling policy.";
  return `${method} Recorded sampling policy: ${model.sampling_policy || "not published"}; initialization policy: ${model.initialization_policy || "not published"}.`;
}

function pipelineExecutionDescription(rows, environment = {}) {
  const modes = new Set(rows.map((row) => row.execution_mode ||
    (row.stage === "native_production_pipeline" ? "native" : environment.execution_mode || "legacy")));
  const descriptions = [];
  if (modes.has("native")) descriptions.push("Native trials compare an entirely CPU analysis with GPU bulk processing, including normalization, sampling/scaling and anomaly scoring. Independent CPU reference checks, including graph validation, run after timing.");
  if (modes.has("legacy")) descriptions.push("Historical hybrid trials retain CPU preparation, sampling/scaling and anomaly work; the GPU graph's CPU reference is included in their elapsed time. These timings do not measure the newer native implementation.");
  return descriptions.join(" ");
}

function communityPalette(nodes) {
  const labels = [...new Set(nodes.filter((row) => !row.is_isolated && row.community != null).map((row) => Number(row.community)))].sort((a, b) => a - b);
  return new Map(labels.map((label, index) => [label, colors[index] || `hsl(${(index * 137.508) % 360}, 65%, 42%)`]));
}

function validatedMeasured(rows) {
  return rows.filter((row) => row.phase === "measured" && Number.isFinite(row.elapsed_s) && row.elapsed_s > 0 &&
    ["passed", "pass", "true", "ok"].includes(String(row.validation).toLowerCase()));
}

function pipelineMedians(rows) {
  const groups = new Map();
  for (const row of validatedMeasured(rows)) {
    const key = JSON.stringify([row.experiment_id, row.protocol_hash, row.stage, row.timing_scope, row.input_rows, row.backend]);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return [...groups.values()].map((values) => {
    const times = values.map((row) => row.elapsed_s).sort((a, b) => a - b);
    const middle = Math.floor(times.length / 2);
    return { ...values[0], elapsed_s: times.length % 2 ? times[middle] : (times[middle - 1] + times[middle]) / 2,
      repetitions: times.length, minimum_s: times[0], maximum_s: times[times.length - 1] };
  });
}

function memoryGiB(value) {
  return Number.isFinite(value) && value >= 0 ? number(value / 2 ** 30, 3) : "Unavailable";
}

function pipelineQualityRow(row) {
  const result = { ...row };
  for (const key of ["training_inertia_per_row", "calibration_silhouette", "test_silhouette", "test_mean_squared_center_distance"])
    result[key] = Number.isFinite(row[key]) ? number(row[key], 6) : "Unavailable";
  for (const key of ["selection_status", "test_status", "diagnostics_status"])
    result[key] = String(row[key] || "not_recorded").replaceAll("_", " ");
  return result;
}

function cpuEvidenceDescription(evidence = {}) {
  if (!evidence.measured_native_cpu_trials) return "";
  return `Native CPU isolation: ${number(evidence.guard_and_no_cuda_imports_confirmed)} of ${number(evidence.measured_native_cpu_trials)} measured CPU trials recorded an active CUDA import guard and no imported CUDA libraries. GPU process memory was sampled as zero in ${number(evidence.zero_sampled_gpu_memory_trials)} trials, nonzero in ${number(evidence.nonzero_sampled_gpu_memory_trials)}, and unavailable in ${number(evidence.gpu_memory_unavailable_trials)}. Sampled zero is evidence from the recorded observations, not continuous memory monitoring.`;
}

function pipelineResourceRow(row) {
  const background = row.background_gpu_processes ?? row.gpu_background_processes;
  return {
    ...row,
    elapsed_s: number(row.elapsed_s, 6),
    host_gib: memoryGiB(row.peak_host_rss_sampled_bytes),
    gpu_gib: memoryGiB(row.peak_gpu_memory_sampled_bytes),
    rss_interval_ms: Number.isFinite(row.rss_sample_interval_s) ? number(row.rss_sample_interval_s * 1000, 1) : "Unavailable",
    gpu_interval_ms: Number.isFinite(row.gpu_sample_interval_s) ? number(row.gpu_sample_interval_s * 1000, 1) : "Unavailable",
    background: background == null ? "Not recorded" : typeof background === "boolean" ? (background ? "Observed" : "None observed") : Array.isArray(background) ? `${background.length} observed` : escapeHtml(background),
    months: row.months ?? row.month_count,
  };
}

function environmentDetails(environment) {
  const gpu = environment?.gpu?.gpu || environment?.gpu?.name || "Not recorded";
  return `<p>GPU: <strong>${escapeHtml(gpu)}</strong>. Host RAM: <strong>${memoryGiB(environment?.host_memory_bytes)} GiB</strong>. CPU cores: <strong>${number(environment?.cpu_physical_cores)} physical / ${number(environment?.cpu_logical_cores)} logical</strong>. Configured CPU threads: <strong>${number(environment?.config?.cpu_threads)}</strong>.</p><details class="technical-details"><summary>Hardware, software and experiment settings</summary><pre class="code-block">${escapeHtml(JSON.stringify(environment || {}, null, 2))}</pre></details>`;
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
      `KMeans groups trips by log(1 + distance), log(1 + duration), and pickup hour encoded around a 24-hour circle, so 23:00 and 00:00 are close. Features are standardized using training data only. ${status === "calibration_selected" ? `${clusterCount ? `${clusterCount} groups were` : "The group count was"} selected on calibration data (${periodLabel(modelSplits(), "calibration")}) after fitting on training data (${periodLabel(modelSplits(), "train")}).` : "See model status for the fitting and selection periods."} Zone IDs are not clustering inputs. Group numbers are arbitrary labels, not neighborhoods or known passenger purposes. Counts and means describe eligible trips in the scoring period. ${modelPreparationDescription(state.summary.meta)}`,
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

async function renderGraph(data, request) {
  if (!data?.nodes?.length || !byId("graph-section")) {
    if (byId("graph-section")) byId("graph-section").innerHTML = panel("Zone network analysis", notice("Graph results have not been published for this snapshot."));
    return;
  }
  const metadata = data.metadata || {};
  const rows = [...data.nodes].sort((a, b) => b.pagerank - a.pagerank);
  const palette = communityPalette(rows);
  byId("graph-section").innerHTML =
    panel("Zone network analysis",
      notice(`Fixed full-period graph: ${number(metadata.mapped_zone_count ?? rows.length)} mapped zones, ${number(metadata.directed_edge_count)} directed cross-zone pairs, ${number(metadata.retained_trip_count)} retained trips. Backend: ${backendLabel(metadata.backend)}. Origin, destination and top-pair controls above do not change these results.`) +
      `<p>Graph validation: <strong>${escapeHtml((metadata.validation?.status || "not recorded").replaceAll("_", " "))}</strong>${metadata.validation?.reference ? ` · ${escapeHtml(metadata.validation.reference)}` : ""}.</p>` +
      `<p><strong>PageRank</strong> measures a zone's incoming network connectivity, taking the strengths of its incoming connections into account. Edge weights are recorded trip counts; the scores sum to one across all mapped zones. PageRank is not a trip count.</p><p><strong>Louvain communities</strong> group zones with strong weighted connections. For this calculation, A → B and B → A are added into one undirected pair. Colors and group numbers are categories, not a ranking, and the groups need not match borough boundaries.</p><div class="graph-controls"><label for="graph-view">Color the zone map by</label><select id="graph-view"><option value="community">Louvain community</option><option value="pagerank">PageRank connectivity</option></select></div>${chartDiv("graph-map")}<div id="graph-legend" class="community-legend" aria-label="Community colors"></div>`,
      `Directed pairs require at least ${number(metadata.parameters?.min_trips_per_edge ?? 1)} trips after aggregation. ${number(metadata.self_trip_count)} within-zone trips are reported separately and excluded from both graph algorithms. ${number(metadata.isolated_zone_count)} zones have no retained cross-zone edge; they remain included. This small zone graph does not imply a GPU speed advantage.`) +
    panel("Zone connectivity and communities", table(rows.map((row) => ({
      ...row, zone: row.zone || zoneName(row.zone_id), pagerank: number(row.pagerank, 6),
      community_label: row.is_isolated ? "Isolated" : `Group ${Number(row.community) + 1}`,
    })), [
      ["zone", "Zone"], ["borough", "Borough"], ["pagerank", "PageRank"], ["community_label", "Community"],
      ["in_degree", "Origins linked in"], ["out_degree", "Destinations linked out"],
      ["in_trip_strength", "Incoming cross-zone trips"], ["out_trip_strength", "Outgoing cross-zone trips"],
      ["self_trip_count", "Within-zone trips"],
    ], 50) + `<p class="artifact-links"><a href="data/graph.json">Download all mapped zones and graph definitions</a></p>`,
      `The top ${Math.min(50, rows.length)} of ${number(rows.length)} mapped zones by PageRank; the map includes all zones. In/out degree counts distinct connected zones; strength counts trips. Weighted undirected modularity: ${number(metadata.modularity, 3)}. Community labels can differ across CPU and GPU runs.`);
  const geometry = await load("zones.geojson", true);
  if (request !== state.request || !byId("graph-view")) return;
  function drawGraph() {
    const community = byId("graph-view").value === "community";
    const anchors = new Map(state.summary.zones.map((row) => [Number(row.zone_id), row]));
    const groups = community ? [...groupRows(rows, "community")].map(([label, values]) => ({
      name: values.every((row) => row.is_isolated) ? "Isolated zone" : `Group ${Number(label) + 1}`,
      values, color: values.every((row) => row.is_isolated) ? "#8793a4" : palette.get(Number(label)),
    })) : [{ name: "PageRank", values: rows }];
    const traces = groups.map((group) => {
      const values = geometry ? group.values : group.values.filter((row) => {
        const point = anchors.get(Number(row.zone_id));
        return Number.isFinite(point?.longitude) && Number.isFinite(point?.latitude);
      });
      const hover = values.map((row) => [escapeHtml(row.zone || zoneName(row.zone_id)), row.pagerank,
        row.in_trip_strength, row.out_trip_strength, row.is_isolated ? "Isolated" : `Group ${Number(row.community) + 1}`]);
      const trace = { name: group.name, customdata: hover, showlegend: false,
        hovertemplate: "%{customdata[0]}<br>PageRank %{customdata[1]:.6f}<br>%{customdata[4]}<br>Incoming trips %{customdata[2]:,}<br>Outgoing trips %{customdata[3]:,}<extra></extra>" };
      if (geometry) return { ...trace, type: "choroplethmap", geojson: geometry,
        featureidkey: "properties.zone_id", locations: values.map((row) => row.zone_id),
        z: values.map((row) => community ? 1 : row.pagerank),
        colorscale: community ? [[0, group.color], [1, group.color]] : [[0, "#edf3ff"], [1, blue]],
        showscale: !community, colorbar: { title: { text: "PageRank" }, thickness: 12, len: .7 },
        marker: { opacity: .85, line: { width: .5, color: "white" } } };
      return { ...trace, type: "scattermap", mode: "markers",
        lon: values.map((row) => anchors.get(Number(row.zone_id)).longitude),
        lat: values.map((row) => anchors.get(Number(row.zone_id)).latitude),
        marker: { size: 10, color: community ? group.color : values.map((row) => row.pagerank),
          colorscale: [[0, "#a7c2f4"], [1, blue]], showscale: !community } };
    });
    byId("graph-legend").innerHTML = community ? [...palette].map(([label, color]) =>
      `<span><i style="background:${escapeHtml(color)}"></i>Group ${label + 1}</span>`).join("") +
      (rows.some((row) => row.is_isolated) ? `<span><i style="background:#8793a4"></i>Isolated zones</span>` : "") :
      "Darker blue means a larger weighted PageRank score.";
    plot("graph-map", traces, { height: 470,
      map: { style: "white-bg", center: { lat: 40.74, lon: -73.96 }, zoom: 9.5 },
      margin: { l: 0, r: 0, t: 0, b: 0 }, uirevision: `${state.summary.meta.snapshot_id}-graph` });
  }
  byId("graph-view").onchange = drawGraph;
  drawGraph();
}

async function zones(request) {
  const [data, graphData] = await Promise.all([load("od.json"), load("graph.json", true)]);
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
    ) + `<div id="graph-section"></div>`;
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
  if (request === state.request) await renderGraph(graphData, request);
}

async function anomalies(request) {
  const [volumeData, durationData] = await Promise.all([
    load("anomalies.json"), load("duration-anomalies.json", true),
  ]);
  if (request !== state.request) return;
  const hasDuration = Boolean(state.summary.meta.models?.duration_anomalies || durationData?.status_counts?.length);
  byId("view-controls").innerHTML =
    `<div class="compact-control"><label for="alert-metric">Metric</label><select id="alert-metric"><option value="volume">Pickup volume</option><option value="duration"${hasDuration ? "" : " disabled"}>Travel duration${hasDuration ? "" : " · not published"}</option></select></div><div class="wide-control"><label for="alert-zone">Zone · exported alerts only</label><select id="alert-zone">${zoneOptions()}</select></div>`;
  const splitLabel = (split) => ({ calibration: "Calibration", test: "Test", train: "Training" })[split] || split;
  function renderMetric() {
    const metric = byId("alert-metric").value;
    const data = metric === "duration" ? durationData : volumeData;
    if (!data) {
      byId("page-content").innerHTML = notice("This metric has not been published for this snapshot.");
      return;
    }
    const definition = alertDefinition(metric, state.summary.meta, data);
    const { duration, unit, splits, scoreThreshold, absoluteThreshold, scaleFloor, minHistory, minTrips } = definition;
    const observedMeaning = duration ?
      "the mean eligible-trip duration, in minutes, for trips starting in one pickup zone during one hour" :
      "the number of recorded pickups in one zone during one hour";
    const baselineMeaning = duration ?
      "that zone's median of sufficiently populated hourly mean durations at the same weekday and hour during training. Each supported historical hour has equal weight" :
      "that zone's median pickup count at the same weekday and hour during training";
    const supportNote = duration ?
      `<p>Both training and scored hours need at least <strong>${number(minTrips)} eligible trips</strong>. Hours with fewer trips, missing coverage or insufficient history are gaps, not zero-minute trips. The eligible trip filters are the fixed study limits recorded in the model.</p><p>Longer mean duration can reflect a change in destinations or trip lengths. This detector does not adjust for that trip mix, so it does not establish congestion, slower traffic or passenger delay.</p>` :
      "<p>Pickup volume measures completed recorded trips. It does not measure people unable to find a taxi or all travel demand.</p>";
    byId("page-content").innerHTML =
      notice("Alerts are statistical deviations, not verified incidents, fraud, congestion or causal explanations. Unreviewed means a person has not yet checked the explanation.") +
      notice(`This metric exports the ${number(data.alert_limit)} largest alert hours and at most ${number(data.incident_limit)} merged incidents. Zone filtering selects within that bounded export; it does not rerank every source hour.`) +
      panel("How to read an alert",
        `<p><strong>Training (${escapeHtml(periodLabel(splits, "train"))})</strong> establishes the baseline. <strong>Calibration (${escapeHtml(periodLabel(splits, "calibration"))})</strong> selects the threshold under an alert-volume budget. <strong>Test (${escapeHtml(periodLabel(splits, "test"))})</strong> uses the frozen detector on later hours. That alert budget is not a false-positive probability.</p><p><strong>Observed</strong> is ${observedMeaning}. <strong>Baseline</strong> is ${baselineMeaning}. <strong>History</strong> counts matching observed training hours, not trips or previous alerts.</p><p><code>score = (observed − baseline) / max(1.4826 × MAD, ${number(scaleFloor)})</code></p><p>MAD is the median absolute difference of historical ${duration ? "hourly means" : "counts"} from their median. The scale floor is <strong>${number(scaleFloor)} ${unit}</strong>. Positive means ${duration ? "longer duration" : "more pickups"}; negative means ${duration ? "shorter duration" : "fewer pickups"}. An alert needs <strong>|score| ≥ ${number(scoreThreshold)}</strong>, a change of at least <strong>${number(absoluteThreshold)} ${unit}</strong>, and at least <strong>${number(minHistory)} matching historical hours</strong>. The score is not a percentage, probability or cause.</p>${supportNote}`) +
      panel("Scoring coverage", table(data.status_counts, [["status", "Status"], ["zone_hours", "Zone-hours"]], 20),
        `Threshold alert hours: ${number(data.total_alert_hours)}. Merged incidents: ${number(data.total_incidents)}. Missing or unsupported hours are not evidence of normal activity.`) +
      panel("Observed versus expected", `<label for="alert-choice">Inspect an exported alert</label><select id="alert-choice"></select>${chartDiv("alert-context")}`,
        `${duration ? "Mean eligible-trip duration in minutes" : "Recorded pickup count"} and the historical median, up to 48 hours on either side. Missing and unsupported observations remain gaps.`) +
      panel("Largest exported alert hours", `<div id="alert-table"></div>`) +
      panel("Merged incidents", `<div id="incident-table"></div>`,
        "Adjacent alert hours are grouped using the saved detector policy. Unreviewed is the display name for the saved unresolved label; computation has finished, but no real-world event has been confirmed.");
    function drawContext() {
      const context = data.contexts[byId("alert-choice").value];
      if (!context?.length) {
        emptyPlot("alert-context", "No exported alert context for this selection.");
        return;
      }
      plot("alert-context", [
        { type: "scatter", mode: "lines", x: context.map((row) => row.hour), y: context.map((row) => row.observed),
          name: definition.observedLabel, line: { color: blue, width: 2.5 }, connectgaps: false },
        { type: "scatter", mode: "lines", x: context.map((row) => row.hour), y: context.map((row) => row.baseline),
          name: "Historical median", line: { color: teal, width: 2, dash: "dash" }, connectgaps: false },
        { type: "scatter", mode: "markers", x: context.filter((row) => row.is_alert).map((row) => row.hour),
          y: context.filter((row) => row.is_alert).map((row) => row.observed), name: "Statistical alert",
          marker: { symbol: "diamond", size: 8, color: "#b91c1c" } },
      ], { hovermode: "x unified", xaxis: { title: { text: "NYC source local time" } },
        yaxis: { title: { text: duration ? "Mean trip duration (minutes)" : "Recorded pickups" }, rangemode: "tozero" } });
    }
    function draw() {
      const zone = byId("alert-zone").value;
      const alerts = data.alerts.filter((row) => zone === "all" || Number(row.zone_id) === Number(zone));
      byId("alert-choice").innerHTML = alerts.length ? alerts.map((row) =>
        `<option value="${escapeHtml(row.context_id)}">${escapeHtml(zoneName(row.zone_id))} · ${escapeHtml(row.hour.slice(0, 16).replace("T", " "))} · score ${number(row.score, 2)} · ${escapeHtml(splitLabel(row.split))}</option>`).join("") :
        `<option value="">No exported alerts in this selection</option>`;
      byId("alert-choice").disabled = !alerts.length;
      byId("alert-table").innerHTML = table(alerts.map((row) => ({ ...row, zone: zoneName(row.zone_id), split: splitLabel(row.split) })), [
        ["zone", "Zone"], ["hour", "Local hour"], ["split", "Split"],
        ["observed", definition.observedLabel], ["baseline", definition.baselineLabel],
        ...(duration ? [["eligible_trip_count", "Eligible trips in hour"]] : []),
        ["score", "Score"], ["threshold", "Threshold"], ["history_count", "Matching historical hours"],
      ], data.alert_limit);
      const incidents = data.incidents.filter((row) => zone === "all" || Number(row.zone_id) === Number(zone)).map((row) =>
        ({ ...row, zone: zoneName(row.zone_id), label: row.label === "unresolved" ? "Unreviewed" : row.label }));
      byId("incident-table").innerHTML = table(incidents, [
        ["zone", "Zone"], ["start_hour", "Start"], ["end_hour", "End"], ["direction", "Direction"],
        ["affected_hours", "Hours"], ["peak_score", "Peak score"], ["label", "Review label"],
      ], data.incident_limit);
      drawContext();
    }
    byId("alert-zone").onchange = draw;
    byId("alert-choice").onchange = drawContext;
    draw();
  }
  byId("alert-metric").onchange = renderMetric;
  renderMetric();
}

function renderPipelinePerformance(data) {
  const target = byId("pipeline-section");
  if (!target) return;
  if (!data?.trials?.length) {
    target.innerHTML = panel("Full pipeline scaling", notice("No complete-pipeline CPU/GPU experiment has been published in this snapshot. The stage measurements below cover narrower work."));
    return;
  }
  const rows = validatedMeasured(data.trials), points = pipelineMedians(data.trials);
  const measured = data.trials.filter((row) => row.phase === "measured");
  const warmups = data.trials.filter((row) => row.phase === "warmup");
  const comparisons = data.comparisons || [];
  const cpuEvidence = cpuEvidenceDescription(data.cpu_execution_evidence);
  const monthByRows = new Map(measured.map((row) => [Number(row.input_rows), row.months ?? row.month_count]));
  target.innerHTML =
    panel("Full pipeline scaling",
      notice(`${number(rows.length)} validated measured runs; ${number(warmups.length)} warmups and ${number(measured.length - rows.length)} unvalidated measured runs excluded from the timing chart. Stage timing/backend selectors do not change these full-pipeline panels.`) + chartDiv("pipeline-chart"),
      `Starts with local raw TLC files and runs checks, cleaning, features, aggregations, clustering, both anomaly detectors, graph analysis and artifact writes. ${pipelineExecutionDescription(data.trials, data.environment)} Downloads, separate benchmark validation and website publication are excluded. Six months adds calibration/test work absent from one/three-month runs, so scaling reflects more records and more analysis. Points are measured medians; whiskers show the observed minimum and maximum.`) +
    panel("Matched full pipeline CPU/GPU comparisons", comparisons.length ? table(comparisons.map((row) => ({
      ...comparisonDisplay(row), months: monthByRows.get(Number(row.input_rows)),
      protocol: row.protocol_hash ? String(row.protocol_hash).slice(0, 12) : "Not recorded",
    })), [
      ["months", "Months"], ["input_rows", "Raw rows"], ["cpu_seconds", "CPU seconds"],
      ["gpu_seconds", "GPU seconds"], ["time_ratio", "CPU time / GPU time"], ["faster", "Faster backend"],
      ["experiment_id", "Experiment"], ["protocol", "Protocol ID"],
    ]) : notice("No CPU/GPU pair has passed matching-workload validation for this experiment yet."),
      `Each ratio pairs the same experiment, input rows, stage, timing scope and protocol. Warmups are excluded. A ratio below one means the CPU finished faster. These pipeline comparisons are separate from the older stage experiments below. ${cpuEvidence}`) +
    panel("Clustering quality by measured trial", data.clustering_quality?.length ? table(data.clustering_quality.map(pipelineQualityRow), [
      ["months", "Months"], ["backend", "Backend"], ["repeat", "Repeat"], ["selected_k", "Selected groups"],
      ["training_rows", "Training rows"], ["training_inertia_per_row", "Training inertia / row"],
      ["calibration_rows", "Calibration rows"], ["calibration_silhouette", "Calibration silhouette"],
      ["test_rows", "Test rows"], ["test_silhouette", "Test silhouette"],
      ["test_mean_squared_center_distance", "Test mean squared center distance"],
      ["selection_status", "Selection status"], ["test_status", "Test status"],
      ["validation", "Trial validation"], ["diagnostics_status", "Diagnostics"],
    ], 200) : notice("Clustering-quality diagnostics were not published for these historical trial records."),
      "Every measured trial is shown, including failures; warmups are excluded. January and January–March contain no April calibration or May–June test data. Training inertia and test center distance are squared distances in standardized feature space, not miles or minutes; lower is more compact for comparable feature inputs and group counts. Higher silhouette indicates better geometric separation, not verified passenger categories. Matching inputs and passed output checks do not guarantee identical fitted clusters or equal model quality. Missing metrics are unavailable, not zero.") +
    panel("Full pipeline sampled resources", table(measured.map(pipelineResourceRow), [
      ["months", "Months"], ["backend", "Backend"], ["repeat", "Repeat"], ["input_rows", "Raw rows"],
      ["elapsed_s", "Seconds"], ["host_gib", "Sampled host peak (GiB)"], ["gpu_gib", "Sampled GPU peak (GiB)"],
      ["rss_interval_ms", "Requested host interval (ms)"], ["gpu_interval_ms", "Requested GPU interval (ms)"],
      ["rss_samples", "Host samples"], ["gpu_samples", "GPU samples"], ["background", "Other GPU activity"],
      ["validation", "Validation"],
    ], 200),
      "These are the largest observed samples, not guaranteed peak memory. Host RSS covers the monitored process tree; GPU memory is sampled for the trial's processes. Short spikes, missing process readings and incomplete sampling can understate usage. Unavailable is not zero. Other GPU activity can affect timings. GiB means bytes divided by 1,073,741,824.") +
    panel("Full pipeline experiment conditions", environmentDetails(data.environment), "These conditions belong to the pipeline experiment. Stage experiments below retain their own environment and configuration.") +
    `<p class="muted artifact-links"><a href="data/pipeline-performance.json">Download full pipeline trials and matching comparisons</a></p>`;
  if (!points.length) {
    emptyPlot("pipeline-chart", "No validated measured full-pipeline timings are available.");
    return;
  }
  const series = new Map();
  for (const point of points) {
    const key = JSON.stringify([point.experiment_id, point.backend, point.stage, point.timing_scope]);
    if (!series.has(key)) series.set(key, []);
    series.get(key).push(point);
  }
  const multipleExperiments = new Set(points.map((point) => point.experiment_id)).size > 1;
  const traces = [...series.values()].map((values) => {
    values.sort((a, b) => a.input_rows - b.input_rows);
    const duplicateSizes = new Set(values.map((row) => row.input_rows)).size !== values.length;
    return { type: "scatter", mode: duplicateSizes ? "markers" : "lines+markers",
      x: values.map((row) => row.input_rows), y: values.map((row) => row.elapsed_s),
      name: `${backendLabel(values[0].backend)}${multipleExperiments ? ` · ${values[0].experiment_id}` : ""}`,
      customdata: values.map((row) => [row.months ?? row.month_count, row.repetitions, row.minimum_s, row.maximum_s, row.protocol_hash]),
      marker: { size: 9, color: values[0].backend === "cpu" ? blue : teal },
      line: { color: values[0].backend === "cpu" ? blue : teal, width: 2 },
      error_y: { type: "data", symmetric: false,
        array: values.map((row) => row.maximum_s - row.elapsed_s),
        arrayminus: values.map((row) => row.elapsed_s - row.minimum_s) },
      hovertemplate: "%{x:,} raw rows · %{customdata[0]} months<br>Median %{y:.3f} s · %{customdata[1]} measured runs<br>Observed range %{customdata[2]:.3f}–%{customdata[3]:.3f} s<extra>%{fullData.name}</extra>" };
  });
  plot("pipeline-chart", traces, { height: 400,
    xaxis: { title: { text: "Raw source records processed" }, tickformat: ".2s", rangemode: "tozero" },
    yaxis: { title: { text: "Full pipeline elapsed seconds" }, rangemode: "tozero" } });
}

async function renderPerformance(request) {
  const [data, pipelineData] = await Promise.all([load("performance.json"), load("pipeline-performance.json", true)]);
  if (request !== state.request) return;
  if (!data.trials.length) {
    byId("view-controls").innerHTML = "";
    byId("page-content").innerHTML =
      panel("Where the GPU was used", processingDescription(state.summary.meta)) +
      `<div id="pipeline-section"></div>` +
      panel("Stage measurements", notice("No separate stage benchmark measurements have been published."));
    renderPipelinePerformance(pipelineData);
    return;
  }
  const scopes = [...new Set(data.trials.map((row) => row.timing_scope))],
    backends = [...new Set(data.trials.map((row) => row.backend))];
  byId("view-controls").innerHTML =
    `<div class="compact-control"><label for="trial-scope">Stage timing scope</label><select id="trial-scope"><option value="all">All scopes</option>${scopes.map((scope) => `<option>${escapeHtml(scope)}</option>`).join("")}</select></div><div class="compact-control"><label for="trial-backend">Stage backend</label><select id="trial-backend"><option value="all">All backends</option>${backends.map((backend) => `<option>${escapeHtml(backend)}</option>`).join("")}</select></div>`;
  byId("page-content").innerHTML =
    notice(
      `Published stage trials: ${number(data.total_trials)}. Date and zone filters do not apply. Full pipeline experiments below have their own records and scope; compare matching experiment, input size and protocol.`,
    ) +
    panel(
      "Where the GPU was used",
      processingDescription(state.summary.meta),
    ) + `<div id="pipeline-section"></div>` +
    panel(
      "Stage elapsed time",
      chartDiv("trial-chart"),
      "These historical stage experiments retain their original configurations. Every point is a recorded correctness-validated trial. A loaded kernel or aggregation timing is not a full-pipeline speedup.",
    ) +
    panel(
      "Matched stage CPU/GPU comparisons",
      `<div id="speedup-table"></div>`,
      "CPU time / GPU time is a ratio of median seconds across all validated trials. It is always positive: above 1 means the GPU was faster; below 1 means the CPU was faster; 1 means equal times. The faster-backend column expresses the advantage for the winner (for example, a ratio of 0.5 means CPU faster by 2×). Seconds have six decimal places; original measurements remain in data/performance.json.",
    ) +
    panel("Raw stage trials", `<div id="trial-table"></div>`) +
    panel(
      "Stage benchmark environment",
      environmentDetails(data.environment),
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
  renderPipelinePerformance(pipelineData);
}

async function methods(request) {
  const [data, pipelineData, graphData] = await Promise.all([
    load("methods.json"), load("pipeline-performance.json", true), load("graph.json", true),
  ]);
  if (request !== state.request) return;
  const meta = state.summary.meta;
  const hasDuration = Boolean(meta.models?.duration_anomalies);
  const hasGraph = Boolean(graphData?.nodes?.length);
  const comparisons = pipelineData?.comparisons || [];
  const maximumComparedRows = Math.max(0, ...comparisons.map((row) => Number(row.input_rows) || 0));
  const pipelineScope = comparisons.length ?
    `Published matching CPU/GPU pipeline comparisons reach ${number(maximumComparedRows)} raw records${meta.raw_rows && maximumComparedRows >= meta.raw_rows ? ", including the full published input size" : ""}. They measure local raw-file processing through models and artifact writes; source downloads and website publication are excluded. Resource figures are sampled observations, not exact hardware high-water marks.` :
    "This snapshot has no published matching full-pipeline CPU/GPU comparisons. Its stage experiments cover narrower operations and cannot establish whole-system speedup.";
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
      "Implemented scope and remaining limitations",
      processingDescription(meta) +
      `<p>The system acquires TLC records, prepares features, aggregates activity, groups trips and detects unusual pickup volume. ${hasDuration ? "The duration detector also compares hourly mean eligible-trip minutes with a frozen historical baseline. Changing destinations and trip lengths can affect that mean; duration alerts alone do not prove congestion." : "Travel-duration anomaly results have not been published in this snapshot."}</p><p>${hasGraph ? "Zone network analysis includes weighted directed PageRank and weighted undirected Louvain communities. The map separates this fixed-period graph analysis from filtered origin–destination lines. Community numbers are labels, not ranked quality." : "Graph algorithm results have not been published in this snapshot; the origin–destination lines alone do not establish graph analysis."}</p><p>${escapeHtml(pipelineScope)}</p><p>Completed taxi trips indicate recorded activity, not unmet travel demand. Alerts remain unreviewed until checked against independent evidence. The small zone graph need not run faster on a GPU, and individual stage wins do not guarantee a faster complete pipeline.</p>${meta.models?.clustering?.eligibility ? "<p>The fixed trip plausibility limits were chosen after inspecting full-period chart outliers. Revised May–June clustering and duration results are retrospective analyses; an additional untouched period is needed for independent confirmation.</p>" : ""}`,
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
      table(Object.entries(data.models || {}).map(([name, model]) => ({
        model: name.replaceAll("_", " "), status: model.status,
        backend: model.backend || model.computation_backend || "Not recorded", metric: model.metric || "See definition",
      })), [["model", "Analysis"], ["status", "Status"], ["backend", "Backend"], ["metric", "Metric"]]) +
      `<details class="technical-details"><summary>Model parameters and evaluation details</summary><pre class="code-block">${escapeHtml(JSON.stringify(data.models, null, 2))}</pre></details>`,
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
