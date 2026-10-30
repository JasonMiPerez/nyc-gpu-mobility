"""Exercise new public metric, graph and pipeline views using their actual JS."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_public_extended_views_and_calculations():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to exercise the published JavaScript")
    script = Path(__file__).parents[1] / "src/mobility/dashboard/static/app.js"
    checks = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map(), plots = [];
function element(id, tag = 'div') {
  if (elements.has(id)) return elements.get(id);
  const result = { id, tag, style: {}, value: '', disabled: false, _html: '' };
  Object.defineProperty(result, 'innerHTML', {
    get() { return this._html; },
    set(html) {
      this._html = html;
      for (const match of html.matchAll(/<([\w-]+)\b[^>]*\bid="([^"]+)"[^>]*>/g)) {
        element(match[2], match[1]);
      }
      for (const match of html.matchAll(/<select\b[^>]*\bid="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
        elements.get(match[1]).innerHTML = match[2];
      }
      if (this.tag === 'select') this.value = html.match(/<option\b[^>]*\bvalue="([^"]*)"/)?.[1] || '';
    },
  });
  elements.set(id, result);
  return result;
}
for (const id of ['view-controls', 'page-content', 'graph-section', 'pipeline-section']) element(id);
const context = vm.createContext({
  fetch: () => new Promise(() => {}), console,
  document: { getElementById: (id) => elements.get(id) },
  Plotly: { newPlot: (target, traces, layout) => { plots.push({ id: target.id, traces, layout }); } },
});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const splits = { train_start: '2024-01-01', train_end: '2024-04-01',
  calibration_start: '2024-04-01', calibration_end: '2024-05-01',
  test_start: '2024-05-01', test_end: '2024-07-01' };
context.summary = { meta: { snapshot_id: 'test', raw_rows: 100, splits,
  models: { anomalies: { threshold: 4, splits }, duration_anomalies: { threshold: 6, splits,
    duration_anomalies_config: { min_trips_per_hour: 20, scale_floor_minutes: 2, absolute_threshold_minutes: 5 } } },
  processing: { aggregation_backend: 'gpu', clustering_backend: 'gpu', anomaly_backend: 'cpu',
    duration_anomaly_backend: 'cpu', graph_backend: 'gpu' } },
  zones: [{ zone_id: 1, zone: 'North', borough: 'Test', longitude: -74, latitude: 40.7 },
    { zone_id: 2, zone: 'South', borough: 'Test', longitude: -73.9, latitude: 40.8 }] };
vm.runInContext('state.summary = summary; state.request = 7;', context);
const volume = context.alertDefinition('volume', context.summary.meta, { alerts: [] });
const duration = context.alertDefinition('duration', context.summary.meta, { alerts: [] });
assert.equal(volume.unit, 'pickups');
assert.equal(volume.scaleFloor, 5);
assert.equal(volume.absoluteThreshold, 20);
assert.equal(duration.unit, 'minutes');
assert.equal(duration.scoreThreshold, 6);
assert.equal(duration.minTrips, 20);
assert.equal(duration.scaleFloor, 2);
assert.equal(duration.absoluteThreshold, 5);
const base = { phase: 'measured', validation: 'passed', experiment_id: 'run', protocol_hash: 'A',
  stage: 'production_pipeline', timing_scope: 'raw_files', input_rows: 100, months: 1, backend: 'cpu' };
const trials = [{ ...base, elapsed_s: 20 }, { ...base, elapsed_s: 30 },
  { ...base, phase: 'warmup', elapsed_s: 1 }, { ...base, validation: 'failed', elapsed_s: 2 },
  { ...base, protocol_hash: 'B', elapsed_s: 500 }, { ...base, backend: 'gpu', elapsed_s: 10 }];
const medians = context.pipelineMedians(trials);
assert.equal(medians.length, 3); // Different protocol cannot be pooled into A.
assert.equal(medians.find((row) => row.protocol_hash === 'A' && row.backend === 'cpu').elapsed_s, 25);
assert.equal(medians.find((row) => row.protocol_hash === 'B').elapsed_s, 500);
assert.equal(context.validatedMeasured(trials).length, 4);
assert.equal(context.memoryGiB(null), 'Unavailable');
assert.equal(context.memoryGiB(0), '0.000');
assert.equal(context.memoryGiB(2 ** 30), '1.000');
const resources = context.pipelineResourceRow({ peak_host_rss_sampled_bytes: 2 ** 30,
  peak_gpu_memory_sampled_bytes: null, rss_sample_interval_s: .01, background_gpu_processes: false });
assert.equal(resources.host_gib, '1.000');
assert.equal(resources.gpu_gib, 'Unavailable');
assert.equal(resources.rss_interval_ms, '10.0');
assert.equal(resources.background, 'None observed');
assert.equal(context.pipelineResourceRow({}).background, 'Not recorded');
const palette = context.communityPalette([{ community: 9 }, { community: 2 }, { community: 0, is_isolated: true }]);
assert.equal(palette.size, 2);
assert.notEqual(palette.get(2), palette.get(9));
assert.equal(palette.get(9), context.communityPalette([{ community: 2 }, { community: 9 }]).get(9));
const description = context.processingDescription(context.summary.meta);
assert(description.includes('aggregations GPU'));
assert(description.includes('graph analysis GPU'));
assert(!description.includes('built on the CPU'));
assert(context.processingDescription({}).includes('Not published'));
context.volumeData = { alerts: [{ zone_id: 1, context_id: '0', hour: '2024-06-01T00:00:00', split: 'test',
  observed: 50, baseline: 10, score: 8, threshold: 4, history_count: 12 }],
  incidents: [], contexts: { '0': [{ hour: '2024-06-01T00:00:00', observed: 50, baseline: 10, is_alert: true }] },
  status_counts: [{ status: 'scored', zone_hours: 1 }], alert_limit: 100, incident_limit: 200, total_alert_hours: 1, total_incidents: 0 };
context.durationData = { ...context.volumeData,
  alerts: [{ ...context.volumeData.alerts[0], observed: 30, baseline: 10, eligible_trip_count: 25 }],
  contexts: { '0': [{ hour: '2024-05-31T23:00:00', observed: null, baseline: null, is_alert: false },
    { hour: '2024-06-01T00:00:00', observed: 30, baseline: 10, is_alert: true }] } };
vm.runInContext(`state.cache.set('anomalies.json', Promise.resolve(volumeData));
  state.cache.set('duration-anomalies.json', Promise.resolve(durationData));
  state.cache.set('zones.geojson', Promise.resolve(null));`, context);
(async () => {
  await context.anomalies(7);
  assert(elements.has('alert-zone') && elements.has('alert-choice') && elements.has('alert-metric'));
  elements.get('alert-metric').value = 'duration';
  elements.get('alert-metric').onchange();
  assert(elements.get('page-content').innerHTML.includes('20 eligible trips'));
  assert(elements.get('page-content').innerHTML.includes('2 minutes'));
  assert(elements.get('page-content').innerHTML.includes('5 minutes'));
  assert(elements.get('alert-table').innerHTML.includes('Eligible trips in hour'));
  const durationPlot = plots.filter((plot) => plot.id === 'alert-context').at(-1);
  assert.equal(durationPlot.traces[0].y[0], null); // Never substitute zero for missing duration.
  assert.equal(durationPlot.layout.yaxis.title.text, 'Mean trip duration (minutes)');
  await context.renderGraph({ nodes: [
    { zone_id: 1, zone: 'North', pagerank: .8, community: 4, in_trip_strength: 8, out_trip_strength: 2 },
    { zone_id: 2, zone: 'South', pagerank: .2, community: 5, is_isolated: true, in_trip_strength: 0, out_trip_strength: 0 },
  ], metadata: { backend: 'gpu', mapped_zone_count: 2, directed_edge_count: 1,
    retained_trip_count: 8, self_trip_count: 2, isolated_zone_count: 1, modularity: .25 } }, 7);
  assert(elements.get('graph-section').innerHTML.includes('do not change these results'));
  const communityPlot = plots.filter((plot) => plot.id === 'graph-map').at(-1);
  assert(communityPlot.traces.every((trace) => typeof trace.marker.color === 'string'));
  elements.get('graph-view').value = 'pagerank';
  elements.get('graph-view').onchange();
  assert.equal(plots.filter((plot) => plot.id === 'graph-map').at(-1).traces.length, 1);
  context.renderPipelinePerformance({ trials, comparisons: [], environment: {} });
  assert(elements.get('pipeline-section').innerHTML.includes('not guaranteed peak memory'));
  assert(elements.get('pipeline-section').innerHTML.includes('No CPU/GPU pair has passed'));
  assert(plots.some((plot) => plot.id === 'pipeline-chart'));
})().catch((error) => { console.error(error); process.exitCode = 1; });
"""
    subprocess.run([node, "-e", checks, str(script)], check=True, capture_output=True, text=True)
