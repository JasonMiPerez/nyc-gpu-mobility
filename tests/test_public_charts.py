"""Run the browser's actual view calculations against boundary examples."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_public_sample_views_and_timing_ratios():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to exercise the published JavaScript")
    script = Path(__file__).parents[1] / "src/mobility/dashboard/static/app.js"
    checks = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({ fetch: () => new Promise(() => {}) });
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const rows = [
  {trip_distance_miles: 0, duration_seconds: 60},
  {trip_distance_miles: 1.735, duration_seconds: 744.5},
  {trip_distance_miles: 50, duration_seconds: 7200},
  {trip_distance_miles: 50.1, duration_seconds: 60},
  {trip_distance_miles: 1, duration_seconds: 7201},
  {trip_distance_miles: 100125.14, duration_seconds: 780},
  {trip_distance_miles: null, duration_seconds: 60},
  {trip_distance_miles: 3, duration_seconds: -1},
];
const typical = context.sampleSelection(rows, 'typical');
assert.equal(typical.valid.length, 6);
assert.equal(typical.shown.length, 3);
assert.equal(typical.outside, 3);
assert.equal(typical.invalid, 2);
assert.equal(typical.shown[0].trip_distance_miles, 0);
for (const mode of ['full', 'log']) {
  const full = context.sampleSelection(rows, mode);
  assert.equal(full.shown.length, 6);
  assert.equal(full.outside, 0);
  assert.equal(full.shown[5].trip_distance_miles, 100125.14);
}
const bins = context.sampleDistanceBins(typical.valid);
assert.equal(bins.counts.reduce((sum, n) => sum + n, 0), 6);
assert.equal(bins.counts[0], 1); // Zero remains counted.
assert.equal(bins.counts[6], 1); // 50 belongs to 20–50.
assert.equal(bins.counts[7], 2); // Both extreme distances remain in overflow.
const logAxis = context.logAxis([0, 1, 100125.14], 'Miles');
assert.equal(logAxis.tickvals[0], 0);
assert.equal(logAxis.ticktext[0], '0');
assert(logAxis.tickvals.every(Number.isFinite));
const cpuWin = context.comparisonDisplay({cpu_seconds: .004186, gpu_seconds: .004763});
assert.equal(cpuWin.cpu_seconds, '0.004186');
assert.equal(cpuWin.gpu_seconds, '0.004763');
assert.equal(cpuWin.time_ratio, '0.879×');
assert.equal(cpuWin.faster, 'CPU faster · 1.14×');
const gpuWin = context.comparisonDisplay({cpu_seconds: 2, gpu_seconds: 1});
assert.equal(gpuWin.time_ratio, '2.000×');
assert.equal(gpuWin.faster, 'GPU faster · 2.00×');
assert.equal(context.comparisonDisplay({cpu_seconds: 0, gpu_seconds: 1}).faster, 'Unavailable');
"""
    subprocess.run([node, "-e", checks, str(script)], check=True, capture_output=True, text=True)
