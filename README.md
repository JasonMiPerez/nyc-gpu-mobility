# NYC Mobility Observatory

A working system for exploring recorded NYC yellow taxi activity and comparing
CPU, GPU-library, and custom CUDA computation. The website reads validated saved
results; large downloads, training, and GPU jobs run separately.

The first complete study covers January–June 2024: **20,332,093 official TLC source
records**. The current results include six trip clusters selected on April data
and a seasonal anomaly baseline evaluated on May–June. Counts describe recorded
taxi trips, not all NYC travel. Statistical alerts do not establish their cause.

Public website: [NYC Mobility Observatory](https://nyc-mobility-observatory-jp.briny-fern-5873.chatgpt.site).
It serves a compact interactive snapshot. The full Python app below supports
additional hourly and date-specific origin–destination queries.

## Open the local dashboard

On this Mac the environment and published results are already available:

```bash
cd /Users/jasonperez/Downloads/nyc-gpu-mobility
.venv/bin/python -m mobility dashboard
```

Open **http://127.0.0.1:8050**. Stop the server with Control-C.
You can also run `scripts/start_dashboard.command` from Finder or Terminal.

The six pages are Overview, Mobility patterns, Zones & connections, Unusual
activity, CPU & GPU performance, and Data & methods. Choose dates/zones and click
**Apply filters**. Static model summaries retain their labeled model period.

## Install on another CPU computer

Use Python 3.11 or newer; the local build was tested with Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
mobility doctor
python -m pytest -q
```

`environments/cpu-macos-py313.lock.txt` records the exact tested Mac environment,
including development/browser tools. The GPU environment is separate; follow
[docs/gpu.md](docs/gpu.md) rather than installing arbitrary CUDA packages.
To recreate that exact Mac/Python 3.13 environment, install the lock file with
`python -m pip install -r environments/cpu-macos-py313.lock.txt`, then install this
project with `python -m pip install --no-deps -e .`.

An offline fixture checks the full application without downloading NYC records:

```bash
mobility demo
mobility dashboard
```

The fixture is prominently labeled **synthetic** and uses a separate workspace.
Running it changes the active local dashboard snapshot; existing real snapshots
remain available and can be selected with `dashboard --snapshot PATH`.

## Reproduce the real study

Run these commands from the project directory with the environment active:

```bash
mobility --config configs/default.yaml data fetch
mobility zones prepare
mobility --config configs/default.yaml pipeline run --publish
mobility dashboard
```

The downloader verifies existing files rather than downloading them again.
Every run retains source checksums, the resolved config, quality counts, model
parameters, and validation results. Failed runs never replace the active website
snapshot. The pipeline prints its run directory on completion.

For a quick real-data check use `configs/january.yaml`. January alone cannot
support the configured later calibration/test periods, so it is labeled exploratory.
For a full-year study use `configs/full-year.yaml`. Batch sizes limit working
memory; the data and canonical outputs still need disk space.

```bash
mobility --config configs/default.yaml benchmark run --run artifacts/runs/RUN_ID --publish
mobility validate --run artifacts/runs/RUN_ID
mobility status
```

On a verified NVIDIA environment, add `--gpu` to the benchmark command. For the
GPU pipeline use `pipeline run --backend gpu`; set `processing.custom_cuda: true`
in a config to integrate the custom kernel. GPU-unavailable errors remain errors.

## What to read in the code

| File or directory | Responsibility |
|---|---|
| `src/mobility/data/acquire.py` | Official downloads, resumable transfers, checksums, zones |
| `src/mobility/data/quality.py` | Source normalization, flags, endpoint eligibility, DST |
| `src/mobility/aggregate.py` | Independent endpoint counts, OD counts, exact daily histograms |
| `src/mobility/pipeline.py` | Bounded processing, reconciliation, immutable publication |
| `src/mobility/models/` | Chronological KMeans and seasonal median/MAD anomalies |
| `src/mobility/backends/` | Explicit GPU operations and guarded custom CUDA integration |
| `cuda/` | Standalone compiled CUDA correctness checks |
| `src/mobility/benchmarks/` | Equivalent workloads, synchronization, raw repeated timings |
| `src/mobility/serving/` | Safe, bounded queries over published results |
| `src/mobility/dashboard/` | Full Dash app and compact public snapshot export |
| `tests/` | Data semantics, leakage, publication, queries, models, GPU parity |

Read [methods](docs/methods.md) for analytical definitions and limitations,
[GPU setup](docs/gpu.md) for server work, and [deployment](docs/deployment.md)
for public/static and Python hosting. [Case studies](docs/case-studies.md) explain
three real alerts, [release results](docs/release-results.md) summarize the
measured validation and benchmark evidence, and the
[proposal audit](docs/proposal-coverage.md) distinguishes completed components
from remaining work. Dataset source: the
[official NYC TLC catalog](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

## Scope

Custom CUDA is part of the implemented system. Team size and semester schedules
are not requirements. Yellow taxis are the supported source adapter; ride-hailing,
weather, graph algorithms, distributed execution, and multi-GPU processing remain
unimplemented. The proposal's cuGraph component and travel-duration anomalies
still need implementation. The code fails explicitly for unsupported source families.

All benchmark claims must identify their workload and timing scope. A faster
kernel does not imply an equally fast full pipeline. The public static snapshot
offers a smaller, labeled filter set; the full Dash app supports detailed
hourly queries without sending raw trip tables to visitors.
