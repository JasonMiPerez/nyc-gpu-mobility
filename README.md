# NYC Mobility Observatory

A working system for exploring recorded NYC yellow taxi activity and comparing
CPU, GPU-library, and custom CUDA computation. The website reads validated saved
results; large downloads, training, and GPU jobs run separately.

The study covers January–June 2024: **20,332,093 official TLC source records**.
The implementation includes trip clustering, separate pickup-count and mean-duration
anomaly detectors, weighted PageRank, Louvain communities, and CPU/GPU performance
experiments. Counts describe recorded taxi trips, not all NYC travel or unmet demand.
Statistical alerts remain unreviewed until someone checks their explanation.

The completed six-month GPU run `20261030T164900-a9f2d0b4` fits six cuML clusters
and predicts **20,230,079 eligible trips**, reports **550 duration alert hours**
merged into **374 unreviewed candidate incidents**, and executes cuGraph PageRank
and Louvain on **263 mapped zones**. The earlier CPU study, quality refit, January
GPU validation and stage benchmarks remain documented as historical results.

Full-pipeline experiment `pipeline-eb6b177f59b1` passed all **18 measured trials**
at one, three and six months, with six discarded warmups. At six months, median
CPU/GPU times were **342.734 / 334.459 seconds**: a modest **1.025× CPU/GPU ratio**,
not the larger speedups seen for some isolated stages. See the completed
[scaling report](docs/pipeline-scaling-results.md), [release results](docs/release-results.md)
and [proposal audit](docs/proposal-coverage.md) for evidence and scope. Verification
for this expansion: **159 local tests passed**, seven GPU-only tests skipped on
the Mac, and **36 server checks passed**. Completion of these runs is separate
from deploying their results to the public site.

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
**Apply filters**. Trip-group summaries and graph ranks retain their labeled
model period. Published duration results appear separately from pickup-count
alerts; graph results add PageRank and categorical community maps. The performance
page separates stage timings from full raw-file pipeline trials. Older snapshots
explicitly identify results they do not contain.

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
memory; the data and canonical outputs still need disk space. Duration anomalies
and zone-graph analysis are enabled in the default configuration.

```bash
mobility --config configs/default.yaml benchmark run --run artifacts/runs/RUN_ID --publish
mobility validate --run artifacts/runs/RUN_ID
mobility status
```

On a verified NVIDIA environment, add `--gpu` to the benchmark command. For the
GPU pipeline use `pipeline run --backend gpu`; set `processing.custom_cuda: true`
in a config to integrate the custom kernel. GPU-unavailable errors remain errors;
the GPU graph path requires cuGraph and does not silently fall back to NetworkX.

The stage benchmark above is separate from the full raw-file pipeline experiment.
On a prepared NVIDIA environment with the same six source months already acquired,
the latter can be reproduced with:

```bash
python scripts/benchmark_pipeline_scaling.py \
  --config configs/default.yaml \
  --output artifacts/pipeline-scaling-new \
  --month-counts 1 3 6 --backends cpu gpu --warmups 1 --repeats 3
```

The output directory must be new. Each trial uses a fresh process and records
its source identities, configuration, elapsed time, validation and resource samples.
Only completed, matching, correctness-validated measured trials support a ratio.
January and January–March lack the configured later evaluation periods; the
six-month case adds held-out evaluation work as well as more rows. See
[extended methods](docs/extended-methods.md) for the exact timing and memory scopes,
and the [completed scaling report](docs/pipeline-scaling-results.md) for all
repetitions, interquartile ranges and sampled resource measurements.

## What to read in the code

| File or directory | Responsibility |
|---|---|
| `src/mobility/data/acquire.py` | Official downloads, resumable transfers, checksums, zones |
| `src/mobility/data/quality.py` | Source normalization, flags, endpoint eligibility, DST |
| `src/mobility/aggregate.py` | Independent endpoint counts, OD counts, exact daily histograms |
| `src/mobility/pipeline.py` | Bounded processing, reconciliation, immutable publication |
| `src/mobility/models/clustering.py` | Chronological KMeans, fixed study filters and bounded samples |
| `src/mobility/models/anomalies.py` | Seasonal pickup-count alerts and retrospective evaluation |
| `src/mobility/models/duration_anomalies.py` | Hourly mean-duration baselines, support rules and alerts |
| `src/mobility/models/graph_analysis.py` | Weighted PageRank, Louvain communities and CPU/GPU validation |
| `src/mobility/backends/` | Explicit GPU operations and guarded custom CUDA integration |
| `cuda/` | Standalone compiled CUDA correctness checks |
| `src/mobility/benchmarks/` | Matched stage workloads and full raw-file pipeline experiments |
| `src/mobility/serving/` | Safe, bounded queries over published results |
| `src/mobility/dashboard/` | Full Dash app and compact public snapshot export |
| `tests/` | Data semantics, leakage, publication, queries, models, GPU parity |

Read [methods](docs/methods.md) for analytical definitions and limitations,
[extended methods](docs/extended-methods.md) for duration alerts, zone graphs and
full-pipeline benchmarks,
[GPU setup](docs/gpu.md) for server work, and [deployment](docs/deployment.md)
for public/static and Python hosting. [Case studies](docs/case-studies.md) explain
three real alerts, [release results](docs/release-results.md) summarize the
measured validation and benchmark evidence, and the
[proposal audit](docs/proposal-coverage.md) distinguishes completed components
from remaining work. Dataset source: the
[official NYC TLC catalog](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

## Scope

Custom CUDA, cuDF aggregation, cuML clustering and cuGraph analysis are implemented.
This is a hybrid pipeline: source normalization, timestamps/DST checks, primary
quality masks, model sampling/scaling, and the two anomaly detectors run on CPU.
The GPU graph path also runs a CPU reference for correctness. A GPU backend does
not mean every operation runs on the device, and a small taxi-zone graph is not
assumed to benefit from GPU execution.

Duration alerts compare hourly mean trip minutes within a pickup zone, weekday and
hour. Changes in route, distance or destination mix can change that mean without
establishing congestion. Community IDs are arbitrary labels, not ranked zone quality.
The study filters were revised after historical data inspection, so the resulting
chronological holdout is retrospective rather than a fresh prospective validation.

Team size and semester schedules are not requirements. Yellow taxis are the
supported source adapter; ride-hailing, weather, distributed execution, and
multi-GPU processing remain extensions. The code rejects unsupported source families.

All benchmark claims must identify their workload and timing scope. A faster
kernel does not imply an equally fast full pipeline. Recorded RSS and GPU memory
are sampled observations, not exact peaks; short spikes can be missed and RSS can
count shared pages more than once. The public static snapshot
offers a smaller, labeled filter set; the full Dash app supports detailed
hourly queries without sending raw trip tables to visitors.
