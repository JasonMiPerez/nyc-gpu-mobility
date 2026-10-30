# Deployment

The analytical pipeline and website have separate lifecycles. Publish only a
validated snapshot. A public viewer cannot run downloads, shell commands, training,
or GPU jobs.

## Full Python dashboard

The local entry point is `mobility dashboard`. For a production Python host:

```bash
export MOBILITY_SNAPSHOT=/absolute/path/to/published/snapshot
gunicorn app:server --bind 127.0.0.1:8050 --workers 2 --threads 2 --timeout 60
```

Put the process behind the hosting provider's HTTPS reverse proxy. `/healthz`
returns the snapshot identifier and status. Debug mode is off. A running app
uses one immutable snapshot; restart it deliberately when switching versions.

The Dockerfile packages the CPU-only serving environment:

```bash
docker build -t nyc-mobility .
docker run --rm -p 8050:8050 \
  -v /absolute/path/to/snapshot:/snapshot:ro nyc-mobility
```

Docker deployment is provided as a reproducible configuration; it is not proof
that a particular host or reverse proxy has been provisioned. Do not expose a
Dash development server as the production service.

## Compact public snapshot

`mobility.dashboard.static_export.export_public(snapshot, destination)` creates
a standalone website from validated results. It includes its chart library
locally, needs no Python server or GPU, and is suitable for static hosting.

The exported site provides exact daily activity filters, fixed-period OD
summaries, bounded sampled trip plots, model descriptions, anomaly summaries,
quality context, and measured benchmark tables. Every section declares which
filters it supports. It does not claim the full Dash app's hourly query surface.

The exporter allowlists public metadata and omits server paths, SSH details,
record IDs, raw trip files, and credentials. Large raw Parquet files and canonical
partitions are never part of the website bundle. Re-export and publish a new
immutable version when updating results; do not edit live data in place.

## Recovery

Run manifests are under `artifacts/runs/`. Published snapshots are under
`artifacts/snapshots/`; `artifacts/current.json` is the active local pointer.
To use a previous snapshot without changing that pointer:

```bash
mobility dashboard --snapshot artifacts/snapshots/SNAPSHOT_ID
```

Keep at least the latest verified snapshot, source manifest, environment locks,
model metadata, and raw benchmark trials. Raw source files can be reacquired
only if their frozen checksum still matches. Never delete the only archived
copy of final experiment evidence.
