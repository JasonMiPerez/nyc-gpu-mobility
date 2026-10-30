"""Reproduce the small aggregate tables used in docs/case-studies.md.

This reads a completed research run. It never changes model results or assigns
human review labels, and it does not read raw trip-level records.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


CASES = [
    ("lincoln-square", 142, "2024-06-16 23:00", "2024-06-17 02:00"),
    ("east-village", 79, "2024-05-25 00:00", "2024-05-25 01:00"),
    ("laguardia", 138, "2024-06-06 00:00", "2024-06-06 02:00"),
]


def export_cases(run: Path, output: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("data_kind") != "real":
        raise ValueError("Case studies require a completed real-data run.")
    tables = run / "tables"
    scores = pd.read_parquet(tables / "anomaly_scores.parquet")
    incidents = pd.read_parquet(tables / "anomaly_incidents.parquet")
    zones = pd.read_parquet(tables / "zones.parquet").set_index("zone_id")
    hourly, weekly, summaries = [], [], []
    for case_id, zone_id, start, end in CASES:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        incident = incidents.loc[
            incidents.zone_id.eq(zone_id)
            & incidents.start_hour.eq(start)
            & incidents.end_hour.eq(end)
            & incidents.split.eq("test")
        ]
        if len(incident) != 1:
            raise ValueError(f"Expected exactly one matching incident for {case_id}.")
        incident = incident.iloc[0]
        context = scores.loc[
            scores.zone_id.eq(zone_id)
            & scores.hour.between(start - pd.Timedelta(hours=2), end + pd.Timedelta(hours=2))
        ].copy()
        selected = context.loc[context.hour.between(start, end)]
        if not selected.is_alert.all() or len(selected) != incident.affected_hours:
            raise ValueError(f"Hourly rows do not reconcile with {case_id}.")
        if float(selected.deviation.sum()) != float(incident.total_deviation):
            raise ValueError(f"Deviation total does not reconcile for {case_id}.")
        for key, value in {
            "case_id": case_id,
            "run_id": manifest["run_id"],
            "zone_name": zones.loc[zone_id, "zone"],
        }.items():
            context[key] = value
        context["in_selected_incident"] = context.hour.between(start, end)
        hourly.append(context)
        peak = selected.loc[selected.score.abs().idxmax()]
        comparison_hours = [peak.hour + pd.Timedelta(days=days) for days in (-14, -7, 0, 7, 14)]
        comparison = scores.loc[
            scores.zone_id.eq(zone_id) & scores.hour.isin(comparison_hours)
        ].copy()
        comparison["case_id"] = case_id
        comparison["run_id"] = manifest["run_id"]
        comparison["zone_name"] = zones.loc[zone_id, "zone"]
        weekly.append(comparison)
        summaries.append(
            {
                "case_id": case_id,
                "run_id": manifest["run_id"],
                "model_id": incident.model_id,
                "zone_id": zone_id,
                "zone_name": zones.loc[zone_id, "zone"],
                "start_hour": start,
                "last_affected_hour": end,
                "affected_hours": int(incident.affected_hours),
                "peak_hour": peak.hour,
                "peak_observed": int(peak.observed),
                "peak_baseline": float(peak.baseline),
                "peak_scale": float(peak.scale),
                "peak_signed_score": float(peak.score),
                "observed_total": int(selected.observed.sum()),
                "baseline_total": float(selected.baseline.sum()),
                "total_deviation": float(selected.deviation.sum()),
                "minimum_history": int(selected.history_count.min()),
                "review_status": "assistant_prepared_pending_user_review",
                "artifact_label": incident.label,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(hourly, ignore_index=True).to_csv(output / "case-study-hours.csv", index=False)
    pd.concat(weekly, ignore_index=True).to_csv(
        output / "case-study-weekly-context.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(output / "case-study-summary.csv", index=False)
    print(
        f"Wrote {len(summaries)} case summaries and {sum(map(len, hourly))} context hours to {output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs"))
    arguments = parser.parse_args()
    export_cases(arguments.run, arguments.output)
