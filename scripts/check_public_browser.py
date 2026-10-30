"""Check the actual static website with Chromium; run after exporting a snapshot.

Install the optional Playwright tool and its Chromium browser first. The website
must be served over HTTP. Screenshots and the check report stay in artifacts/qa.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import expect, sync_playwright

from mobility.pipeline import current_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8051/")
    args = parser.parse_args()
    snapshot = current_snapshot(Path("artifacts"))
    zone = pd.read_parquet(snapshot / "zone_hour.parquet")
    expected_all = int(zone.pickup_count.sum())
    selected = zone.loc[
        zone.zone_id.eq(161)
        & zone.hour.ge(pd.Timestamp("2024-01-01"))
        & zone.hour.lt(pd.Timestamp("2024-02-01"))
    ]
    expected_filtered = int(selected.pickup_count.sum())
    output = Path("artifacts/qa")
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors, records = [], []
        chart_checks = {}
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="networkidle", timeout=60000)
        summary = page.request.get(args.url.rstrip("/") + "/data/summary.json").json()
        assert summary["meta"]["snapshot_id"] == snapshot.name
        expect(page.locator("#page-content")).to_contain_text(f"{expected_all:,}", timeout=60000)
        assert page.evaluate("typeof window.performance.now") == "function"
        for name in ["overview", "patterns", "zones", "anomalies", "performance", "methods"]:
            started = time.perf_counter()
            page.locator(f'button[data-page="{name}"]').click()
            expect(page.locator("#page-content")).not_to_contain_text(
                "Loading the published view", timeout=60000
            )
            page.wait_for_load_state("networkidle")
            content = page.locator("#page-content").inner_text()
            assert "could not be loaded" not in content.lower(), content
            assert len(content) > 20, content
            if name == "patterns":
                page.wait_for_function("document.getElementById('sample-scatter').data?.length > 0")
                chart_checks["typical_axes"] = page.locator("#sample-scatter").evaluate(
                    "node => ({x: node.layout.xaxis.range, y: node.layout.yaxis.range})"
                )
                assert chart_checks["typical_axes"] == {"x": [0, 50], "y": [0, 120]}
                page.locator("#sample-view").select_option("full")
                expect(page.locator("#sample-view-summary")).to_contain_text("5,000")
                page.wait_for_function(
                    "document.getElementById('sample-scatter').data.reduce((n, t) => n + t.x.length, 0) === 5000"
                )
                assert (
                    page.locator("#sample-histogram").evaluate(
                        "node => node.data[0].y.reduce((a, b) => a + b, 0)"
                    )
                    == 5000
                )
                page.locator("#sample-view").select_option("log")
                page.wait_for_function(
                    "document.getElementById('sample-scatter').layout.xaxis.title.text.includes('log')"
                )
                chart_checks["full_and_log_points"] = 5000
                page.locator("#sample-view").select_option("typical")
            elif name == "zones":
                expect(page.locator("#od-limit")).to_have_value("50")
                expect(page.locator("#od-summary")).to_contain_text("50")
                page.locator("#od-limit").select_option("25")
                expect(page.locator("#od-summary")).to_contain_text("25")
                chart_checks["od_limit_control"] = True
                if page.locator("#graph-view").count():
                    expect(page.locator("#graph-view")).to_have_value("community")
                    expect(page.locator("#graph-section")).to_contain_text("263")
                    page.locator("#graph-view").select_option("pagerank")
                    page.wait_for_function(
                        "document.getElementById('graph-map').data[0].colorbar?.title?.text === 'PageRank'"
                    )
                    page.locator("#graph-view").select_option("community")
                    chart_checks["graph_community_and_pagerank"] = True
            elif name == "anomalies":
                expect(page.locator("#page-content")).to_contain_text("Unreviewed")
                expect(page.locator("#page-content")).to_contain_text("MAD")
                duration_option = page.locator('#alert-metric option[value="duration"]')
                if duration_option.count() and duration_option.is_enabled():
                    page.locator("#alert-metric").select_option("duration")
                    expect(page.locator("#page-content")).to_contain_text("20 eligible trips")
                    expect(page.locator("#alert-table")).to_contain_text("Eligible trips in hour")
                    page.wait_for_function(
                        "document.getElementById('alert-context').layout.yaxis.title.text === 'Mean trip duration (minutes)'"
                    )
                    page.screenshot(path=str(output / "public-duration.png"), full_page=True)
                    chart_checks["duration_metric_units_and_support"] = True
                    page.locator("#alert-metric").select_option("volume")
            elif name == "performance":
                expect(page.locator("#speedup-table")).to_contain_text("CPU faster")
                expect(page.locator("#speedup-table")).to_contain_text("0.004186")
                expect(page.locator("#speedup-table")).to_contain_text("0.004763")
                chart_checks["cpu_winner_and_precision"] = True
                if page.locator("#pipeline-chart").count():
                    page.wait_for_function(
                        "document.getElementById('pipeline-chart').data?.length > 0"
                    )
                    expect(page.locator("#page-content")).to_contain_text("Sampled host peak (GiB)")
                    chart_checks["full_pipeline_scaling"] = True
                    pipeline = page.request.get(
                        args.url.rstrip("/") + "/data/pipeline-performance.json"
                    ).json()
                    quality = page.locator("#pipeline-section section.panel").filter(
                        has=page.get_by_role("heading", name="Clustering quality by measured trial")
                    )
                    expect(quality.locator("tbody tr")).to_have_count(
                        min(200, len(pipeline["clustering_quality"]))
                    )
                    expect(quality).to_contain_text("Test silhouette")
                    expect(quality).to_contain_text("Unavailable")
                    chart_checks["clustering_quality_trials"] = len(pipeline["clustering_quality"])
                    evidence = pipeline["cpu_execution_evidence"]
                    count = evidence["measured_native_cpu_trials"]
                    if count:
                        assert evidence["guard_and_no_cuda_imports_confirmed"] == count
                        assert evidence["zero_sampled_gpu_memory_trials"] == count
                        expect(page.locator("#pipeline-section")).to_contain_text(
                            f"Native CPU isolation: {count} of {count}"
                        )
                        chart_checks["native_cpu_isolation_trials"] = count
            records.append({"page": name, "elapsed_s": round(time.perf_counter() - started, 3)})
            page.screenshot(path=str(output / f"public-{name}.png"), full_page=True)
        page.locator('button[data-page="overview"]').click()
        expect(page.locator("#page-content")).to_contain_text("Mapped pickups")
        page.locator("#start-date").fill("2024-01-01")
        page.locator("#end-date").fill("2024-01-31")
        page.locator("#zone-select").select_option("161")
        page.locator("#apply").click()
        expect(page.locator(".kpi-value").first).to_have_text(f"{expected_filtered:,}")
        with page.expect_download() as download:
            page.locator("#export").click()
        csv = output / "public-filter-export.csv"
        download.value.save_as(csv)
        exported = pd.read_csv(csv)
        assert int(exported.pickups.sum()) == expected_filtered
        assert exported.zone_id.eq(161).all()
        page.locator("#reset").click()
        expect(page.locator(".kpi-value").first).to_have_text(f"{expected_all:,}")
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "public-mobile.png"), full_page=True)
        widths = page.evaluate(
            "({viewport: innerWidth, content: document.documentElement.scrollWidth})"
        )
        assert widths["content"] <= widths["viewport"], widths
        assert not errors, errors
        report = {
            "url": args.url,
            "snapshot_id": snapshot.name,
            "pages": records,
            "errors": errors,
            "mobile_widths": widths,
            "filter_expected_pickups": expected_filtered,
            "csv_verified": True,
            "chart_checks": chart_checks,
        }
        (output / "browser-public.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
