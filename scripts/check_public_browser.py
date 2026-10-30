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
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="networkidle", timeout=60000)
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
            "pages": records,
            "errors": errors,
            "mobile_widths": widths,
            "filter_expected_pickups": expected_filtered,
            "csv_verified": True,
        }
        (output / "browser-public.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
