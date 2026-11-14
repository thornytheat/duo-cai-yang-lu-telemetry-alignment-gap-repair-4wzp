"""全线断采归因：同一份判断贯穿网格、缺口、摘要与质量报告。"""
from __future__ import annotations

import pytest

from app import gaps, grid, incidents, quality, summary
from app.telemetry import Dataset, Series

BASE = 1_750_000_000.0 // 60 * 60
B0 = int(BASE // 60)


def make_dataset(series_points: dict[tuple[str, str], list[tuple[float, float]]]) -> Dataset:
    series = {
        (device, metric): Series(device_id=device, metric=metric, points=points, interval_s=60, source="test")
        for (device, metric), points in series_points.items()
    }
    return Dataset(
        dataset_id="ds-test",
        sha256="test",
        source="test",
        rows=sum(len(p) for p in series_points.values()),
        files=[],
        series=series,
        window_start=BASE,
        window_end=BASE + 9 * 60,
    )


def test_line_outage_is_every_series_missing_same_bucket():
    # 两台设备同一分钟都没点 -> 该分钟是全线断采；只有一台缺 -> 不是。
    points_a = [(B0 * 60 + 0, 1.0), (B0 * 60 + 120, 3.0)]  # 缺 B0+1
    points_b = [(B0 * 60 + 0, 2.0), (B0 * 60 + 120, 4.0)]  # 缺 B0+1
    points_b_ok = [(B0 * 60 + 0, 2.0), (B0 * 60 + 60, 5.0), (B0 * 60 + 120, 4.0)]
    ds = make_dataset({("A", "temperature"): points_a, ("B", "temperature"): points_b_ok})
    assert incidents.line_outage_buckets(ds, B0, B0 + 2) == set()

    ds2 = make_dataset({("A", "temperature"): points_a, ("B", "temperature"): points_b})
    assert incidents.line_outage_buckets(ds2, B0, B0 + 2) == {B0 + 1}
    desc = incidents.describe(ds2, start_bucket=B0, end_bucket=B0 + 2)
    assert desc["count"] == 1
    assert desc["cells"] == 1
    assert desc["affected_devices"] == ["A", "B"]
    window = desc["windows"][0]
    assert window["cells"] == 1
    assert all(d["availability"] == 0.0 for d in window["devices"])


def test_single_device_outage_is_not_line_outage():
    # B 设备在 B0+1..B0+3 断采，但 A 设备每分钟都有点 -> 单设备断采。
    points_a = [(B0 * 60 + i * 60, 1.0) for i in range(6)]
    points_b = [(B0 * 60, 2.0), (B0 * 60 + 240, 3.0)]
    ds = make_dataset({("A", "temperature"): points_a, ("B", "temperature"): points_b})
    assert incidents.line_outage_buckets(ds, B0, B0 + 4) == set()
    assert incidents.describe(ds, start_bucket=B0, end_bucket=B0 + 4)["count"] == 0


def test_pipeline_judgement_is_shared_across_grid_gaps_summary_and_quality():
    # 5 分钟窗口：B0+1..B0+3 全线断采（3 格），B0+4 起点恢复。
    points = [(B0 * 60, 10.0), (B0 * 60 + 240, 20.0)]
    ds = make_dataset({("A", "temperature"): points})
    outage = incidents.describe(ds, start_bucket=B0, end_bucket=B0 + 4)
    cells = grid.resample(
        points, 60, B0, B0 + 4, line_outage_buckets=outage["buckets"]
    )
    grid_stats = grid.stats(cells)
    assert grid_stats["line_outage_cells"] == 3

    gap_result = gaps.find_gaps(cells, line_outage_buckets=outage["buckets"])
    assert gap_result["line_outage_gaps"] == 1
    assert gap_result["interpolated_cells"] == 0
    # 全线断采单独计入不可用格，不计入「未修复设备缺口」。
    assert gap_result["unrepaired_cells"] == 0
    assert gap_result["line_outage_cells"] == 3
    assert all(cells[i]["value"] is None for i in range(1, 4))

    hourly = summary.hourly_summary(cells)
    hour = hourly[0]
    assert hour["unavailable_cells"] == 3
    assert hour["availability"] == pytest.approx(2 / 5)

    cov = quality.coverage_report(cells, 5, gap_result, line_outage=outage)
    assert cov["valid_cells"] == 2
    assert cov["unavailable_cells"] == 3
    assert cov["line_outage_cells"] == 3
    assert cov["coverage"] == pytest.approx(2 / 5)
    assert cov["grade"] == "C"


def test_summary_status_and_availability_show_link_downtime():
    points = [(B0 * 60, 10.0)] + [(B0 * 60 + i * 60, 10.0 + i) for i in range(4, 6)]
    ds = make_dataset({("A", "temperature"): points})
    outage = incidents.describe(ds, start_bucket=B0, end_bucket=B0 + 5)
    cells = grid.resample(points, 60, B0, B0 + 5, line_outage_buckets=outage["buckets"])
    gaps.find_gaps(cells, line_outage_buckets=outage["buckets"])
    hours = summary.hourly_summary(cells)
    assert len(hours) == 1
    assert hours[0]["cells"] == 6
    assert hours[0]["unavailable_cells"] == 3
    assert hours[0]["availability"] == pytest.approx(3 / 6)
    assert hours[0]["status"] == "partial"
    assert hours[0]["min"] == pytest.approx(10.0)
    assert hours[0]["max"] == pytest.approx(15.0)


def test_report_rollup_lists_devices_and_availability_for_outage_window():
    points_a = [(B0 * 60, 1.0), (B0 * 60 + 180, 2.0)]
    points_b = [(B0 * 60, 3.0), (B0 * 60 + 180, 4.0)]
    ds = make_dataset({("A", "temperature"): points_a, ("B", "temperature"): points_b})
    outage = incidents.describe(ds)
    report = quality.report_summary([], line_outage=outage)
    assert report["line_outage_windows"][0]["cells"] == 2
    rollup = quality.device_rollup(
        [
            {"device_id": "A", "metric": "temperature", "interval_s": 60,
             "coverage": quality.coverage_report(
                 grid.resample(points_a, 60, B0, B0 + 2, outage["buckets"]), 3,
                 gaps.find_gaps(grid.resample(points_a, 60, B0, B0 + 2, outage["buckets"]), outage["buckets"]),
                 line_outage=outage),
             "gaps": gaps.find_gaps(grid.resample(points_a, 60, B0, B0 + 2, outage["buckets"]), outage["buckets"])},
        ],
        line_outage=outage,
    )
    assert rollup[0]["line_outage_windows"]
    assert rollup[0]["line_outage_windows"][0]["availability"] == 0.0


def test_single_device_short_gap_is_repaired_but_line_gap_is_not():
    """同一长度（3 格）：单设备断采可插补，全线断采一律留空不可用。"""
    # A 每分钟有点；B 在 B0+1..B0+3 缺（A 有点 → 单设备短缺口，应插补）。
    a = [(B0 * 60 + i * 60, 10.0 + i) for i in range(6)]
    b = [(B0 * 60, 20.0), (B0 * 60 + 240, 24.0)]
    ds = make_dataset({("A", "temperature"): a, ("B", "temperature"): b})
    outage = incidents.describe(ds, start_bucket=B0, end_bucket=B0 + 4)
    assert outage["count"] == 0

    cells_b = grid.resample(b, 60, B0, B0 + 4, line_outage_buckets=outage["buckets"])
    gap_b = gaps.find_gaps(cells_b, line_outage_buckets=outage["buckets"])
    assert gap_b["device_gap_count"] == 1
    assert gap_b["interpolated_cells"] == 3
    assert cells_b[2]["source"] == "interpolated"

    # 同样的 3 格形状，但 A/B 同时缺 → 全线断采，不插补。
    b2 = [(B0 * 60, 20.0), (B0 * 60 + 240, 24.0)]
    a2 = [(B0 * 60, 10.0), (B0 * 60 + 240, 14.0)]
    ds2 = make_dataset({("A", "temperature"): a2, ("B", "temperature"): b2})
    outage2 = incidents.describe(ds2, start_bucket=B0, end_bucket=B0 + 4)
    assert outage2["cells"] == 3
    cells_b2 = grid.resample(b2, 60, B0, B0 + 4, line_outage_buckets=outage2["buckets"])
    gap_b2 = gaps.find_gaps(cells_b2, line_outage_buckets=outage2["buckets"])
    assert gap_b2["line_outage_gaps"] == 1
    assert gap_b2["interpolated_cells"] == 0
    assert all(cells_b2[i]["value"] is None for i in (1, 2, 3))
    assert all(cells_b2[i]["unavailable"] for i in (1, 2, 3))


def test_pipeline_end_to_end_uses_same_judgement(monkeypatch):
    """走完整流水线：网格标记、缺口归因、摘要状态、覆盖率口径完全一致。"""
    from app import pipeline

    def stub_drift(*args, **kwargs):
        return {
            "available": True,
            "offset_s": 0.0,
            "rate_ppm": 0.0,
            "rmse_s": 0.0,
            "anchors": 4,
            "threshold_s": 5.0,
            "drifted": False,
            "verdict": "stub",
            "suggestion": "stub",
        }

    monkeypatch.setattr(pipeline.drift, "estimate_drift", stub_drift)
    # 两条序列在 B0+1..B0+3 同时缺 → 全线断采；B 在 B0+6 单独缺 → 单设备短缺口（可插补）。
    # B 的 B0+5 / B0+7 采样点相距 61s（≤ 90s），不会触发 hold，能落到缺口模块。
    a = [(B0 * 60 + i * 60, 10.0 + i) for i in range(8) if i not in (1, 2, 3)]
    b = [(B0 * 60, 20.0), (B0 * 60 + 240, 24.0), (B0 * 60 + 300 + 59, 25.0), (B0 * 60 + 420, 27.0)]
    ds = make_dataset({("A", "temperature"): a, ("B", "temperature"): b})
    ds.window_start = B0 * 60
    ds.window_end = B0 * 60 + 7 * 60

    payload = pipeline.run_alignment(ds, "B", "temperature")
    assert payload["line_outage"]["count"] == 1
    assert payload["line_outage"]["cells"] == 3
    assert payload["line_outage"]["affected_devices"] == ["A", "B"]
    assert payload["grid"]["line_outage_cells"] == 3
    assert payload["gaps"]["line_outage_gaps"] == 1
    assert payload["gaps"]["interpolated_cells"] == 1
    assert payload["coverage"]["unavailable_cells"] == 3
    assert payload["coverage"]["line_outage_labels"]
    hour = payload["hourly"][0]
    assert hour["unavailable_cells"] == 3
    assert hour["status"] == "partial"
    # 每一层口径都能指出「采集链路停过」。
    assert payload["coverage"]["coverage"] < 1.0
    assert payload["coverage"]["availability"] < 1.0
