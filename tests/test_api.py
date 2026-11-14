"""API 端到端：导入、对齐、结果页、错误输入 4xx、质量报告与 CSV 导出。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        # 主流程第一步：幂等导入样例遥测
        resp = test_client.post("/api/import")
        assert resp.status_code == 200, resp.text
        yield test_client


def align(client: TestClient, **payload):
    return client.post("/api/align", json=payload)


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["app"] == "tsgrid"


def test_import_is_idempotent(client: TestClient):
    first = client.post("/api/import").json()
    second = client.post("/api/import").json()
    assert first["dataset_id"] == second["dataset_id"]
    assert first["sha256"] == second["sha256"]
    assert first["rows"] > 20000
    assert len(first["devices"]) == 3
    assert {d["device_id"] for d in first["devices"]} == {"LINE-A-01", "LINE-A-02", "LINE-A-03"}
    assert first["window"]["cells"] == 360


def test_drifted_device_offset_is_about_37_seconds(client: TestClient):
    resp = align(client, device_id="LINE-A-03", metric="temperature", offset_threshold=5)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["drift"]["available"] is True
    assert abs(data["drift"]["offset_s"] - 37.0) < 1.0
    assert data["drift"]["drifted"] is True
    assert data["drift"]["rmse_s"] < 1.0
    assert data["reference"]["device_id"] == "LINE-A-01"
    # 温度序列：两段全线断采（3 格 + 5 格）都归因 line、留空不可用，不再插补 3 格那段。
    assert data["gaps"]["count"] == 2
    assert data["gaps"]["line_outage_gaps"] == 2
    assert data["gaps"]["device_gap_count"] == 0
    assert data["gaps"]["interpolated_cells"] == 0
    # 全线断采不计入「未修复设备缺口」，单独列为不可用格。
    assert data["gaps"]["unrepaired_cells"] == 0
    assert data["gaps"]["line_outage_cells"] == 8
    assert data["coverage"]["grade"] == "B"
    assert data["coverage"]["unavailable_cells"] == 8
    assert data["line_outage"]["count"] == 2
    assert data["line_outage"]["affected_devices"] == ["LINE-A-01", "LINE-A-02", "LINE-A-03"]
    assert data["coverage"]["expected_cells"] == 360
    assert data["interval_s"] == 60
    assert data["run_id"].startswith("run-")


def test_low_rate_grades_and_carried_counts(client: TestClient):
    temperature = align(client, device_id="LINE-A-02", metric="temperature").json()
    assert temperature["drift"]["drifted"] is False          # 5s 设备时钟正常
    assert temperature["coverage"]["grade"] == "C"           # 全线断采 8 格 + 单设备断采 12 格
    assert temperature["coverage"]["carried_cells"] == 30    # 3 分钟降级段被 hold
    assert temperature["gaps"]["count"] == 3
    assert temperature["gaps"]["line_outage_gaps"] == 2
    assert temperature["gaps"]["device_gap_count"] == 1

    pressure = align(client, device_id="LINE-A-03", metric="pressure").json()
    assert pressure["coverage"]["grade"] == "C"
    assert abs(pressure["drift"]["offset_s"] - 37.0) < 1.0
    assert pressure["gaps"]["unrepaired_cells"] == 25
    assert pressure["gaps"]["line_outage_cells"] == 8


def test_hourly_summary_shows_line_outage(client: TestClient):
    data = align(client, device_id="LINE-A-03", metric="temperature").json()
    hours = {h["hour_start"][11:13]: h for h in data["hourly"]}
    assert hours["03"]["unavailable_cells"] == 3
    assert hours["03"]["line_outage_cells"] == 3
    assert hours["04"]["unavailable_cells"] == 5
    assert hours["03"]["status"] == "partial"
    # 摘要口径与质量报告一致：可用率 = 有效格 / 该小时格数。
    assert hours["03"]["availability"] == (60 - 3) / 60


def test_reference_device_reports_no_self_comparison(client: TestClient):
    data = align(client, device_id="LINE-A-01", metric="temperature").json()
    assert data["reference"]["self_reference"] is True
    assert data["drift"]["available"] is False
    assert data["drift"]["offset_s"] is None


def test_threshold_change_flips_verdict(client: TestClient):
    strict = align(client, device_id="LINE-A-02", metric="temperature", offset_threshold=0.005).json()
    assert strict["drift"]["drifted"] is True


def test_unknown_device_returns_4xx_with_reason(client: TestClient):
    resp = align(client, device_id="LINE-A-99", metric="temperature")
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "LINE-A-99" in detail
    assert "可用" in detail


def test_metric_not_on_device_returns_4xx(client: TestClient):
    resp = align(client, device_id="LINE-A-01", metric="pressure")
    assert resp.status_code == 400
    assert "pressure" in resp.json()["detail"]


def test_bad_threshold_returns_4xx(client: TestClient):
    resp = align(client, device_id="LINE-A-03", metric="temperature", offset_threshold="abc")
    assert resp.status_code == 400
    assert "阈值" in resp.json()["detail"]

    resp_negative = align(client, device_id="LINE-A-03", metric="temperature", offset_threshold=-5)
    assert resp_negative.status_code == 400


def test_missing_fields_return_4xx(client: TestClient):
    assert client.post("/api/align", json={"metric": "temperature"}).status_code == 400
    assert client.post("/api/align", json={"device_id": "LINE-A-03"}).status_code == 400


def test_result_page_renders_evidence(client: TestClient):
    data = align(client, device_id="LINE-A-03", metric="temperature").json()
    resp = client.get(f"/result/{data['run_id']}")
    assert resp.status_code == 200
    html = resp.text
    for expected in ["时钟漂移估计", "校正建议", "1 分钟网格预览", "缺口明细", "小时摘要", "覆盖率与分级", "37."]:
        assert expected in html


def test_unknown_run_returns_404(client: TestClient):
    assert client.get("/result/run-doesnotexist").status_code == 404
    assert client.get("/api/runs/run-doesnotexist").status_code == 404


def test_pages_render(client: TestClient):
    for path in ["/", "/workbench", "/review"]:
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert "遥测" in resp.text or "对齐" in resp.text


def test_report_and_csv_export(client: TestClient):
    report = client.get("/api/report").json()
    assert len(report["rows"]) == 5
    grades = {row["coverage"]["grade"] for row in report["rows"]}
    assert grades == {"B", "C"}  # 全线断采压低所有序列覆盖率
    assert len(report["devices"]) == 3
    # gap_count 只数单设备缺口；全线断采单独汇总。
    assert report["summary"]["total_gaps"] == 2
    assert report["summary"]["total_device_gaps"] == 2
    assert report["line_outage"]["count"] == 2
    assert report["line_outage"]["cells"] == 8
    assert report["line_outage"]["affected_devices"] == ["LINE-A-01", "LINE-A-02", "LINE-A-03"]
    for window in report["line_outage"]["windows"]:
        assert {d["device_id"] for d in window["devices"]} == {"LINE-A-01", "LINE-A-02", "LINE-A-03"}
        assert all(d["availability"] == 0.0 for d in window["devices"])
    assert report["summary"]["total_unavailable"] > 0

    csv_resp = client.get("/api/report.csv")
    assert csv_resp.status_code == 200
    assert csv_resp.headers["content-type"].startswith("text/csv")
    lines = [line for line in csv_resp.text.strip().splitlines() if line]
    assert lines[0].startswith("device_id,metric,interval_s")
    assert len(lines) == 1 + 5 + 1 + 1 + 6  # 表头 + 5 序列 + 段标记 + 段表头 + 2 段 × 3 设备
    assert "line_outage_windows" in lines[0]
    assert "# line_outages" in lines[6]
    assert csv_resp.headers["X-Report-Id"] == report["report_id"]


def test_rerun_same_batch_does_not_add_reports_or_change_conclusion(client: TestClient):
    first = align(client, device_id="LINE-A-03", metric="temperature").json()
    second = align(client, device_id="LINE-A-03", metric="temperature").json()
    # 重跑同一批：run_id 相同（不新增运行记录），结论完全一致。
    assert first["run_id"] == second["run_id"]
    assert first["coverage"] == second["coverage"]
    assert first["gaps"] == second["gaps"]

    report_a = client.get("/api/report").json()
    report_b = client.get("/api/report").json()
    assert report_a["report_id"] == report_b["report_id"]
    assert report_a["reports_stored"] == report_b["reports_stored"]
    assert report_a["summary"] == report_b["summary"]


def test_repair_keeps_original_gap_record(client: TestClient):
    # LINE-A-02 温度 05:10–05:21 是单设备长缺口（12 格）：不插补，但原始记录必须保留。
    data = align(client, device_id="LINE-A-02", metric="temperature").json()
    device_gaps = [g for g in data["gaps"]["items"] if g["attribution"] == "device"]
    assert len(device_gaps) == 1
    gap = device_gaps[0]
    assert gap["cells"] == 12
    assert gap["kind"] == "unrepaired"
    assert gap["filled_values"] == []
    assert gap["original"]["cells"] == 12
    records = [r for r in data["gaps"]["records"] if r["attribution"] == "device"]
    assert len(records) == 1
    assert records[0]["start_time"].endswith("05:10:00+08:00")


def test_alignment_pipeline_stages_recorded(client: TestClient):
    data = align(client, device_id="LINE-A-03", metric="temperature").json()
    names = [stage["name"] for stage in data["stages"]]
    assert names == ["数据导入", "漂移估计", "网格重采样", "缺口判定", "小时摘要", "质量评分"]
    assert data["total_ms"] > 0
    assert len(data["hourly"]) == 6
    assert len(data["preview"]) == 60
