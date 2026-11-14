"""缺口判定：单设备短缺口插补、长缺口未修复、首尾不外推、hold 不算缺口、全线断采不修补。"""
from __future__ import annotations

import pytest

from app import gaps, grid

BASE = 1_750_000_000.0 // 60 * 60
B0 = int(BASE // 60)


def build(present_values: dict[int, float], end_offset: int) -> list[dict]:
    """按「哪些桶有原始点」直接构造 1 分钟网格（便于精确控制缺口形状）。"""
    cells = []
    for offset in range(end_offset + 1):
        value = present_values.get(offset)
        cells.append(
            {
                "bucket": B0 + offset,
                "bucket_start": grid.bucket_start_iso(B0 + offset),
                "value": value,
                "mean": value,
                "max": value,
                "min": value,
                "samples": 1 if value is not None else 0,
                "source": "sample" if value is not None else "empty",
                "carried": False,
                "filled": False,
                "line_outage": False,
                "unavailable": False,
                "first_ts": B0 * 60 + offset * 60 + 5 if value is not None else None,
                "last_ts": B0 * 60 + offset * 60 + 5 if value is not None else None,
            }
        )
    return cells


def test_three_cell_gap_is_linearly_interpolated():
    cells = build({0: 10.0, 1: 11.0, 5: 15.0, 6: 16.0}, end_offset=6)
    result = gaps.find_gaps(cells)
    assert result["count"] == 1
    assert result["device_gap_count"] == 1
    assert result["line_outage_gaps"] == 0
    item = result["items"][0]
    assert item["cells"] == 3
    assert item["kind"] == "interpolated"
    assert item["attribution"] == "device"
    assert result["interpolated_cells"] == 3
    assert result["unrepaired_cells"] == 0
    # 线性插补：17.0/4=0.25 递增；起始值 11.0
    assert [round(v, 4) for v in item["filled_values"]] == [12.0, 13.0, 14.0]
    assert cells[2]["value"] == pytest.approx(12.0)
    assert cells[4]["source"] == "interpolated"


def test_five_cell_gap_is_marked_unrepaired():
    cells = build({0: 10.0, 1: 11.0, 7: 17.0}, end_offset=7)
    result = gaps.find_gaps(cells)
    assert result["count"] == 1
    item = result["items"][0]
    assert item["cells"] == 5
    assert item["kind"] == "unrepaired"
    assert item["attribution"] == "device"
    assert result["unrepaired_cells"] == 5
    assert item["filled_values"] == []
    assert cells[3]["value"] is None  # 不填值


def test_three_cell_line_outage_is_not_interpolated():
    # 3 格空段全是全线断采桶：即便 <=3 格也不插补，留空并列为不可用。
    cells = build({0: 10.0, 1: 11.0, 5: 15.0, 6: 16.0}, end_offset=6)
    outage = {B0 + 2, B0 + 3, B0 + 4}
    for idx in (2, 3, 4):
        cells[idx]["line_outage"] = True
        cells[idx]["unavailable"] = True
    result = gaps.find_gaps(cells, line_outage_buckets=outage)
    assert result["line_outage_gaps"] == 1
    assert result["device_gap_count"] == 0
    item = result["items"][0]
    assert item["kind"] == "line_outage"
    assert item["attribution"] == "line"
    assert item["unavailable"] is True
    assert result["interpolated_cells"] == 0
    assert all(cells[i]["value"] is None for i in (2, 3, 4))
    assert all(cells[i]["unavailable"] for i in (2, 3, 4))


def test_repaired_gap_keeps_original_record():
    cells = build({0: 10.0, 1: 11.0, 5: 15.0, 6: 16.0}, end_offset=6)
    before = {c["bucket"]: c["value"] for c in cells}
    result = gaps.find_gaps(cells)
    assert len(result["records"]) == 1
    record = result["records"][0]
    assert record["cells"] == 3
    assert record["kind"] == "interpolated"
    assert record["attribution"] == "device"
    assert record["start_bucket"] == B0 + 2
    # 修补后网格有值，但原始记录仍在（不被覆盖/删除）。
    assert cells[2]["value"] is not None
    assert result["items"][0]["original"]["cells"] == 3
    # 原始空段仍可还原：记录覆盖的桶在修补前全部是空的。
    assert all(before[B0 + i] is None for i in (2, 3, 4))
    assert result["original_empty_cells"] == 3


def test_head_gap_is_not_extrapolated():
    cells = build({2: 12.0, 3: 13.0}, end_offset=3)
    result = gaps.find_gaps(cells)
    assert result["count"] == 1
    item = result["items"][0]
    assert item["cells"] == 2
    assert item["kind"] == "unrepaired"
    assert item["boundary"] is True
    assert cells[0]["value"] is None
    assert cells[1]["value"] is None


def test_tail_gap_is_not_extrapolated():
    cells = build({0: 10.0, 1: 11.0}, end_offset=3)
    result = gaps.find_gaps(cells)
    assert result["count"] == 1
    item = result["items"][0]
    assert item["cells"] == 2
    assert item["kind"] == "unrepaired"
    assert item["boundary"] is True
    assert cells[3]["value"] is None


def test_no_gap_when_all_cells_have_raw_points():
    cells = build({i: float(i) for i in range(6)}, end_offset=5)
    result = gaps.find_gaps(cells)
    assert result["count"] == 0
    assert result["interpolated_cells"] == 0
    assert result["items"] == []


def test_two_gaps_are_reported_separately():
    cells = build({0: 0.0, 1: 1.0, 5: 5.0, 6: 6.0, 12: 12.0}, end_offset=12)
    result = gaps.find_gaps(cells)
    assert result["count"] == 2
    kinds = sorted(item["kind"] for item in result["items"])
    assert kinds == ["interpolated", "unrepaired"]
    assert [item["cells"] for item in result["items"]] == [3, 5]


def test_hold_filled_cells_are_not_reported_as_gap():
    # 60s 设备：第 2 分钟缺报但前后采样点相距 120s -> hold 沿用，不算缺口
    points = [(B0 * 60 + 5, 20.0), (B0 * 60 + 65, 21.0), (B0 * 60 + 185, 23.0), (B0 * 60 + 245, 24.0)]
    cells = grid.resample(points, interval_s=60, start_bucket=B0, end_bucket=B0 + 4)
    assert cells[2]["carried"] is True
    result = gaps.find_gaps(cells)
    assert result["count"] == 0
    assert result["interpolated_cells"] == 0


def test_hold_does_not_bridge_line_outage():
    # 全线断采桶永远不 hold：即便前后原始格相距 <= 3 格、采样间隔 > 1.5 桶。
    points = [(B0 * 60 + 5, 20.0), (B0 * 60 + 185, 23.0)]
    outage = {B0 + 1, B0 + 2}
    cells = grid.resample(
        points, interval_s=60, start_bucket=B0, end_bucket=B0 + 3, line_outage_buckets=outage
    )
    assert [c["source"] for c in cells] == ["sample", "empty", "empty", "sample"]
    assert all(c["unavailable"] for c in cells[1:3])
    assert all(c["carried"] is False for c in cells)
