"""缺口判定、归因与有限修复。

判定口径与 grid / summary / quality 完全一致：都用 incidents 给出的「全线断采桶」。

- 网格内没有原始点（source='empty'）的连续段先按「是不是全线断采桶」切成子段：
  - 子段全是全线断采桶 → 归因 `line`（全线断采）：留空、标记不可用、绝不修补，
    即使长度 <= 3 格也不插补——采集链路整体停过，补出来的值没有依据。
  - 子段不含全线断采桶 → 归因 `device`（单设备断采）：
    - 两侧都是真实原始格且长度 <= 3 格 → 线性插补（interpolated）；
    - 其余（首尾边界、长度 > 3、或一侧挨着全线断采）→ unrepaired，不填值。
- 修补只是在网格格子上写值；缺口记录（起止桶、格数、归因）保持原样，
  并额外给出 `original` 快照与 `records` 汇总，保证原始缺口可追溯、修补不删记录。
"""
from __future__ import annotations

from . import settings


def find_gaps(cells: list[dict], line_outage_buckets: set[int] | None = None) -> dict:
    """在（已 hold 的）网格上找出缺口、归因并给出修复结论。"""
    if line_outage_buckets is None:
        line_outage_buckets = {c["bucket"] for c in cells if c.get("line_outage")}
    original_empty = sum(1 for c in cells if c["source"] == "empty")

    raw_idx = [i for i, c in enumerate(cells) if c["source"] in ("agg", "sample")]
    gaps: list[dict] = []

    if not raw_idx:
        if cells:
            for segment in _split_by_attribution(cells, line_outage_buckets):
                gaps.append(_make_item(segment, cells, at_head=True, at_tail=True))
        _repair(gaps, cells)
        return _result(gaps, cells, original_empty)

    if raw_idx[0] > 0:
        run = cells[: raw_idx[0]]
        for segment in _split_by_attribution(run, line_outage_buckets):
            gaps.append(_make_item(segment, cells, at_head=True, at_tail=False))

    for prev, nxt in zip(raw_idx, raw_idx[1:]):
        run = cells[prev + 1 : nxt]
        if not run or all(c["source"] != "empty" for c in run):
            continue  # 已由 hold 沿用的稀疏采样段，不算缺口
        for segment in _split_by_attribution(run, line_outage_buckets):
            gaps.append(_make_item(segment, cells, at_head=False, at_tail=False))

    if raw_idx[-1] < len(cells) - 1:
        run = cells[raw_idx[-1] + 1 :]
        for segment in _split_by_attribution(run, line_outage_buckets):
            gaps.append(_make_item(segment, cells, at_head=False, at_tail=True))

    _repair(gaps, cells)
    return _result(gaps, cells, original_empty)


def _split_by_attribution(run: list[dict], line_outage_buckets: set[int]) -> list[dict]:
    """把连续空段按「是否全线断采」切成若干子段，返回带 attribution 的段描述。"""
    segments: list[dict] = []
    current: list[dict] = []
    current_attr: str | None = None
    for cell in run:
        attr = "line" if (cell["bucket"] in line_outage_buckets or cell.get("line_outage")) else "device"
        if current_attr is not None and attr != current_attr:
            segments.append({"cells": current, "attribution": current_attr})
            current = []
        current_attr = attr
        current.append(cell)
    if current:
        segments.append({"cells": current, "attribution": current_attr})
    return segments


def _make_item(segment: dict, cells: list[dict], at_head: bool, at_tail: bool) -> dict:
    run = segment["cells"]
    attribution = segment["attribution"]
    length = len(run)
    start_bucket = run[0]["bucket"]
    end_bucket = run[-1]["bucket"]
    prev_cell = _neighbor(cells, start_bucket - 1)
    next_cell = _neighbor(cells, end_bucket + 1)
    has_prev_raw = prev_cell is not None and prev_cell["source"] in ("agg", "sample")
    has_next_raw = next_cell is not None and next_cell["source"] in ("agg", "sample")
    # 只有真正落在序列首/尾的子段才算边界；夹在全线断采里的单设备段不算「首尾」。
    seq_head = at_head and start_bucket == cells[0]["bucket"]
    seq_tail = at_tail and end_bucket == cells[-1]["bucket"]
    boundary = seq_head or seq_tail
    edge_open = not (has_prev_raw and has_next_raw)

    if attribution == "line":
        kind = "line_outage"
        reason = "全线断采：该时段所有设备、所有指标均无原始采样点，留空并列为不可用，不修补"
    elif boundary:
        kind = "unrepaired"
        if seq_head and seq_tail:
            reason = "整个窗口没有原始采样点，无法判定缺口边界"
        elif seq_head:
            reason = "序列开头空缺，首端不外推"
        else:
            reason = "序列结尾空缺，尾端不外推"
    elif edge_open:
        kind = "unrepaired"
        reason = "单设备断采紧邻全线断采，无法在不引入不可用值时插补，整段留空"
    elif length <= settings.INTERPOLATE_MAX_CELLS:
        kind = "interpolated"
        reason = (
            f"连续 {length} 格无原始点（单设备断采），允许线性插补"
            f"（≤{settings.INTERPOLATE_MAX_CELLS} 格）"
        )
    else:
        kind = "unrepaired"
        reason = (
            f"连续 {length} 格无原始点（单设备断采），超过 {settings.INTERPOLATE_MAX_CELLS} 格，"
            "标记未修复"
        )

    return {
        "start_bucket": start_bucket,
        "start_time": run[0]["bucket_start"],
        "end_bucket": end_bucket,
        "end_time": run[-1]["bucket_start"],
        "cells": length,
        "kind": kind,
        "attribution": attribution,
        "boundary": boundary,
        "reason": reason,
        "prev_time": prev_cell["bucket_start"] if has_prev_raw else None,
        "next_time": next_cell["bucket_start"] if has_next_raw else None,
        "filled_values": [],
        "unavailable": attribution == "line",
        # 原始缺口快照：修补后仍保留，便于追溯「这里原本缺了多久、归因是什么」。
        "original": {
            "start_bucket": start_bucket,
            "end_bucket": end_bucket,
            "cells": length,
            "attribution": attribution,
            "kind": kind,
        },
    }


def _repair(gaps: list[dict], cells: list[dict]) -> None:
    """只修补单设备短缺口；全线断采段留空并标记不可用。"""
    by_bucket = {c["bucket"]: c for c in cells}
    for item in gaps:
        run = [by_bucket[b] for b in range(item["start_bucket"], item["end_bucket"] + 1)]
        if item["unavailable"]:
            for cell in run:
                cell["unavailable"] = True
            continue
        if item["kind"] != "interpolated":
            continue
        prev_cell = _neighbor(cells, item["start_bucket"] - 1)
        next_cell = _neighbor(cells, item["end_bucket"] + 1)
        if prev_cell is None or next_cell is None:
            continue
        v0 = prev_cell["value"]
        v1 = next_cell["value"]
        if v0 is None or v1 is None:
            continue
        span = next_cell["bucket"] - prev_cell["bucket"]
        filled: list[float] = []
        for offset, cell in enumerate(run, start=1):
            value = v0 + (v1 - v0) * offset / span
            cell["value"] = value
            cell["mean"] = value
            cell["max"] = value
            cell["min"] = value
            cell["source"] = "interpolated"
            cell["filled"] = True
            filled.append(value)
        item["filled_values"] = filled


def _neighbor(cells: list[dict], bucket: int) -> dict | None:
    for cell in cells:
        if cell["bucket"] == bucket:
            return cell
    return None


def _result(gaps: list[dict], cells: list[dict], original_empty: int) -> dict:
    interpolated = 0
    unrepaired = 0
    boundary = 0
    line_cells = 0
    device_cells = 0
    for item in gaps:
        if item["kind"] == "interpolated":
            interpolated += item["cells"]
            device_cells += item["cells"]
        elif item["attribution"] == "line":
            # 全线断采不是「未修复的设备缺口」，单列不可用，避免与 unrepaired 重复计数。
            line_cells += item["cells"]
        else:
            unrepaired += item["cells"]
            device_cells += item["cells"]
            if item["boundary"]:
                boundary += item["cells"]
    return {
        "items": gaps,
        # 原始缺口记录快照：与修复后的网格解耦，修补不会让记录消失。
        "records": [
            {
                **item["original"],
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "reason": item["reason"],
            }
            for item in gaps
        ],
        "count": len(gaps),
        "device_gap_count": sum(1 for item in gaps if item["attribution"] == "device"),
        "line_outage_gaps": sum(1 for item in gaps if item["attribution"] == "line"),
        "interpolated_cells": interpolated,
        "unrepaired_cells": unrepaired,
        "boundary_cells": boundary,
        "device_gap_cells": device_cells,
        "line_outage_cells": line_cells,
        "unavailable_cells": sum(1 for c in cells if c.get("unavailable")),
        # 修复前网格里的空格数（含全线断采）：用于核对「修补不改写原始缺口记录」。
        "original_empty_cells": original_empty,
    }


def apply_repair(cells: list[dict]) -> list[dict]:
    """返回补齐后的网格（find_gaps 已就地写入插补值，这里做一次显式返回便于测试）。"""
    return cells
