"""质量报告：覆盖率、可用率、分级与跨设备对比。

coverage = 有效格 / 应到格；有效格 = 重采样后**有值**的格子（原始 / 沿用 / 单设备短缺口插补）。
availability = 可用格 / 应到格；可用格 = 原始 + 沿用（不含插补出来的值），
用来区分「真有数据」与「补出来的数据」。
全线断采格留空、不可用，计入分母但不计有效格/可用格，因此采集链路停过会直接压低两项指标。
分级：>= 0.98 A，>= 0.95 B，其余 C。
"""
from __future__ import annotations

import csv
import hashlib
import io
import json

from . import settings


def grade_for(coverage: float | None) -> str:
    if coverage is None:
        return "-"
    if coverage >= settings.GRADE_A_COVERAGE:
        return "A"
    if coverage >= settings.GRADE_B_COVERAGE:
        return "B"
    return "C"


def coverage_report(
    cells: list[dict],
    expected_cells: int,
    gaps: dict,
    line_outage: dict | None = None,
) -> dict:
    values = [c for c in cells if c["value"] is not None]
    raw = [c for c in cells if c["source"] in ("agg", "sample")]
    carried = [c for c in cells if c.get("carried")]
    interpolated = [c for c in cells if c["source"] == "interpolated"]
    missing = [c for c in cells if c["value"] is None]
    unavailable = [c for c in cells if c.get("unavailable")]
    expected = max(1, expected_cells)
    coverage = len(values) / expected
    # 可用 = 原始 + 沿用（不含插补）：与 coverage 分开，避免「补出来的值」抬高可用率。
    available = len(raw) + len(carried)
    return {
        "expected_cells": expected,
        "grid_cells": len(cells),
        "raw_cells": len(raw),
        "carried_cells": len(carried),
        "interpolated_cells": len(interpolated),
        "missing_cells": len(missing),
        "unavailable_cells": len(unavailable),
        "line_outage_cells": sum(1 for c in cells if c.get("line_outage")),
        "valid_cells": len(values),
        "available_cells": available,
        "coverage": coverage,
        "availability": available / expected,
        "raw_coverage": len(raw) / expected,
        "grade": grade_for(coverage),
        "gap_count": gaps.get("device_gap_count", 0),
        "device_gap_count": gaps.get("device_gap_count", 0),
        "line_outage_gap_count": gaps.get("line_outage_gaps", 0),
        "gap_cells": gaps.get("interpolated_cells", 0) + gaps.get("unrepaired_cells", 0),
        "device_gap_cells": gaps.get("device_gap_cells", 0),
        "line_outage_gap_cells": gaps.get("line_outage_cells", 0),
        "unrepaired_cells": gaps.get("unrepaired_cells", 0),
        "line_outage_windows": (line_outage or {}).get("windows", []),
        "line_outage_labels": _window_labels((line_outage or {}).get("windows", [])),
    }


def _window_labels(windows: list[dict]) -> str:
    return "; ".join(f"{w['start_time'][11:16]}–{w['end_time'][11:16]}" for w in windows)


def device_rollup(rows: list[dict], line_outage: dict | None = None) -> list[dict]:
    """按设备汇总（取该设备最差指标的覆盖率与分级），并附该设备全线断采不可用格。"""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["device_id"], []).append(row)
    outage_windows = (line_outage or {}).get("windows", [])
    out: list[dict] = []
    for device, items in grouped.items():
        worst = min(items, key=lambda r: r["coverage"]["coverage"])
        drift_rows = [r for r in items if r.get("drift") and r["drift"].get("offset_s") is not None]
        offset = None
        if drift_rows:
            offset = drift_rows[0]["drift"]["offset_s"]
        device_windows = [
            {
                "start_time": w["start_time"],
                "end_time": w["end_time"],
                "cells": w["cells"],
                "availability": next(
                    (d["availability"] for d in w["devices"] if d["device_id"] == device),
                    None,
                ),
            }
            for w in outage_windows
        ]
        out.append(
            {
                "device_id": device,
                "interval_s": items[0]["interval_s"],
                "metrics": [r["metric"] for r in sorted(items, key=lambda r: r["metric"])],
                "coverage": worst["coverage"]["coverage"],
                "availability": worst["coverage"]["availability"],
                "grade": worst["coverage"]["grade"],
                "worst_metric": worst["metric"],
                "gap_count": sum(r["coverage"]["device_gap_count"] for r in items),
                "carried_cells": sum(r["coverage"]["carried_cells"] for r in items),
                "unrepaired_cells": sum(r["coverage"]["unrepaired_cells"] for r in items),
                "line_outage_cells": sum(r["coverage"]["line_outage_cells"] for r in items),
                "unavailable_cells": sum(r["coverage"]["unavailable_cells"] for r in items),
                "line_outage_windows": device_windows,
                "offset_s": offset,
                "drifted": any(r.get("drift") and r["drift"].get("drifted") for r in items),
            }
        )
    out.sort(key=lambda d: d["device_id"])
    return out


def report_summary(rows: list[dict], line_outage: dict | None = None) -> dict:
    grades = [r["coverage"]["grade"] for r in rows if r["coverage"]["grade"] != "-"]
    counts = {"A": 0, "B": 0, "C": 0}
    for g in grades:
        counts[g] = counts.get(g, 0) + 1
    drifted = [r for r in rows if r.get("drift") and r["drift"].get("drifted")]
    windows = (line_outage or {}).get("windows", [])
    return {
        "pairs": len(rows),
        "grade_counts": counts,
        "drifted_pairs": len(drifted),
        "avg_coverage": sum(r["coverage"]["coverage"] for r in rows) / len(rows) if rows else None,
        "total_gaps": sum(r["coverage"]["gap_count"] for r in rows),
        "total_device_gaps": sum(r["coverage"]["device_gap_count"] for r in rows),
        "total_carried": sum(r["coverage"]["carried_cells"] for r in rows),
        "total_unrepaired": sum(r["coverage"]["unrepaired_cells"] for r in rows),
        "total_unavailable": sum(r["coverage"]["unavailable_cells"] for r in rows),
        "line_outage_windows": windows,
        "line_outage_labels": _window_labels(windows),
        "line_outage_cells": (line_outage or {}).get("cells", 0),
        "affected_devices": (line_outage or {}).get("affected_devices", []),
        "availability": (line_outage or {}).get("availability"),
    }


CSV_COLUMNS = [
    "device_id",
    "metric",
    "interval_s",
    "expected_cells",
    "raw_cells",
    "carried_cells",
    "interpolated_cells",
    "unrepaired_cells",
    "line_outage_cells",
    "unavailable_cells",
    "valid_cells",
    "coverage",
    "availability",
    "grade",
    "gap_count",
    "line_outage_windows",
    "offset_s",
    "rate_ppm",
    "rmse_s",
    "drifted",
]

OUTAGE_CSV_COLUMNS = ["window_start", "window_end", "cells", "device_id", "availability"]


def report_csv(rows: list[dict], line_outage: dict | None = None) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        drift = row.get("drift") or {}
        cov = row["coverage"]
        writer.writerow(
            [
                row["device_id"],
                row["metric"],
                row["interval_s"],
                cov["expected_cells"],
                cov["raw_cells"],
                cov["carried_cells"],
                cov["interpolated_cells"],
                cov["unrepaired_cells"],
                cov["line_outage_cells"],
                cov["unavailable_cells"],
                cov["valid_cells"],
                f"{cov['coverage']:.4f}",
                f"{cov['availability']:.4f}",
                cov["grade"],
                cov["gap_count"],
                cov["line_outage_labels"],
                "" if drift.get("offset_s") is None else f"{drift['offset_s']:.3f}",
                "" if drift.get("rate_ppm") is None else f"{drift['rate_ppm']:.1f}",
                "" if drift.get("rmse_s") is None else f"{drift['rmse_s']:.3f}",
                "" if drift.get("drifted") is None else ("yes" if drift["drifted"] else "no"),
            ]
        )
    # 第二段：全线断采时段 × 受影响设备 × 该时段可用率（与 JSON 报告同一批窗口）。
    writer.writerow([])
    writer.writerow(["# line_outages"] + ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
    writer.writerow(OUTAGE_CSV_COLUMNS)
    for window in (line_outage or {}).get("windows", []):
        for device in window["devices"]:
            writer.writerow(
                [
                    window["start_time"],
                    window["end_time"],
                    window["cells"],
                    device["device_id"],
                    "" if device["availability"] is None else f"{device['availability']:.4f}",
                ]
            )
    return buffer.getvalue()


def report_id(dataset_id: str, threshold_s: float, content: dict) -> str:
    """内容指纹：同一数据集 + 同一阈值 + 同一结论 → 同一个报告 id（重跑不产生新报告）。"""
    canonical = json.dumps(
        {"dataset_id": dataset_id, "threshold_s": threshold_s, "content": content},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "report-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
