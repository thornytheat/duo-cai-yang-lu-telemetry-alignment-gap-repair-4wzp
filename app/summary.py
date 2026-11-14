"""小时摘要：每小时 min / max / mean / p95 + 缺口 / carried / 全线断采 + 可用率。

口径与 grid / gaps / quality 一致：
- 值统计只用「有效格」：原始（agg/sample）、hold 沿用、单设备短缺口插补。
  全线断采格留空、标记不可用，不参与 min/max/mean/p95。
- `gap_cells` 与 `unavailable_cells` 分开计数：前者是缺口格（含插补与未修复），
  后者是采集链路断了、不可用的格；任一非 0 都能在摘要里直接看出。
- `hour_availability` = 可用格 / 应到格，可用格 = 原始 + 沿用（与质量报告的 availability 同口径，
  不含插补），因此补出来的值不会掩盖采集链路停过。

p95 用线性插值分位数（与 statistics.quantiles(method='inclusive') 口径一致，但手写以便
无 numpy 也能精确控制边界）。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from . import settings


def percentile(values: list[float], p: float) -> float | None:
    """线性插值分位数，p ∈ [0, 100]。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    if p <= 0:
        return ordered[0]
    if p >= 100:
        return ordered[-1]
    position = (len(ordered) - 1) * p / 100.0
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    frac = position - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def hourly_summary(cells: list[dict], cells_per_hour: int = 60) -> list[dict]:
    """按小时聚合网格（只统计有效格；空/未修复/全线断采格计入缺口与不可用数）。"""
    buckets: dict[int, list[dict]] = {}
    for cell in cells:
        hour = cell["bucket"] // cells_per_hour
        buckets.setdefault(hour, []).append(cell)

    out: list[dict] = []
    for hour in sorted(buckets):
        group = sorted(buckets[hour], key=lambda c: c["bucket"])
        valid = [c for c in group if c["value"] is not None]
        values = [c["value"] for c in valid]
        interpolated = sum(1 for c in group if c["source"] == "interpolated")
        unavailable = sum(1 for c in group if c.get("unavailable"))
        line_outage = sum(1 for c in group if c.get("line_outage"))
        unrepaired = sum(
            1 for c in group if c["source"] == "empty" and not c.get("unavailable")
        )
        # 缺口格 = 单设备缺口（已插补 + 未修复）；全线断采单列不可用，不重复计入缺口。
        gap_cells = interpolated + unrepaired
        carried = sum(1 for c in group if c["carried"])
        raw = sum(1 for c in group if c["source"] in ("agg", "sample"))
        available = raw + carried
        out.append(
            {
                "hour_index": hour,
                "hour_start": _iso(hour * cells_per_hour * settings.BUCKET_SECONDS),
                "cells": len(group),
                "value_cells": len(values),
                "valid_cells": len(valid),
                "raw_cells": raw,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": sum(values) / len(values) if values else None,
                "p95": percentile(values, 95),
                "gap_cells": gap_cells,
                "interpolated_cells": interpolated,
                "unrepaired_cells": unrepaired,
                "unavailable_cells": unavailable,
                "line_outage_cells": line_outage,
                "carried_cells": carried,
                "available_cells": available,
                "availability": available / len(group) if group else None,
                "status": _status(len(group), len(valid), unavailable),
            }
        )
    return out


def _status(cells: int, valid: int, unavailable: int) -> str:
    if unavailable and unavailable == cells:
        return "unavailable"
    if unavailable:
        return "partial"
    if valid < cells:
        return "gaps"
    return "ok"


def _iso(epoch: float) -> str:
    tz = timezone(timedelta(hours=8))
    return datetime.fromtimestamp(epoch, tz=tz).isoformat(timespec="seconds")
