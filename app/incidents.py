"""全线断采归因：先看采集链路，再看单条序列。

判定口径（全项目唯一一处）：
- 某一分钟桶里**当前数据集的任何序列都没有原始采样点** → 该桶为「全线断采」
  （采集链路整体停过，不能归因到某一台设备，也不能修补）。
- 只要还有任意一条序列在该桶有原始点，其余序列的空缺才可能是「单设备断采」。

统一网格、缺口归因、小时摘要、质量报告都从这里取同一份桶集合，避免各自解释。
"""
from __future__ import annotations

from . import grid, settings


def _window(dataset, start_bucket: int | None, end_bucket: int | None) -> tuple[int, int]:
    start = int(dataset.window_start // settings.BUCKET_SECONDS) if start_bucket is None else start_bucket
    end = int(dataset.window_end // settings.BUCKET_SECONDS) if end_bucket is None else end_bucket
    return start, end


def raw_bucket_set(dataset, start_bucket: int | None = None, end_bucket: int | None = None) -> set[int]:
    """数据集内任意序列有原始点的桶集合（只看窗口内）。"""
    start, end = _window(dataset, start_bucket, end_bucket)
    covered: set[int] = set()
    for series in dataset.series.values():
        for ts, _ in series.points:
            bucket = grid.bucket_of(ts)
            if start <= bucket <= end:
                covered.add(bucket)
    return covered


def line_outage_buckets(
    dataset,
    start_bucket: int | None = None,
    end_bucket: int | None = None,
) -> set[int]:
    """返回窗口内没有任何序列原始点的桶（全线断采）。"""
    start, end = _window(dataset, start_bucket, end_bucket)
    covered = raw_bucket_set(dataset, start, end)
    return {bucket for bucket in range(start, end + 1) if bucket not in covered}


def contiguous_windows(buckets: set[int]) -> list[dict]:
    """把桶集合合并成连续窗口，时间口径与网格一致。"""
    ordered = sorted(buckets)
    if not ordered:
        return []
    runs: list[list[int]] = []
    for bucket in ordered:
        if runs and bucket == runs[-1][-1] + 1:
            runs[-1].append(bucket)
        else:
            runs.append([bucket])
    return [
        {
            "start_bucket": run[0],
            "end_bucket": run[-1],
            "start_time": grid.bucket_start_iso(run[0]),
            "end_time": grid.bucket_start_iso(run[-1]),
            "cells": len(run),
        }
        for run in runs
    ]


def _device_availability(dataset, device: str, start_bucket: int, end_bucket: int) -> dict:
    """设备在该时段的可用率 = 有原始点的「设备 × 指标 × 桶」数 / 应到格。

    按桶去重：1s 设备一个桶里有很多原始点，只能算 1 格。
    """
    metrics = dataset.metrics_for(device)
    expected_per_metric = end_bucket - start_bucket + 1
    expected = expected_per_metric * max(1, len(metrics))
    available = 0
    for metric in metrics:
        series = dataset.series_for(device, metric)
        if series is None:
            continue
        buckets = {
            grid.bucket_of(ts)
            for ts, _ in series.points
            if start_bucket <= grid.bucket_of(ts) <= end_bucket
        }
        available += len(buckets)
    return {
        "device_id": device,
        "metrics": metrics,
        "expected_cells": expected,
        "raw_cells": available,
        "unavailable_cells": expected - available,
        "availability": available / expected if expected else None,
    }


def describe(
    dataset,
    buckets: set[int] | None = None,
    start_bucket: int | None = None,
    end_bucket: int | None = None,
) -> dict:
    """汇总全线断采：窗口 + 受影响设备 + 各设备在该时段的可用率。"""
    if buckets is None:
        buckets = line_outage_buckets(dataset, start_bucket, end_bucket)
    windows = contiguous_windows(buckets)
    devices = dataset.devices()
    detailed: list[dict] = []
    for window in windows:
        per_device = [
            _device_availability(dataset, device, window["start_bucket"], window["end_bucket"])
            for device in devices
        ]
        expected = sum(item["expected_cells"] for item in per_device)
        raw = sum(item["raw_cells"] for item in per_device)
        detailed.append(
            {
                **window,
                "reason": "全线断采：该时段所有设备、所有指标均无原始采样点，采集链路中断",
                "affected_devices": [item["device_id"] for item in per_device],
                "devices": per_device,
                "expected_cells": expected,
                "raw_cells": raw,
                "availability": raw / expected if expected else None,
            }
        )
    return {
        "buckets": buckets,
        "windows": detailed,
        "count": len(detailed),
        "cells": len(buckets),
        "affected_devices": devices,
        "availability": (
            sum(item["raw_cells"] for item in detailed) / sum(item["expected_cells"] for item in detailed)
            if detailed and sum(item["expected_cells"] for item in detailed)
            else None
        ),
    }


def window_label(windows: list[dict]) -> str:
    """紧凑时段标签，用于 CSV / 页面：03:00–03:02; 04:10–04:14。"""
    return "; ".join(
        f"{window['start_time'][11:16]}–{window['end_time'][11:16]}" for window in windows
    )
