"""JSONL log parsing."""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO

# Common x-axis field names in order of preference
X_AXIS_CANDIDATES = [
    'step', 'iteration', 'iter', 'epoch', 'count',
    'ts', 'time', 'timestamp', 'datetime',
    'i', 't', 'x',
]

_TIMESTAMP_FORMATS = (
    '%Y/%m/%d %H:%M:%S.%f',
    '%Y/%m/%d %H:%M:%S',
    '%Y-%m-%d %H:%M:%S.%f',
    '%Y-%m-%d %H:%M:%S',
    '%Y/%m/%dT%H:%M:%S.%f',
    '%Y/%m/%dT%H:%M:%S',
)


def parse_timestamp(value) -> float | None:
    """Convert a value to epoch seconds. Returns None if not parseable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
    except ValueError:
        pass
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def _normalize_epoch(series: dict) -> dict:
    """If x-values look like epoch seconds, rebase them to seconds-from-start."""
    all_x = []
    for value in series.values():
        if isinstance(value, dict):
            for xs, _ in value.values():
                all_x.extend(xs)
        else:
            xs, _ = value
            all_x.extend(xs)
    if not all_x or min(all_x) <= 1e9:
        return series
    base = min(all_x)
    for key, value in series.items():
        if isinstance(value, dict):
            for gk, (xs, ys) in value.items():
                value[gk] = ([x - base for x in xs], ys)
        else:
            xs, ys = value
            series[key] = ([x - base for x in xs], ys)
    return series


def _clip_pair(xs: list[float], ys: list[float], xmin, xmax):
    """Keep only points whose x falls within [xmin, xmax] (either bound optional)."""
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        if xmin is not None and x < xmin:
            continue
        if xmax is not None and x > xmax:
            continue
        out_x.append(x)
        out_y.append(y)
    return out_x, out_y


def clip_x_range(series: dict, xmin: float | None = None, xmax: float | None = None) -> dict:
    """
    Restrict a series or panel dict to an x-axis window.

    Accepts both flat ({name: (xs, ys)}) and nested
    ({tag: {run: (xs, ys)}}) shapes, mirroring the other helpers here.
    Returns a new dict; empty series are dropped.
    """
    if xmin is None and xmax is None:
        return series

    clipped: dict = {}
    for key, value in series.items():
        if isinstance(value, dict):
            inner = {}
            for gk, (xs, ys) in value.items():
                cx, cy = _clip_pair(xs, ys, xmin, xmax)
                if cx:
                    inner[gk] = (cx, cy)
            if inner:
                clipped[key] = inner
        else:
            xs, ys = value
            cx, cy = _clip_pair(xs, ys, xmin, xmax)
            if cx:
                clipped[key] = (cx, cy)
    return clipped


def smooth_values(values: list[float], weight: float) -> list[float]:
    """
    Apply exponential moving average smoothing.

    Args:
        values: List of values to smooth
        weight: Smoothing weight 0-1 (0=no smoothing, 1=infinite smoothing)

    Returns:
        Smoothed values
    """
    if weight <= 0 or not values:
        return values

    smoothed = []
    last = values[0]
    for v in values:
        # EMA: smoothed = weight * previous + (1 - weight) * current
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def parse_jsonl(
    source: str | TextIO,
    pattern: str | None = None,
) -> Iterator[dict]:
    """
    Parse JSONL from a file path or file-like object.

    Args:
        source: File path, '-' for stdin, or file-like object
        pattern: Optional regex pattern to filter lines before JSON parsing

    Yields:
        Parsed dictionaries from matching JSONL lines
    """
    regex = re.compile(pattern) if pattern else None

    if isinstance(source, str):
        if source == '-':
            file_obj = sys.stdin
            should_close = False
        else:
            file_obj = open(source, 'r')
            should_close = True
    else:
        file_obj = source
        should_close = False

    try:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue

            # Apply pattern filter if specified
            if regex and not regex.search(line):
                continue

            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    yield data
            except json.JSONDecodeError:
                # Skip non-JSON lines silently
                continue
    finally:
        if should_close:
            file_obj.close()


def detect_x_axis(records: list[dict]) -> str | None:
    """
    Auto-detect the x-axis field from a list of records.

    Returns the first matching candidate that exists in all records
    and contains numeric values.
    """
    if not records:
        return None

    # Find fields present in all records with numeric values
    common_fields = set(records[0].keys())
    for record in records[1:]:
        common_fields &= set(record.keys())

    # Check candidates in order of preference (numeric or timestamp strings)
    for candidate in X_AXIS_CANDIDATES:
        if candidate in common_fields:
            if all(parse_timestamp(r.get(candidate)) is not None for r in records):
                return candidate

    # Fallback: find any numeric field that looks like an index
    for field in sorted(common_fields):
        values = [r.get(field) for r in records]
        if all(isinstance(v, (int, float)) for v in values):
            # Check if values are monotonically increasing (likely an index)
            if all(values[i] <= values[i + 1] for i in range(len(values) - 1)):
                return field

    return None


def extract_series(
    records: list[dict],
    x_axis: str,
    y_fields: list[str] | None = None,
) -> dict[str, tuple[list[float], list[float]]]:
    """
    Extract x and y values for each series from records.

    Args:
        records: List of parsed JSONL records
        x_axis: Field name to use as x-axis
        y_fields: Specific fields to plot, or None for all numeric fields

    Returns:
        Dict mapping field name to (x_values, y_values) tuples
    """
    if not records:
        return {}

    # Determine which fields to plot
    if y_fields:
        fields = y_fields
    else:
        # Auto-detect: all numeric fields except x_axis
        sample = records[0]
        fields = [
            k for k, v in sample.items()
            if k != x_axis and isinstance(v, (int, float))
        ]

    # Extract series data
    series = {}
    for field in fields:
        x_values = []
        y_values = []
        for record in records:
            x = parse_timestamp(record.get(x_axis))
            y = record.get(field)
            if x is not None and isinstance(y, (int, float)) and not isinstance(y, bool):
                x_values.append(x)
                y_values.append(float(y))
        if x_values:
            series[field] = (x_values, y_values)

    return _normalize_epoch(series)


def extract_grouped_series(
    records: list[dict],
    x_axis: str,
    y_fields: list[str] | None,
    group_by: str,
) -> dict[str, dict[str, tuple[list[float], list[float]]]]:
    """
    Extract series split by a categorical field.

    Returns a panel-shaped dict: {y_field: {group_value: (x_values, y_values)}}
    so each distinct value of `group_by` becomes its own curve on the y_field panel.
    """
    if not records:
        return {}

    if y_fields:
        fields = y_fields
    else:
        sample = records[0]
        fields = [
            k for k, v in sample.items()
            if k != x_axis and k != group_by
            and isinstance(v, (int, float)) and not isinstance(v, bool)
        ]

    result: dict[str, dict[str, tuple[list[float], list[float]]]] = {f: {} for f in fields}
    for record in records:
        x = parse_timestamp(record.get(x_axis))
        g = record.get(group_by)
        if x is None or g is None:
            continue
        g_key = f"{group_by}={g}"
        for f in fields:
            y = record.get(f)
            if not isinstance(y, (int, float)) or isinstance(y, bool):
                continue
            bucket = result[f].setdefault(g_key, ([], []))
            bucket[0].append(x)
            bucket[1].append(float(y))

    result = {k: v for k, v in result.items() if v}
    return _normalize_epoch(result)


def extract_paneled_series(
    records: list[dict],
    x_axis: str,
    y_fields: list[str] | None,
    panel_by: str,
) -> dict[str, dict[str, tuple[list[float], list[float]]]]:
    """
    One panel per distinct value of `panel_by`, with each y-field as a curve inside.

    Returns: {panel_value: {y_field: (x_values, y_values)}}
    """
    if not records:
        return {}

    if y_fields:
        fields = y_fields
    else:
        sample = records[0]
        fields = [
            k for k, v in sample.items()
            if k != x_axis and k != panel_by
            and isinstance(v, (int, float)) and not isinstance(v, bool)
        ]

    result: dict[str, dict[str, tuple[list[float], list[float]]]] = {}
    for record in records:
        x = parse_timestamp(record.get(x_axis))
        p = record.get(panel_by)
        if x is None or p is None:
            continue
        p_key = f"{panel_by}={p}"
        panel = result.setdefault(p_key, {})
        for f in fields:
            y = record.get(f)
            if not isinstance(y, (int, float)) or isinstance(y, bool):
                continue
            bucket = panel.setdefault(f, ([], []))
            bucket[0].append(x)
            bucket[1].append(float(y))

    result = {p: v for p, v in result.items() if v}
    return _normalize_epoch(result)
