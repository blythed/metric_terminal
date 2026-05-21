"""JSONL and TensorBoard log parsing."""

import json
import os
import re
import sys
from pathlib import Path
from typing import Iterator, TextIO

# Common x-axis field names in order of preference
X_AXIS_CANDIDATES = ['step', 'iteration', 'iter', 'epoch', 'count', 'i', 't', 'x']


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


def compute_stats(steps: list[float], values: list[float]) -> dict:
    """
    Compute statistics for a series.

    Returns dict with: current, start, min, max, mean, change_pct,
    recent_change_pct, trend, steps
    """
    if not values:
        return {}

    n = len(values)
    current = values[-1]
    start = values[0]
    min_val = min(values)
    max_val = max(values)
    min_step = steps[values.index(min_val)]
    max_step = steps[values.index(max_val)]
    mean_val = sum(values) / n

    # Change from start
    if start != 0:
        change_pct = ((current - start) / abs(start)) * 100
    else:
        change_pct = 0.0 if current == 0 else float('inf')

    # Recent change (last 10% of data or at least 10 points)
    recent_n = max(10, n // 10)
    if n > recent_n:
        recent_start = values[-recent_n]
        if recent_start != 0:
            recent_change_pct = ((current - recent_start) / abs(recent_start)) * 100
        else:
            recent_change_pct = 0.0
    else:
        recent_change_pct = change_pct

    # Trend detection based on recent slope
    if n >= 10:
        recent_vals = values[-min(recent_n, n):]
        # Simple linear regression slope
        x_mean = (len(recent_vals) - 1) / 2
        y_mean = sum(recent_vals) / len(recent_vals)
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(recent_vals))
        denominator = sum((i - x_mean) ** 2 for i in range(len(recent_vals)))
        slope = numerator / denominator if denominator != 0 else 0

        # Normalize slope by mean to get relative change
        if y_mean != 0:
            rel_slope = slope / abs(y_mean)
        else:
            rel_slope = 0

        if abs(recent_change_pct) < 1:
            trend = 'plateau'
        elif rel_slope > 0.001:
            trend = 'increasing'
        elif rel_slope < -0.001:
            trend = 'decreasing'
        else:
            trend = 'plateau'
    else:
        trend = 'insufficient_data'

    return {
        'current': current,
        'start': start,
        'min': {'value': min_val, 'step': min_step},
        'max': {'value': max_val, 'step': max_step},
        'mean': mean_val,
        'change_pct': round(change_pct, 1),
        'recent_change_pct': round(recent_change_pct, 1),
        'trend': trend,
        'steps': n,
    }


def is_tensorboard_dir(path: str) -> bool:
    """Check if path is a TensorBoard log directory."""
    if not os.path.isdir(path):
        return False
    # Look for tfevents files
    for root, dirs, files in os.walk(path):
        for f in files:
            if 'tfevents' in f:
                return True
    return False


def is_tensorboard_file(path: str) -> bool:
    """Check if path is a TensorBoard event file."""
    return os.path.isfile(path) and 'tfevents' in os.path.basename(path)


def parse_tensorboard(
    path: str,
    tags: list[str] | None = None,
    run_name: str | None = None,
) -> dict[str, dict[str, tuple[list[float], list[float]]]]:
    """
    Parse TensorBoard logs and return series data grouped by tag.

    Args:
        path: Path to TensorBoard log directory or event file
        tags: Optional list of specific tags to extract
        run_name: Optional name for this run (used when comparing multiple experiments)

    Returns:
        Dict mapping tag name to dict of {run_name: (steps, values)}
        This allows multiple runs to be overlaid per metric.
    """
    try:
        from tbparse import SummaryReader
    except ImportError:
        raise ImportError(
            "tbparse is required for TensorBoard support. "
            "Install with: pip install tbparse"
        )

    reader = SummaryReader(path, extra_columns={'dir_name'})
    df = reader.scalars

    if df.empty:
        return {}

    available_tags = df['tag'].unique()

    # Filter to requested tags if specified
    if tags:
        target_tags = [t for t in tags if t in available_tags]
    else:
        target_tags = list(available_tags)

    # Group by tag, then by run (dir_name)
    result = {}
    for tag in target_tags:
        tag_df = df[df['tag'] == tag]
        runs = tag_df['dir_name'].unique()

        tag_series = {}
        for run in runs:
            run_df = tag_df[tag_df['dir_name'] == run].sort_values('step')
            steps = run_df['step'].tolist()
            values = run_df['value'].tolist()
            if steps:
                # Determine run label
                if run_name:
                    # Use provided run name, append subdir if present
                    label = f"{run_name}/{run}" if run else run_name
                else:
                    label = run if run else 'default'
                tag_series[label] = (steps, values)

        if tag_series:
            result[tag] = tag_series

    return result


def merge_tensorboard_panels(
    *panel_dicts: dict[str, dict[str, tuple[list[float], list[float]]]]
) -> dict[str, dict[str, tuple[list[float], list[float]]]]:
    """
    Merge multiple panel dicts from different TensorBoard directories.

    Args:
        panel_dicts: Multiple dicts from parse_tensorboard()

    Returns:
        Merged dict with all runs combined per tag
    """
    merged = {}
    for panels in panel_dicts:
        for tag, runs in panels.items():
            if tag not in merged:
                merged[tag] = {}
            merged[tag].update(runs)
    return merged


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

    # Check candidates in order of preference
    for candidate in X_AXIS_CANDIDATES:
        if candidate in common_fields:
            # Verify it contains numeric values
            if all(isinstance(r.get(candidate), (int, float)) for r in records):
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
            x = record.get(x_axis)
            y = record.get(field)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                x_values.append(float(x))
                y_values.append(float(y))
        if x_values:
            series[field] = (x_values, y_values)

    return series
