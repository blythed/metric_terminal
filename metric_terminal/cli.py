"""Command-line interface."""

import argparse
import glob
import os
import sys
from pathlib import Path

import json as json_module

from .parser import (
    parse_jsonl,
    detect_x_axis,
    extract_series,
    is_tensorboard_dir,
    is_tensorboard_file,
    parse_tensorboard,
    merge_tensorboard_panels,
    smooth_values,
    compute_stats,
)
from .chart import LineChart
from .renderers import get_renderer


def get_terminal_size() -> tuple[int, int]:
    """Get terminal width and height."""
    try:
        size = os.get_terminal_size()
        return size.columns, size.lines
    except OSError:
        return 80, 24


def format_value(v: float, width: int = 12) -> str:
    """Format a numeric value for display."""
    if abs(v) >= 1000:
        return f"{v:>{width},.1f}"
    elif abs(v) >= 1:
        return f"{v:>{width}.3f}"
    elif abs(v) >= 0.001:
        return f"{v:>{width}.4f}"
    else:
        return f"{v:>{width}.2e}"


def format_stats_table(all_stats: dict) -> str:
    """
    Format statistics as a table.

    Single run: vertical format
    Multiple runs: side-by-side table
    """
    lines = []

    for tag, runs in all_stats.items():
        run_names = list(runs.keys())
        num_runs = len(run_names)

        if num_runs == 0:
            continue

        # Determine column width based on run names
        col_width = max(12, max(len(n) for n in run_names) + 2)

        if num_runs == 1:
            # Single run - vertical format
            run_name = run_names[0]
            stats = runs[run_name]
            lines.append(f"{tag}:")
            lines.append(f"  current:  {format_value(stats['current'])}")
            lines.append(f"  start:    {format_value(stats['start'])}")
            lines.append(f"  min:      {format_value(stats['min']['value'])} (step {int(stats['min']['step'])})")
            lines.append(f"  max:      {format_value(stats['max']['value'])} (step {int(stats['max']['step'])})")
            lines.append(f"  mean:     {format_value(stats['mean'])}")
            lines.append(f"  change:   {stats['change_pct']:+.1f}%")
            lines.append(f"  recent:   {stats['recent_change_pct']:+.1f}%")
            lines.append(f"  trend:    {stats['trend']}")
            lines.append(f"  steps:    {stats['steps']}")
            lines.append("")
        else:
            # Multiple runs - side-by-side table
            # Header
            header = f"{tag:<20}" + "".join(f"{n:>{col_width}}" for n in run_names)
            lines.append(header)

            # Rows
            rows = [
                ('current', lambda s: format_value(s['current'], col_width)),
                ('start', lambda s: format_value(s['start'], col_width)),
                ('min', lambda s: format_value(s['min']['value'], col_width)),
                ('max', lambda s: format_value(s['max']['value'], col_width)),
                ('mean', lambda s: format_value(s['mean'], col_width)),
                ('change', lambda s: f"{s['change_pct']:>{col_width-1}.1f}%"),
                ('recent', lambda s: f"{s['recent_change_pct']:>{col_width-1}.1f}%"),
                ('trend', lambda s: f"{s['trend']:>{col_width}}"),
                ('steps', lambda s: f"{s['steps']:>{col_width}}"),
            ]

            for row_name, formatter in rows:
                row = f"  {row_name:<18}"
                for run_name in run_names:
                    stats = runs[run_name]
                    row += formatter(stats)
                lines.append(row)
            lines.append("")

    return "\n".join(lines)


def get_run_name(path: str) -> str:
    """Extract a short run name from a path."""
    p = Path(path).resolve()
    # Use parent directory name if path ends with common tensorboard dir names
    if p.name.lower() in ('tensorboard', 'logs', 'events'):
        return p.parent.name
    return p.name


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog='metric-terminal',
        description='Display metrics as ASCII charts in the terminal',
    )
    parser.add_argument(
        'file',
        nargs='*',
        default=['-'],
        help='Input file(s), TensorBoard log directory(s), or - for stdin',
    )
    parser.add_argument(
        '-p', '--pattern',
        help='Regex pattern to filter lines (JSONL mode only)',
    )
    parser.add_argument(
        '-x', '--x-axis',
        dest='x_axis',
        help='Field to use as x-axis (auto-detect if omitted)',
    )
    parser.add_argument(
        '-y', '--y-axis',
        dest='y_fields',
        action='append',
        help='Field(s)/tag(s) to plot (can be specified multiple times)',
    )
    parser.add_argument(
        '-w', '--width',
        type=int,
        help='Chart width (default: terminal width)',
    )
    parser.add_argument(
        '-H', '--height',
        type=int,
        help='Chart height per panel (default: auto based on terminal)',
    )
    parser.add_argument(
        '--ascii',
        action='store_true',
        help='Use plain ASCII (no Unicode/colors)',
    )
    parser.add_argument(
        '--title',
        help='Chart title',
    )
    parser.add_argument(
        '--tensorboard', '--tb',
        action='store_true',
        help='Force TensorBoard mode (auto-detected if path contains tfevents)',
    )
    parser.add_argument(
        '--overlay',
        action='store_true',
        help='Overlay all metrics on single chart (default for JSONL)',
    )
    parser.add_argument(
        '--list-tags',
        action='store_true',
        help='List available TensorBoard tags and exit',
    )
    parser.add_argument(
        '-s', '--smooth',
        type=float,
        default=0.0,
        metavar='WEIGHT',
        help='Smoothing weight 0-1 (0=none, 0.9=heavy smoothing, like TensorBoard)',
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics instead of graph (side-by-side table for multiple runs)',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output stats as JSON (use with --stats)',
    )

    args = parser.parse_args(argv)

    # Handle empty file list and expand globs
    raw_files = args.file if args.file else ['-']
    files = []
    for f in raw_files:
        if f == '-':
            files.append(f)
        elif '*' in f or '?' in f or '[' in f:
            # Expand glob pattern
            expanded = sorted(glob.glob(f))
            if expanded:
                files.extend(expanded)
            else:
                # No matches, keep original (will error later)
                files.append(f)
        else:
            files.append(f)

    # Determine dimensions
    term_width, term_height = get_terminal_size()
    width = args.width if args.width else term_width
    # Auto-size height: use ~60% of terminal height, minimum 15
    height = args.height if args.height else max(15, int(term_height * 0.6))

    # Detect input type - check if any file is TensorBoard
    is_tb = args.tensorboard or any(
        is_tensorboard_dir(f) or is_tensorboard_file(f)
        for f in files if f != '-'
    )

    renderer = get_renderer(ascii_only=args.ascii)

    if is_tb:
        # TensorBoard mode - separate panels per metric by default
        # Filter to only existing TensorBoard directories (exclude unexpanded globs)
        tb_dirs = [
            f for f in files
            if f != '-'
            and '*' not in f and '?' not in f and '[' not in f  # Skip unexpanded globs
            and (is_tensorboard_dir(f) or is_tensorboard_file(f))
        ]

        if not tb_dirs:
            print('No TensorBoard logs found', file=sys.stderr)
            return 1

        # Parse each directory and merge
        all_panels = []
        for f in tb_dirs:
            try:
                # Use directory name as run identifier when comparing multiple
                run_name = get_run_name(f) if len(tb_dirs) > 1 else None
                # For --list-tags, parse all tags (don't filter)
                tags_to_parse = None if args.list_tags else args.y_fields
                panels = parse_tensorboard(f, tags_to_parse, run_name=run_name)
                all_panels.append(panels)
            except ImportError as e:
                print(f'Error: {e}', file=sys.stderr)
                return 1
            except Exception as e:
                print(f'Error reading TensorBoard logs from {f}: {e}', file=sys.stderr)
                return 1

        if not all_panels:
            print('No TensorBoard logs found', file=sys.stderr)
            return 1

        # Merge all panels
        panels = merge_tensorboard_panels(*all_panels)

        # Apply smoothing if requested
        if args.smooth > 0:
            for tag in panels:
                for run_name in panels[tag]:
                    steps, values = panels[tag][run_name]
                    panels[tag][run_name] = (steps, smooth_values(values, args.smooth))

        # If --list-tags, just print available tags and exit
        if args.list_tags:
            print('Available tags:')
            for tag in sorted(panels.keys()):
                runs = list(panels[tag].keys())
                print(f'  {tag}  ({len(runs)} run{"s" if len(runs) > 1 else ""})')
            return 0

        if not panels:
            print('No scalar data found in TensorBoard logs', file=sys.stderr)
            return 1

        # Stats mode - show statistics instead of graph
        if args.stats:
            all_stats = {}
            for tag, runs in panels.items():
                all_stats[tag] = {}
                for run_name, (steps, values) in runs.items():
                    all_stats[tag][run_name] = compute_stats(steps, values)

            if args.json:
                print(json_module.dumps(all_stats, indent=2))
            else:
                print(format_stats_table(all_stats))
            return 0

        if args.overlay:
            # Flatten to single chart overlay mode
            chart = LineChart(title=args.title or '', x_label='step')
            for tag, runs in panels.items():
                for run_name, (steps, values) in runs.items():
                    label = f"{tag}" if run_name == 'default' else f"{tag}/{run_name}"
                    chart.add_series(label, steps, values)
            output = renderer.render(chart, width, height)
        else:
            # Separate panels per metric (TensorBoard style)
            from .renderers.plotext_renderer import PlotextRenderer
            if isinstance(renderer, PlotextRenderer):
                output = renderer.render_panels(panels, width, height, x_label='step')
            else:
                # Fallback for ASCII renderer - just render each separately
                outputs = []
                for tag, runs in panels.items():
                    chart = LineChart(title=tag, x_label='step')
                    for run_name, (steps, values) in runs.items():
                        chart.add_series(run_name, steps, values)
                    outputs.append(renderer.render(chart, width, height))
                output = '\n'.join(outputs)

        print(output)
        return 0

    # JSONL mode - overlay by default (use first file only)
    input_file = files[0]
    try:
        records = list(parse_jsonl(input_file, args.pattern))
    except FileNotFoundError:
        print(f'Error: File not found: {input_file}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130

    if not records:
        print('No matching records found', file=sys.stderr)
        return 1

    # Detect or use specified x-axis
    x_axis = args.x_axis or detect_x_axis(records)
    if not x_axis:
        print('Error: Could not detect x-axis field. Use -x to specify.', file=sys.stderr)
        return 1

    # Extract series
    series_data = extract_series(records, x_axis, args.y_fields)
    if not series_data:
        print('Error: No numeric data found to plot', file=sys.stderr)
        return 1

    # Apply smoothing if requested
    if args.smooth > 0:
        for name in series_data:
            x_vals, y_vals = series_data[name]
            series_data[name] = (x_vals, smooth_values(y_vals, args.smooth))

    # Build chart
    chart = LineChart(
        title=args.title or '',
        x_label=x_axis,
    )
    for name, (x_vals, y_vals) in series_data.items():
        chart.add_series(name, x_vals, y_vals)

    # Render
    output = renderer.render(chart, width, height)
    print(output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
