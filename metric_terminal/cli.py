"""Command-line interface."""

import argparse
import glob
import os
import sys
import threading
import time
from collections import deque

from .parser import (
    parse_jsonl,
    detect_x_axis,
    extract_series,
    extract_grouped_series,
    extract_paneled_series,
    clip_x_range,
    smooth_values,
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


def parse_xlim(value: str) -> tuple[float | None, float | None]:
    """
    Parse an x-axis range string into (min, max) bounds.

    Accepts 'MIN:MAX' with either side optional ('0:200', ':200', '1000:'),
    or a single value 'MAX' as shorthand for ':MAX'.
    """
    s = value.strip()
    if ':' in s:
        lo, _, hi = s.partition(':')
    else:
        lo, hi = '', s
    try:
        xmin = float(lo) if lo.strip() else None
        xmax = float(hi) if hi.strip() else None
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid x-axis range {value!r}; expected MIN:MAX (e.g. 0:200)"
        )
    if xmin is not None and xmax is not None and xmin > xmax:
        xmin, xmax = xmax, xmin
    return xmin, xmax


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
        help='Input file(s), or - for stdin',
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
        '--xlim', '--xrange',
        dest='xlim',
        type=parse_xlim,
        metavar='MIN:MAX',
        help='Restrict the x-axis to a range, e.g. --xlim 0:200, :200, or 1000: '
             '(open-ended bounds allowed)',
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
        '--overlay',
        action='store_true',
        help='Overlay all metrics on a single chart instead of one panel per metric',
    )
    layout_group = parser.add_mutually_exclusive_group()
    layout_group.add_argument(
        '-g', '--group-by',
        dest='group_by',
        help='One panel per y-field, one curve per distinct value of this field (e.g. -g gpu)',
    )
    layout_group.add_argument(
        '--panel-by',
        dest='panel_by',
        help='One panel per distinct value of this field, y-fields as curves inside each panel',
    )
    parser.add_argument(
        '-f', '--follow',
        action='store_true',
        help='Stream JSONL input and refresh the plot as new records arrive',
    )
    parser.add_argument(
        '--window',
        type=int,
        default=1000,
        help='Max records to retain in --follow mode (default: 1000)',
    )
    parser.add_argument(
        '--refresh-ms',
        dest='refresh_ms',
        type=int,
        default=200,
        help='Refresh interval in milliseconds for --follow mode (default: 200)',
    )
    parser.add_argument(
        '--columns', '--cols',
        type=int,
        default=1,
        help='Arrange panels in N columns (default: 1, stacked vertically)',
    )
    parser.add_argument(
        '-s', '--smooth',
        type=float,
        default=0.0,
        metavar='WEIGHT',
        help='Smoothing weight 0-1 (0=none, 0.9=heavy smoothing)',
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

    renderer = get_renderer(ascii_only=args.ascii)

    # JSONL mode (use first file only)
    input_file = files[0]
    if args.follow:
        return run_follow(args, input_file, width, height, renderer)

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

    x_axis = args.x_axis or detect_x_axis(records)
    if not x_axis:
        print('Error: Could not detect x-axis field. Use -x to specify.', file=sys.stderr)
        return 1

    output = render_jsonl(records, x_axis, args, width, height, renderer)
    if output is None:
        print('Error: No numeric data found to plot', file=sys.stderr)
        return 1
    print(output)
    return 0


def render_jsonl(records, x_axis, args, width, height, renderer) -> str | None:
    """Render a list of JSONL records to a string. Returns None on no data."""
    import math as _math

    from .renderers.plotext_renderer import PlotextRenderer

    columns = max(1, getattr(args, 'columns', 1))

    if args.group_by or args.panel_by:
        if args.group_by:
            panels = extract_grouped_series(records, x_axis, args.y_fields, args.group_by)
        else:
            panels = extract_paneled_series(records, x_axis, args.y_fields, args.panel_by)
        if args.xlim:
            panels = clip_x_range(panels, *args.xlim)
        if not panels:
            return None
        if args.smooth > 0:
            for tag in panels:
                for run in panels[tag]:
                    xs, ys = panels[tag][run]
                    panels[tag][run] = (xs, smooth_values(ys, args.smooth))

        # Auto-fit per-panel height for grid layout when -H wasn't specified
        panel_height = height
        if columns > 1 and not args.height:
            _, term_h = get_terminal_size()
            rows = _math.ceil(len(panels) / columns)
            panel_height = max(7, (term_h - 2) // max(rows, 1))

        if isinstance(renderer, PlotextRenderer):
            return renderer.render_panels(panels, width, panel_height, x_label=x_axis, columns=columns)
        outputs = []
        for tag, runs in panels.items():
            chart = LineChart(title=tag, x_label=x_axis)
            for run_name, (xs, ys) in runs.items():
                chart.add_series(run_name, xs, ys)
            outputs.append(renderer.render(chart, width, panel_height))
        return '\n'.join(outputs)

    series_data = extract_series(records, x_axis, args.y_fields)
    if args.xlim:
        series_data = clip_x_range(series_data, *args.xlim)
    if not series_data:
        return None
    if args.smooth > 0:
        for name in series_data:
            xs, ys = series_data[name]
            series_data[name] = (xs, smooth_values(ys, args.smooth))

    if args.overlay or len(series_data) == 1:
        chart = LineChart(title=args.title or '', x_label=x_axis)
        for name, (xs, ys) in series_data.items():
            chart.add_series(name, xs, ys)
        return renderer.render(chart, width, height)

    panels = {name: {'default': (xs, ys)} for name, (xs, ys) in series_data.items()}
    panel_height = height
    if columns > 1 and not args.height:
        _, term_h = get_terminal_size()
        rows = _math.ceil(len(panels) / columns)
        panel_height = max(7, (term_h - 2) // max(rows, 1))
    if isinstance(renderer, PlotextRenderer):
        return renderer.render_panels(panels, width, panel_height, x_label=x_axis, columns=columns)
    outputs = []
    for name, (xs, ys) in series_data.items():
        chart = LineChart(title=name, x_label=x_axis)
        chart.add_series(name, xs, ys)
        outputs.append(renderer.render(chart, width, panel_height))
    return '\n'.join(outputs)


def run_follow(args, input_file, width, height, renderer) -> int:
    """Stream JSONL from input_file, keep a bounded window, re-render on a timer."""
    buffer: deque = deque(maxlen=max(1, args.window))
    lock = threading.Lock()
    stop = threading.Event()
    reader_done = threading.Event()

    def reader():
        try:
            for rec in parse_jsonl(input_file, args.pattern):
                if stop.is_set():
                    return
                with lock:
                    buffer.append(rec)
        except FileNotFoundError:
            sys.stderr.write(f'Error: File not found: {input_file}\n')
        except Exception as e:
            sys.stderr.write(f'reader error: {e}\n')
        finally:
            reader_done.set()

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    refresh_s = max(0.02, args.refresh_ms / 1000.0)
    x_axis = args.x_axis
    sys.stdout.write('\x1b[?25l')  # hide cursor
    sys.stdout.flush()

    try:
        while True:
            with lock:
                snapshot = list(buffer)

            if snapshot:
                if not x_axis:
                    x_axis = detect_x_axis(snapshot)
                if x_axis:
                    output = render_jsonl(snapshot, x_axis, args, width, height, renderer)
                    if output is not None:
                        sys.stdout.write('\x1b[H\x1b[J')  # cursor home + clear to end
                        sys.stdout.write(output)
                        sys.stdout.write('\n')
                        sys.stdout.flush()

            if reader_done.is_set() and not snapshot:
                # Input closed and nothing was ever buffered
                sys.stderr.write('No matching records found\n')
                return 1

            time.sleep(refresh_s)
    except KeyboardInterrupt:
        stop.set()
        return 0
    finally:
        sys.stdout.write('\x1b[?25h')  # show cursor
        sys.stdout.flush()


if __name__ == '__main__':
    sys.exit(main())
