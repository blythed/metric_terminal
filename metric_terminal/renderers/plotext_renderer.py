"""Plotext-based renderer for smooth terminal charts."""

import math
import re

import plotext as plt

from .base import Renderer
from ..chart import LineChart

_ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')


def _visible_width(s: str) -> int:
    """Length of a string with ANSI escape sequences stripped."""
    return len(_ANSI_RE.sub('', s))


def _pad_visible(s: str, width: int) -> str:
    """Pad string with spaces so its visible (ANSI-stripped) length equals width."""
    pad = width - _visible_width(s)
    return s + (' ' * pad if pad > 0 else '')


# Colors for different series
COLORS = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white', 'orange']


class PlotextRenderer(Renderer):
    """Render charts using plotext for smooth terminal graphics."""

    def render(self, chart: LineChart, width: int, height: int) -> str:
        """Render the chart using plotext."""
        if chart.is_empty():
            return "No data to display"

        # Clear any previous plot
        plt.clear_figure()

        # Use clear theme (no background, integrates with terminal)
        plt.theme('clear')

        # Set size
        plt.plotsize(width, height)

        # Plot each series using braille for thin lines
        for idx, series in enumerate(chart.series):
            color = COLORS[idx % len(COLORS)]
            # Sort by x values
            points = sorted(zip(series.x_values, series.y_values))
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            plt.plot(x_vals, y_vals, color=color, marker='braille')

        # Add manual right-aligned legend
        x_max = chart.x_max
        y_max = chart.y_max
        y_range = chart.y_range or 1.0
        max_name_len = max(len(s.name) for s in chart.series)
        num_series = len(chart.series)
        # Top margin grows with entry count so the first label doesn't clip the chart border
        top_margin = y_range * (0.08 + 0.005 * num_series)
        available_height = y_range - top_margin - (y_range * 0.05)
        legend_spacing = min(y_range * 0.12, available_height / max(num_series - 1, 1))
        legend_start_y = y_max - top_margin

        for idx, series in enumerate(chart.series):
            color = COLORS[idx % len(COLORS)]
            legend_y = legend_start_y - (idx * legend_spacing)
            padded_name = series.name.ljust(max_name_len)
            plt.text(f"── {padded_name}", x=x_max, y=legend_y, color=color, alignment='right')

        # Set labels
        if chart.x_label:
            plt.xlabel(chart.x_label)

        if chart.title:
            plt.title(chart.title)

        # Build the plot string
        return plt.build()

    def _draw_panel(self, tag: str, runs: dict, x_label: str) -> None:
        """Draw a single panel's content onto the currently-selected (sub)plot."""
        if not runs:
            return

        all_y = []
        all_x = []
        for steps, values in runs.values():
            all_y.extend(values)
            all_x.extend(steps)
        if not all_y:
            return

        y_min, y_max = min(all_y), max(all_y)
        y_range = y_max - y_min if y_max != y_min else 1.0
        x_max = max(all_x) if all_x else 1.0

        run_names = list(runs.keys())
        for idx, (run_name, (steps, values)) in enumerate(runs.items()):
            color = COLORS[idx % len(COLORS)]
            points = sorted(zip(steps, values))
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            plt.plot(x_vals, y_vals, color=color, marker='braille')

        plt.title(tag)
        plt.xlabel(x_label)

        if len(runs) > 1:
            max_name_len = max(len(n) for n in run_names)
            num_entries = len(run_names)
            top_margin = y_range * (0.08 + 0.005 * num_entries)
            available_height = y_range - top_margin - (y_range * 0.05)
            legend_spacing = min(y_range * 0.12, available_height / max(num_entries - 1, 1))
            legend_start_y = y_max - top_margin

            for idx, run_name in enumerate(run_names):
                color = COLORS[idx % len(COLORS)]
                legend_y = legend_start_y - (idx * legend_spacing)
                padded_name = run_name.ljust(max_name_len)
                plt.text(f"── {padded_name}", x=x_max, y=legend_y, color=color, alignment='right')

    def render_panels(
        self,
        panels: dict[str, dict[str, tuple[list[float], list[float]]]],
        width: int,
        height_per_panel: int,
        x_label: str = 'step',
        columns: int = 1,
    ) -> str:
        """
        Render multiple panels, one per metric.

        Args:
            panels: Dict mapping metric name to {run_name: (x_vals, y_vals)}
            width: Width in characters (total figure width)
            height_per_panel: Height per panel row
            x_label: Label for x-axis
            columns: Number of columns in the grid (default 1 = stacked vertically)
        """
        if not panels:
            return "No data to display"

        columns = max(1, columns)
        panel_items = [(tag, runs) for tag, runs in panels.items() if runs]
        if not panel_items:
            return "No data to display"

        # Render each panel individually, then stitch them into a grid.
        # We avoid plotext's native subplots() because it under-allocates vertical
        # space to the first row when subplots have x-labels, collapsing it.
        sub_width = width if columns == 1 else max(20, width // columns)
        panel_strings = []
        for tag, runs in panel_items:
            plt.clear_figure()
            plt.theme('clear')
            plt.plotsize(sub_width, height_per_panel)
            self._draw_panel(tag, runs, x_label)
            panel_strings.append(plt.build())

        if columns == 1:
            return '\n'.join(panel_strings)

        rows_out = []
        for row_start in range(0, len(panel_strings), columns):
            row_panels = panel_strings[row_start:row_start + columns]
            row_lines = [p.split('\n') for p in row_panels]
            max_lines = max(len(lines) for lines in row_lines)
            for lines in row_lines:
                lines.extend([''] * (max_lines - len(lines)))
            padded = [[_pad_visible(line, sub_width) for line in lines] for lines in row_lines]
            for i in range(max_lines):
                rows_out.append(''.join(panel[i] for panel in padded))

        return '\n'.join(rows_out)
