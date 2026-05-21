"""Plotext-based renderer for smooth terminal charts."""

import plotext as plt

from .base import Renderer
from ..chart import LineChart


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
        y_range = chart.y_range
        max_name_len = max(len(s.name) for s in chart.series)
        num_series = len(chart.series)
        available_height = y_range * 0.6
        legend_spacing = min(y_range * 0.12, available_height / num_series)
        legend_start_y = y_max - (y_range * 0.05)

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

    def render_panels(
        self,
        panels: dict[str, dict[str, tuple[list[float], list[float]]]],
        width: int,
        height_per_panel: int,
        x_label: str = 'step',
    ) -> str:
        """
        Render multiple panels, one per metric.

        Args:
            panels: Dict mapping metric name to {run_name: (x_vals, y_vals)}
            width: Width in characters
            height_per_panel: Height per panel
            x_label: Label for x-axis

        Returns:
            Rendered string with all panels stacked
        """
        if not panels:
            return "No data to display"

        outputs = []

        for tag, runs in panels.items():
            # Clear and set up this panel
            plt.clear_figure()
            plt.theme('clear')
            plt.plotsize(width, height_per_panel)

            # Skip if no data
            if not runs:
                continue

            # Collect all values for legend positioning
            all_y = []
            all_x = []
            for steps, values in runs.values():
                all_y.extend(values)
                all_x.extend(steps)

            if not all_y:
                continue

            y_min, y_max = min(all_y), max(all_y)
            y_range = y_max - y_min if y_max != y_min else 1.0
            x_max = max(all_x) if all_x else 1.0

            # Plot each run
            run_names = list(runs.keys())
            for idx, (run_name, (steps, values)) in enumerate(runs.items()):
                color = COLORS[idx % len(COLORS)]
                points = sorted(zip(steps, values))
                x_vals = [p[0] for p in points]
                y_vals = [p[1] for p in points]
                plt.plot(x_vals, y_vals, color=color, marker='braille')

            # Title is the metric name
            plt.title(tag)
            plt.xlabel(x_label)

            # Add manual right-aligned legend if multiple runs
            if len(runs) > 1:
                max_name_len = max(len(n) for n in run_names)
                # Start below the title area and space entries to fit
                num_entries = len(run_names)
                available_height = y_range * 0.6  # Use 60% of chart for legend
                legend_spacing = min(y_range * 0.12, available_height / num_entries)
                legend_start_y = y_max - (y_range * 0.05)  # Start slightly below top

                for idx, run_name in enumerate(run_names):
                    color = COLORS[idx % len(COLORS)]
                    legend_y = legend_start_y - (idx * legend_spacing)
                    padded_name = run_name.ljust(max_name_len)
                    plt.text(f"── {padded_name}", x=x_max, y=legend_y, color=color, alignment='right')

            outputs.append(plt.build())

        return '\n'.join(outputs)
