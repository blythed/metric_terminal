"""Plain ASCII renderer (no dependencies)."""

from .base import Renderer
from ..chart import LineChart


# Characters for different series (used for line segments)
SERIES_CHARS = ['*', '+', 'o', 'x', '#', '@', '%', '&', '=', '~']


class AsciiRenderer(Renderer):
    """Render charts using plain ASCII characters with smooth lines."""

    def render(self, chart: LineChart, width: int, height: int) -> str:
        """Render the chart to ASCII art."""
        if chart.is_empty():
            return "No data to display"

        # Reserve space for y-axis labels and legend
        y_label_width = 8
        legend_width = max(len(s.name) for s in chart.series) + 4 if chart.series else 0
        plot_width = width - y_label_width - legend_width - 1

        if plot_width < 10:
            plot_width = 10

        # Build the plot grid - stores (char, series_idx) or None
        grid = [[None for _ in range(plot_width)] for _ in range(height)]

        # Get bounds with a small margin
        x_min, x_max = chart.x_min, chart.x_max
        y_min, y_max = chart.y_min, chart.y_max

        # Add margin to y range
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1.0
            y_min -= 0.5
            y_max += 0.5
        else:
            margin = y_range * 0.05
            y_min -= margin
            y_max += margin
            y_range = y_max - y_min

        x_range = x_max - x_min
        if x_range == 0:
            x_range = 1.0

        # Plot each series with smooth lines
        for series_idx, series in enumerate(chart.series):
            # Sort points by x value for line drawing
            points = sorted(zip(series.x_values, series.y_values))

            # Convert to grid coordinates
            grid_points = []
            for x, y in points:
                grid_x = int((x - x_min) / x_range * (plot_width - 1))
                grid_y = int((y_max - y) / y_range * (height - 1))
                grid_x = max(0, min(plot_width - 1, grid_x))
                grid_y = max(0, min(height - 1, grid_y))
                grid_points.append((grid_x, grid_y))

            # Draw smooth lines between consecutive points
            for i in range(len(grid_points) - 1):
                x0, y0 = grid_points[i]
                x1, y1 = grid_points[i + 1]
                self._draw_smooth_line(grid, x0, y0, x1, y1, series_idx)

        # Build output lines
        lines = []

        # Title
        if chart.title:
            lines.append(chart.title.center(width))
            lines.append('')

        # Y-axis labels and plot
        y_labels = self._generate_y_labels(y_min, y_max, height)

        for row_idx in range(height):
            label = y_labels[row_idx].rjust(y_label_width - 1)

            # Build row data
            row_chars = []
            for col_idx in range(plot_width):
                cell = grid[row_idx][col_idx]
                if cell:
                    row_chars.append(cell[0])
                else:
                    row_chars.append(' ')
            row_data = ''.join(row_chars)

            # Add legend on the right for first few rows
            legend_part = ''
            if row_idx < len(chart.series):
                char = SERIES_CHARS[row_idx % len(SERIES_CHARS)]
                legend_part = f' {char} {chart.series[row_idx].name}'

            lines.append(f'{label}|{row_data}{legend_part}')

        # X-axis
        x_axis_line = ' ' * y_label_width + '+' + '-' * plot_width
        lines.append(x_axis_line)

        # X-axis labels
        x_labels = self._generate_x_labels(x_min, x_max, plot_width)
        x_label_line = ' ' * y_label_width + ' ' + x_labels
        lines.append(x_label_line)

        # X-axis name
        if chart.x_label:
            lines.append(' ' * (y_label_width + plot_width // 2) + chart.x_label)

        return '\n'.join(lines)

    def _draw_smooth_line(self, grid, x0, y0, x1, y1, series_idx):
        """Draw a smooth line between two points using directional characters."""
        dx = x1 - x0
        dy = y1 - y0

        # Handle single point
        if dx == 0 and dy == 0:
            char = SERIES_CHARS[series_idx % len(SERIES_CHARS)]
            grid[y0][x0] = (char, series_idx)
            return

        steps = max(abs(dx), abs(dy))

        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            x = int(round(x0 + t * dx))
            y = int(round(y0 + t * dy))

            if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
                # Calculate local direction for this segment
                if i < steps:
                    next_t = (i + 1) / steps
                    next_x = int(round(x0 + next_t * dx))
                    next_y = int(round(y0 + next_t * dy))
                    local_dx = next_x - x
                    local_dy = next_y - y
                else:
                    # Use previous direction for last point
                    prev_t = (i - 1) / steps if i > 0 else 0
                    prev_x = int(round(x0 + prev_t * dx))
                    prev_y = int(round(y0 + prev_t * dy))
                    local_dx = x - prev_x
                    local_dy = y - prev_y

                char = self._get_line_char(local_dx, local_dy, series_idx)

                # Only set if empty or same series (allow overwriting own line)
                if grid[y][x] is None or grid[y][x][1] == series_idx:
                    grid[y][x] = (char, series_idx)

    def _get_line_char(self, dx, dy, series_idx):
        """Get the appropriate line character based on direction."""
        # Pure horizontal
        if dy == 0:
            return '-'
        # Pure vertical
        if dx == 0:
            return '|'
        # Diagonal down-right or up-left (y increases downward)
        if (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
            return '\\'
        # Diagonal up-right or down-left
        return '/'

    def _generate_y_labels(self, y_min: float, y_max: float, height: int) -> list[str]:
        """Generate y-axis labels for each row."""
        labels = []
        for i in range(height):
            y = y_max - (i / (height - 1)) * (y_max - y_min) if height > 1 else y_max
            labels.append(self._format_number(y))
        return labels

    def _generate_x_labels(self, x_min: float, x_max: float, width: int) -> str:
        """Generate x-axis label string."""
        min_str = self._format_number(x_min)
        max_str = self._format_number(x_max)
        mid = (x_min + x_max) / 2
        mid_str = self._format_number(mid)

        result = [' '] * width
        for i, c in enumerate(min_str):
            if i < width:
                result[i] = c
        start_pos = width - len(max_str)
        for i, c in enumerate(max_str):
            if start_pos + i < width:
                result[start_pos + i] = c
        mid_pos = width // 2 - len(mid_str) // 2
        if mid_pos > len(min_str) + 1 and mid_pos + len(mid_str) < start_pos - 1:
            for i, c in enumerate(mid_str):
                result[mid_pos + i] = c

        return ''.join(result)

    def _format_number(self, value: float) -> str:
        """Format a number for display."""
        abs_val = abs(value)
        if abs_val == 0:
            return '0'
        elif abs_val < 0.001:
            return f'{value:.0e}'
        elif abs_val < 0.01:
            return f'{value:.3f}'
        elif abs_val < 1:
            return f'{value:.2f}'
        elif abs_val < 10:
            return f'{value:.2f}'
        elif abs_val < 100:
            return f'{value:.1f}'
        elif abs_val < 1000:
            return f'{value:.0f}'
        else:
            return f'{value:.0e}'
