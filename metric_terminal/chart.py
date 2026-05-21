"""Chart data model."""

from dataclasses import dataclass, field


@dataclass
class Series:
    """A single data series for plotting."""

    name: str
    x_values: list[float] = field(default_factory=list)
    y_values: list[float] = field(default_factory=list)

    def add_point(self, x: float, y: float) -> None:
        """Add a data point to the series."""
        self.x_values.append(x)
        self.y_values.append(y)

    @property
    def x_min(self) -> float:
        """Minimum x value."""
        return min(self.x_values) if self.x_values else 0.0

    @property
    def x_max(self) -> float:
        """Maximum x value."""
        return max(self.x_values) if self.x_values else 0.0

    @property
    def y_min(self) -> float:
        """Minimum y value."""
        return min(self.y_values) if self.y_values else 0.0

    @property
    def y_max(self) -> float:
        """Maximum y value."""
        return max(self.y_values) if self.y_values else 0.0

    def __len__(self) -> int:
        return len(self.x_values)


@dataclass
class LineChart:
    """A line chart with multiple overlaid series."""

    title: str = ""
    x_label: str = ""
    series: list[Series] = field(default_factory=list)

    def add_series(self, name: str, x_values: list[float], y_values: list[float]) -> Series:
        """Add a new series to the chart."""
        s = Series(name=name, x_values=list(x_values), y_values=list(y_values))
        self.series.append(s)
        return s

    @property
    def x_min(self) -> float:
        """Global minimum x value across all series."""
        if not self.series:
            return 0.0
        return min(s.x_min for s in self.series)

    @property
    def x_max(self) -> float:
        """Global maximum x value across all series."""
        if not self.series:
            return 0.0
        return max(s.x_max for s in self.series)

    @property
    def y_min(self) -> float:
        """Global minimum y value across all series."""
        if not self.series:
            return 0.0
        return min(s.y_min for s in self.series)

    @property
    def y_max(self) -> float:
        """Global maximum y value across all series."""
        if not self.series:
            return 0.0
        return max(s.y_max for s in self.series)

    @property
    def x_range(self) -> float:
        """Range of x values."""
        return self.x_max - self.x_min

    @property
    def y_range(self) -> float:
        """Range of y values."""
        return self.y_max - self.y_min

    def is_empty(self) -> bool:
        """Check if the chart has any data."""
        return not self.series or all(len(s) == 0 for s in self.series)
