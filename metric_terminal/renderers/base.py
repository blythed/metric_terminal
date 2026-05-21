"""Base renderer interface."""

from abc import ABC, abstractmethod

from ..chart import LineChart


class Renderer(ABC):
    """Abstract base class for chart renderers."""

    @abstractmethod
    def render(self, chart: LineChart, width: int, height: int) -> str:
        """
        Render a chart to a string.

        Args:
            chart: The chart to render
            width: Width in characters
            height: Height in characters (for the plot area)

        Returns:
            A string representation of the chart
        """
        pass
