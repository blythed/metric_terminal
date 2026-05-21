"""metric_terminal - Display metrics as ASCII charts in the terminal."""

from .chart import LineChart, Series
from .parser import parse_jsonl, detect_x_axis, extract_series
from .renderers import get_renderer, Renderer, AsciiRenderer

__version__ = '0.1.0'

__all__ = [
    'LineChart',
    'Series',
    'parse_jsonl',
    'detect_x_axis',
    'extract_series',
    'get_renderer',
    'Renderer',
    'AsciiRenderer',
]
