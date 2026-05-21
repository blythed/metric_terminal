"""Chart renderers."""

from .base import Renderer
from .ascii import AsciiRenderer

__all__ = ['Renderer', 'AsciiRenderer', 'get_renderer']


def get_renderer(ascii_only: bool = False) -> Renderer:
    """
    Get the appropriate renderer based on settings and availability.

    Args:
        ascii_only: Force ASCII-only rendering (basic fallback)

    Returns:
        A Renderer instance
    """
    if ascii_only:
        return AsciiRenderer()

    # Try plotext first (best quality)
    try:
        from .plotext_renderer import PlotextRenderer
        return PlotextRenderer()
    except ImportError:
        pass

    # Fallback to basic ASCII
    return AsciiRenderer()
