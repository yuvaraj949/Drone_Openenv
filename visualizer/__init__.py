"""
Visualizer package for Drone Traffic Control.

Two renderers available:
  TerminalRenderer  — live Rich table output per step (zero dependencies beyond 'rich')
  GridAnimator      — matplotlib episode .gif (requires 'matplotlib')
"""

from .terminal_vis import TerminalRenderer
from .grid_vis import GridAnimator

__all__ = ["TerminalRenderer", "GridAnimator"]
