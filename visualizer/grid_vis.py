"""
Matplotlib-based episode animator for Drone Traffic Control.

Captures one frame per step and saves a .gif (or .mp4) of the full episode.

Usage (from inference.py):
    animator = GridAnimator(rows=3, cols=3, task_name="easy")
    animator.capture(obs)            # call after every step (including reset)
    animator.save("episode.gif")     # call after episode ends
"""

from __future__ import annotations

import io
import warnings
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # headless — safe in Docker / HF Spaces
# Suppress "Glyph missing from font" warnings (emoji on Windows with DejaVu)
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np

from environment.models import Observation


# ── colour constants ──────────────────────────────────────────────────────────
_EMPTY_CLR     = "#1a1a2e"   # dark navy
_SINGLE_CLR    = "#16213e"   # slight highlight
_CONGESTED_CLR = "#e94560"   # red — 2+ drones
_BOTTLENECK_CLR = "#f5a623"  # amber — bottleneck marker
_DELIVERY_CLR  = "#0f3460"   # deep blue — delivered zone
_OBSTACLE_CLR  = "#2c2c54"   # purple-grey — blocked zone

_NORMAL_DRONE  = "#4fc3f7"   # light blue
_EMERG_DRONE   = "#ef5350"   # red
_DEAD_DRONE    = "#757575"   # grey
_DELIVERED_DOT = "#66bb6a"   # green tick


class GridAnimator:
    """
    Captures per-step frames and saves them as an animated .gif.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions matching the task config.
    task_name : str
        Shown in the figure title.
    bottleneck_zones : list of str, optional
        Zones to highlight with a border.
    figsize : tuple, optional
        Matplotlib figure size (width, height) in inches.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        task_name: str = "easy",
        bottleneck_zones: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (10, 7),
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.task_name = task_name
        self.bottleneck_zones = set(bottleneck_zones or [])
        self.figsize = figsize
        self._frames: List[bytes] = []   # PNG bytes per frame
        self._row_labels = [chr(ord("A") + r) for r in range(rows)]

    # ── public API ────────────────────────────────────────────────────────────

    def capture(self, obs: Observation, blocked_zones: Optional[List[str]] = None) -> None:
        """Render current observation to a PNG frame and store it."""
        blocked = set(blocked_zones or [])
        fig = self._draw_frame(obs, blocked)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        self._frames.append(buf.read())

    def save(self, path: str = "episode_animation.gif", fps: int = 2) -> str:
        """
        Save all captured frames as an animated GIF.

        Returns the resolved output path.
        """
        if not self._frames:
            raise RuntimeError("No frames captured — call capture() at least once.")

        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError(
                "Pillow is required for GIF export: pip install Pillow"
            )

        images = [Image.open(io.BytesIO(f)).convert("RGBA") for f in self._frames]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=False,
        )
        return path

    def get_frames_as_pil(self):
        """Return frames as a list of PIL Images (for Gradio display)."""
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("Pillow is required: pip install Pillow")
        return [Image.open(io.BytesIO(f)) for f in self._frames]

    def frame_count(self) -> int:
        return len(self._frames)

    # ── internal drawing ──────────────────────────────────────────────────────

    def _draw_frame(self, obs: Observation, blocked: set) -> Figure:
        fig = plt.figure(figsize=self.figsize, facecolor="#0d0d1a")
        gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
        ax_grid  = fig.add_subplot(gs[0])
        ax_table = fig.add_subplot(gs[1])

        self._draw_grid(ax_grid, obs, blocked)
        self._draw_legend(ax_table, obs)

        task_label = self.task_name.upper()
        fig.suptitle(
            f"🚁 Drone Traffic Control — {task_label}   |   Step {obs.step}   |   "
            f"Collisions: {obs.collisions}",
            color="white", fontsize=12, fontweight="bold", y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _draw_grid(self, ax: plt.Axes, obs: Observation, blocked: set) -> None:
        ax.set_facecolor("#0d0d1a")
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)
        ax.invert_yaxis()

        cmap = obs.congestion_map

        # Draw zone cells
        for r in range(self.rows):
            for c in range(self.cols):
                zone  = f"{self._row_labels[r]}{c + 1}"
                count = cmap.get(zone, 0)
                is_blocked = zone in blocked
                is_bottleneck = zone in self.bottleneck_zones

                if is_blocked:
                    clr = _OBSTACLE_CLR
                elif count >= 2:
                    clr = _CONGESTED_CLR
                elif count == 1:
                    clr = _SINGLE_CLR
                else:
                    clr = _EMPTY_CLR

                rect = mpatches.FancyBboxPatch(
                    (c - 0.45, r - 0.45), 0.90, 0.90,
                    boxstyle="round,pad=0.04",
                    facecolor=clr,
                    edgecolor=_BOTTLENECK_CLR if is_bottleneck else "#2a2a4a",
                    linewidth=2.5 if is_bottleneck else 0.8,
                    zorder=1,
                )
                ax.add_patch(rect)

                # Zone label
                delivered_here = any(
                    d.destination == zone and d.delivered for d in obs.drones
                )
                label_color = _DELIVERED_DOT if delivered_here else "#555577"
                ax.text(c - 0.38, r - 0.38, zone, color=label_color,
                        fontsize=7, va="top", ha="left", zorder=2,
                        fontfamily="monospace")

        # Draw connections (grid edges as thin lines)
        for r in range(self.rows):
            for c in range(self.cols):
                if c < self.cols - 1:
                    ax.plot([c, c + 1], [r, r], color="#2a2a4a", lw=0.5, zorder=0)
                if r < self.rows - 1:
                    ax.plot([c, c], [r, r + 1], color="#2a2a4a", lw=0.5, zorder=0)

        # Draw drones
        for drone in obs.drones:
            row_idx = ord(drone.location[0]) - ord("A")
            col_idx = int(drone.location[1:]) - 1

            # Offset overlapping drones in the same cell
            others_in_cell = [
                d for d in obs.drones
                if d.location == drone.location and not d.delivered and d.id <= drone.id
            ]
            offset_x = (len(others_in_cell) - 1) * 0.12
            offset_y = (len(others_in_cell) - 1) * 0.12

            if drone.delivered:
                colour = _DELIVERED_DOT
                marker = "*"
                size   = 220
            elif drone.battery <= 0:
                colour = _DEAD_DRONE
                marker = "x"
                size   = 120
            elif drone.priority == 2:
                colour = _EMERG_DRONE
                marker = "^"
                size   = 220
            else:
                colour = _NORMAL_DRONE
                marker = "o"
                size   = 160

            ax.scatter(
                col_idx + offset_x, row_idx + offset_y,
                c=colour, marker=marker, s=size, zorder=5,
                edgecolors="white", linewidths=0.7,
            )

            # Drone ID label
            if not drone.delivered:
                ax.text(
                    col_idx + offset_x + 0.07,
                    row_idx + offset_y - 0.07,
                    drone.id,
                    color="white", fontsize=6.5, zorder=6, fontweight="bold",
                )

            # Arrow toward destination
            if not drone.delivered and drone.battery > 0:
                dst_row = ord(drone.destination[0]) - ord("A")
                dst_col = int(drone.destination[1:]) - 1
                if (dst_row, dst_col) != (row_idx, col_idx):
                    dx = (dst_col - col_idx) * 0.18
                    dy = (dst_row - row_idx) * 0.18
                    ax.annotate(
                        "", xy=(col_idx + dx, row_idx + dy),
                        xytext=(col_idx, row_idx),
                        arrowprops=dict(arrowstyle="-|>", color=colour,
                                        lw=0.9, connectionstyle="arc3,rad=0.0"),
                        zorder=4,
                    )

        ax.set_xticks(range(self.cols))
        ax.set_xticklabels([str(c + 1) for c in range(self.cols)], color="#888899")
        ax.set_yticks(range(self.rows))
        ax.set_yticklabels(self._row_labels, color="#888899")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _draw_legend(self, ax: plt.Axes, obs: Observation) -> None:
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")

        delivered_n = sum(1 for d in obs.drones if d.delivered)
        total_n     = len(obs.drones)
        emerg_n     = sum(1 for d in obs.drones if d.priority == 2)
        emerg_done  = sum(1 for d in obs.drones if d.priority == 2 and d.delivered)

        # Progress bar
        pct = delivered_n / max(total_n, 1)
        bar_bg  = mpatches.FancyBboxPatch((0.05, 0.88), 0.90, 0.04,
                                          boxstyle="round,pad=0.01",
                                          facecolor="#2a2a4a", transform=ax.transAxes,
                                          clip_on=False)
        bar_fg  = mpatches.FancyBboxPatch((0.05, 0.88), 0.90 * pct, 0.04,
                                          boxstyle="round,pad=0.01",
                                          facecolor=_DELIVERED_DOT, transform=ax.transAxes,
                                          clip_on=False)
        ax.add_patch(bar_bg)
        ax.add_patch(bar_fg)

        lines = [
            ("white", f"Delivered:  {delivered_n}/{total_n}", 0.82),
            ("white", f"Emergency:  {emerg_done}/{emerg_n}", 0.76),
            (_CONGESTED_CLR, f"Collisions: {obs.collisions}", 0.70),
            ("", "", 0.63),   # spacer
            (_NORMAL_DRONE,  "(o) Normal drone",    0.58),
            (_EMERG_DRONE,   "(^) Emergency drone", 0.52),
            (_DELIVERED_DOT, "(*) Delivered",        0.46),
            (_DEAD_DRONE,    "(X) Dead battery",     0.40),
            ("", "", 0.33),
            (_BOTTLENECK_CLR, "(#) Bottleneck zone", 0.28),
            (_CONGESTED_CLR,  "[#] Congested (2+)",  0.22),
            (_OBSTACLE_CLR,   "[X] Blocked (obstacle)", 0.16),
        ]
        for colour, label, y in lines:
            if not label:
                continue
            ax.text(0.05, y, label, transform=ax.transAxes,
                    color=colour or "white", fontsize=8.5, va="center",
                    fontfamily="monospace")

        # Battery bars for each drone
        y_start = 0.08
        ax.text(0.05, y_start + 0.02, "Battery:", transform=ax.transAxes,
                color="#888899", fontsize=7.5)
        for i, drone in enumerate(sorted(obs.drones, key=lambda d: d.id)):
            bpct = drone.battery / 100.0
            colour = _DELIVERED_DOT if bpct > 0.5 else (_BOTTLENECK_CLR if bpct > 0.2 else _EMERG_DRONE)
            y_pos = y_start - 0.04 - i * 0.055
            if y_pos < 0:
                break
            bar = mpatches.FancyBboxPatch(
                (0.20, y_pos), 0.70 * bpct, 0.03,
                boxstyle="round,pad=0.005",
                facecolor=colour, transform=ax.transAxes, clip_on=False,
            )
            ax.add_patch(bar)
            ax.text(0.05, y_pos + 0.015, drone.id, transform=ax.transAxes,
                    color="white", fontsize=6.5, va="center")
