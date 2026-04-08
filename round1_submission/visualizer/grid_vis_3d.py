"""
3D Matplotlib-based episode animator for Drone Traffic Control.
Uses continuous altitude (Option B) for spatial rendering.
"""

from __future__ import annotations

import io
import warnings
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image

from environment.models import Observation

# -- constants --
_NORMAL_DRONE  = "#4fc3f7"
_EMERG_DRONE   = "#ef5350"
_DEAD_DRONE    = "#757575"
_DELIVERED_DOT = "#66bb6a"
_GRID_COLOR     = "#2a2a4a"

class GridAnimator3D:
    def __init__(
        self,
        rows: int,
        cols: int,
        max_alt: float = 20.0,
        task_name: str = "easy",
        figsize: Tuple[float, float] = (12, 8),
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.max_alt = max_alt
        self.task_name = task_name
        self.figsize = figsize
        self._frames: List[bytes] = []
        self._row_labels = [chr(ord("A") + r) for r in range(rows)]

    def capture(self, obs: Observation, blocked_zones: Optional[List[str]] = None) -> None:
        fig = self._draw_frame(obs, set(blocked_zones or []))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        self._frames.append(buf.read())

    def save(self, path: str = "episode_3d.webp", fps: int = 2) -> str:
        if not self._frames:
            return ""
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
        return [Image.open(io.BytesIO(f)) for f in self._frames]

    def _draw_frame(self, obs: Observation, blocked: set) -> Figure:
        fig = plt.figure(figsize=self.figsize, facecolor="#0d0d1a")
        gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
        ax_3d = fig.add_subplot(gs[0], projection='3d')
        ax_info = fig.add_subplot(gs[1])
        
        self._draw_3d_space(ax_3d, obs)
        self._draw_info(ax_info, obs)
        
        fig.suptitle(
            f"3D Drone Traffic Control - {self.task_name.upper()} | Step {obs.step} | Collisions: {obs.collisions}",
            color="white", fontsize=12, fontweight="bold", y=0.98
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _draw_3d_space(self, ax: Axes3D, obs: Observation) -> None:
        ax.set_facecolor("#0d0d1a")
        ax.xaxis.pane.set_facecolor('#0d0d1a')
        ax.yaxis.pane.set_facecolor('#0d0d1a')
        ax.zaxis.pane.set_facecolor('#0d0d1a')
        
        # Grid boundaries
        ax.set_xlim(0, self.cols - 1)
        ax.set_ylim(0, self.rows - 1)
        ax.set_zlim(0, self.max_alt)
        
        # Draw floor grid
        t = np.linspace(0, self.cols - 1, self.cols)
        s = np.linspace(0, self.rows - 1, self.rows)
        X, Y = np.meshgrid(t, s)
        ax.plot_wireframe(X, Y, np.zeros_like(X), color=_GRID_COLOR, alpha=0.3, lw=0.5)

        # Draw Wind Vector Arrow
        if hasattr(obs, "wind_vector") and any(v != 0 for v in obs.wind_vector):
            wx, wy, wz = obs.wind_vector
            # Scale wind for visualization
            w_scale = 1.0
            # Plot at top corner
            ax.quiver(self.cols-1, self.rows-1, self.max_alt, 
                      wx*w_scale, wy*w_scale, wz*w_scale, 
                      color='yellow', linewidth=2, label='Wind', length=2.0)
            ax.text(self.cols-1, self.rows-1, self.max_alt + 1, "WIND", color='yellow', fontsize=8, fontweight='bold')

        # Plot drones
        for d in obs.drones:
            r = ord(d.location[0]) - ord("A")
            c = int(d.location[1:]) - 1
            z = d.altitude
            
            color = _NORMAL_DRONE
            if d.delivered: color = _DELIVERED_DOT
            elif d.battery <= 0: color = _DEAD_DRONE
            elif d.priority == 2: color = _EMERG_DRONE
            
            # 3D Marker
            ax.scatter([c], [r], [z], c=color, s=100, edgecolors='white', alpha=0.9)
            
            # Vertical projection line to floor
            ax.plot([c, c], [r, r], [0, z], color=color, linestyle='--', alpha=0.4, lw=0.8)
            
            # ID Label
            if not d.delivered:
                ax.text(c, r, z + 1, d.id, color='white', fontsize=7, fontweight='bold')

        # Axis styling
        ax.set_xlabel("Columns", color="#888899", fontsize=8)
        ax.set_ylabel("Rows", color="#888899", fontsize=8)
        ax.set_zlabel("Altitude (m)", color="#888899", fontsize=8)
        ax.tick_params(axis='x', colors='#555577', labelsize=7)
        ax.tick_params(axis='y', colors='#555577', labelsize=7)
        ax.tick_params(axis='z', colors='#555577', labelsize=7)
        
        # View angle
        ax.view_init(elev=25, azim=-45)

    def _draw_info(self, ax: plt.Axes, obs: Observation) -> None:
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")
        
        y = 0.9
        ax.text(0.05, y, "Flight Status:", color="white", fontweight="bold", transform=ax.transAxes)
        y -= 0.1
        
        for d in sorted(obs.drones, key=lambda x: x.id):
            status = "FLYING"
            if d.delivered: status = "DONE"
            elif d.battery <= 0: status = "CRASHED"
            
            info_str = f"{d.id}: {d.location} @ {d.altitude:.1f}m [{status}]"
            color = "white"
            if d.delivered: color = _DELIVERED_DOT
            elif d.battery <= 0: color = _EMERG_DRONE
            
            ax.text(0.05, y, info_str, color=color, fontsize=7, transform=ax.transAxes, fontfamily="monospace")
            y -= 0.05
            if y < 0.05: break
