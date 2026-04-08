"""
Terminal visualizer using the 'rich' library.

Renders a live, colour-coded grid + drone status table after every step.

Usage (from inference.py):
    renderer = TerminalRenderer(env)
    renderer.render(obs, reward, info)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from environment.models import Observation, Reward


# ->-> colour palette ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->
_EMERGENCY_COLOUR = "bold red"
_NORMAL_COLOUR    = "bold cyan"
_DELIVERED_COLOUR = "bold green"
_DEAD_COLOUR      = "dim white"
_HOT_ZONE         = "on dark_red"      # 3+ drones
_WARM_ZONE        = "on dark_orange3"  # 2 drones
_COOL_ZONE        = "on grey23"        # 1 drone
_EMPTY_ZONE       = ""                 # 0 drones

_PRIORITY_ICON = {1: "->?", 2: "->?"}
_BATTERY_ICON  = lambda b: "->?" if b > 30 else ("->-> " if b > 10 else "->?")


class TerminalRenderer:
    """
    Rich-powered terminal renderer for one episode.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions (used to draw the zone map).
    task_name : str
        Displayed in the header panel.
    console : Console, optional
        Inject a custom Rich Console (useful for testing / capturing output).
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        task_name: str = "unknown",
        console: Optional[Console] = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.task_name = task_name
        self.con = console or Console()
        self._step_history: List[float] = []

    # ->-> public API ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

    def render(
        self,
        obs: Observation,
        reward: Optional[Reward] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print the current environment state to the terminal."""
        info = info or {}
        if reward is not None:
            self._step_history.append(reward.total)

        grid_panel  = self._build_grid_panel(obs)
        drone_panel = self._build_drone_table(obs)
        stats_panel = self._build_stats_panel(obs, reward, info)

        self.con.print(Columns([grid_panel, drone_panel], equal=False, expand=False))
        self.con.print(stats_panel)
        self.con.rule(style="dim")

    def render_final(self, final_score: float, state: Dict[str, Any]) -> None:
        """Print the final episode summary."""
        delivered = sum(1 for d in state["drones"] if d["delivered"])
        total     = len(state["drones"])
        collisions = state.get("collisions", 0)
        steps      = state.get("step", "?")

        summary = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        summary.add_column(style="bold yellow")
        summary.add_column(style="white")
        summary.add_row("Final Score",  f"[bold green]{final_score:.4f}[/bold green]")
        summary.add_row("Delivered",    f"{delivered}/{total}")
        summary.add_row("Collisions",   str(collisions))
        summary.add_row("Steps Used",   str(steps))
        if self._step_history:
            summary.add_row("Total Reward", f"{sum(self._step_history):.2f}")
            summary.add_row("Mean Reward",  f"{sum(self._step_history)/len(self._step_history):.3f}")

        self.con.print(Panel(summary, title="[bold white]->? Episode Complete[/bold white]",
                             border_style="green", expand=False))

    # ->-> grid panel ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

    def _build_grid_panel(self, obs: Observation) -> Panel:
        cmap = obs.congestion_map
        row_labels = [chr(ord("A") + r) for r in range(self.rows)]

        lines: List[Text] = []
        # Column header
        header = Text("   ")
        for c in range(1, self.cols + 1):
            header.append(f"  {c}  ", style="bold white")
        lines.append(header)

        for row in row_labels:
            row_text = Text(f" {row} ")
            for c in range(1, self.cols + 1):
                zone  = f"{row}{c}"
                count = cmap.get(zone, 0)
                # Colour cell by congestion
                if count >= 3:
                    bg = _HOT_ZONE
                elif count == 2:
                    bg = _WARM_ZONE
                elif count == 1:
                    bg = _COOL_ZONE
                else:
                    bg = _EMPTY_ZONE

                # Find drones in this zone
                ids = [
                    d.id for d in obs.drones
                    if d.location == zone and not d.delivered
                ]
                label = ",".join(ids) if ids else "??  "
                label = label[:4].ljust(4)
                row_text.append(f" {label}", style=bg)
            lines.append(row_text)

        grid_text = Text("\n").join(lines)
        return Panel(grid_text, title=f"[bold]??-?  Airspace - Step {obs.step}[/bold]",
                     border_style="blue", expand=False)

    # ->-> drone table ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

    def _build_drone_table(self, obs: Observation) -> Panel:
        t = Table(box=box.SIMPLE_HEAD, show_lines=False, expand=False)
        t.add_column("ID",   style="bold", width=4)
        t.add_column("Loc",  width=4)
        t.add_column("-> Dst", width=5)
        t.add_column("Bat",  width=6)
        t.add_column("Pri",  width=4)
        t.add_column("->",    width=3)

        for d in sorted(obs.drones, key=lambda x: (-x.priority, x.id)):
            if d.delivered:
                style = _DELIVERED_COLOUR
                check = "->"
            elif d.battery <= 0:
                style = _DEAD_COLOUR
                check = "->?"
            elif d.priority == 2:
                style = _EMERGENCY_COLOUR
                check = ""
            else:
                style = _NORMAL_COLOUR
                check = ""

            icon = _PRIORITY_ICON.get(d.priority, "")
            bat_icon = _BATTERY_ICON(d.battery)
            t.add_row(
                f"{icon}{d.id}",
                d.location,
                d.destination,
                f"{bat_icon}{d.battery:.0f}%",
                "EMG" if d.priority == 2 else "nrm",
                check,
                style=style,
            )

        return Panel(t, title="[bold]->? Drones[/bold]", border_style="blue", expand=False)

    # ->-> stats panel ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

    def _build_stats_panel(
        self,
        obs: Observation,
        reward: Optional[Reward],
        info: Dict[str, Any],
    ) -> Panel:
        parts: List[str] = []
        if reward is not None:
            colour = "green" if reward.total >= 0 else "red"
            parts.append(f"Reward: [{colour}]{reward.total:+.2f}[/{colour}]")
            det = reward.details
            if det.collision_penalty < 0:
                parts.append(f"[red]Collisions: {int(abs(det.collision_penalty / 2))}[/red]")
            if det.emergency_bonus > 0:
                parts.append(f"[yellow]Emergency bonus: +{det.emergency_bonus:.0f}[/yellow]")
            if det.deliveries > 0:
                parts.append(f"[green]Delivered this step: {int(det.deliveries)}[/green]")

        blocked = info.get("blocked_zones", [])
        if blocked:
            parts.append(f"[orange1]Blocked zones: {blocked}[/orange1]")

        cum_col = info.get("cumulative_collisions", obs.collisions)
        total   = len(obs.drones)
        deliv   = info.get("delivered", sum(1 for d in obs.drones if d.delivered))
        parts.append(f"Progress: {deliv}/{total} drones")
        if cum_col:
            parts.append(f"[red]Total collisions: {cum_col}[/red]")

        text = Text("  ->  ".join(parts))
        return Panel(text, border_style="dim", expand=True)
