"""
Task configurations for the Drone Traffic Control environment.

Each task defines:
  - grid size and topology
  - number of drones / emergencies
  - max episode steps
  - optional dynamic obstacles (hard task)
  - a human-readable description

Tasks scale in complexity:
  easy   -> 3x3 grid,  3 drones, 1 emergency
  medium -> 4x4 grid,  5 drones, 2 emergencies, bottleneck zones
  hard   -> 5x5 grid, 10 drones, 3 emergencies, dynamic obstacles
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Graph builder helpers
# ---------------------------------------------------------------------------

def _build_grid_graph(rows: int, cols: int) -> Dict[str, List[str]]:
    """Create a full grid adjacency list. Zones are labelled row-letter + col-number."""
    row_labels = [chr(ord("A") + r) for r in range(rows)]
    graph: Dict[str, List[str]] = {}

    for r, row in enumerate(row_labels):
        for c in range(1, cols + 1):
            zone = f"{row}{c}"
            neighbours: List[str] = []
            # left / right
            if c > 1:
                neighbours.append(f"{row}{c - 1}")
            if c < cols:
                neighbours.append(f"{row}{c + 1}")
            # up / down (previous / next row)
            if r > 0:
                neighbours.append(f"{chr(ord(row) - 1)}{c}")
            if r < rows - 1:
                neighbours.append(f"{chr(ord(row) + 1)}{c}")
            graph[zone] = neighbours

    return graph


def _all_zones(rows: int, cols: int) -> List[str]:
    """Return all zone labels for an r?-c grid."""
    row_labels = [chr(ord("A") + r) for r in range(rows)]
    return [f"{row}{c}" for row in row_labels for c in range(1, cols + 1)]


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, dict] = {
    # ------------------------------------------------------------------
    # EASY - 3?-3 grid, 3 drones (1 emergency), max 30 steps
    # ------------------------------------------------------------------
    "easy": {
        "name": "easy",
        "description": (
            "3-drone scenario on a 3?-3 grid with one emergency drone. "
            "Goal: route all drones to their destinations without collision."
        ),
        "rows": 3,
        "cols": 3,
        "num_drones": 3,
        "num_emergencies": 1,
        "max_steps": 30,
        "bottleneck_zones": [],   # zones restricted to 1 drone at a time
        "dynamic_obstacles": [],  # list of (step_range, zone) that become blocked
        "battery_drain_per_step": 5.0,
        "emergency_deadline": 20,   # steps within which emergency must arrive
    },

    # ------------------------------------------------------------------
    # MEDIUM - 4?-4 grid, 5 drones (2 emergencies), bottlenecks, max 40 steps
    # ------------------------------------------------------------------
    "medium": {
        "name": "medium",
        "description": (
            "5-drone scenario on a 4?-4 grid with two emergency drones and "
            "bottleneck zones (B2, C3) that can hold at most one drone. "
            "Tests congestion awareness and priority routing."
        ),
        "rows": 4,
        "cols": 4,
        "num_drones": 5,
        "num_emergencies": 2,
        "max_steps": 40,
        "bottleneck_zones": ["B2", "C3"],
        "dynamic_obstacles": [],
        "battery_drain_per_step": 4.0,
        "emergency_deadline": 25,
    },

    # ------------------------------------------------------------------
    # HARD - 5?-5 grid, 10 drones (3 emergencies), dynamic obstacles, max 50 steps
    # ------------------------------------------------------------------
    "hard": {
        "name": "hard",
        "description": (
            "10-drone scenario on a 10?-10 grid with three emergency drones. "
            "Optimised for AirSim high-fidelity city transitions."
        ),
        "rows": 10,
        "cols": 10,
        "num_drones": 10,
        "num_emergencies": 3,
        "max_steps": 50,
        "max_altitude": 100.0,
        "bottleneck_zones": ["E5", "F6", "G5"],
        "dynamic_obstacles": [
            (10, 20, "C2"),
            (25, 35, "D4"),
        ],
        "battery_drain_per_step": 1.0,
        "emergency_deadline": 40,
    },
}


def get_task_config(task_name: str) -> dict:
    """Return the task config dict and inject a pre-built adjacency graph."""
    if task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown task '{task_name}'. Valid options: {list(TASK_CONFIGS.keys())}"
        )
    cfg = TASK_CONFIGS[task_name].copy()
    cfg["graph"] = _build_grid_graph(cfg["rows"], cfg["cols"])
    cfg["all_zones"] = _all_zones(cfg["rows"], cfg["cols"])
    return cfg
