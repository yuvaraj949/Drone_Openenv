"""
Task configurations for the Drone Traffic Control environment.
"""

from __future__ import annotations

from typing import Dict, List


def _build_grid_graph(rows: int, cols: int) -> Dict[str, List[str]]:
    """
    Build a 4-neighbour grid adjacency list.

    Zone labels follow:
    - Rows: A, B, C, ...
    - Columns: 1, 2, 3, ...
    Example: A1, A2, B1, B2
    """
    graph: Dict[str, List[str]] = {}

    for r in range(rows):
        row_label = chr(ord("A") + r)
        for c in range(1, cols + 1):
            zone = f"{row_label}{c}"
            neighbours: List[str] = []

            # Left / right
            if c > 1:
                neighbours.append(f"{row_label}{c - 1}")
            if c < cols:
                neighbours.append(f"{row_label}{c + 1}")

            # Up / down
            if r > 0:
                neighbours.append(f"{chr(ord('A') + r - 1)}{c}")
            if r < rows - 1:
                neighbours.append(f"{chr(ord('A') + r + 1)}{c}")

            graph[zone] = neighbours

    return graph


def _all_zones(rows: int, cols: int) -> List[str]:
    """Return all zone labels for the grid."""
    zones: List[str] = []
    for r in range(rows):
        row_label = chr(ord("A") + r)
        for c in range(1, cols + 1):
            zones.append(f"{row_label}{c}")
    return zones


TASK_CONFIGS: Dict[str, dict] = {
    "easy": {
        "id": "easy",
        "name": "easy",
        "description": (
            "3-drone scenario on a 3x3 grid with one emergency drone. "
            "Goal: route all drones to their destinations without collision."
        ),
        "rows": 3,
        "cols": 3,
        "num_drones": 3,
        "num_emergencies": 1,
        "max_steps": 30,
        "bottleneck_zones": [],
        "dynamic_obstacles": [],
        "battery_drain_per_step": 5.0,
        "emergency_deadline": 20,
        "max_altitude": 100.0,
    },
    "medium": {
        "id": "medium",
        "name": "medium",
        "description": (
            "5-drone scenario on a 4x4 grid with two emergency drones and "
            "bottleneck zones that can hold at most one drone."
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
        "max_altitude": 100.0,
    },
    "hard": {
        "id": "hard",
        "name": "hard",
        "description": (
            "10-drone scenario on a 5x5 grid with three emergency drones. "
            "Includes dynamic obstacles and priority routing."
        ),
        "rows": 5,
        "cols": 5,
        "num_drones": 10,
        "num_emergencies": 3,
        "max_steps": 50,
        "bottleneck_zones": ["E5", "D4", "C5"],
        "dynamic_obstacles": [
            (10, 20, "C2"),
            (25, 35, "D4"),
        ],
        "battery_drain_per_step": 1.0,
        "emergency_deadline": 40,
        "max_altitude": 100.0,
    },
}


TASKS = [
    {"id": "easy", "config": TASK_CONFIGS["easy"]},
    {"id": "medium", "config": TASK_CONFIGS["medium"]},
    {"id": "hard", "config": TASK_CONFIGS["hard"]},
]


def get_task_config(task_name: str) -> dict:
    """
    Return a copy of the selected task config with derived fields added.
    """
    if task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown task '{task_name}'. Valid options: {list(TASK_CONFIGS.keys())}"
        )

    cfg = dict(TASK_CONFIGS[task_name])
    cfg["graph"] = _build_grid_graph(cfg["rows"], cfg["cols"])
    cfg["all_zones"] = _all_zones(cfg["rows"], cfg["cols"])
    return cfg