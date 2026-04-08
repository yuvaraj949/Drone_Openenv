"""
Task Registry for the Drone Autonomous Dispatcher environment.
Defines grid size, drone counts, deadlines, and dynamic constraints.
"""

from typing import Dict, List, Optional, Tuple

TASK_REGISTRY: Dict[str, dict] = {
    "easy": {
        "name": "easy",
        "description": "3 drones, one emergency, 3x3 grid. Reach target without collision.",
        "rows": 3,
        "cols": 3,
        "num_drones": 3,
        "num_emergencies": 1,
        "max_steps": 30,
        "energy_drain": 5.0,
        "deadline": 20,
    },
    "medium": {
        "name": "medium",
        "description": "5 drones, two emergencies, 4x4 grid with bottleneck zones (B2, C3).",
        "rows": 4,
        "cols": 4,
        "num_drones": 5,
        "num_emergencies": 2,
        "max_steps": 40,
        "bottlenecks": ["B2", "C3"],
        "energy_drain": 4.0,
        "deadline": 25,
    },
    "hard": {
        "name": "hard",
        "description": "10 drones, three emergencies, 5x5 grid with dynamic NFZs (No-Fly Zones).",
        "rows": 5,
        "cols": 5,
        "num_drones": 10,
        "num_emergencies": 3,
        "max_steps": 50,
        "bottlenecks": ["C3", "D4"],
        "dynamic_nfz": [
            (10, 20, "B2"),
            (25, 35, "D4"),
        ],
        "energy_drain": 2.0,
        "deadline": 35,
    }
}

def build_grid_graph(rows: int, cols: int) -> Dict[str, List[str]]:
    """Adjacency list for a rectangular grid."""
    row_labels = [chr(65 + r) for r in range(rows)]
    graph: Dict[str, List[str]] = {}
    for r, row in enumerate(row_labels):
        for c in range(1, cols + 1):
            zone = f"{row}{c}"
            moves = []
            if c > 1: moves.append(f"{row}{c-1}")
            if c < cols: moves.append(f"{row}{c+1}")
            if r > 0: moves.append(f"{chr(ord(row)-1)}{c}")
            if r < rows - 1: moves.append(f"{chr(ord(row)+1)}{c}")
            graph[zone] = moves
    return graph

def get_all_zones(rows: int, cols: int) -> List[str]:
    return [f"{chr(65+r)}{c}" for r in range(rows) for c in range(1, cols+1)]

def get_config(task: str) -> dict:
    if task not in TASK_REGISTRY:
        raise ValueError(f"Task '{task}' not found. Available: {list(TASK_REGISTRY.keys())}")
    cfg = TASK_REGISTRY[task].copy()
    cfg["graph"] = build_grid_graph(cfg["rows"], cfg["cols"])
    cfg["all_zones"] = get_all_zones(cfg["rows"], cfg["cols"])
    return cfg
