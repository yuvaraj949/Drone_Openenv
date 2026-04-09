# Task configuration for Drone Traffic Control environment

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class TaskConfig:
    task_id: str
    difficulty: str
    description: str
    max_steps: int
    grader_fn: Callable


TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        difficulty="easy",
        description="3 drones, 3x3 grid, zero obstacles. Basic navigation check.",
        max_steps=30,
        grader_fn=None,  # Will be loaded dynamically from tasks.graders
    ),
    "medium": TaskConfig(
        task_id="medium",
        difficulty="medium",
        description="5 drones, 4x4 grid, battery drains + bottleneck zones.",
        max_steps=40,
        grader_fn=None,
    ),
    "hard": TaskConfig(
        task_id="hard",
        difficulty="hard",
        description="10 drones, 5x5 grid, dynamic No-Fly Zones + emergency priorities.",
        max_steps=50,
        grader_fn=None,
    ),
}


def get_task(task_id: str) -> TaskConfig:
    """Get task config by ID."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}. Valid: {list(TASKS.keys())}")
    return TASKS[task_id]
