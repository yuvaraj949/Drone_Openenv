# Task configuration for Drone Traffic Control environment
# Each task has its own grader function

from dataclasses import dataclass
from typing import Callable, Dict, Any

try:
    from tasks.graders import grade_task_easy, grade_task_medium, grade_task_hard
except Exception:
    try:
        from environment.graders import grade_task_easy, grade_task_medium, grade_task_hard
    except Exception:
        from graders import grade_task_easy, grade_task_medium, grade_task_hard


@dataclass
class TaskConfig:
    task_id: str
    difficulty: str
    description: str
    max_steps: int
    grader_fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        difficulty="easy",
        description="3 drones, 3x3 grid, zero obstacles. Basic navigation check.",
        max_steps=30,
        grader_fn=grade_task_easy,
    ),
    "medium": TaskConfig(
        task_id="medium",
        difficulty="medium",
        description="5 drones, 4x4 grid, battery drains + bottleneck zones.",
        max_steps=40,
        grader_fn=grade_task_medium,
    ),
    "hard": TaskConfig(
        task_id="hard",
        difficulty="hard",
        description="10 drones, 5x5 grid, dynamic No-Fly Zones + emergency priorities.",
        max_steps=50,
        grader_fn=grade_task_hard,
    ),
}


def get_task(task_id: str) -> TaskConfig:
    """Get task config by ID."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}. Valid: {list(TASKS.keys())}")
    return TASKS[task_id]
