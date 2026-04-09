# Grader functions for each task in the Drone Traffic Control environment

from typing import Any, Dict


def grade_task_easy(env_state: Dict[str, Any], task_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Grader for easy task.
    Easy: 3 drones, 3x3 grid - focus on basic delivery rate.
    """
    from environment.graders import _grade_task
    return _grade_task(env_state, task_config, difficulty="easy")


def grade_task_medium(env_state: Dict[str, Any], task_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Grader for medium task.
    Medium: 5 drones, 4x4 grid with bottlenecks - includes collision penalty.
    """
    from environment.graders import _grade_task
    return _grade_task(env_state, task_config, difficulty="medium")


def grade_task_hard(env_state: Dict[str, Any], task_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Grader for hard task.
    Hard: 10 drones, 5x5 grid with dynamic obstacles - full evaluation.
    """
    from environment.graders import _grade_task
    return _grade_task(env_state, task_config, difficulty="hard")
