# Grader wrapper functions for each task

from typing import Any, Dict


def grade_task_easy(env_state: Dict[str, Any], task_config: Any = None) -> Dict[str, Any]:
    """Grader for easy task."""
    try:
        from environment.graders import grade_task
        return grade_task(env_state, task_config)
    except Exception:
        return {"score": 0.01}


def grade_task_medium(env_state: Dict[str, Any], task_config: Any = None) -> Dict[str, Any]:
    """Grader for medium task."""
    try:
        from environment.graders import grade_task
        return grade_task(env_state, task_config)
    except Exception:
        return {"score": 0.01}


def grade_task_hard(env_state: Dict[str, Any], task_config: Any = None) -> Dict[str, Any]:
    """Grader for hard task."""
    try:
        from environment.graders import grade_task
        return grade_task(env_state, task_config)
    except Exception:
        return {"score": 0.01}
