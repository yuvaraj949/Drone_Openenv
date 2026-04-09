"""
Grader logic for the Drone Traffic Control environment.

The grader receives a completed episode state and returns a normalized
score in [0.01, 0.99] plus diagnostic metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def grade_task(env_state: Dict[str, Any], task_config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Main grader function used by the validator.
    
    Expected signature: grade_task(env_state, task_config=None, **kwargs)
    Returns a dict with 'score' key containing float between 0.01 and 0.99
    """
    try:
        task_id = None
        if task_config:
            task_id = task_config.get("id") or task_config.get("task_id")
        elif kwargs:
            task_id = kwargs.get("task_id") or kwargs.get("id")
        
        return _compute_score(env_state, task_config, task_id)
    except Exception as e:
        return {"score": 0.01, "error": str(e)}


def _compute_score(
    env_state: Dict[str, Any],
    task_config: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute normalized score from episode state."""
    
    if not isinstance(env_state, dict):
        return {"score": 0.01}

    drones: List[Dict[str, Any]] = env_state.get("drones", [])
    total_drones = len(drones)

    collisions = int(env_state.get("collisions", 0) or 0)
    step = int(env_state.get("step", 0) or 0)

    if total_drones == 0:
        return {"score": 0.01}

    def is_delivered(drone: Dict[str, Any]) -> bool:
        return bool(
            drone.get("delivered")
            or drone.get("location") == drone.get("destination")
        )

    delivered = sum(1 for d in drones if is_delivered(d))
    delivery_rate = delivered / total_drones

    max_collisions = max(total_drones, 1)
    collision_score = max(0.0, 1.0 - (collisions / max_collisions))

    emergency_drones = [d for d in drones if int(d.get("priority", 1) or 1) == 2]
    if emergency_drones:
        deadline = int((task_config or {}).get("emergency_deadline", 25))
        on_time = sum(
            1
            for d in emergency_drones
            if is_delivered(d) and int(d.get("steps_taken", step) or step) <= deadline
        )
        emergency_score = on_time / len(emergency_drones)
    else:
        emergency_score = 1.0

    max_steps = int((task_config or {}).get("max_steps", 50))
    max_steps = max(max_steps, 1)
    efficiency_score = max(0.0, 1.0 - (step / max_steps))

    score = (
        0.50 * delivery_rate
        + 0.25 * collision_score
        + 0.15 * emergency_score
        + 0.10 * efficiency_score
    )

    clamped_score = max(0.01, min(0.99, float(score)))

    return {
        "score": round(clamped_score, 4),
        "delivered": delivered,
        "collisions": collisions,
    }


def grade_task_easy(env_state: Dict[str, Any], task_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Grader for easy task - delegates to main grade_task."""
    return grade_task(env_state, task_config)


def grade_task_medium(env_state: Dict[str, Any], task_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Grader for medium task - delegates to main grade_task."""
    return grade_task(env_state, task_config)


def grade_task_hard(env_state: Dict[str, Any], task_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Grader for hard task - delegates to main grade_task."""
    return grade_task(env_state, task_config)


def grade_episode_log(episode_rewards: List[float]) -> Dict[str, float]:
    """Summarize a list of per-step rewards."""
    if not episode_rewards:
        return {"total_reward": 0.0, "mean_reward": 0.0}

    total_reward = float(sum(episode_rewards))
    return {
        "total_reward": round(total_reward, 4),
        "mean_reward": round(total_reward / len(episode_rewards), 4),
    }
