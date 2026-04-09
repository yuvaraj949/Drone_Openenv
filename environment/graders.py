"""
Grader logic for the Drone Traffic Control environment.

The grader receives a completed episode state and returns a normalized
score in [0.01, 0.99] plus diagnostic metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def grade_task(*args, **kwargs) -> Dict[str, Any]:
    """
    Public grader entrypoint.

    Supported calling styles:
    - grade_task(env_state, task_config)
    - grade_task({"env_state": ..., "task_config": ...})
    """
    if len(args) == 1 and not kwargs and isinstance(args[0], dict):
        payload = args[0]
        env_state = payload.get("env_state", payload)
        task_config = payload.get("task_config")
        return _grade_task(env_state, task_config)

    if len(args) >= 1:
        env_state = args[0]
        task_config = args[1] if len(args) > 1 else kwargs.get("task_config")
        return _grade_task(env_state, task_config)

    env_state = kwargs.get("env_state", {})
    task_config = kwargs.get("task_config")
    return _grade_task(env_state, task_config)


def _grade_task(
    env_state: Dict[str, Any],
    task_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute a normalized score from completed episode state.
    """
    if not isinstance(env_state, dict):
        return {
            "score": 0.01,
            "delivered": 0,
            "collisions": 0,
            "delivery_rate": 0.0,
            "emergency_score": 0.0,
            "efficiency_score": 0.0,
        }

    drones: List[Dict[str, Any]] = env_state.get("drones", [])
    total_drones = len(drones)

    collisions = int(env_state.get("collisions", 0) or 0)
    step = int(env_state.get("step", 0) or 0)

    if total_drones == 0:
        return {
            "score": 0.01,
            "delivered": 0,
            "collisions": collisions,
            "delivery_rate": 0.0,
            "emergency_score": 0.0,
            "efficiency_score": 0.0,
        }

    def is_delivered(drone: Dict[str, Any]) -> bool:
        return bool(
            drone.get("delivered")
            or drone.get("location") == drone.get("destination")
        )

    delivered = sum(1 for d in drones if is_delivered(d))
    delivery_rate = delivered / total_drones

    # Collision score: fewer collisions is better.
    max_collisions = max(total_drones, 1)
    collision_score = max(0.0, 1.0 - (collisions / max_collisions))

    # Emergency score: emergency drones should be delivered on time.
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

    # Efficiency score: earlier completion is better.
    max_steps = int((task_config or {}).get("max_steps", 50))
    max_steps = max(max_steps, 1)
    efficiency_score = max(0.0, 1.0 - (step / max_steps))

    score = (
        0.50 * delivery_rate
        + 0.25 * collision_score
        + 0.15 * emergency_score
        + 0.10 * efficiency_score
    )

    if not isinstance(score, (int, float)) or score != score:
        score = 0.01

    clamped_score = max(0.01, min(0.99, float(score)))

    return {
        "score": round(clamped_score, 4),
        "delivered": delivered,
        "collisions": collisions,
        "delivery_rate": round(delivery_rate, 4),
        "emergency_score": round(emergency_score, 4),
        "efficiency_score": round(efficiency_score, 4),
    }


def grade_episode_log(episode_rewards: List[float]) -> Dict[str, float]:
    """Summarize a list of per-step rewards."""
    if not episode_rewards:
        return {
            "total_reward": 0.0,
            "mean_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
        }

    total_reward = float(sum(episode_rewards))
    return {
        "total_reward": round(total_reward, 4),
        "mean_reward": round(total_reward / len(episode_rewards), 4),
        "min_reward": round(float(min(episode_rewards)), 4),
        "max_reward": round(float(max(episode_rewards)), 4),
    }