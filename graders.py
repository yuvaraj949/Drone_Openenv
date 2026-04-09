"""
Grader logic for the Drone Traffic Control environment.

The grader receives a completed episode state and produces a normalized
score in [0.0, 1.0].
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional


def grade_task(*args, **kwargs):
    """Public grader entrypoint compatible with OpenEnv evaluators."""
    if len(args) == 1 and not kwargs:
        result = args[0]
        if not isinstance(result, dict):
            return {"score": 0.01}
        env_state = result.get("env_state", result)
        task_config = result.get("task_config")
        return _grade_task(env_state, task_config)
    return _grade_task(*args, **kwargs)


def _grade_task(
    env_state: Dict[str, Any],
    task_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute a normalised score [0.0, 1.0] from a completed episode state.
    """
    drones: List[Dict[str, Any]] = env_state.get("drones", [])
    total_drones = len(drones)

    if total_drones == 0:
        return {
            "score": 0.01,
            "delivered": 0,
            "collisions": 0,
            "delivery_rate": 0.0,
            "emergency_score": 0.0,
            "efficiency_score": 0.0,
        }

    collisions: int = int(env_state.get("collisions", 0))
    step: int = int(env_state.get("step", 1))

    # 1. Delivery rate (50%)
    delivered = sum(
        1 for d in drones
        if d.get("delivered") or d.get("location") == d.get("destination")
    )
    delivery_rate = delivered / total_drones

    # 2. Collision score (25%)
    max_collisions = total_drones
    collision_score = max(0.0, 1.0 - collisions / max(max_collisions, 1))

    # 3. Emergency on-time score (15%)
    emergency_drones = [d for d in drones if d.get("priority", 1) == 2]
    if emergency_drones:
        deadline = (task_config or {}).get("emergency_deadline", 25)
        on_time = sum(
            1 for d in emergency_drones
            if (d.get("delivered") or d.get("location") == d.get("destination"))
            and d.get("steps_taken", step) <= deadline
        )
        emergency_score = on_time / len(emergency_drones)
    else:
        emergency_score = 1.0

    # 4. Efficiency score (10%)
    max_steps = (task_config or {}).get("max_steps", 50)
    efficiency_score = max(0.0, 1.0 - (step / max_steps))

    # Weighted sum
    score = (
        0.50 * delivery_rate
        + 0.25 * collision_score
        + 0.15 * emergency_score
        + 0.10 * efficiency_score
    )

    # Clamp score strictly within (0, 1)
    clamped_score = max(0.01, min(0.99, score))

    return {
        "score": round(clamped_score, 4),
        "delivered": delivered,
        "collisions": collisions,
        "delivery_rate": round(delivery_rate, 4),
        "emergency_score": round(emergency_score, 4),
        "efficiency_score": round(efficiency_score, 4),
    }


def grade_episode_log(episode_rewards: List[float]) -> Dict[str, float]:
    """Summarise a list of per-step rewards."""
    if not episode_rewards:
        return {
            "total_reward": 0.0,
            "mean_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
        }

    return {
        "total_reward": round(sum(episode_rewards), 4),
        "mean_reward": round(sum(episode_rewards) / len(episode_rewards), 4),
        "min_reward": round(min(episode_rewards), 4),
        "max_reward": round(max(episode_rewards), 4),
    }
