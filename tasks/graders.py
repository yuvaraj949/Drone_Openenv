# Grader functions for OpenEnv validator
# Each task has its own grader with exact signature: (trajectory, task, **kwargs) -> float

def grade_task_easy(trajectory, task, **kwargs):
    """
    Grader for easy task.
    Expected signature: (trajectory, task, **kwargs) -> float
    """
    try:
        if trajectory is None:
            return 0.01
        
        drones = trajectory.get("drones", [])
        total = len(drones)

        if total == 0:
            return 0.01

        delivered = sum(1 for d in drones if d.get("delivered"))
        return float(max(0.01, min(0.99, delivered / total)))

    except Exception:
        return 0.01


def grade_task_medium(trajectory, task, **kwargs):
    """
    Grader for medium task.
    Expected signature: (trajectory, task, **kwargs) -> float
    """
    try:
        if trajectory is None:
            return 0.01
        
        drones = trajectory.get("drones", [])
        total = len(drones)

        if total == 0:
            return 0.01

        delivered = sum(1 for d in drones if d.get("delivered"))
        collisions = trajectory.get("collisions", 0)
        
        # Medium: include collision penalty
        delivery_score = delivered / total
        collision_penalty = min(0.3, collisions * 0.1)
        
        return float(max(0.01, min(0.99, delivery_score - collision_penalty)))

    except Exception:
        return 0.01


def grade_task_hard(trajectory, task, **kwargs):
    """
    Grader for hard task.
    Expected signature: (trajectory, task, **kwargs) -> float
    """
    try:
        if trajectory is None:
            return 0.01
        
        drones = trajectory.get("drones", [])
        total = len(drones)

        if total == 0:
            return 0.01

        delivered = sum(1 for d in drones if d.get("delivered"))
        collisions = trajectory.get("collisions", 0)
        step = trajectory.get("step", 50)
        
        # Hard: full scoring with efficiency
        delivery_score = delivered / total
        collision_penalty = min(0.3, collisions * 0.1)
        efficiency = max(0, 1 - step / 50)
        
        score = (0.6 * delivery_score) + (0.3 * efficiency) - collision_penalty
        
        return float(max(0.01, min(0.99, score)))

    except Exception:
        return 0.01
