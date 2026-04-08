"""
Programmatic grader for the Drone Autonomous Dispatcher environment.
Evaluates agent performance on a scale of 0.0 to 1.0.
"""

from typing import Any, Dict

def grade_task(state: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Calculate final normalized score.
    Weights:
      - 50% Delivery Success (rate of drones delivered)
      - 25% Safety (penalizes collisions)
      - 15% Battery Efficiency (remaining battery of delivered drones)
      - 10% Time Efficiency (steps taken / max_steps)
    """
    drones = state.get("drones", [])
    num_drones = len(drones)
    if num_drones == 0: return 0.0
    
    delivered = [d for d in drones if d.get("delivered", False)]
    delivery_rate = len(delivered) / num_drones
    
    # Safety: Linear penalty for collisions. Max 5 collisions to reach 0 safety.
    colls = state.get("collisions", 0)
    safety_score = max(0.0, 1.0 - (colls * 0.2))
    
    # Battery: Average remaining battery of DELIVERED drones
    if delivered:
        bat_score = sum(d.get("battery", 0.0) for d in delivered) / (100.0 * len(delivered))
    else:
        bat_score = 0.0
        
    # Time: Normalized inversely to steps taken
    steps = state.get("step", 0)
    max_steps = config.get("max_steps", 100)
    time_score = max(0.0, 1.0 - (steps / max_steps))
    
    # Weighted Sum
    final_score = (
        (delivery_rate * 0.50) +
        (safety_score * 0.25) +
        (bat_score * 0.15) +
        (time_score * 0.10)
    )
    
    return round(min(max(final_score, 0.0), 1.0), 4)
