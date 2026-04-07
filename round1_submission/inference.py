"""
Baseline Inference Script — Drone Autonomous Dispatcher
======================================================
MANDATORY: Follows Round 1 [START], [STEP], [END] stdout format.
Agent: Greedy-BFS with 3D-lite altitude highways.
"""

import asyncio
import os
from collections import deque
from typing import List, Optional, Dict, Tuple

from environment.drone_env import DroneDispatchEnv
from environment.models import Action, DroneAction, HOVER
from environment.graders import grade_task
from environment.tasks import get_config

# Environment variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
TASK_NAME = os.getenv("TASK", "easy")
MODEL_NAME = "Greedy-BFS-Baseline"

MAX_STEPS = get_config(TASK_NAME)["max_steps"]

# ---------------------------------------------------------------------------
# BFS Logic
# ---------------------------------------------------------------------------

def bfs_next_zone(current: str, destination: str, graph: Dict[str, List[str]], blocked: set) -> str:
    if current == destination:
        return current
    
    queue = deque([(current, [current])])
    visited = {current}
    
    while queue:
        zone, path = queue.popleft()
        for nb in graph.get(zone, []):
            if nb in visited or nb in blocked:
                continue
            new_path = path + [nb]
            if nb == destination:
                return new_path[1]
            visited.add(nb)
            queue.append((nb, new_path))
    return current

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# Main Episode Runner
# ---------------------------------------------------------------------------

async def main() -> None:
    env = DroneDispatchEnv(task=TASK_NAME)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env="drone_dispatch", model=MODEL_NAME)

    try:
        obs = await env.reset()
        graph = obs.graph
        
        # 3D-lite: assigning altitude highways to avoid mid-air collisions
        # D1 -> 15m, D2 -> 18m, D3 -> 21m, etc.
        safety_margin = 3.0
        drone_highways = {d.id: 10.0 + (i * safety_margin) for i, d in enumerate(obs.drones)}

        for step_idx in range(1, MAX_STEPS + 1):
            drone_actions = []
            claimed_zones = set()
            
            # Sort by priority then ID to ensure deterministic behavior
            sorted_drones = sorted(obs.drones, key=lambda d: (-d.priority, d.id))
            
            for drone in sorted_drones:
                if drone.delivered: continue
                
                # Horizontal Move (BFS)
                next_zone = bfs_next_zone(drone.location, drone.destination, graph, claimed_zones)
                move = HOVER if next_zone == drone.location else next_zone
                claimed_zones.add(next_zone)
                
                # Vertical Move (Climb/Descend to assigned highway)
                target_alt = drone_highways.get(drone.id, 15.0)
                climb = 0.0
                if abs(drone.altitude - target_alt) > 0.5:
                    climb = 2.0 if drone.altitude < target_alt else -2.0
                
                drone_actions.append(DroneAction(drone_id=drone.id, move_to=move, climb=climb))

            action = Action(actions=drone_actions)
            obs, reward_obj, done, info = await env.step(action)
            
            reward = reward_obj.total
            rewards.append(reward)
            steps_taken = step_idx
            
            # Formatted action string for logs
            act_str = ";".join([f"{a.drone_id}:{a.move_to}" for a in action.actions])
            log_step(step=step_idx, action=act_str, reward=reward, done=done, error=None)

            if done:
                break

        # Final grading
        st = await env.state()
        score = grade_task(st, env.cfg)
        success = score >= 0.7  # Higher threshold for BFS baseline

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
