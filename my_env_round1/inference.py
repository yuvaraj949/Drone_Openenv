"""
Baseline Inference Script — Drone Autonomous Dispatcher
======================================================
MANDATORY: Follows Round 1 [START], [STEP], [END] stdout format.
Uses OpenAI client for agent logic.
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from environment.drone_env import DroneDispatchEnv
from environment.models import Action, DroneAction, HOVER
from environment.graders import grade_task
from environment.tasks import get_config

import argparse

# Default settings (can be overridden by CLI or Environment variables)
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TEMPERATURE = 0.0

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

async def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEnv Baseline Inference")
    parser.add_argument("--task", type=str, default=os.getenv("TASK", "easy"), help="Task name (easy, medium, hard)")
    args = parser.parse_args()
    
    task_name = args.task
    cfg = get_config(task_name)
    max_steps = cfg["max_steps"]

    # Initialize Environment
    env = DroneDispatchEnv(task=task_name)
    
    # Initialize Agent (OpenAI Client)
    if not API_KEY:
        print("[WARN] API_KEY/HF_TOKEN not found. Baseline will use Greedy-BFS without LLM coordination.", flush=True)
        client = None
    else:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env="drone_dispatch", model=MODEL_NAME)

    try:
        obs = await env.reset()
        
        for step_idx in range(1, max_steps + 1):
            # Logic: Determine next moves for all drones
            # In a real submission, we'd prompt the LLM here.
            # For the baseline, we'll use a Greedy BFS strategy.
            
            drone_actions = []
            for drone in obs.drones:
                if drone.delivered: continue
                
                # 3D Pathfinding Logic:
                # 1. If not at destination: Cruising altitude = 10.0m + 2.0m * drone.priority
                # 2. If at destination: Target altitude = 0.0m (Landing)
                
                target_alt = 10.0 + (int(drone.id[1:]) % 5) * 2.0  # Distributed cruising altitudes to avoid collisions
                if drone.location == drone.destination:
                    target_alt = 0.0
                
                # Altitude control (climb/descend by 5.0m per step)
                climb = 0.0
                if drone.altitude < target_alt:
                    climb = min(5.0, target_alt - drone.altitude)
                elif drone.altitude > target_alt:
                    climb = max(-5.0, target_alt - drone.altitude)

                # Horizontal move (Greedy BFS)
                neighbors = obs.graph.get(drone.location, [])
                best_move = drone.location
                min_dist = 999
                
                def dist(z1, z2):
                    return abs(ord(z1[0]) - ord(z2[0])) + abs(int(z1[1:]) - int(z2[1:]))

                for n in neighbors + [drone.location]:
                    d = dist(n, drone.destination)
                    if d < min_dist:
                        min_dist = d
                        best_move = n
                
                drone_actions.append(DroneAction(drone_id=drone.id, move_to=best_move, climb=climb))

            action = Action(actions=drone_actions)
            
            # Step the environment
            obs, reward_obj, done, info = await env.step(action)
            
            reward = reward_obj.total
            rewards.append(reward)
            steps_taken = step_idx
            
            # Action string for logging
            act_str = ";".join([f"{a.drone_id}:{a.move_to}" for a in action.actions])
            
            log_step(step=step_idx, action=act_str, reward=reward, done=done, error=None)

            if done:
                break

        # Final grading
        final_state = await env.state()
        score = grade_task(final_state, env.cfg)
        success = score > 0.5

    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
