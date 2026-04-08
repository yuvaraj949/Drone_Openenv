"""
Baseline Inference Script — Drone Autonomous Dispatcher
======================================================
MANDATORY: Follows Round 1 [START], [STEP], [END] stdout format.
Agent: LLM-based Dispatcher (using OpenAI Client).
"""

import os
import sys
import json
from typing import List, Optional, Dict
from dotenv import load_dotenv

# Ensure local environment package is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables from .env file
load_dotenv()

from environment.drone_env import DroneTrafficEnv
from environment.models import Action, DroneAction, HOVER
from environment.graders import grade_task
from openai import OpenAI

# ---------------------------------------------------------------------------
# Requirement: API_BASE_URL, MODEL_NAME, HF_TOKEN
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

CLIENT = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

TASK_NAME = os.getenv("TASK", "easy")

# ---------------------------------------------------------------------------
# Logging Utilities
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
# LLM Agent Logic
# ---------------------------------------------------------------------------

def get_llm_action(obs) -> Action:
    """
    Asks the LLM to provide drone actions based on the current observation.
    """
    system_prompt = (
        "You are an autonomous drone traffic dispatcher. Your goal is to move drones to their destinations "
        "without collisions and while managing battery. "
        "Rules: Drones can move to an adjacent zone or 'hover'. Every step costs battery. "
        "Emergencies have priority (2). Bottleneck zones allow only 1 drone at a time. "
        "Respond ONLY with a JSON object matching the format: "
        "{\"actions\": [{\"drone_id\": \"D1\", \"move_to\": \"B2\", \"vertical_command\": 0.0}, ...]}"
    )

    # Simplified observation for context
    obs_json = {
        "step": obs.step,
        "drones": [
            {
                "id": d.id, "location": d.location, "destination": d.destination,
                "battery": round(d.battery, 1), "priority": d.priority, "altitude": round(d.altitude, 1)
            } for d in obs.drones if not d.delivered
        ],
        "congestion": obs.congestion_map,
        "graph": obs.graph_edges
    }

    try:
        response = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Current observation: {json.dumps(obs_json)}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content)
        return Action(**data)
    except Exception as e:
        # Fallback to HOVER on error to satisfy the loop
        return Action(actions=[DroneAction(drone_id=d.id, move_to=HOVER) for d in obs.drones if not d.delivered])

# ---------------------------------------------------------------------------
# Episode Runner
# ---------------------------------------------------------------------------

def main() -> None:
    # Initialize Environment
    env = DroneTrafficEnv(task=TASK_NAME)
    cfg = env.cfg

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env="drone_traffic", model=MODEL_NAME)

    try:
        obs = env.reset()
        max_steps = cfg["max_steps"]

        # 3D-lite: assigning altitude highways as a safety wrapper
        safety_margin = 3.0
        drone_highways = {d.id: 10.0 + (i * safety_margin) for i, d in enumerate(obs.drones)}

        for step_idx in range(1, max_steps + 1):
            # Select Action using LLM
            action = get_llm_action(obs)

            # Post-process actions for 3D-lite altitude control (safety highway)
            for drone_act in action.actions:
                drone_state = next((d for d in obs.drones if d.id == drone_act.drone_id), None)
                if drone_state:
                    target_alt = drone_highways.get(drone_state.id, 15.0)
                    if abs(drone_state.altitude - target_alt) > 0.5:
                        drone_act.vertical_command = 2.0 if drone_state.altitude < target_alt else -2.0
                    else:
                        drone_act.vertical_command = 0.0

            # Step Environment
            obs, reward_obj, done, info = env.step(action)

            reward = reward_obj.total
            rewards.append(reward)
            steps_taken = step_idx

            # Formatted action string for logs (drone_id:move_to)
            act_str = ";".join([f"{a.drone_id}:{a.move_to}" for a in action.actions])
            log_step(step=step_idx, action=act_str, reward=reward, done=done, error=None)

            if done:
                break

        # Final Grading
        st = env.state()
        grading_result = grade_task(st, cfg)
        score = grading_result.get("score", 0.0)
        success = score >= 0.5 

    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()

