"""
Baseline Inference Script — Drone Autonomous Dispatcher
======================================================
MANDATORY: Follows Round 1 [START], [STEP], [END] stdout format.
Agent: Trained DDQN Model (Double DQN with PER).
"""

import os
import sys
from typing import List, Optional, Dict

# Ensure local environment package is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from environment.drone_env import DroneTrafficEnv
from environment.models import Action, DroneAction, HOVER
from environment.graders import grade_task
from environment.tasks import get_task_config
from environment.dqn_agent import DDQNAgent

# Environment variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
TASK_NAME = os.getenv("TASK", "easy")
MODEL_NAME = "Trained-DDQN-v1"

# ---------------------------------------------------------------------------
# Agent Configuration (matching training setup in config.ini)
# ---------------------------------------------------------------------------

AGENT_CONFIG = {
    'general': {'device': 'cpu'},
    'network': {'hidden_sizes': '512, 512', 'activation': 'relu'},
    'dqn': {
        'gamma': 0.99, 'learning_rate': 0.001, 'batch_size': 128,
        'update_target_interval': 500, 'train_interval': 4,
        'wait_before_train': 500, 'buffer_len': 10000,
        'epsilon_start': 0.05, 'epsilon_end': 0.05, 'epsilon_decay_steps': 25000
    },
    'logging': {'tensorboard_dir': 'runs/ddqn'}
}

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
# Episode Runner
# ---------------------------------------------------------------------------

def main() -> None:
    # Initialize Environment
    env = DroneTrafficEnv(task=TASK_NAME)
    cfg = env.cfg
    zone_names = env.all_zones

    # Initialize Agent
    agent = DDQNAgent(
        cfg=AGENT_CONFIG,
        num_zones=len(zone_names),
        zone_names=zone_names,
        graph=env.graph,
        task_cfg=cfg
    )

    # Load Trained Weights
    model_path = os.path.join(os.path.dirname(__file__), "models", "ddqn_final.pt")
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"[WARN] Model checkpoint not found at {model_path}. Using random initialization.")

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env="drone_traffic", model=MODEL_NAME)

    try:
        obs = env.reset()
        max_steps = cfg["max_steps"]

        # 3D-lite: assigning altitude highways to avoid mid-air collisions (as a wrapper/safety)
        safety_margin = 3.0
        drone_highways = {d.id: 10.0 + (i * safety_margin) for i, d in enumerate(obs.drones)}

        for step_idx in range(1, max_steps + 1):
            # Select Action using Trained DDQN
            # The agent expects an Observation object and returns an Action object
            action = agent.select_action(obs, training=False, step=step_idx)

            # Post-process actions for 3D-lite altitude control
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
        success = score >= 0.5 # Success threshold in 0-1 range

    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
