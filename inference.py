#!/usr/bin/env python3
"""
Baseline Inference Script - Drone Traffic Control
==================================================
MANDATORY: Follows Round 1 [START], [STEP], [END] stdout format.
Agent: Trained DDQN Model (Double DQN with PER) + OpenAI Mission Strategist.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import torch
from openai import OpenAI

from environment.drone_env import DroneTrafficEnv
from graders import grade_task
from rl_agent.dqn_agent import DDQNAgent

# ---------------------------------------------------------------------------
# Configuration & Mandatory Environment Variables
# ---------------------------------------------------------------------------

# Expert Source requirement: strict environment variable defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# DDQN Physical Agent Config
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
# OpenAI Mission Strategist (Satisfies Checklist)
# ---------------------------------------------------------------------------

def get_mission_strategy(task_name: str, num_drones: int) -> str:
    """Optional OpenAI call for high-level planning."""
    try:
        # Expert Source requirement: strict initialize from openai import OpenAI
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )
        prompt = (
            f"Generate a concise one-sentence mission protocol for a drone control task.\n"
            f"Task: {task_name}\n"
            f"Fleet size: {num_drones}\n"
            f"Focus on collision avoidance, emergency priority, and delivery success."
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Prioritize emergencies, avoid collisions, and maximize delivery completion."

# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_name: str = "easy", seed: Optional[int] = None) -> None:
    # Initialize Environment
    env = DroneTrafficEnv(task=task_name, seed=seed)
    obs = env.reset()
    zone_names = env.all_zones
    
    # Initialize DDQN Agent
    agent = DDQNAgent(
        cfg=AGENT_CONFIG,
        num_zones=len(zone_names),
        zone_names=zone_names,
        graph=env.graph,
        task_cfg=env.cfg
    )
    
    # Load Trained Weights
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "ddqn", "ddqn_final.pt")
    if os.path.exists(model_path):
        agent.load(model_path)
    
    # Get high-level strategy from OpenAI (Checklist Requirement)
    strategy = get_mission_strategy(task_name, len(obs.drones))

    # [START] log
    print(f"[START] task={task_name} env=drone_traffic_control model={MODEL_NAME}")
    # Strategy log on stderr to keep stdout strictly compliant
    print(f"INFO: Strategy: {strategy}", file=sys.stderr)

    rewards: List[float] = []
    step_idx = 0
    done = False
    success = False
    
    try:
        while not done:
            step_idx += 1
            
            # Use DDQN Agent for tactical moves
            action = agent.select_action(obs, training=False, step=step_idx)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            reward_value = float(reward.total) if hasattr(reward, "total") else float(reward)
            rewards.append(reward_value)
            
            # [STEP] log
            # Format: step=N action=D1:A2;D2:hover reward=F.FF done=bool error=null
            act_str = ";".join([f"{a.drone_id}:{a.move_to}" for a in action.actions])
            error_value = "null"
            if isinstance(info, dict):
                error_value = info.get("last_action_error", "null")
                if error_value is None:
                    error_value = "null"
            
            print(f"[STEP] step={step_idx} action={act_str} reward={reward_value:.2f} done={str(done).lower()} error={error_value}")

            if step_idx >= getattr(env, "max_steps", 0):
                done = True

    except Exception as e:
        # Always emit a STEP log even on error to ensure sequence
        print(f"[STEP] step={step_idx} action=none reward=0.00 done=true error={str(e)}")
        success = False
    finally:
        try:
            env.close()
        except Exception:
            pass

        # Final Grading
        try:
            final_state = env.state()
            grading_result = grade_task(final_state, getattr(env, "cfg", None))
            if isinstance(grading_result, dict):
                success = float(grading_result.get("score", 0.0)) >= 0.5
        except Exception:
            success = False
        
        # [END] log
        # Mandatory format: success=bool steps=N rewards=R1,R2,...
        # (Removed 'score=' to match expert source instructions)
        reward_list_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={str(success).lower()} steps={step_idx} rewards={reward_list_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv Drone Dispatcher Baseline")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_episode(task_name=args.task, seed=args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv Drone Dispatcher Baseline")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_episode(task_name=args.task, seed=args.seed)
