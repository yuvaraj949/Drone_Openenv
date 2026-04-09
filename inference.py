#!/usr/bin/env python3
"""
Baseline Inference Script - Drone Traffic Control
==================================================
Required stdout format:
[START]
[STEP]
[END]
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import List, Optional

# Suppress noisy library warnings before imports that may trigger them
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from openai import OpenAI

from environment.drone_env import DroneTrafficEnv
from environment.graders import grade_task
from rl_agent.dqn_agent import DDQNAgent


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_TOTAL_REWARD = 1000.0  # Normalized reward scale for scoring



# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

AGENT_CONFIG = {
    "general": {"device": "cpu"},
    "network": {"hidden_sizes": "512, 512", "activation": "relu"},
    "dqn": {
        "gamma": 0.99,
        "learning_rate": 0.001,
        "batch_size": 128,
        "update_target_interval": 500,
        "train_interval": 4,
        "wait_before_train": 500,
        "buffer_len": 10000,
        "epsilon_start": 0.05,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 25000,
    },
    "logging": {"tensorboard_dir": "runs/ddqn"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_bool(v: bool) -> str:
    return "true" if v else "false"


def fmt_reward(v: float) -> str:
    return f"{float(v):.2f}"


def fmt_error(err: Optional[str]) -> str:
    if err is None or err == "":
        return "null"
    return str(err).replace("\n", " ").replace("\r", " ")


def emit_start(task_name: str, env_name: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={env_name} model={model_name}")


def emit_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={fmt_reward(reward)} done={fmt_bool(done)} error={fmt_error(error)}"
    )


def emit_end(success: bool, steps: int, rewards: List[float], score: float) -> None:
    rewards_str = ",".join(fmt_reward(r) for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def action_to_string(action) -> str:
    try:
        if hasattr(action, "actions"):
            parts = []
            for a in action.actions:
                drone_id = getattr(a, "drone_id", "unknown")
                move_to = getattr(a, "move_to", "none")
                parts.append(f"{drone_id}:{move_to}")
            return ";".join(parts) if parts else "noop"

        if isinstance(action, dict):
            return str(action).replace("\n", " ")

        return str(action).replace("\n", " ")
    except Exception:
        return "noop"


def get_mission_strategy(task_name: str, num_drones: int) -> str:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        prompt = (
            "Generate a concise one-sentence mission protocol for a drone control task.\n"
            f"Task: {task_name}\n"
            f"Fleet size: {num_drones}\n"
            "Focus on collision avoidance, emergency priority, and delivery success."
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return (
            content.strip()
            if content
            else "Prioritize emergencies, avoid collisions, and maximize delivery completion."
        )
    except Exception:
        return "Prioritize emergencies, avoid collisions, and maximize delivery completion."


# ---------------------------------------------------------------------------
# Main episode runner
# ---------------------------------------------------------------------------

def run_episode(task_name: str = "easy", seed: Optional[int] = None) -> None:
    env = None
    agent = None
    rewards: List[float] = []
    step_idx = 0
    success = False
    raw_score = 0.0

    try:
        env = DroneTrafficEnv(task=task_name, seed=seed)
        obs = env.reset()

        zone_names = getattr(env, "all_zones", [])
        agent = DDQNAgent(
            cfg=AGENT_CONFIG,
            num_zones=len(zone_names),
            zone_names=zone_names,
            graph=env.graph,
            task_cfg=env.cfg,
        )

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models", "ddqn", "ddqn_final.pt")
        if os.path.exists(model_path):
            agent.load(model_path)

        _ = get_mission_strategy(task_name, len(getattr(obs, "drones", [])))

        emit_start(task_name, "drone-traffic-control", MODEL_NAME)

        done = False
        max_steps = getattr(env, "max_steps", 0) or 0

        while not done:
            step_idx += 1

            action = agent.select_action(obs, training=False, step=step_idx)
            obs, reward, done, info = env.step(action)

            reward_value = float(getattr(reward, "total", reward))
            rewards.append(reward_value)

            step_error = None
            if isinstance(info, dict):
                step_error = info.get("last_action_error", None)

            emit_step(
                step=step_idx,
                action=action_to_string(action),
                reward=reward_value,
                done=done,
                error=step_error,
            )

            if max_steps and step_idx >= max_steps:
                done = True

        try:
            final_state = env.state()
            grading_result = grade_task(final_state, getattr(env, "cfg", None))
            if isinstance(grading_result, dict):
                raw_score = float(grading_result.get("score", 0.0))
                success = raw_score >= 0.5
        except Exception:
            success = False

    except Exception as e:
        print(f"# ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        # Calculate score based on total rewards
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.01
        score = min(max(score, 0.01), 0.99)

        emit_end(success=success, steps=step_idx, rewards=rewards, score=score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv Drone Dispatcher Baseline")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_episode(task_name=args.task, seed=args.seed)