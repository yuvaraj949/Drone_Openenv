from __future__ import annotations
import os

# Silencing TensorFlow warnings (oneDNN, CPU instructions, etc.)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import io
import json
import tempfile
import time
from typing import Any, Dict, Generator, List, Tuple, Optional

import numpy as np
import gradio as gr

from environment.drone_env import DroneTrafficEnv
from environment.graders import grade_episode_log, grade_task
from environment.models import Action, DroneAction, DroneState, HOVER, Observation
from collections import deque, defaultdict
import torch
from rl_agent.dqn_agent import DDQNAgent
import configparser

def read_config(path: str) -> dict:
    parser = configparser.ConfigParser()
    parser.read(path)
    return {section: dict(parser.items(section)) for section in parser.sections()}


# ---------------------------------------------------------------------------
# Agent (same greedy BFS as inference.py — importable here too)
# ---------------------------------------------------------------------------

def _bfs_next_zone(current, destination, graph, blocked_zones=None):
    blocked = set(blocked_zones or [])
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
                return new_path[1] if len(new_path) > 1 else current
            visited.add(nb)
            queue.append((nb, new_path))
    return current


def _act_greedy(obs: Observation) -> Action:
    graph = obs.graph_edges
    claimed: List[str] = []
    actions: List[DroneAction] = []
    for drone in sorted([d for d in obs.drones if not d.delivered], key=lambda d: (-d.priority, d.id)):
        if drone.battery <= 0.0:
            actions.append(DroneAction(drone_id=drone.id, move_to=HOVER))
            continue
        nz = _bfs_next_zone(drone.location, drone.destination, graph, claimed)
        move = HOVER if nz == drone.location else nz
        claimed.append(nz)
        actions.append(DroneAction(drone_id=drone.id, move_to=move))
    return Action(actions=actions)

# Cache for RL agent instance to avoid re-loading weights every step
_RL_AGENT_CACHE: Dict[str, DDQNAgent] = {}

def _get_rl_agent(env: DroneTrafficEnv) -> Optional[DDQNAgent]:
    global _RL_AGENT_CACHE
    model_path = "models/ddqn/ddqn_final.pt"
    if not os.path.exists(model_path):
        return None
    
    # We cache based on number of zones (actions) because the agent output dimension
    # must match. If it doesn't match, we start fresh or fallback.
    zone_names = list(env.cfg["graph"].keys())
    agent_key = f"agent_{len(zone_names)}"
    
    if agent_key not in _RL_AGENT_CACHE:
        try:
            cfg = read_config("rl_agent/config.ini")
            agent = DDQNAgent(cfg, len(zone_names), zone_names)
            agent.load(model_path)
            _RL_AGENT_CACHE[agent_key] = agent
        except Exception as e:
            print(f"Error loading agent for {len(zone_names)} zones: {e}")
            return None
    return _RL_AGENT_CACHE[agent_key]


def _act(obs: Observation, agent_type: str = "Greedy BFS", env: Optional[DroneTrafficEnv] = None) -> Action:
    if agent_type == "DDQN (RL)":
        agent = _get_rl_agent(env)
        if agent:
            return agent.select_action(obs, training=False)
            
    if agent_type == "PPO (Physics RL)":
        from rl_agent.ppo_agent import PPOAgent
        # PPO agent is initialized here for simplicity in this demo
        # In production, we would use a cache similar to DDQN
        agent = PPOAgent(state_dim=12, action_dim=3)
        return agent.select_action(obs)
    
    return _act_greedy(obs)

# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode_gradio(
    task: str,
    seed_str: str,
    algo: str = "Greedy BFS",
    use_3d: bool = False,
    physics_mode: bool = False,
    wind_intensity: float = 1.0,
    wind_gust: float = 0.5,
    use_airsim: bool = False,
    airsim_ip: str = "127.0.0.1",
    airsim_port: int = 41451,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Run a full episode synchronously.

    Returns
    -------
    log_text   : str — full step-by-step log
    gif_frames : list of PIL Images — one per step
    summary    : dict — final scores and stats
    """
    from visualizer.grid_vis import GridAnimator

    seed = int(seed_str) if (seed_str and seed_str.strip().isdigit()) else None

    if use_airsim:
        from environment.airsim_env import AirSimDroneEnv
        env = AirSimDroneEnv(ip=airsim_ip, port=int(airsim_port), task=task)
        # Attempt connection
        try:
            env.connect()
        except Exception as e:
            return f"❌ AirSim Connection Failed: {e}", None, {}
    elif physics_mode:
        from environment.physics_env import PhysicsDroneEnv
        env = PhysicsDroneEnv(task=task, seed=seed)
        env.base_wind = np.array([wind_intensity, wind_intensity * 0.3, 0.0])
        env.wind_gust_std = wind_gust
    else:
        env = DroneTrafficEnv(task=task, seed=seed)
    
    obs = env.reset()

    lines: List[str] = []
    
    if algo == "PEDRA (Legacy TF1)":
        from environment.pedra_bridge import PedraAirSimMock
        import sys
        # Point to the legacy folder so it finds its relative imports
        pedra_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pedra_legacy'))
        if pedra_path not in sys.path:
            sys.path.insert(0, pedra_path)

        # Mock AirSim behavior
        client = PedraAirSimMock(env)
        
        # In a real TF1 environment, you'd run the actual DeepQLearning loop here.
        lines.append("[INFO] Using PEDRA (Legacy) Bridge. Translating 3D commands to grid...")
        lines.append("[INFO] AirSimMock initialized. Client connected.")
    
    if use_3d:
        from visualizer.grid_vis_3d import GridAnimator3D
        animator = GridAnimator3D(
            rows=env.cfg["rows"],
            cols=env.cfg["cols"],
            max_alt=env.cfg.get("max_altitude", 15.0),
            task_name=task,
        )
    else:
        from visualizer.grid_vis import GridAnimator
        animator = GridAnimator(
            rows=env.cfg["rows"],
            cols=env.cfg["cols"],
            task_name=task,
            bottleneck_zones=env.cfg.get("bottleneck_zones", []),
        )
    animator.capture(obs)

    lines.append(f"[START] Task={task.upper()} | Drones={len(obs.drones)} | "
                 f"MaxSteps={env.max_steps} | Seed={seed}")
    lines.append("-" * 56)
    for d in obs.drones:
        tag = "🚨 EMERGENCY" if d.priority == 2 else "📦 normal"
        lines.append(f"  {d.id}: {d.location} → {d.destination} [{tag}]")
    lines.append("")

    step_rewards: List[float] = []
    done = False

    while not done:
        action = _act(obs, agent_type=algo, env=env)
        # 3D Vertical Separation Logic: Assigned 'Highways' for 10 drones
        if use_3d:
            safety_margin = env.cfg.get("safety_margin", 2.5)
            # Maintain assigned altitude highway to avoid vertical collisions
            for i, a in enumerate(action.actions):
                try:
                    # Extract index from 'Drone0', 'Drone1', etc.
                    d_idx = int(''.join(filter(str.isdigit, a.drone_id)))
                except:
                    d_idx = i
                
                # Target altitude is the highway offset
                target_alt = d_idx * safety_margin
                
                # Find actual drone state to check current altitude
                drone = next((d for d in obs.drones if d.id == a.drone_id), None)
                if drone:
                    # Move towards the target altitude highway
                    diff = target_alt - drone.altitude
                    # We limit vertical speed to 2.0m per step for smoothness
                    a.vertical_command = float(np.clip(diff, -2.0, 2.0))

        obs, reward, done, info = env.step(action)
        step_rewards.append(reward.total)

        lines.append(f"Step {obs.step}: TotalReward={reward.total:+.2f}")
        for d in obs.drones:
            if not d.delivered:
                lines.append(f"  {d.id} @ {d.location} (Batt: {d.battery:.1f}%, X:{d.x:.1f}, Y:{d.y:.1f}, Z:{d.altitude:.1f}m)")
            else:
                lines.append(f"  {d.id} ✅ DELIVERED (X:{d.x:.1f}, Y:{d.y:.1f}, Z:{d.altitude:.1f}m)")
        
        if info.get("cumulative_collisions", 0) > 0:
            lines.append(f"  💥 COLLISION DETECTED: {info.get('cumulative_collisions', 0)} occurrence(s)")
        
        animator.capture(obs)

    lines.append("-" * 56)
    lines.append("[DONE] Episode finished.")
    
    summary = grade_task(env.state(), env.cfg)
    
    lines.append(f"Final Score: {summary['score']:.4f}")
    lines.append(f"Deliveries: {summary['delivered']} / {len(obs.drones)}")
    lines.append(f"Collisions: {summary['collisions']}")
    
    # Save animation as WebP to return a file path (required by gr.Image)
    tmp_file = tempfile.NamedTemporaryFile(suffix=".webp", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()
    
    animator.save(tmp_path, fps=2)
    
    return "\n".join(lines), tmp_path, summary


# ---------------------------------------------------------------------------
# Gradio Blocks UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# 🚁 Drone Traffic Control (OpenEnv)
Manage a fleet of delivery drones. Avoid collisions, respect bottlenecks, and prioritise emergency payloads.
"""

TASK_INFO = {
    "easy": "3x3 Grid | 3 Drones | 1 Emergency | 30 Steps",
    "medium": "4x4 Grid | 5 Drones | 2 Emergency | 40 Steps | Bottlenecks",
    "hard": "5x5 Grid | 10 Drones | 3 Emergency | 50 Steps | Dynamic Obstacles",
}

with gr.Blocks(title="Drone Traffic Control — OpenEnv") as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Configuration")
            task_dropdown = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Task Difficulty",
                interactive=True,
            )
            task_info_box = gr.Markdown(TASK_INFO["easy"])
            seed_input = gr.Textbox(
                value="42",
                label="Random Seed (integer, or leave blank for random)",
                placeholder="e.g. 42",
            )
            algo_radio = gr.Radio(
                choices=["Greedy BFS", "DDQN (RL)", "PEDRA (Legacy TF1)", "PPO (Physics RL)"],
                value="Greedy BFS",
                label="Routing Algorithm",
                info="Choose between deterministic search, modern DDQN, or original PEDRA implementation.",
            )
            use_3d_check = gr.Checkbox(label="3D Space (Continuous Altitude)", value=True)
            physics_check = gr.Checkbox(label="Physics Mode (Wind & Dynamics)", value=False)
            
            with gr.Accordion("🌬️ Wind & Physics Settings", open=False):
                wind_base = gr.Slider(0, 5, value=1.0, label="Base Wind Intensity")
                wind_gust = gr.Slider(0, 2, value=0.5, label="Gust Variability")

            with gr.Accordion("🎮 AirSim (Unreal Engine) Settings", open=False):
                use_airsim_check = gr.Checkbox(label="Connect to AirSim", value=False)
                airsim_ip_input = gr.Textbox(value="127.0.0.1", label="AirSim IP")
                airsim_port_input = gr.Number(value=41451, label="AirSim Port")

            run_btn = gr.Button("▶  Run Episode", variant="primary", scale=1)

            gr.Markdown("---")
            gr.Markdown("### 📊 Results")
            score_out = gr.Label(label="Overall Performance Score")
            stats_out = gr.JSON(label="Detailed Stats")

        with gr.Column(scale=2):
            gr.Markdown("## 📺 Live Simulation")
            vis_out = gr.Image(label="Grid Visualization (Frame Slider)", type="pil")
            
            gr.Markdown("## 📜 Step Log")
            log_out = gr.Textbox(
                label="",
                lines=15,
                max_lines=25,
                interactive=False,
            )

    # Interactivity
    task_dropdown.change(fn=lambda t: TASK_INFO[t], inputs=task_dropdown, outputs=task_info_box)

    run_btn.click(
        fn=run_episode_gradio,
        inputs=[
            task_dropdown, seed_input, algo_radio, use_3d_check, 
            physics_check, wind_base, wind_gust, 
            use_airsim_check, airsim_ip_input, airsim_port_input
        ],
        outputs=[log_out, vis_out, score_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
