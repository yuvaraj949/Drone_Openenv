"""
OpenEnv-compatible FastAPI server for Drone Traffic Control environment.
Implements /reset, /step, /state, and /run_demo endpoints.
"""

from __future__ import annotations
import asyncio
import os
import sys
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from environment.drone_env import DroneTrafficEnv
from environment.models import Action, Observation, Reward
from environment.dqn_agent import DDQNAgent
from environment.graders import grade_task

# Global environment and agent instances
_env: Optional[DroneTrafficEnv] = None
_agent: Optional[DDQNAgent] = None
_task: str = "easy"

# Agent Configuration (matching baseline training setup)
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


def get_env_and_agent():
    """Get or create the global environment and agent instances."""
    global _env, _agent, _task
    if _env is None:
        _env = DroneTrafficEnv(task=_task)
    if _agent is None:
        zone_names = _env.all_zones
        _agent = DDQNAgent(
            cfg=AGENT_CONFIG,
            num_zones=len(zone_names),
            zone_names=zone_names,
            graph=_env.graph,
            task_cfg=_env.cfg
        )
        # Load Trained Weights
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "ddqn_final.pt")
        if os.path.exists(model_path):
            _agent.load(model_path)
    return _env, _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for environment and agent initialization."""
    # Initialize basic components on startup
    get_env_and_agent()
    yield
    # Shutdown logic
    global _env
    if _env is not None and hasattr(_env, "close"):
        if asyncio.iscoroutinefunction(_env.close):
            await _env.close()
        else:
            _env.close()


app = FastAPI(
    title="Drone Traffic Control - OpenEnv",
    description="Autonomous drone dispatcher environment with live demo",
    version="1.1.0",
    lifespan=lifespan,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the landing page."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/reset")
async def reset(task: Optional[str] = None) -> Dict[str, Any]:
    global _env, _task, _agent
    if task:
        _task = task
        _env = DroneTrafficEnv(task=task)
        _agent = None # Reset agent for new task config
    env, agent = get_env_and_agent()
    obs = env.reset()
    return {"observation": obs.model_dump()}


@app.post("/run_demo")
async def run_demo(task: Optional[str] = "easy") -> Dict[str, Any]:
    """Execute one full episode and return the mission trace for the UI."""
    env = DroneTrafficEnv(task=task)
    zone_names = env.all_zones
    agent = DDQNAgent(
        cfg=AGENT_CONFIG,
        num_zones=len(zone_names),
        zone_names=zone_names,
        graph=env.graph,
        task_cfg=env.cfg
    )
    
    # Load model Weights
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "ddqn_final.pt")
    if os.path.exists(model_path):
        agent.load(model_path)
    
    trace: List[Dict[str, Any]] = []
    obs = env.reset()
    max_steps = env.cfg["max_steps"]
    
    # Static altitude highways for 3D-lite mode (same as inference.py)
    safety_margin = 3.0
    drone_highways = {d.id: 10.0 + (i * safety_margin) for i, d in enumerate(obs.drones)}
    
    for step_idx in range(1, max_steps + 1):
        # Model selects action
        action = agent.select_action(obs, training=False, step=step_idx)
        
        # Post-process for 3D logic
        for drone_act in action.actions:
            drone_state = next((d for d in obs.drones if d.id == drone_act.drone_id), None)
            if drone_state:
                target_alt = drone_highways.get(drone_state.id, 15.0)
                if abs(drone_state.altitude - target_alt) > 0.5:
                    drone_act.vertical_command = 2.0 if drone_state.altitude < target_alt else -2.0
                else:
                    drone_act.vertical_command = 0.0

        # Step
        obs_new, reward_obj, done, info = env.step(action)
        
        trace.append({
            "step": step_idx,
            "actions": [f"{a.drone_id}: {a.move_to}" for a in action.actions],
            "reward": round(reward_obj.total, 2),
            "done": done,
            "drones": [{"id": d.id, "zone": d.zone, "altitude": round(d.altitude, 1), "battery": round(d.battery, 1)} for d in obs_new.drones]
        })
        
        obs = obs_new
        if done:
            break
            
    # Final metrics
    st = env.state()
    grading = grade_task(st, env.cfg)
    
    return {
        "trace": trace,
        "score": round(grading.get("score", 0.0), 3),
        "success": grading.get("score", 0.0) >= 0.5,
        "metrics": {
            "delivered": st.get("num_delivered", 0),
            "collisions": st.get("num_collisions", 0),
            "total_reward": round(sum(t["reward"] for t in trace), 1)
        }
    }


@app.post("/step")
async def step(action: Dict[str, Any]) -> Dict[str, Any]:
    env, _ = get_env_and_agent()
    try:
        action_obj = Action(**action)
        obs, reward_obj, done, info = env.step(action_obj)
        return {
            "observation": obs.model_dump(),
            "reward": reward_obj.model_dump(),
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/state")
async def state() -> Dict[str, Any]:
    env, _ = get_env_and_agent()
    state_dict = env.state()
    return {"state": state_dict}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


def main():
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
