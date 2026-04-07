"""
OpenEnv-compatible FastAPI server for Drone Traffic Control environment.
Implements /reset, /step, and /state endpoints as required by OpenEnv spec.
"""

from __future__ import annotations
import asyncio
import os
import sys
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from environment.drone_env import DroneTrafficEnv
from environment.models import Action, Observation, Reward

# Global environment instance
_env: Optional[DroneTrafficEnv] = None
_task: str = "easy"  # Default task


def get_env() -> DroneTrafficEnv:
    """Get or create the global environment instance."""
    global _env
    if _env is None:
        _env = DroneTrafficEnv(task=_task)
    return _env


app = FastAPI(
    title="Drone Traffic Control — OpenEnv",
    description="Autonomous drone dispatcher environment",
    version="1.0.0",
)


@app.on_event("startup")
async def startup():
    """Initialize environment on server startup."""
    global _env
    _task_env = os.getenv("TASK", "easy")
    _env = DroneTrafficEnv(task=_task_env)


@app.on_event("shutdown")
async def shutdown():
    """Clean up environment on shutdown."""
    global _env
    if _env is not None:
        await _env.close() if hasattr(_env, "close") else None


@app.post("/reset")
async def reset(task: Optional[str] = None) -> Dict[str, Any]:
    """
    Reset the environment and return the initial observation.

    Query Parameters:
    - task: optional task name ("easy", "medium", "hard")

    Returns:
    - observation: Initial Observation as JSON
    """
    global _env, _task
    if task:
        _task = task
        _env = DroneTrafficEnv(task=task)
    env = get_env()
    obs = env.reset()
    return {"observation": obs.model_dump()}


@app.post("/step")
async def step(action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step the environment with the given action.

    Body:
    - action: Action JSON

    Returns:
    - observation: Observation as JSON
    - reward: Reward as JSON
    - done: Boolean
    - info: Additional info dict
    """
    env = get_env()
    try:
        # Parse action from dict
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
    """
    Get the current environment state for grading.

    Returns:
    - state: Complete state dict
    """
    env = get_env()
    state_dict = env.state()
    return {"state": state_dict}


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


def main():
    """Main entry point for running the server."""
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

