"""
OpenEnv Server entry point.
Exposes the environment via FastAPI and Uvicorn.
"""

import uvicorn
from fastapi import FastAPI
from openenv_core import serve
from environment.drone_env import DroneDispatchEnv

app = FastAPI(title="Drone Dispatch OpenEnv Server")

# Initialize environment
env = DroneDispatchEnv(task="easy")

# Register OpenEnv routes
serve(app, env)

def main():
    """Main execution entry point."""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
