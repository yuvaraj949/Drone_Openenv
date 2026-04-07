import asyncio
import os
import sys

# Ensure the local environment package is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from environment.drone_env import DroneDispatchEnv
from environment.models import Action, DroneAction

async def test_env():
    print("Testing DroneDispatchEnv (3D-lite)...")
    env = DroneDispatchEnv(task="easy")
    obs = await env.reset()
    print(f"Initial Observation: {len(obs.drones)} drones, Step: {obs.step}")
    
    # Take a hover step
    action = Action(actions=[
        DroneAction(drone_id=d.id, move_to="hover", climb=0.0) 
        for d in obs.drones
    ])
    obs, reward, done, info = await env.step(action)
    print(f"Step 1: Reward={reward.total}, Done={done}, Collisions={info['collisions']}")
    
    st = await env.state()
    print(f"Current State: {st['delivered']} deliveries")
    print("Test PASSED.")

if __name__ == "__main__":
    asyncio.run(test_env())
