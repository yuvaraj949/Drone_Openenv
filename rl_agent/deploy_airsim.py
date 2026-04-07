import sys
import os
import torch
import numpy as np
import time

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.airsim_env import AirSimDroneEnv
from rl_agent.ppo_agent import PPOAgent
from environment.models import Action, DroneAction

def deploy_airsim(model_path="models/ppo/ppo_curriculum.pt", task="hard"):
    # Connect to AirSim
    # Note: Requires a running AirSim binary (AirSimNH.exe)
    try:
        env = AirSimDroneEnv(task=task)
        env.connect()
    except Exception as e:
        print(f"[ERROR] Could not connect to AirSim: {e}")
        print("Please ensure AirSim is running and IP/Port are correct.")
        return

    # Load trained model
    agent = PPOAgent(state_dim=36, action_dim=3)
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"[WARN] No model at {model_path}. Running with random weights.")

    print(f"\n--- DEPLOYING PPO POLICY TO AIRSIM (Task: {task}) ---")
    
    obs = env.reset()
    done = False
    
    try:
        while not done:
            drone_actions = []
            
            # Select action for each vehicle in the simulation
            for i, drone in enumerate(obs.drones):
                if drone.delivered or drone.battery <= 0:
                    drone_actions.append(DroneAction(drone_id=drone.id, move_to="hover", thrust_vector=[0, 0, 0]))
                    continue
                
                state = agent._extract_state(obs, i)
                state_t = torch.FloatTensor(state).to(agent.device).unsqueeze(0)
                
                with torch.no_grad():
                    dist, _ = agent.policy(state_t)
                    # Use mean thrust for production deployment
                    action_thrust = dist.mean
                
                # Scale from [-1, 1] to simulator thrust (e.g., 20N)
                thrust_list = (action_thrust.cpu().numpy()[0] * 10.0).tolist()
                drone_actions.append(DroneAction(drone_id=drone.id, move_to="hover", thrust_vector=thrust_list))
            
            # Execute in AirSim
            env_action = Action(actions=drone_actions)
            obs, reward, done, info = env.step(env_action)
            
            print(f"Step {info['step']} | Collisions: {info['collisions']} | Delivered: {sum(d.delivered for d in obs.drones)}")
            
    except KeyboardInterrupt:
        print("\nDeployment stopped by user.")
    
    print("\n[DEPLOYMENT COMPLETE]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo/ppo_curriculum.pt")
    parser.add_argument("--task", type=str, default="hard")
    args = parser.parse_args()
    
    deploy_airsim(model_path=args.model, task=args.task)
