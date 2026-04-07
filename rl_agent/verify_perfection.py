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

def verify_perfection(model_path="models/ppo/ppo_airsim.pt", num_episodes=10):
    # Testing on HARD task for absolute validation
    try:
        env = AirSimDroneEnv(task="hard")
        env.connect()
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    agent = PPOAgent(state_dim=36, action_dim=3)
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"No model found at {model_path}")
        return

    stats = {
        "success_rate": 0,
        "total_deliveries": 0,
        "total_collisions": 0,
        "avg_battery_consumed": []
    }

    for ep in range(1, num_episodes + 1):
        print(f"\nEpisode {ep}/{num_episodes}...")
        obs = env.reset()
        done = False
        ep_collisions = 0
        
        while not done:
            drone_actions = []
            for i, drone in enumerate(obs.drones):
                if drone.delivered:
                    drone_actions.append(DroneAction(drone_id=drone.id, move_to="hover", thrust_vector=[0,0,0]))
                    continue
                
                state = agent._extract_state(obs, i)
                state_t = torch.FloatTensor(state).to(agent.device).unsqueeze(0)
                with torch.no_grad():
                    dist, _ = agent.policy(state_t)
                    action_thrust = dist.mean # Deterministic for evaluation
                
                v_vec = (action_thrust.cpu().numpy()[0] * 5.0).tolist()
                drone_actions.append(DroneAction(drone_id=drone.id, move_to="hover", thrust_vector=v_vec))
            
            obs, reward, done, info = env.step(Action(actions=drone_actions))
            ep_collisions += info.get("collisions", 0)
        
        delivered = sum(d.delivered for d in obs.drones)
        stats["total_deliveries"] += delivered
        stats["total_collisions"] += ep_collisions
        if delivered == len(obs.drones) and ep_collisions == 0:
            stats["success_rate"] += 1
            
        print(f"EP {ep} Result: Delivered {delivered}/{len(obs.drones)} | Collisions: {ep_collisions}")

    # Final Report
    print("\n" + "="*40)
    print("PERFECTION BENCHMARK REPORT")
    print("="*40)
    print(f"Overall Success Rate: {stats['success_rate']/num_episodes*100:.1f}%")
    print(f"Total Collisions: {stats['total_collisions']}")
    print(f"Avg Deliveries: {stats['total_deliveries']/num_episodes:.1f}/{len(obs.drones)}")
    print("="*40)

if __name__ == "__main__":
    verify_perfection()
