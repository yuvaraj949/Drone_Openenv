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

def train_airsim(max_episodes=500, update_interval=2000):
    # Direct training in Unreal Engine via AirSim
    # Note: Training in simulator is slow, so we optimize update frequency
    env = AirSimDroneEnv(task="easy") # Start with Easy for simulator stability
    agent = PPOAgent(state_dim=36, action_dim=3)
    
    model_path = "models/ppo/ppo_airsim.pt"
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Resuming AirSim-native training from {model_path}")
    # elif os.path.exists("models/ppo/ppo_curriculum.pt"):
    #     agent.load("models/ppo/ppo_curriculum.pt")
    #     print("Warm-starting AirSim training from curriculum physics model...")

    print(f"\n--- STARTING DIRECT AIRSIM TRAINING ---")
    
    memory_states = []
    memory_actions = []
    memory_logprobs = []
    memory_rewards = []
    memory_is_terminals = []
    
    timestep = 0
    
    for ep in range(1, max_episodes + 1):
        try:
            obs = env.reset()
        except Exception as e:
            print(f"[AirSim Error] Reset failed: {e}. Retrying in 5s...")
            time.sleep(5)
            continue
            
        done = False
        ep_reward = 0
        
        while not done:
            timestep += 1
            
            actions_list = []
            active_drones_indices = []
            
            # 1. Collective Action Selection
            for i, d in enumerate(obs.drones):
                if d.delivered or d.battery <= 0:
                    continue
                
                active_drones_indices.append(i)
                state = agent._extract_state(obs, i)
                state_t = torch.FloatTensor(state).to(agent.device).unsqueeze(0)
                
                with torch.no_grad():
                    dist, _ = agent.policy_old(state_t)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                
                memory_states.append(state_t.squeeze(0))
                memory_actions.append(action.squeeze(0))
                memory_logprobs.append(log_prob.squeeze(0))
                
                # Action Mapping: [-1, 1] thrust -> [-5, 5] velocity vector
                # Using velocity vector directly in AirSim for stable RL convergence
                v_vector = (action.cpu().numpy()[0] * 5.0).tolist()
                actions_list.append((d.id, v_vector))
            
            if not actions_list:
                break
            
            # 2. Step in AirSim
            env_action = Action(actions=[
                DroneAction(drone_id=aid, move_to="hover", thrust_vector=v_vec) 
                for aid, v_vec in actions_list
            ])
            
            try:
                next_obs, reward, done, info = env.step(env_action)
            except Exception as e:
                print(f"[AirSim Error] Step failed: {e}. Reconnecting...")
                try: 
                    env.connect()
                except: 
                    pass
                break
                
            # 3. Store transitions
            # All drones share the step reward for city-level optimization
            for i in range(len(active_drones_indices)):
                memory_rewards.append(reward.total) # Step-wise reward
                memory_is_terminals.append(done)
                
            obs = next_obs
            ep_reward += reward.total
            
            # 4. Update Policy
            if timestep % update_interval == 0 and len(memory_states) > 0:
                print(f"\n--- PPO Update at Timestep {timestep} (AirSim City) ---")
                agent.update(
                    memory_states, 
                    memory_actions, 
                    memory_logprobs, 
                    memory_rewards, 
                    memory_is_terminals
                )
                memory_states = []
                memory_actions = []
                memory_logprobs = []
                memory_rewards = []
                memory_is_terminals = []
                agent.save(model_path)

        if ep % 2 == 0:
            print(f"AirSim EP {ep:03d} | Reward: {ep_reward:+.2f} | Collisions: {info.get('collisions', 0)} | Deliv: {sum(d.delivered for d in obs.drones)}")

    # Final Save
    agent.save(model_path)
    print("\n--- AirSim Training Complete ---")

if __name__ == "__main__":
    train_airsim()
