"""
PPO Training Loop for PhysicsDroneEnv
=====================================
Trains the PPO agent on continuous 3D control using point-mass physics.
"""

import sys
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Silencing TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.physics_env import PhysicsDroneEnv
from rl_agent.ppo_agent import PPOAgent, Memory

def train_ppo():
    # Parameters
    task = "easy"
    max_episodes = 2000
    update_timestep = 2000 # Update policy every N timesteps
    log_interval = 10
    save_interval = 100
    
    # Environment & Agent
    env = PhysicsDroneEnv(task=task, seed=42)
    state_dim = 12
    action_dim = 3
    
    agent = PPOAgent(state_dim, action_dim, lr=3e-4)
    memory = Memory()
    
    # Logging
    writer = SummaryWriter(log_dir="runs/ppo")
    model_dir = "models/ppo"
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Starting PPO Training: {max_episodes} episodes on task '{task}'")
    
    time_step = 0
    for ep in range(1, max_episodes + 1):
        obs = env.reset()
        ep_reward = 0
        
        for t in range(env.max_steps):
            time_step += 1
            
            # Select action
            action = agent.select_action(obs, memory)
            
            # Count how many drones are active (added states to memory)
            num_acting_drones = len([d for d in obs.drones if not d.delivered and d.battery > 0])
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Save reward and is_terminal for each drone's state
            for _ in range(num_acting_drones):
                memory.rewards.append(reward.total)
                memory.is_terminals.append(done)
            
            # Update policy if enough timesteps collected
            if time_step % update_timestep == 0:
                agent.update(memory)
                memory.clear()
            
            obs = next_obs
            ep_reward += reward.total
            
            if done:
                break
        
        # Log to TensorBoard
        writer.add_scalar("Env/Episode_Reward", ep_reward, ep)
        writer.add_scalar("Env/Collisions", env.state()["collisions"], ep)
        
        if ep % log_interval == 0:
            print(f"Episode {ep:04d} | Reward: {ep_reward:+.2f} | TimeStep: {time_step}")
            
        if ep % save_interval == 0:
            path = os.path.join(model_dir, f"ppo_chkpt_ep{ep}.pt")
            agent.save(path)
            print(f"Saved checkpoint: {path}")

    # Final save
    agent.save(os.path.join(model_dir, "ppo_final.pt"))
    print("PPO Training Complete.")

if __name__ == "__main__":
    train_ppo()
