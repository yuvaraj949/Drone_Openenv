import sys
import os
import torch
import numpy as np
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.physics_env import PhysicsDroneEnv
from rl_agent.ppo_agent import PPOAgent

def train_physics(task="easy", max_episodes=2000, update_interval=1500, model_path="models/ppo/ppo_final.pt"):
    env = PhysicsDroneEnv(task=task)
    # State dim 36: Self(7) + Wind(3) + Neighbors(3x6=18) + Obstacles(2x4=8)
    agent = PPOAgent(state_dim=36, action_dim=3)
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded existing model from {model_path}")
    
    print(f"Starting PPO Physics Training: {max_episodes} episodes on task '{task}'")
    
    memory_states = []
    memory_actions = []
    memory_logprobs = []
    memory_rewards = []
    memory_is_terminals = []
    
    timestep = 0
    
    for ep in range(1, max_episodes + 1):
        obs = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            timestep += 1
            
            # 1. Collective Action for all drones
            actions_list = []
            
            # We use a shared policy for all drones
            # In each step, we collect a transition for every active drone
            for i, d in enumerate(obs.drones):
                if d.delivered or d.battery <= 0:
                    continue
                
                state = agent._extract_state(obs, i)
                state_t = torch.FloatTensor(state).to(agent.device).unsqueeze(0)
                
                with torch.no_grad():
                    dist, _ = agent.policy_old(state_t)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                
                memory_states.append(state_t.squeeze(0))
                memory_actions.append(action.squeeze(0))
                memory_logprobs.append(log_prob.squeeze(0))
                
                # Scale action to thrust (N)
                thrust_list = (action.cpu().numpy()[0] * 10.0).tolist()
                actions_list.append((d.id, thrust_list))
            
            # Step environment
            from environment.models import Action, DroneAction
            env_action = Action(actions=[
                DroneAction(drone_id=aid, move_to="hover", thrust_vector=atv) 
                for aid, atv in actions_list
            ])
            
            next_obs, reward, done, info = env.step(env_action)
            
            # Shared reward for all drones in this step for simplicity in this demo
            # In a more advanced setting, each drone would have its own reward
            for i in range(len(actions_list)):
                memory_rewards.append(reward.total) # Step reward
                memory_is_terminals.append(done)
            
            obs = next_obs
            ep_reward += reward.total
            
            # Update PPO
            if timestep % update_interval == 0 and len(memory_states) > 0:
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
                print(f"--- PPO Update at Timestep {timestep} ---")

        if ep % 20 == 0:
            print(f"Episode {ep:04d} | Reward: {ep_reward:+.2f} | Collisions: {info.get('collisions', 0)}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save(model_path)
    print(f"Training Complete. Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="easy")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--model", type=str, default="models/ppo/ppo_final.pt")
    args = parser.parse_args()
    
    train_physics(task=args.task, max_episodes=args.episodes, model_path=args.model)
