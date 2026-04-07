"""
DDQN Training Loop
=====================
Trains the DDQN agent on the DroneTrafficEnv.
"""

import sys
import os

# Silencing TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import configparser
import argparse
from typing import List

from environment.drone_env import DroneTrafficEnv
from environment.graders import grade_task, grade_episode_log
from rl_agent.dqn_agent import DDQNAgent


def read_config(path: str) -> dict:
    parser = configparser.ConfigParser()
    parser.read(path)
    return {section: dict(parser.items(section)) for section in parser.sections()}

def train(resume: bool = True):
    cfg = read_config("rl_agent/config.ini")
    gen_cfg = cfg['general']
    task_name = gen_cfg['task']
    
    env = DroneTrafficEnv(task=task_name, seed=int(gen_cfg['seed']))
    
    # Extract zones list for the agent
    zone_names = list(env.cfg["graph"].keys())
    
    agent = DDQNAgent(cfg, len(zone_names), zone_names)
    
    max_episodes = int(gen_cfg['max_episodes'])
    model_dir = cfg['logging']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    
    final_model_path = os.path.join(model_dir, f"ddqn_{task_name}.pt")
    
    if resume and os.path.exists(final_model_path):
        print(f"Resuming from existing model: {final_model_path}")
        agent.load(final_model_path)
    
    print(f"Starting Training: {max_episodes} episodes on task '{task_name}'")
    
    for ep in range(1, max_episodes + 1):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            action = agent.select_action(obs, training=True)
            
            next_obs, reward, done, info = env.step(action)
            
            # Add to PER memory
            agent.store_transition(obs, action, reward.total, next_obs, done)
            
            # Step training
            agent.train_step()
            
            obs = next_obs
            ep_reward += reward.total
            
        # Log episode stats
        agent.writer.add_scalar("Env/Episode_Reward", ep_reward, ep)
        agent.writer.add_scalar("Env/Collisions", env.state()["collisions"], ep)
        
        if ep % 10 == 0:
            print(f"Episode {ep:04d} | Reward: {ep_reward:+.2f} | EPS: {agent.epsilon:.3f} | Steps: {info['step']}")
            
        if ep % int(cfg['logging']['save_every_episodes']) == 0:
            path = os.path.join(model_dir, f"ddqn_chkpt_ep{ep}.pt")
            agent.save(path)
            
    # Final save
    agent.save(final_model_path)
    print(f"Training Complete. Model saved to {final_model_path}")


if __name__ == "__main__":
    train()
