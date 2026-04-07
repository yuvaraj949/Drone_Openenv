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

def train():
    cfg = read_config("rl_agent/config.ini")
    gen_cfg = cfg['general']
    
    env = DroneTrafficEnv(task=gen_cfg['task'], seed=int(gen_cfg['seed']))
    
    # Extract zones list for the agent
    zone_names = list(env.cfg["graph"].keys())
    
    agent = DDQNAgent(cfg, len(zone_names), zone_names, graph=env.graph, task_cfg=env.cfg)
    
    max_episodes = int(gen_cfg['max_episodes'])
    model_dir = cfg['logging']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Starting Training: {max_episodes} episodes on task '{gen_cfg['task']}'")
    
    for ep in range(1, max_episodes + 1):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            # Pass current step to agent
            step = info['step'] if 'info' in locals() else 0
            action = agent.select_action(obs, training=True, step=step)
            
            next_obs, reward, done, info = env.step(action)
            
            # Add to PER memory with step
            agent.store_transition(obs, action, reward.total, next_obs, done, step=step)
            
            # Step training
            agent.train_step()
            
            obs = next_obs
            ep_reward += reward.total
            
        # Log episode stats
        agent.writer.add_scalar("Env/Episode_Reward", ep_reward, ep)
        agent.writer.add_scalar("Env/Collisions", env.state()["collisions"], ep)
        
        if ep % 10 == 0:
            print(f"Episode {ep:04d} | Reward: {ep_reward:+.2f} | EPS: {agent.epsilon:.3f} | Steps: {info['step']}")
            
        # Periodic Evaluation
        if ep % 50 == 0:
            print(f"--- Evaluating at Episode {ep} ---")
            eval_obs = env.reset()
            eval_done = False
            eval_reward = 0
            eval_steps = 0
            while not eval_done:
                eval_action = agent.select_action(eval_obs, training=False, step=eval_steps)
                eval_obs, r, eval_done, eval_info = env.step(eval_action)
                eval_reward += r.total
                eval_steps += 1
            print(f"Eval Reward: {eval_reward:+.2f} | Collisions: {eval_info['cumulative_collisions']} | Delivered: {eval_info['delivered']}/3")
            agent.writer.add_scalar("Eval/Reward", eval_reward, ep)
            agent.writer.add_scalar("Eval/Delivered", eval_info['delivered'], ep)

        if ep % int(cfg['logging']['save_every_episodes']) == 0:
            path = os.path.join(model_dir, f"ddqn_chkpt_ep{ep}.pt")
            agent.save(path)
            
    # Final save
    agent.save(os.path.join(model_dir, "ddqn_final.pt"))
    print("Training Complete.")


if __name__ == "__main__":
    train()
