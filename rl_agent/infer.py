"""
DDQN Inference / Testing
========================
Loads a trained checkpoint and runs the agent in the environment.
Can use the terminal visualizer to watch it live!
"""

import sys
import os

# Silencing TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Ensure the root project directory is on the path regardless of how this is run.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import configparser
import time

from environment.drone_env import DroneTrafficEnv
from rl_agent.dqn_agent import DDQNAgent
from visualizer.terminal_vis import TerminalRenderer


def read_config(path: str) -> dict:
    parser = configparser.ConfigParser()
    parser.read(path)
    return {section: dict(parser.items(section)) for section in parser.sections()}

def test_agent(model_path: str, use_rich: bool = True):
    cfg = read_config("rl_agent/config.ini")
    gen_cfg = cfg['general']
    
    # We will test on easy task for validation
    task = gen_cfg['task']
    env = DroneTrafficEnv(task=task, seed=int(gen_cfg['seed']))
    zone_names = list(env.cfg["graph"].keys())
    
    # Init agent
    agent = DDQNAgent(cfg, len(zone_names), zone_names)
    
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"[WARN] No checkpoint at {model_path}. Running with random initialized weights.")

    # Init visualizer
    renderer = TerminalRenderer(
        rows=env.cfg["rows"], 
        cols=env.cfg["cols"], 
        task_name=env.task_name
    ) if use_rich else None
    
    print(f"\n--- Testing DDQN Agent on task: {task} ---")
    
    obs = env.reset()
    if renderer: renderer.render(obs, reward=None, info={})
    
    done = False
    total_reward = 0.0
    
    while not done:
        # epsilon=0 for pure exploitation
        agent.epsilon = 0.0
        
        # Get action from model
        action = agent.select_action(obs, training=False)
        
        # Step env
        obs, reward, done, info = env.step(action)
        total_reward += reward.total
        
        if renderer:
            time.sleep(0.5) # slow down for watching
            renderer.render(obs, reward=reward, info=info)
            
    print("\n[Testing Complete]")
    print(f"Total Reward: {total_reward:+.2f}")
    print(f"Collisions: {info['cumulative_collisions']}")
    print(f"Delivered: {info['delivered']} / {len(obs.drones)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained DDQN Agent")
    parser.add_argument("--model", type=str, default="models/ddqn/ddqn_final.pt", help="Path to checkpoint")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich terminal rendering")
    args = parser.parse_args()
    
    test_agent(args.model, not args.no_rich)
