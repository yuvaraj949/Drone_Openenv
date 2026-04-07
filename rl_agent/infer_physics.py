import sys
import os
import argparse
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.physics_env import PhysicsDroneEnv
from rl_agent.ppo_agent import PPOAgent
from visualizer.terminal_vis import TerminalRenderer
from visualizer.grid_vis import GridAnimator

def test_physics_agent(model_path: str, use_visualize: bool = False, gif_path: str = "rl_physics_episode.gif"):
    env = PhysicsDroneEnv(task="easy", seed=42)
    
    agent = PPOAgent(state_dim=36, action_dim=3)
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"[WARN] No checkpoint at {model_path}. Running with random initialized weights.")

    animator = GridAnimator(
        rows=env.cfg["rows"],
        cols=env.cfg["cols"],
        task_name=f"{env.task_name} (Physics)"
    ) if use_visualize else None
    
    print(f"\n--- Testing PPO Physics Agent ---")
    
    obs = env.reset()
    if animator: animator.capture(obs)
    
    done = False
    total_reward = 0.0
    
    while not done:
        # Get action from model for all active drones
        from environment.models import Action, DroneAction
        drone_actions = []
        
        for i, d in enumerate(obs.drones):
            if d.delivered or d.battery <= 0:
                drone_actions.append(DroneAction(drone_id=d.id, move_to="hover", thrust_vector=[0, 0, 0]))
                continue
                
            # Use policy_old or policy (they are identical after training)
            state = agent._extract_state(obs, i)
            state_t = torch.FloatTensor(state).to(agent.device).unsqueeze(0)
            
            with torch.no_grad():
                dist, _ = agent.policy(state_t)
                action_thrust = dist.mean # Use mean for testing (no noise)
                
            thrust_list = (action_thrust.cpu().numpy()[0] * 10.0).tolist()
            drone_actions.append(DroneAction(drone_id=d.id, move_to="hover", thrust_vector=thrust_list))
            
        env_action = Action(actions=drone_actions)
        
        # Step env
        obs, reward, done, info = env.step(env_action)
        total_reward += reward.total
        
        if animator:
            animator.capture(obs)
            
    print("\n[Testing Complete]")
    print(f"Total Reward: {total_reward:+.2f}")
    print(f"Delivered: {sum(1 for d in obs.drones if d.delivered)} / {len(obs.drones)}")
    
    if animator:
        saved = animator.save(gif_path)
        print(f"[VIS] Animation saved to {saved}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained PPO Physics Agent")
    parser.add_argument("--model", type=str, default="models/ppo/ppo_final.pt", help="Path to checkpoint")
    parser.add_argument("--visualize", action="store_true", help="Generate animated GIF of the episode")
    parser.add_argument("--gif-path", type=str, default="rl_physics_test.gif", help="Path to save the GIF")
    args = parser.parse_args()
    
    test_physics_agent(args.model, args.visualize, args.gif_path)
