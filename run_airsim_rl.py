"""
AirSim RL Agent Runner - Use Trained Models in Unreal Engine
This script loads your trained DDQN models and runs them in AirSim.
"""

import sys
import os
import argparse

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.airsim_env import AirSimDroneEnv
from environment.models import Action, DroneAction, HOVER
from visualizer.grid_vis_3d import GridAnimator3D
from rl_agent.dqn_agent import DDQNAgent
import configparser
import torch


def read_config(path: str) -> dict:
    parser = configparser.ConfigParser()
    parser.read(path)
    return {section: dict(parser.items(section)) for section in parser.sections()}


def load_trained_agent(model_path: str, task: str = "easy"):
    """Load a trained DDQN agent for the specified task."""
    print(f"Loading trained model: {model_path}")

    # Get environment config to determine action space
    from environment.drone_env import DroneTrafficEnv
    env = DroneTrafficEnv(task=task)
    zone_names = list(env.cfg["graph"].keys())

    # Load agent config
    cfg = read_config("rl_agent/config.ini")

    # Create agent with same architecture
    agent = DDQNAgent(cfg, len(zone_names), zone_names)

    # Load trained weights
    agent.load(model_path)

    print(f"✅ Model loaded successfully for {len(zone_names)} zones")
    return agent, env


def run_airsim_with_rl_agent(
    task: str = "easy",
    model_path: str = "models/ddqn/ddqn_final.pt",
    ip: str = "127.0.0.1",
    port: int = 41451,
    steps: int = None,
    seed: int = 42
):
    """
    Run trained RL agent in AirSim environment.

    Parameters:
    -----------
    task : str
        "easy", "medium", or "hard"
    model_path : str
        Path to trained model checkpoint
    ip : str
        AirSim server IP
    port : int
        AirSim server port
    steps : int or None
        Max steps (None = use default from task)
    seed : int
        Random seed for reproducibility
    """

    print(f"\n{'='*70}")
    print(f"🚁 AirSim RL Agent Runner - Trained DDQN in Unreal Engine")
    print(f"{'='*70}")
    print(f"Task: {task.upper()}")
    print(f"Model: {model_path}")
    print(f"Connecting to AirSim at {ip}:{port}")
    print(f"{'='*70}\n")

    try:
        # Load trained agent
        print("🤖 Loading trained RL agent...")
        agent, env_config = load_trained_agent(model_path, task)
        print("✅ Agent loaded and ready!\n")

        # Create AirSim environment
        print("🎮 Connecting to AirSim...")
        airsim_env = AirSimDroneEnv(ip=ip, port=port, task=task)
        airsim_env.connect()
        print("✅ Connected to AirSim!\n")

        # Reset and start
        print("🔄 Resetting simulation...")
        obs = airsim_env.reset()
        print(f"✅ Simulation ready!\n")

        print(f"📊 Environment Info:")
        print(f"  - Drones: {len(obs.drones)}")
        print(f"  - Max Steps: {airsim_env.max_steps}")
        print(f"  - Grid Size: {airsim_env.cfg['rows']}x{airsim_env.cfg['cols']}")
        print(f"  - Zones: {len(airsim_env.all_zones)}")
        print()

        # Set up visualization
        animator = GridAnimator3D(
            rows=airsim_env.cfg["rows"],
            cols=airsim_env.cfg["cols"],
            max_alt=airsim_env.cfg.get("max_altitude", 15.0),
            task_name=task,
        )
        animator.capture(obs)

        # Run simulation with trained agent
        step = 0
        total_reward = 0
        collisions = 0
        delivered_count = 0

        print("🎬 Starting RL agent simulation...\n")

        while not obs.done and step < airsim_env.max_steps:
            # Get action from trained RL agent
            action = agent.select_action(obs, training=False)

            # Step environment
            obs, reward, done, info = airsim_env.step(action)

            step += 1
            total_reward += reward.total
            collisions = info.get("cumulative_collisions", 0)
            delivered_count = sum(1 for d in obs.drones if d.delivered)
            animator.capture(obs)

            # Print status every 5 steps
            if step % 5 == 0:
                print(f"Step {step:3d} | Reward: {reward.total:+7.2f} | "
                      f"Delivered: {delivered_count}/{len(obs.drones)} | "
                      f"Collisions: {collisions}")

        # Summary
        print("\n" + "="*70)
        print("✅ SIMULATION COMPLETED")
        print("="*70)
        print(f"Total Steps: {step}")
        print(f"Total Reward: {total_reward:+.2f}")
        print(f"Drones Delivered: {delivered_count}/{len(obs.drones)}")
        print(f"Collision Count: {collisions}")
        print("="*70 + "\n")

        return True

    except ConnectionError as e:
        print(f"\n❌ CONNECTION ERROR: {e}")
        print("\n📋 Troubleshooting Steps:")
        print("  1. Make sure AirSim binary is running in a separate window")
        print("  2. Check that it's listening on port 41451:")
        print("     netstat -ano | findstr :41451")
        print("  3. Verify settings.json exists at C:\\Users\\Hp\\Documents\\AirSim\\settings.json")
        print("  4. Restart AirSim binary completely\n")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trained RL agent in AirSim")
    parser.add_argument("--task", choices=["easy", "medium", "hard"],
                       default="easy", help="Task difficulty")
    parser.add_argument("--model", default="models/ddqn/ddqn_final.pt",
                       help="Path to trained model checkpoint")
    parser.add_argument("--ip", default="127.0.0.1", help="AirSim server IP")
    parser.add_argument("--port", type=int, default=41451, help="AirSim server port")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    success = run_airsim_with_rl_agent(
        task=args.task,
        model_path=args.model,
        ip=args.ip,
        port=args.port,
        seed=args.seed,
    )

    sys.exit(0 if success else 1)
