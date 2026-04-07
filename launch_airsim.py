"""
AirSim 3D Drone Simulation Launcher
This script helps you connect to a running AirSim instance and run simulations.
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.airsim_env import AirSimDroneEnv
from environment.models import Action, DroneAction, HOVER
from visualizer.grid_vis_3d import GridAnimator3D
from collections import deque
from typing import List


def bfs_next_zone(current, destination, graph, blocked_zones=None):
    """Simple pathfinding for drone navigation."""
    blocked = set(blocked_zones or [])
    if current == destination:
        return current
    queue = deque([(current, [current])])
    visited = {current}
    while queue:
        zone, path = queue.popleft()
        for nb in graph.get(zone, []):
            if nb in visited or nb in blocked:
                continue
            new_path = path + [nb]
            if nb == destination:
                return new_path[1] if len(new_path) > 1 else current
            visited.add(nb)
            queue.append((nb, new_path))
    return current


def run_airsim_simulation(task="easy", seed=42, ip="127.0.0.1", port=41451, steps=None):
    """
    Connect to AirSim and run a simulation.
    
    Parameters:
    -----------
    task : str
        "easy", "medium", or "hard"
    seed : int
        Random seed for reproducibility
    ip : str
        AirSim server IP (default: localhost)
    port : int
        AirSim server port (default: 41451)
    steps : int or None
        Max steps (None = use default from task)
    """
    
    print(f"\n{'='*60}")
    print(f"🚁 AirSim 3D Drone Traffic Control Simulator")
    print(f"{'='*60}")
    print(f"Task: {task.upper()}")
    print(f"Connecting to AirSim at {ip}:{port}")
    print(f"{'='*60}\n")
    
    try:
        # Create environment
        env = AirSimDroneEnv(ip=ip, port=port, task=task)
        
        # Try to connect
        print("⏳ Connecting to AirSim...")
        env.connect()
        print("✅ Connected successfully!\n")
        
        # Reset and start
        print("🔄 Resetting simulation...")
        obs = env.reset()
        print(f"✅ Simulation ready!\n")
        
        print(f"📊 Environment Info:")
        print(f"  - Drones: {len(obs.drones)}")
        print(f"  - Max Steps: {env.max_steps}")
        print(f"  - Grid Size: {env.cfg['rows']}x{env.cfg['cols']}")
        print(f"  - Zones: {len(env.all_zones)}")
        print()
        
        # Set up visualization
        animator = GridAnimator3D(
            rows=env.cfg["rows"],
            cols=env.cfg["cols"],
            max_alt=env.cfg.get("max_altitude", 15.0),
            task_name=task,
        )
        animator.capture(obs)
        
        # Run simulation
        step = 0
        total_reward = 0
        collisions = 0
        delivered_count = 0
        
        print("🎬 Starting simulation...\n")
        
        while not obs.done and step < env.max_steps:
            # Simple greedy pathfinding
            claimed = []
            actions = []
            
            for drone in sorted(
                [d for d in obs.drones if not d.delivered],
                key=lambda d: (-d.priority, d.id),
            ):
                if drone.battery <= 0:
                    actions.append(DroneAction(drone_id=drone.id, move_to=HOVER))
                    continue
                
                # Find next zone via BFS
                next_zone = bfs_next_zone(
                    drone.location,
                    drone.destination,
                    obs.graph_edges,
                    claimed,
                )
                move = HOVER if next_zone == drone.location else next_zone
                claimed.append(next_zone)
                actions.append(DroneAction(drone_id=drone.id, move_to=move))
            
            # Step environment
            action = Action(actions=actions)
            obs, reward, done, info = env.step(action)
            
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
        print("\n" + "="*60)
        print("✅ SIMULATION COMPLETED")
        print("="*60)
        print(f"Total Steps: {step}")
        print(f"Total Reward: {total_reward:+.2f}")
        print(f"Drones Delivered: {delivered_count}/{len(obs.drones)}")
        print(f"Collision Count: {collisions}")
        print("="*60 + "\n")
        
        return True
        
    except ConnectionError as e:
        print(f"\n❌ CONNECTION ERROR: {e}")
        print("\n📋 Troubleshooting Steps:")
        print("  1. Download AirSim City environment from:")
        print("     https://github.com/microsoft/airsim/releases")
        print("  2. Extract the City environment binary")
        print("  3. Run the AirSimNH.exe or similar binary")
        print("  4. Make sure settings.json is in C:\\Users\\Hp\\Documents\\AirSim\\")
        print("  5. Come back and run this script again\n")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch AirSim 3D simulation")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], 
                       default="easy", help="Task difficulty")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ip", default="127.0.0.1", help="AirSim server IP")
    parser.add_argument("--port", type=int, default=41451, help="AirSim server port")
    
    args = parser.parse_args()
    
    success = run_airsim_simulation(
        task=args.task,
        seed=args.seed,
        ip=args.ip,
        port=args.port,
    )
    
    sys.exit(0 if success else 1)
