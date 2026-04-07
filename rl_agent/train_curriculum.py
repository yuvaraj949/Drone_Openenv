import os
import sys

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_agent.train_physics import train_physics

def run_curriculum():
    # Define Curriculum Stages
    # Stage 1: Easy (3 drones) - Warmstart/Baseline
    # Stage 2: Medium (5 drones + bottlenecks)
    # Stage 3: Hard (10 drones + dynamic obstacles)
    
    stages = [
        {"task": "easy",   "episodes": 200},
        {"task": "medium", "episodes": 500},
        {"task": "hard",   "episodes": 1000},
    ]
    
    model_path = "models/ppo/ppo_curriculum.pt"
    
    for i, stage in enumerate(stages):
        print("\n" + "="*50)
        print(f"CURRICULUM STAGE {i+1}/{len(stages)}: {stage['task'].upper()}")
        print("="*50)
        
        train_physics(
            task=stage['task'], 
            max_episodes=stage['episodes'], 
            model_path=model_path
        )
        
        print(f"Completed Stage: {stage['task']}")

if __name__ == "__main__":
    run_curriculum()
