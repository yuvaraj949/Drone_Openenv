# 🚁 Setup & Training Guide

This guide provides step-by-step instructions for setting up the **Drone Traffic Control** environment and training the Reinforcement Learning agents.

## 📦 1. Installation

### Prerequisites

- Python 3.8+ (3.10 recommended)
- `pip` (latest version)
- (Optional) CUDA-capable GPU for faster RL training.

### Environment Setup

1. **Clone the repository** (if you haven't already):

   ```bash
   git clone <repository_url>
   cd drone_traffic_control
   ```

2. **Create a virtual environment**:

   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   pip install torch tensorboard rich matplotlib Pillow airsim
   ```

---

## 🎮 2. AirSim Simulation Setup

To run the project in high-fidelity 3D mode, you need the AirSim binary and the correct configuration.

### A. Download the Environment

1. Go to the [AirSim Releases](https://github.com/microsoft/airsim/releases) page.
2. Download the **City environment** binaries:
   - `City_environ.zip.001`
   - `City_environ.zip.002`
3. Combine and extract them into a folder.

### B. Configure `settings.json`

AirSim looks for a configuration file in your Documents folder (`C:\Users\<User>\Documents\AirSim\settings.json` on Windows).

Copy and paste the following configuration to enable 10 drones:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1.0,
  "Vehicles": {
    "Drone1": { "VehicleType": "SimpleFlight", "X": 0, "Y": 0, "Z": -2 },
    "Drone2": { "VehicleType": "SimpleFlight", "X": 5, "Y": 0, "Z": -2 },
    "Drone3": { "VehicleType": "SimpleFlight", "X": 10, "Y": 0, "Z": -2 },
    "Drone4": { "VehicleType": "SimpleFlight", "X": 15, "Y": 0, "Z": -2 },
    "Drone5": { "VehicleType": "SimpleFlight", "X": 20, "Y": 0, "Z": -2 },
    "Drone6": { "VehicleType": "SimpleFlight", "X": 25, "Y": 0, "Z": -2 },
    "Drone7": { "VehicleType": "SimpleFlight", "X": 30, "Y": 0, "Z": -2 },
    "Drone8": { "VehicleType": "SimpleFlight", "X": 35, "Y": 0, "Z": -2 },
    "Drone9": { "VehicleType": "SimpleFlight", "X": 40, "Y": 0, "Z": -2 },
    "Drone10": { "VehicleType": "SimpleFlight", "X": 45, "Y": 0, "Z": -2 }
  }
}
```

---

## 🧠 3. Training the Agents

### A. Training the Grid Agent (DDQN)

The DDQN agent is designed for discrete grid navigation. It is the best choice for "Easy" and "Medium" tasks.

1. **Configure parameters** (Optional):
   Edit `rl_agent/config.ini` to adjust `max_episodes`, `learning_rate`, or the `task` type.

2. **Run the training script**:

   ```bash
   python rl_agent/train.py
   ```

3. **Monitor progress**:
   Open a new terminal and run TensorBoard to see rewards and collision rates:

   ```bash
   tensorboard --logdir runs/ddqn
   ```

### B. Training the Physics Agent (PPO)

The PPO agent is designed for continuous 3D control (Physics Mode) and handles stochastic wind.

1. **Ensure Physics requirements are met**:
   You need `pydantic` and `torch` installed.

2. **Run training** (Implementation check):

   *Note: Currently, the PPO training loop is intended for researchers to extend. You can trigger it by modifying `rl_agent/ppo_agent.py` or running your custom PPO loop against `environment/physics_env.py`.*

---

## 🚀 4. Running Inference

### Baseline (Greedy BFS)

To run the deterministic baseline that uses shortest-path logic:

   ```bash
   python inference.py --task easy --rich --visualize
   ```

### RL Agent (Trained Model)

To test your trained DDQN model:

   ```bash
   python rl_agent/infer.py --model models/ddqn/ddqn_final.pt
   ```

### Gradio Dashboard (Interactive)

The easiest way to explore all features (3D, Physics, AirSim):

   ```bash
   python app.py
   ```

   Open `http://localhost:7860` in your browser.

---

## 💡 Troubleshooting

- **"Drones only go up":** Fixed in the latest update. Drones now maintain an altitude "highway" (e.g., Drone 1 at 2.5m, Drone 2 at 5.0m) to avoid vertical collisions.
- **"Agent is dumb":** If the agent performs poorly, it likely needs more training episodes. Increase `max_episodes` in `config.ini` and ensure `epsilon_decay` allows for enough exploration.
- **"AirSim Connection Failed":** Ensure the `City_environ` app is running and your `settings.json` matches the vehicle names in your code.
