# 🚁 Autonomous Drone Traffic Control — OpenEnv Blueprint

A production-ready reinforcement learning environment and multi-agent routing suite, modernized with **PEDRA**-inspired architectures.

## 🏗️ Project Layout

```text
drone_traffic_control/
├── airsim_bridge/        # Native AirSim 3D Mapping
├── environment/          # Core OpenEnv (reset/step)
│   ├── pedra_bridge.py   # NEW: Mock AirSim Client for legacy support
│   ├── drone_env.py      # Main Environment
│   └── ... 
├── pedra_legacy/         # ORIGINAL PEDRA REPOSITORY (TF1/Legacy Archive)
├── rl_agent/             # Modern PyTorch DDQN + PER
├── visualizer/           # Terminal & GIF Animators
├── app.py                # Unified Gradio Dashboard
├── Dockerfile            # Container for HF Spaces
└── README.md             # Project Blueprint
```

---

## 🚀 Getting Started

**Looking for the full installation and training guide?** See [setup.md](setup.md) for detailed instructions on virtual environments, training DQN/PPO agents, and benchmarking results.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Baseline Agent

```bash
# Easy task (default)
python inference.py

# Medium task with fixed seed
python inference.py --task medium --seed 42

# Rich live terminal rendering
python inference.py --task easy --rich

# Save animated GIF of the episode
python inference.py --task hard --seed 7 --visualize
```

### 3. Launch Gradio Hub

```bash
python app.py
# Opens at http://localhost:7860
```

---

## 🗺️ Airspace Model

The airspace is represented as a **grid graph** of discrete zones:

```text
3×3 (easy):        4×4 (medium):          5×5 (hard):
A1 — A2 — A3      A1 — A2 — A3 — A4      A1 … A5
 |    |    |       |    |    |    |        …
B1 — B2 — B3      B1 — B2*— B3 — B4      E1 … E5
 |    |    |       |    |    |    |
C1 — C2 — C3      C1 — C2 — C3*— C4
```

`*` = bottleneck zone (capacity: 1 drone)

---

## 📊 Tasks & Scoring

| Task   | Grid | Drones | Emergencies | Max Steps |
|--------|------|--------|-------------|-----------|
| easy   | 3×3  | 3      | 1           | 30        |
| medium | 4×4  | 5      | 2           | 40        |
| hard   | 5×5  | 10     | 3           | 50        |

**Scoring:** `0.50 × delivery + 0.25 × collision_avoidance + 0.15 × emergency_speed + 0.10 × efficiency`

---

## 🧠 Stage 4B — Full PEDRA Legacy Integration

We have consolidated the original **PEDRA** repository into this project under `pedra_legacy/`. To allow the original TF1 algorithms to run against our modern discrete grid, we implemented a **Mock AirSim Bridge**.

### The Bridge (`environment/pedra_bridge.py`)

This module provides a `PedraAirSimMock` class that mimics `airsim.MultirotorClient`. It intercepts 3D commands like `moveToPositionAsync(x, y, z)` and translates them into our grid movements.

### Key Features

1.  **Structural Consolidation**: PEDRA is now a first-class module in the `drone_traffic_control` tree.
2.  **Algorithm Selector**: The Gradio UI (`app.py`) now includes a **"PEDRA (Legacy TF1)"** option that triggers the bridge.
3.  **Modernized Brain**: We recommend the `rl_agent/` module (PyTorch 2.0) for new development.

### Training the RL Agent

1.  **Install RL dependencies**

```bash
pip install torch tensorboard
```

2.  **Train the DDQN Agent**

```bash
python rl_agent/train.py
```

3.  **View Statistics**

```bash
tensorboard --logdir runs/ddqn
```

---

## 📈 Baseline Scores (Greedy BFS)

| Task | Score | Collisions | Delivered |
|------|-------|------------|-----------|
| Easy (seed=42) | **0.9933** | 0 | 3/3 |
| Medium (seed=42) | **0.9875** | 0 | 5/5 |
| Hard (seed=0) | **0.9860** | 0 | 10/10 |

---

## 📜 License

MIT — free to use, modify, and build upon for research and hackathons.
