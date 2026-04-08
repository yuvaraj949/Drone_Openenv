# Autonomous Drone Dispatcher 3D — OpenEnv

*Documentation & AirSim Gallery now integrated directly into the Gradio Dashboard.*


A production-ready reinforcement learning environment and multi-agent routing suite, modernized with **PEDRA**-inspired architectures.

## 🏗️ Project Layout

```text
drone_traffic_control/
├── airsim_bridge/        # Native AirSim 3D Mapping
├── environment/          # Core OpenEnv (reset/step)
├── pedra_legacy/         # ORIGINAL PEDRA REPOSITORY
├── rl_agent/             # Modern PyTorch DDQN + PER
├── visualizer/           # Terminal & GIF Animators
├── app.py                # Unified Gradio Dashboard
├── airsim_demo.gif       # LIVE DEMO
└── README.md             # Project Blueprint
```

---

## 🚀 Getting Started

**Looking for the full installation and training guide?** See [setup.md](setup.md) for detailed instructions.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Baseline Agent

```bash
python inference.py --task easy --rich
```

### 3. Launch Gradio Hub

```bash
python app.py
```

---

## 🧠 AirSim Integration & PEDRA Legacy

We have consolidated the original **PEDRA** repository and implemented a **Mock AirSim Bridge** to allow legacy algorithms to run on our modernized discrete grid.

### Features
- **Structural Consolidation**: PEDRA is now a first-class module in the tree.
- **Algorithm Selector**: Gradio UI now includes "PEDRA (Legacy TF1)" option.
- **Modernized Brain**: PyTorch 2.0 based DDQN in `rl_agent/`.

---

## 📜 License

MIT — free to use, modify, and build upon.
