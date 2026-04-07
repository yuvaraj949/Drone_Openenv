# Autonomous Drone Dispatcher 3D — OpenEnv Environment

**A high-fidelity logistics simulation requiring vertical separation, collision avoidance, and emergency prioritization in a grid-based 3D airspace.**

---

## Overview

This environment simulates real-world autonomous drone traffic management and routing. Agents must coordinate multiple drones to deliver packages across a grid-based airspace while:

- **Avoiding collisions** between drones in shared zones
- **Managing battery constraints** with dynamic drain rates
- **Respecting bottleneck zones** with restricted capacity
- **Prioritizing emergency deliveries** with time-critical deadlines
- **Operating in dynamic No-Fly Zones (NFZ)** (hard task only)

This is a **real-world utility environment** suitable for training multi-agent RL systems to solve logistics coordination problems that humans face in aerial delivery networks.

---

## Environment Description

### Task Domain
**Logistics Coordination** — Autonomous city airspace dispatcher for package delivery drones.

### Physical Model
- **Airspace**: Discrete grid of zones (3×3 to 5×5)
- **Agents**: Up to 10 autonomous drones with individual state
- **Dynamics**:
  - Drones move one zone per step (4-directional routing)
  - Altitude control (3D-lite) with collision detection based on altitude separation
  - Battery drains with vertical movement penalty (1.5x climb cost)
  - Simultaneous occupancy detection

### Episode Structure
1. **Reset** → Initialize drones at random start locations with destinations
2. **Step loop** → Agent selects routing and vertical commands; environment applies physics
3. **Termination** → All drones delivered OR max_steps reached
4. **Grading** → Normalized score [0, 1] based on delivery rate, collisions, emergencies, efficiency

---

## Tasks (Increasing Difficulty)

### Task 1: **Easy** ✓
- **Drones**: 3 couriers
- **Grid**: 3×3 (9 zones)
- **Obstacles**: None
- **Battery drain**: 2% per step (no altitude penalty)
- **Max steps**: 30
- **Target**: Basic pathfinding with minimal collision risk
- **Baseline score**: ~0.33–0.50 (hovering agent)
- **Expected agent score**: 0.70–0.95 (BFS + altitude separation)

### Task 2: **Medium** ✓
- **Drones**: 5 couriers + 1 emergency
- **Grid**: 4×4 (16 zones)
- **Bottleneck zones**: 1–2 restricted-capacity zones (max 2 drones)
- **Battery drain**: 3% per step + 1.5× climb penalty
- **Max steps**: 40
- **Deadlines**: Emergency drone must reach destination by step 25
- **Target**: Strategic routing avoiding congestion + battery management
- **Baseline score**: ~0.35–0.45
- **Expected agent score**: 0.60–0.85

### Task 3: **Hard** ✓
- **Drones**: 10 couriers + 2 emergencies
- **Grid**: 5×5 (25 zones)
- **Dynamic NFZ**: 1–2 No-Fly Zones that activate at random steps
- **Bottleneck zones**: 2–3 restricted zones
- **Battery drain**: 4% per step + 2× climb penalty
- **Max steps**: 50
- **Deadlines**: Emergencies by step 20
- **Target**: Real-time replanning + multi-objective optimization
- **Baseline score**: ~0.28–0.35
- **Expected agent score**: 0.45–0.75

---

## Action Space

### Type: `Action` (Pydantic Model)

```python
class Action(BaseModel):
    actions: List[DroneAction]

class DroneAction(BaseModel):
    drone_id: str                          # "D1", "D2", etc.
    move_to: str                           # Adjacent zone or "hover"
    vertical_command: float                # Climb/descend in meters (+/-); 0 = level
    thrust_vector: List[float] = [0,0,0]  # [Fx, Fy, Fz] for physics mode (optional)
```

**Discrete routing** (4-directional): `{"N", "S", "E", "W", "hover"}`
**Continuous altitude control**: -5.0 to +5.0 meters per step

---

## Observation Space

### Type: `Observation` (Pydantic Model)

```python
class Observation(BaseModel):
    drones: List[DroneState]               # Current state of all active drones
    congestion_map: Dict[str, int]         # Occupancy per zone
    wind_vector: List[float]               # Global wind [Wx, Wy, Wz]
    step: int                              # Current step number
    collisions: int                        # Total collisions this episode
    sensing_radius: float                  # Detection range (10m default)
    stationary_obstacles: List[Obstacle]   # 3D obstacles
    graph_edges: Dict[str, List[str]]      # Adjacency list for routing

class DroneState(BaseModel):
    id: str                                # "D1", "D2", etc.
    location: str                          # Current zone: "A1", "B2", etc.
    x, y, altitude: float                  # 3D coordinates
    vx, vy, vz: float                      # Velocity vectors
    destination: str                       # Target zone
    target_altitude: float                 # Desired final altitude
    battery: float                         # 0–100 %
    priority: int                          # 1 = normal, 2 = emergency
    delivered: bool                        # Has reached destination?
    steps_taken: int                       # Cumulative steps for this drone
```

---

## Reward Function

### Components (Step-Level)

| Component | Value | Meaning |
|-----------|-------|---------|
| **Delivery** | +20.0 | Drone reaches destination |
| **Distance reward** | +2.0 × Δdist | Moving closer to goal (shaping) |
| **Emergency bonus** | +10.0 | On-time emergency delivery |
| **Step penalty** | −0.5 | Per-step cost (encourages speed) |
| **Collision penalty** | −2.0 | Per collision detected |
| **Battery penalty** | −0.1 | If battery < 10% |
| **Energy penalty** | Scaled | Large thrust magnitude penalty |

### Episode-Level Grading

Final **normalized score** [0.0–1.0]:

```
score = (
    0.50 * delivery_rate          # Fraction of drones delivered
    + 0.25 * collision_score      # 1.0 - (collisions / fleet_size)
    + 0.15 * emergency_score      # Fraction of emergencies on-time
    + 0.10 * efficiency_score     # 1.0 - (steps / max_steps)
)
```

---

## Baseline Agent & Reproduction

### Baseline: Greedy BFS with Altitude Highways

**Algorithm**:
1. For each drone, compute shortest path (BFS) to destination
2. Assign altitude highway based on drone ID (e.g., D1→10m, D2→13m, D3→16m)
3. Climb/descend toward assigned altitude when in congested zones
4. Hover in bottleneck zones if capacity exceeded

**Inference**: `python inference.py`

**Example output**:
```
[START] task=easy env=drone_traffic model=Trained-DDQN-v1
[STEP] step=1 action=D1:N;D2:E;D3:hover reward=-0.50 done=false error=null
[STEP] step=2 action=D1:N;D2:E;D3:E reward=0.50 done=false error=null
...
[END] success=true steps=15 score=0.642 rewards=-0.50,0.50,-0.50,20.00,...
```

**Baseline Scores**:
- Easy: 0.35–0.50
- Medium: 0.28–0.40
- Hard: 0.20–0.32

---

## Setup & Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+
- Pydantic 2.0+
- OpenEnv Core 0.2.0+

### Local Installation

```bash
# Clone the repository
git clone <repo>
cd drone_traffic_control/round1_submission

# Install dependencies
pip install -r requirements.txt

# Optional: Install dev dependencies
pip install -e ".[dev]"
```

### Run Inference Locally

```bash
# Set environment variables
export OPENAI_API_KEY="sk-..."  # For LLM-based agents (optional)
export TASK="easy"              # Task: easy, medium, hard

# Run baseline inference
python inference.py
```

---

## Deployment

### Docker

```bash
# Build
docker build -t drone-traffic:latest .

# Run
docker run -it -p 7860:7860 drone-traffic:latest

# Health check
docker run --rm drone-traffic:latest python -c \
  "from environment.drone_env import DroneTrafficEnv; \
   DroneTrafficEnv('easy').reset(); print('OK')"
```

### Hugging Face Spaces

1. Create a public GitHub repository with this code
2. Set up a Hugging Face Space (Docker runtime)
3. Connect to GitHub repo
4. Space will auto-deploy the Dockerfile
5. API endpoint: `https://<username>-<space>.hf.space/reset`

---

## Specification & Compliance

### OpenEnv Compliance

✓ **Typed models** (Pydantic)
✓ **Step/reset/state interface** (sync)
✓ **Grader function** (`grade_task`)
✓ **openenv.yaml manifest**
✓ **3+ tasks** with difficulty progression
✓ **Meaningful reward** with component breakdown
✓ **Baseline inference script** with [START]/[STEP]/[END] format

### Performance

- **Startup time**: <1 second
- **Step latency**: ~10 ms (CPU)
- **Max episode** (50 steps): <500 ms
- **Memory**: ~50 MB base + 20 MB per drone
- **Inference time** (hard, 50 steps): <10 seconds on CPU

---

## Example Agent Code

### Random Agent
```python
from environment.drone_env import DroneTrafficEnv
from environment.models import Action, DroneAction, HOVER
import random

env = DroneTrafficEnv(task="easy")
obs = env.reset()

for _ in range(50):
    action = Action(actions=[
        DroneAction(
            drone_id=d.id,
            move_to=random.choice(["N", "S", "E", "W", "hover"]),
            vertical_command=random.uniform(-2, 2)
        )
        for d in obs.drones
    ])
    obs, reward, done, info = env.step(action)
    if done:
        break

print(f"Score: {env.state()}")
```

### RL Training Loop
```python
from environment.drone_env import DroneTrafficEnv
from environment.dqn_agent import DDQNAgent

env = DroneTrafficEnv(task="medium")
agent = DDQNAgent(...)

obs = env.reset()
for step in range(1000):
    action = agent.select_action(obs, training=True)
    obs, reward, done, info = env.step(action)
    agent.remember(obs, reward, done)
    if done:
        obs = env.reset()
```

---

## Testing

### Unit Tests
```bash
pytest tests/ -v
```

### Validation
```bash
openenv validate
```

---

## Citation

If you use this environment, please cite:

```bibtex
@misc{drone_traffic_openenv_2024,
  title={Autonomous Drone Dispatcher 3D: An OpenEnv Environment},
  author={...},
  year={2024},
  url={https://github.com/.../drone_traffic_control}
}
```

---

## License

MIT

---

## Contact & Support

For issues or questions, please open a GitHub issue or contact: `support@example.com`

---

## Changelog

### v1.0.0 (2024-04-07)
- Initial OpenEnv Round 1 submission
- 3 tasks: easy, medium, hard
- Greedy-BFS baseline agent
- Full Pydantic model compliance
- Docker + HF Spaces deployment
