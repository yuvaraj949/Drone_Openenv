---
title: Drone Airspace 3D
emoji: 🛸
colorFrom: indigo
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# Autonomous Drone Dispatcher 3D — OpenEnv

*Interactive system documentation and mission gallery are now available directly within the Graduate dashboard.*


A high-fidelity logistics simulation environment for autonomous drone coordination and traffic management in 3D airspace.


## Overview

This environment simulates a real-world drone traffic control scenario where multiple autonomous delivery drones must navigate through a shared 3D grid-based airspace to deliver packages while:
- **Avoiding collisions** with other drones at different altitudes
- **Managing battery** constraints with realistic drain rates
- **Respecting traffic rules** (bottleneck zones, no-fly zones on hard task)
- **Prioritizing emergencies** (medical deliveries on time)

### Real-World Application

This environment models actual autonomous delivery logistics systems being deployed by companies like Amazon Prime Air, Zipline, and Wing Aviation. The key challenges—collision avoidance, battery management, priority scheduling—are fundamental to commercial drone operations.

## Tasks

The environment provides 3 tasks with increasing complexity:

### Easy: Basic Coordination
- **Grid**: 3×3 zones, 3 drones
- **Challenge**: Navigate 3 drones from start to destination without collisions
- **Typical Score**: 0.4–0.8

### Medium: Resource Constraints
- **Grid**: 4×4 zones, 5 drones, 2 emergencies
- **Challenge**: Battery drains; bottleneck zones limit simultaneous occupancy
- **Typical Score**: 0.2–0.6

### Hard: Dynamic Obstacles + Priorities
- **Grid**: 5×5 zones, 10 drones, 3 emergencies
- **Challenge**: Dynamic no-fly zones; emergencies must be delivered within 25 steps
- **Typical Score**: 0.0–0.4

## Action Space

Each step, send an `Action` with `DroneAction` commands:

```python
class DroneAction(BaseModel):
    drone_id: str                  # e.g. "D1"
    move_to: str                   # Adjacent zone or "hover"
    vertical_command: float        # Climb/descend (meters)
```

## Observation Space

Receive an `Observation` after each step:

```python
class Observation(BaseModel):
    drones: List[DroneState]       # All drone states
    obstacles: List[Obstacle]      # Static/dynamic obstacles
    step: int                      # Current step
    grid_size: Tuple[int, int]    # Grid dimensions
```

## Reward Structure

- **Base**: -0.5 per step (efficiency incentive)
- **Collision**: -1.0 per collision
- **Battery**: -0.5 if battery < 20%
- **Delivery**: +5.0 upon destination
- **Emergency**: +2.0 for on-time delivery

## Grading Formula

| Metric | Weight |
|--------|--------|
| Delivery Rate | 50% |
| Collision Score | 25% |
| Emergency Score | 15% |
| Efficiency | 10% |

Final score: normalized reward in [0, 1]

## Installation

### Local
```bash
pip install -r requirements.txt
python inference.py
```

### Docker
```bash
docker build -t drone-traffic-control .
docker run -p 7860:7860 drone-traffic-control
```

## Usage

### Baseline Run
```bash
export TASK=easy
python inference.py
```

Output:
```
[START] task=easy env=drone_traffic model=Trained-DDQN-v1
[STEP] step=1 action=D1:B2;D2:A3;D3:hover reward=-0.50 done=false error=null
[END] success=true steps=18 score=0.680 rewards=-0.50,1.50,...
```

### Programmatic
```python
from environment.drone_env import DroneTrafficEnv
from environment.models import Action, DroneAction

env = DroneTrafficEnv(task="easy")
obs = env.reset()

for step in range(30):
    actions = [DroneAction(drone_id=d.id, move_to="hover") for d in obs.drones]
    obs, reward, done, info = env.step(Action(actions=actions))
    if done:
        break
```

### OpenEnv Server
```bash
curl -X POST http://localhost:7860/reset?task=easy
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" \
  -d '{"actions": [{"drone_id": "D1", "move_to": "B2", "vertical_command": 0.0}]}'
curl -X POST http://localhost:7860/state
```

## Baseline Performance

| Task | Score | Delivered | Collisions | Steps |
|------|-------|-----------|-----------|-------|
| easy | 0.68 | 2/3 | 0 | 18 |
| medium | 0.35 | 2/5 | 1 | 28 |
| hard | 0.12 | 1/10 | 3 | 45 |

(DDQN baseline with greedy-BFS routing)

## Project Structure

```
round1_submission/
├── environment/
│   ├── drone_env.py       # Main environment
│   ├── models.py          # Pydantic models
│   ├── tasks.py           # Task configs
│   ├── graders.py         # Grading logic
│   ├── dqn_agent.py       # DDQN agent
│   └── per_memory.py      # Replay buffer
├── server/
│   ├── app.py             # FastAPI server
│   ├── __init__.py
│   └── __main__.py
├── models/
│   └── ddqn_final.pt      # Trained weights
├── inference.py           # Baseline script
├── openenv.yaml           # Spec file
├── requirements.txt       # Dependencies
└── README.md
```

## Dependencies

- `openenv-core>=0.2.0`
- `pydantic>=2.0`
- `torch>=2.0`
- `numpy>=1.20`
- `fastapi`, `uvicorn`

## Performance

- **Episode time**: 2–5 seconds (CPU)
- **Memory**: ~200 MB
- **Throughput**: 10–20 episodes/min

## Real-World Relevance

This environment addresses core challenges in actual autonomous delivery systems:
- Multi-agent path planning with constraints
- Resource management (battery, time)
- Collision avoidance in shared airspace
- Priority task scheduling

Companies deploying these systems (Amazon, Wing, Zipline) face exactly these optimization problems daily.

## License

MIT License. See LICENSE for details.
