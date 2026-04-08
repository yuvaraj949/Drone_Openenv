"""
Pydantic data models for the Drone Traffic Control environment.

All inter-component communication uses these typed models to ensure
strict schema validation compatible with OpenEnv conventions.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Drone-level models
# ---------------------------------------------------------------------------

class DroneState(BaseModel):
    """Complete state of a single drone."""

    id: str = Field(..., description="Unique drone identifier, e.g. 'D1'")
    location: str = Field(..., description="Current grid zone, e.g. 'A1'")
    x: float = Field(default=0.0, description="Current X coordinate (meters)")
    y: float = Field(default=0.0, description="Current Y coordinate (meters)")
    altitude: float = Field(default=0.0, description="Current altitude in meters")
    vx: float = Field(default=0.0, description="Current velocity X (m/s)")
    vy: float = Field(default=0.0, description="Current velocity Y (m/s)")
    vz: float = Field(default=0.0, description="Current velocity Z (m/s)")
    destination: str = Field(..., description="Target grid zone, e.g. 'C3'")
    target_altitude: float = Field(default=0.0, description="Desired destination altitude")
    battery: float = Field(
        ..., ge=0.0, le=100.0, description="Battery level 0->100 %"
    )
    priority: int = Field(
        ..., ge=1, le=2, description="1 = normal courier, 2 = emergency"
    )
    delivered: bool = Field(
        default=False, description="True once the drone has reached its destination"
    )
    steps_taken: int = Field(
        default=0, description="Number of steps this drone has already moved"
    )

class Obstacle(BaseModel):
    """Stationary or dynamic obstacle in the 3D airspace."""
    id: str
    x: float
    y: float
    z: float
    radius: float


class DroneAction(BaseModel):
    """Routing command for a single drone."""

    drone_id: str = Field(..., description="Target drone ID")
    move_to: str = Field(
        ...,
        description="Adjacent zone the drone should move to, or 'hover' to stay in place."
    )
    vertical_command: float = Field(
        default=0.0,
        description="Vertical movement: +N for climb, -N for descend (meters)."
    )
    thrust_vector: List[float] = Field(
        default=[0.0, 0.0, 0.0],
        description="Forces [Fx, Fy, Fz] applied to the drone in Physics Mode."
    )


# ---------------------------------------------------------------------------
# Episode-level models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Full environment observation returned after every step / reset."""

    drones: List[DroneState] = Field(
        ..., description="Current state of all drones in the airspace"
    )
    congestion_map: Dict[str, int] = Field(
        ..., description="Number of drones currently occupying each zone"
    )
    wind_vector: List[float] = Field(
        default=[0.0, 0.0, 0.0],
        description="Current global wind force vector [Wx, Wy, Wz]"
    )
    step: int = Field(
        default=0, description="Current episode step counter"
    )
    collisions: int = Field(
        default=0, description="Cumulative collision count for this episode"
    )
    sensing_radius: float = Field(
        default=10.0, description="Max distance drones can sense each other (meters)"
    )
    stationary_obstacles: List[Obstacle] = Field(
        default_factory=list, description="Obstacles in the 3D airspace"
    )
    graph_edges: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Adjacency list of the airspace graph (read-only context)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "drones": [
                    {
                        "id": "D1",
                        "location": "A1",
                        "destination": "C3",
                        "battery": 100.0,
                        "priority": 1,
                        "delivered": False,
                        "steps_taken": 0,
                    }
                ],
                "congestion_map": {"A1": 1, "A2": 0},
                "step": 0,
                "collisions": 0,
                "graph_edges": {"A1": ["A2", "B1"]},
            }
        }


HOVER = "hover"  # Sentinel value: drone stays in its current zone


class Action(BaseModel):
    """Routing decisions for all active drones in one step."""

    actions: List[DroneAction] = Field(
        ..., description="One action per active drone"
    )

    @validator("actions")
    def no_duplicate_drone_ids(cls, v: List[DroneAction]) -> List[DroneAction]:
        ids = [a.drone_id for a in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate drone_id in Action.actions - each drone may have at most one action per step.")
        return v


class RewardDetails(BaseModel):
    """Fine-grained breakdown of the step reward."""

    deliveries: float = Field(default=0.0, description="+1 per drone delivered this step")
    step_penalty: float = Field(default=0.0, description="-0.5 applied each step to encourage speed")
    distance_reward: float = Field(default=0.0, description="Reward for moving closer to destination")
    energy_penalty: float = Field(default=0.0, description="Penalty for high thrust magnitude")
    collision_penalty: float = Field(default=0.0, description="-2 per collision detected this step")
    emergency_bonus: float = Field(default=0.0, description="+1 per emergency drone delivered on-time")
    battery_penalty: float = Field(default=0.0, description="-0.1 per drone with battery < 10 %")


class Reward(BaseModel):
    """Reward signal returned after each environment step."""

    total: float = Field(..., description="Scalar sum of all reward components")
    details: RewardDetails = Field(
        default_factory=RewardDetails,
        description="Breakdown of individual reward components"
    )
    done: bool = Field(
        default=False, description="True when the episode has ended"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total": 0.5,
                "details": {
                    "deliveries": 1.0,
                    "step_penalty": -0.5,
                    "collision_penalty": 0.0,
                    "emergency_bonus": 0.0,
                    "battery_penalty": 0.0,
                },
                "done": False,
            }
        }
