"""
Pydantic data models for the Drone Autonomous Dispatcher environment.
Strictly follows OpenEnv schema conventions.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Drone & Airspace Models
# ---------------------------------------------------------------------------

class DroneState(BaseModel):
    """Current state of a single drone."""
    id: str = Field(..., description="Unique drone ID (e.g. 'D1')")
    location: str = Field(..., description="Current grid zone (e.g. 'A1')")
    altitude: float = Field(default=0.0, description="Altitude in meters")
    destination: str = Field(..., description="Target grid zone")
    battery: float = Field(..., ge=0.0, le=100.0, description="Battery %")
    priority: int = Field(..., ge=1, le=2, description="1=Normal, 2=Emergency")
    delivered: bool = Field(default=False, description="Delivery completion status")
    steps_taken: int = Field(default=0, description="Steps since start")

class Observation(BaseModel):
    """Observation returned after every step/reset."""
    drones: List[DroneState] = Field(..., description="Fleet status")
    congestion: Dict[str, int] = Field(..., description="Drones per zone")
    step: int = Field(..., description="Current step count")
    collisions: int = Field(..., description="Cumulative collisions")
    graph: Dict[str, List[str]] = Field(..., description="Airspace adjacency list")

class DroneAction(BaseModel):
    """Move command for one drone."""
    drone_id: str
    move_to: str = Field(..., description="Target zone or 'hover'")
    climb: float = Field(default=0.0, description="Altitude adjustment (+/- m)")

class Action(BaseModel):
    """Set of actions for all drones."""
    actions: List[DroneAction]

# ---------------------------------------------------------------------------
# Reward Models
# ---------------------------------------------------------------------------

class RewardDetails(BaseModel):
    """Breakdown of step rewards."""
    delivery_bonus: float = Field(default=0.0)
    collision_penalty: float = Field(default=0.0)
    energy_penalty: float = Field(default=0.0)
    progress_reward: float = Field(default=0.0)
    time_penalty: float = Field(default=0.0)

class Reward(BaseModel):
    """Total reward signal."""
    total: float
    details: RewardDetails
    done: bool = False

HOVER = "hover"
