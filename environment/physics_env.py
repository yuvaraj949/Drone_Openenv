"""
PhysicsDroneEnv - 3D Continuous Physics Environment with Stochastic Wind.
Simulates drone dynamics using point-mass physics.
"""

from __future__ import annotations

import random
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Action,
    DroneAction,
    DroneState,
    Observation,
    Reward,
    RewardDetails,
    Obstacle,
)
from .tasks import get_task_config
from .drone_env import DroneTrafficEnv

class PhysicsDroneEnv(DroneTrafficEnv):
    """
    Higher-fidelity drone environment with continuous 3D physics and wind.
    """

    def __init__(self, task: str = "easy", seed: Optional[int] = None) -> None:
        super().__init__(task, seed)
        
        # Physics Parameters
        self.dt = 0.5  # Seconds per step
        self.mass = 1.5  # kg
        self.gravity = 9.81
        self.drag_coeff = 0.1
        self.max_thrust = 25.0  # Newtons
        
        # Wind Parameters
        self.base_wind = np.array([1.5, 0.5, 0.0])  # Base wind vector (m/s)
        self.wind_gust_std = 0.5
        self.current_wind = self.base_wind.copy()

        # Workspace Bounds (from rows/cols)
        self.max_x = self.cfg["cols"] - 1
        self.max_y = self.cfg["rows"] - 1
        self.max_z = self.cfg.get("max_altitude", 20.0)
        
        # Radius-limited sensing
        self.sensing_radius = 10.0
        self.stationary_obstacles = [
            Obstacle(id="TowerA", x=1.5, y=1.5, z=5.0, radius=1.0),
            Obstacle(id="TowerB", x=3.5, y=3.5, z=5.0, radius=1.0)
        ]

    def reset(self) -> Observation:
        obs = super().reset()
        # Initialise velocities to zero
        for d in self._drones:
            d.vx, d.vy, d.vz = 0.0, 0.0, 0.0
            # Set continuous coordinates based on grid location
            r = ord(d.location[0]) - ord("A")
            c = int(d.location[1:]) - 1
            # We treat grid coordinates as meters for physics
            # (In a real sim, zones might be 50m wide, but we'll use 1:1 for simplicity)
            d.vx = 0.0
            d.vy = 0.0
            d.vz = 0.0
        
        self.current_wind = self.base_wind.copy()
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self._step += 1
        
        # Update Wind (stochastic gusts)
        gust = np.random.normal(0, self.wind_gust_std, 3)
        self.current_wind = self.base_wind + gust

        action_map = {a.drone_id: a.thrust_vector for a in action.actions}
        step_reward_components = RewardDetails()
        new_positions: Dict[str, List[DroneState]] = defaultdict(list)

        for d in self._drones:
            if d.delivered or d.battery <= 0:
                continue

            # Thrust Command
            thrust = np.array(action_map.get(d.id, [0.0, 0.0, self.mass * self.gravity]))
            thrust = np.clip(thrust, -self.max_thrust, self.max_thrust)

            # Current Physics State
            pos = self._get_pos(d)
            vel = np.array([d.vx, d.vy, d.vz])
            
            # Forces: Thrust + Gravity + Wind + Drag
            f_grav = np.array([0, 0, -self.mass * self.gravity])
            f_wind = self.current_wind * 2.0  # Wind force multiplier
            f_drag = -self.drag_coeff * vel * np.abs(vel)
            
            total_force = thrust + f_grav + f_wind + f_drag
            accel = total_force / self.mass
            
            # Integrate
            new_vel = vel + accel * self.dt
            new_pos = pos + new_vel * self.dt
            
            # Bounds Check
            new_pos[0] = np.clip(new_pos[0], 0, self.max_x)
            new_pos[1] = np.clip(new_pos[1], 0, self.max_y)
            new_pos[2] = np.clip(new_pos[2], 0, self.max_z)
            
            # Update Drone State
            d.vx, d.vy, d.vz = new_vel
            d.x, d.y = new_pos[0], new_pos[1]
            d.altitude = new_pos[2]
            d.location = self._pos_to_zone(new_pos)
            
            # Battery Drain (proportional to thrust magnitude)
            thrust_mag = np.linalg.norm(thrust)
            drain = (thrust_mag / self.max_thrust) * self.battery_drain * 2.0
            d.battery = max(0, d.battery - drain)
            
            # Energy Penalty (to encourage efficiency)
            step_reward_components.energy_penalty -= (thrust_mag / self.max_thrust) * 0.5

            if d.location == d.destination and d.altitude < 1.0:
                d.delivered = True
                step_reward_components.deliveries += 1.0

            new_positions[d.location].append(d)

        # Collision detection (same as base but can be more precise if needed)
        # Using zone-based for now to maintain consistency with metrics
        safety_margin = self.cfg.get("safety_margin", 2.0)
        step_collisions = 0
        for zone, drones_in_zone in new_positions.items():
            if len(drones_in_zone) < 2: continue
            for i in range(len(drones_in_zone)):
                for j in range(i + 1, len(drones_in_zone)):
                    if abs(drones_in_zone[i].altitude - drones_in_zone[j].altitude) < safety_margin:
                        step_collisions += 1

        self._collisions += step_collisions
        step_reward_components.collision_penalty = -2.0 * step_collisions
        step_reward_components.step_penalty = -0.5

        total_reward = sum([
            step_reward_components.deliveries,
            step_reward_components.collision_penalty,
            step_reward_components.step_penalty,
            step_reward_components.energy_penalty
        ])
        
        self._episode_rewards.append(total_reward)
        obs = self._build_observation()
        done = self._check_done()
        
        return obs, Reward(total=total_reward, details=step_reward_components, done=done), done, {"wind": self.current_wind.tolist()}

    def _get_pos(self, d: DroneState) -> np.ndarray:
        r = ord(d.location[0]) - ord("A")
        c = int(d.location[1:]) - 1
        return np.array([float(c), float(r), d.altitude])

    def _pos_to_zone(self, pos: np.ndarray) -> str:
        c = int(round(pos[0]))
        r = int(round(pos[1]))
        r_char = chr(ord("A") + max(0, min(r, self.cfg["rows"] - 1)))
        c_num = max(1, min(c + 1, self.cfg["cols"]))
        return f"{r_char}{c_num}"

    def _build_observation(self) -> Observation:
        obs = super()._build_observation()
        obs.wind_vector = self.current_wind.tolist()
        obs.sensing_radius = self.sensing_radius
        obs.stationary_obstacles = self.stationary_obstacles
        return obs
