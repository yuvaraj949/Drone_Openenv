"""
AirSimDroneEnv — High-Fidelity Unreal Engine 4 interface for drone traffic.
Wraps the AirSim MultirotorClient to provide a photorealistic training/inference environment.
"""

from __future__ import annotations

import time
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple

# Try Colosseum's PythonClient first (newer, Python 3.13 compatible)
# Using standard pip airsim package (compatible with AirSimNH.exe)
# _colosseum_path = r"C:\Users\yuvar\Downloads\AIRSIM\Plugin\Colosseum\PythonClient"
# if _colosseum_path not in _sys.path:
#     _sys.path.insert(0, _colosseum_path)

_AIRSIM_IMPORT_ERROR = None
try:
    import cosysairsim as airsim
except Exception as _e:
    try:
        import airsim
    except Exception as _e2:
        airsim = None
        _AIRSIM_IMPORT_ERROR = f"cosysairsim error: {str(_e)}\nairsim error: {str(_e2)}"

from .models import (
    Action,
    DroneAction,
    DroneState,
    Observation,
    Reward,
    RewardDetails,
    Obstacle,
)
from .drone_env import DroneTrafficEnv

class AirSimDroneEnv(DroneTrafficEnv):
    """
    AirSim/Unreal Engine interface for high-fidelity drone simulation.
    Configured via app UI (IP/Port).
    """

    def __init__(self, ip: str = "127.0.0.1", port: int = 41451, task: str = "easy"):
        super().__init__(task=task)
        self.ip = ip
        self.port = port
        self.client = None
        self.drone_names = [] # AirSim vehicle names
        self.init_z = -20.0 # Increased altitude to clear trees/buildings (15m)
        self.drone_missions: Dict[str, Dict[str, Any]] = {}

    def connect(self):
        """Initializes connection to the flying Unreal Engine instance."""
        if not airsim:
            err = _AIRSIM_IMPORT_ERROR or 'Unknown import error'
            raise ImportError(
                f"Could not import 'airsim'. Real error:\n{err}\n\n"
                f"Fix: pip install --upgrade airsim"
            )
        
        print(f"[AirSim] Connecting to {self.ip}:{self.port}...")
        self.client = airsim.MultirotorClient(ip=self.ip, port=self.port)
        self.client.confirmConnection()
        
        # We'll use our internal drone IDs as vehicle names in AirSim
        # e.g., 'Drone_1', 'Drone_2' (must match the Settings.json in AirSim)
        # For simplicity, if AirSim only has one drone, we map all to it (sequential testing)
        # or use the internal vehicle list.
        self.drone_names = self.client.listVehicles()
        print(f"[AirSim] Connected. Active Vehicles: {self.drone_names}")
        
        for name in self.drone_names:
            self.client.enableApiControl(True, vehicle_name=name)
            self.client.armDisarm(True, vehicle_name=name)
            self.client.takeoffAsync(vehicle_name=name).join()

    def reset(self) -> Observation:
        """Resets the simulation world and positions drones at start."""
        if not self.client:
            self.connect()
        
        self._step = 0
        self._collisions = 0
        self.client.reset()
        
        # Assign random missions (origin -> destination) for each vehicle
        import random
        # self.all_zones comes from the parent DroneTrafficEnv
        destinations = random.sample(self.all_zones, min(len(self.drone_names), len(self.all_zones)))
        
        self.drone_missions = {}
        for i, name in enumerate(self.drone_names):
            dest = destinations[i] if i < len(destinations) else random.choice(self.all_zones)
            self.drone_missions[name] = {
                "destination": dest,
                "priority": 1 # Forced to Normal for initial task testcaing
            }

        self.delivered_drones: Set[str] = set()
        self.landing_drones: Set[str] = set()

        for name in self.drone_names:
            self.client.enableApiControl(True, vehicle_name=name)
            self.client.armDisarm(True, vehicle_name=name)
            # Takeoff to a safe altitude
            self.client.takeoffAsync(vehicle_name=name)
            
        # 3-second grace period for all drones to reach safe altitude (init_z)
        time.sleep(3.0)
        
        for name in self.drone_names:
            # Assign each drone its highway altitude immediately
            d_idx = int(''.join(filter(str.isdigit, name)))
            target_z = self.init_z - (d_idx * 2.5) # NED Z is negative
            self.client.moveToZAsync(target_z, 3, vehicle_name=name)
        
        # 5-second grace period for all drones to reach their layers
        time.sleep(5.0)

        obs = self._build_observation()
        self.drones_state = obs.drones
        if hasattr(self, 'prev_obs'):
            delattr(self, 'prev_obs') # Reset history
            
        # Bird's eye over center: x=0, y=0, z=-60 (60m up), Pitch=-90deg (down)
        self.client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, -60), airsim.to_quaternion(-1.57, 0, 0)))

        return obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self._step += 1
        
        # Mapping actions to AirSim commands using Directional Velocity
        for a in action.actions:
            v_name = a.drone_id if a.drone_id in self.drone_names else self.drone_names[0]
            
            # If drone is already delivered, stop motors
            if a.drone_id in self.delivered_drones:
                self.client.armDisarm(False, vehicle_name=v_name)
                continue

            state = self.client.getMultirotorState(vehicle_name=v_name)
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            
            # 0. Energy Penalty (Proportional to velocity/thrust)
            v_mag = np.linalg.norm([vel.x_val, vel.y_val, vel.z_val])
            energy_penalty = - (v_mag / 15.0) * 0.1 # Penalty for constant high speed
            
            if state.landed_state == 1 and a.drone_id in self.landing_drones:
                self.delivered_drones.add(a.drone_id)
                continue

            if a.thrust_vector and any(v != 0 for v in a.thrust_vector):
                # RL Training/Inference mode: Use raw thrust/velocity vector
                vx, vy, vz = a.thrust_vector
                self.client.moveByVelocityAsync(vx, vy, vz, duration=1.0, vehicle_name=v_name)
                continue

            target_grid = a.move_to 
            pos = state.kinematics_estimated.position
            
            # 1. Handle HOVER or invalid grid targets
            if target_grid == "hover" or len(target_grid) < 2:
                self.client.hoverAsync(vehicle_name=v_name)
                continue

            # 2. Parse valid grid targets (A1, B2, etc.)
            try:
                col_idx = ord(target_grid[0].upper()) - ord('A')
                row_idx = int(target_grid[1:])
                target_x = (col_idx * 10) - 45 
                target_y = ((row_idx - 1) * 10) - 45
            except (ValueError, IndexError):
                self.client.hoverAsync(vehicle_name=v_name)
                continue
                target_x = (col_idx * 10) - 45 
                target_y = ((row_idx - 1) * 10) - 45
                
                dx = target_x - pos.x_val
                dy = target_y - pos.y_val
                dist = np.sqrt(dx**2 + dy**2)
                
                # Active Vertical Correction: Highway Altitude 20m + offset
                d_idx = int(''.join(filter(str.isdigit, v_name)))
                highway_alt = 20.0 + (d_idx * 2.5)
                local_alt = -pos.z_val
                
                # NAVIGATION TRANSITIONS
                mission = self.drone_missions.get(v_name, {"destination": "A1"})
                
                # PHASE: LANDING (within 3m of goal)
                if target_grid == mission["destination"] and dist < 3.0:
                    if v_name not in self.landing_drones:
                        print(f"[AirSim] {v_name} initiating landing at {target_grid}...")
                        self.client.landAsync(vehicle_name=v_name)
                        self.landing_drones.add(v_name)
                    continue

                # PHASE: CRUISE (Maintain altitude layer)
                vz_vel = np.clip(local_alt - highway_alt, -2.0, 2.0)
                
                if dist > 1.5: 
                    vx = (dx / dist) * 15.0 
                    vy = (dy / dist) * 15.0
                    self.client.moveByVelocityAsync(vx, vy, vz_vel, 1.0, vehicle_name=v_name)
                else:
                    self.client.moveByVelocityAsync(0, 0, vz_vel, 1.0, vehicle_name=v_name)
            else:
                self.client.hoverAsync(vehicle_name=v_name)

        # Wait for simulation to progress (matched to command duration)
        time.sleep(1.0)

        # Check for collisions via AirSim API
        step_collisions = 0
        for name in self.drone_names:
            collision_info = self.client.simGetCollisionInfo(vehicle_name=name)
            if collision_info.has_collided:
                step_collisions += 1
        
        self._collisions += step_collisions
        
        obs = self._build_observation()
        done = self._check_done()
        
        # Calculate Reward Shaping (Distance-to-goal)
        total_step_reward = 0.0
        for d in obs.drones:
            if d.delivered:
                # Big bonus for newly delivered drones
                if d.id not in [prev_d.id for prev_d in self.drones_state if prev_d.delivered]:
                    total_step_reward += 10.0
                continue
            
            # Find destination coordinates in meters
            mission = self.drone_missions.get(d.id, {"destination": "A1"})
            dest = mission["destination"]
            col_idx = ord(dest[0].upper()) - ord('A')
            row_idx = int(dest[1:])
            target_x = (col_idx * 10) - 45 
            target_y = ((row_idx - 1) * 10) - 45
            
            # Distance penalty/reward
            dist = np.sqrt((d.x - target_x)**2 + (d.y - target_y)**2)
            # Small penalty for every step to encourage speed
            total_step_reward -= 0.01 
            # Bonus for getting closer (if we have previous state)
            if hasattr(self, 'prev_obs'):
                prev_d = next((pd for pd in self.prev_obs.drones if pd.id == d.id), None)
                if prev_d:
                    prev_dist = np.sqrt((prev_d.x - target_x)**2 + (prev_d.y - target_y)**2)
                    # Stronger distance gradient (1.0 vs 0.1)
                    total_step_reward += (prev_dist - dist) * 1.0

        if step_collisions > 0:
            total_step_reward -= 20.0 # Extreme collision penalty for "Perfection"
        
        total_step_reward += sum(d.battery_penalty for d in obs.drones if hasattr(d, 'battery_penalty')) # (Optional)
        # Using a shared penalty from v_mag across all drones for city-level efficiency
        total_step_reward += energy_penalty if 'energy_penalty' in locals() else 0
        
        self.prev_obs = obs
        self.drones_state = obs.drones
        reward = Reward(total=total_step_reward, details=RewardDetails(), done=done)
        
        return obs, reward, done, {"airsim": True}

    def _build_observation(self) -> Observation:
        """Constructs Observation from AirSim State."""
        drones_state = []
        for name in self.drone_names:
            state = self.client.getMultirotorState(vehicle_name=name)
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            
            # Map NED coordinates (AirSim) back to our "Grid" space (relative)
            # or just use raw meters.
            # Convert AirSim real-world coordinates (meters) to grid reference
            # Map x: -50..+50m → columns A-J (10 cols), y: -50..+50m → rows 1-10
            col_idx = max(0, min(9, int((pos.x_val + 50) / 10)))
            row_idx = max(1, min(10, int((pos.y_val + 50) / 10) + 1))
            col_letter = chr(ord('A') + col_idx)
            grid_location = f"{col_letter}{row_idx}"

            mission = self.drone_missions.get(name, {"destination": "A1", "priority": 1})
            
            # Persistent delivery check: Once at goal, stays at goal
            is_at_goal = (grid_location == mission["destination"])
            if is_at_goal:
                self.delivered_drones.add(name)

            drones_state.append(DroneState(
                id=name,
                location=grid_location,
                x=pos.x_val,
                y=pos.y_val,
                destination=mission["destination"],
                battery=100.0,
                priority=mission["priority"],
                altitude=-pos.z_val,
                vx=vel.x_val,
                vy=vel.y_val,
                vz=vel.z_val,
                delivered=name in self.delivered_drones
            ))
        
        return Observation(
            step=self._step,
            drones=drones_state,
            congestion_map={},
            graph_edges={},
            wind_vector=[0, 0, 0],
            sensing_radius=10.0,
            stationary_obstacles=[
                # Placeholder AirSim obstacles (can be dynamically mapped from UE4)
                Obstacle(id="AirSimTower1", x=10.0, y=10.0, z=-25.0, radius=2.0),
                Obstacle(id="AirSimTower2", x=-20.0, y=-20.0, z=-25.0, radius=2.0)
            ]
        )

    def state(self) -> Dict[str, Any]:
        """Returns the full environment state for the grader."""
        return {
            "step": self._step,
            "drones": [d.model_dump() for d in self.drones_state],
            "collisions": self._collisions
        }

    def _check_done(self) -> bool:
        # Increase tolerance for Hard Mode (10 drones)
        if self._step >= self.max_steps:
            return True
            
        if self._collisions > (15 if len(self.drone_names) > 5 else 5):
            return True
        
        all_delivered = all(d.delivered for d in self.drones_state)
        return all_delivered
