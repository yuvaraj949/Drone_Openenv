"""
PEDRA-to-OpenEnv AirSim Bridge
==============================
Mocks the airsim.MultirotorClient interface to allow running the original 
PEDRA repository logic against the discrete DroneTrafficEnv.

Maps 3D positions to grid zones and vice-versa.
"""

import math
import numpy as np
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from environment.drone_env import DroneTrafficEnv
from environment.models import Action, DroneAction, HOVER


class Vector3r:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x_val = x
        self.y_val = y
        self.z_val = z

class Quaternionr:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w_val = w
        self.x_val = x
        self.y_val = y
        self.z_val = z

class Pose:
    def __init__(self, position=None, orientation=None):
        self.position = position or Vector3r()
        self.orientation = orientation or Quaternionr()

class Kinematics:
    def __init__(self, position=None):
        self.position = position or Vector3r()

class MultirotorState:
    def __init__(self, position=None):
        self.kinematics_estimated = Kinematics(position)


class PedraAirSimMock:
    """
    Mock AirSim client that bridges PEDRA's expectations to our grid env.
    """
    def __init__(self, env: DroneTrafficEnv, grid_scale: float = 10.0):
        self.env = env
        self.grid_scale = grid_scale  # meters per grid cell
        
        # Internal state to map drone IDs
        self.drone_map: Dict[str, str] = {}
        self._sync_drones()

    def _sync_drones(self):
        """Map drone names to internal env IDs."""
        for i, drone in enumerate(self.env.state()["drones"]):
            # PEDRA usually uses 'drone0', 'drone1', ...
            self.drone_map[f"drone{i}"] = drone["id"]

    def _zone_to_xyz(self, zone: str) -> Tuple[float, float]:
        """Convert 'A1' -> (0, 0) relative to grid_scale."""
        row_char = zone[0].upper()
        col_num = int(zone[1:])
        r = ord(row_char) - ord('A')
        c = col_num - 1
        return float(r * self.grid_scale), float(c * self.grid_scale)

    def _xyz_to_zone(self, x: float, y: float) -> str:
        """Convert (float, float) -> 'A1'."""
        r = int(round(x / self.grid_scale))
        c = int(round(y / self.grid_scale))
        # Bounds check
        r = max(0, min(r, self.env.cfg["rows"] - 1))
        c = max(0, min(c, self.env.cfg["cols"] - 1))
        row_char = chr(ord('A') + r)
        return f"{row_char}{c + 1}"

    # ->-> Mapped AirSim API ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

    def confirmConnection(self):
        return True

    def simGetVehiclePose(self, vehicle_name: str = "") -> Pose:
        """Return the current 3D position of a drone based on grid zone."""
        drone_id = self.drone_map.get(vehicle_name, "D1")
        # Find drone in current obs
        for d in self.env.state()["drones"]:
            if d["id"] == drone_id:
                x, y = self._zone_to_xyz(d["location"])
                return Pose(position=Vector3r(x, y, -5.0)) # altitude -5
        return Pose()

    def getMultirotorState(self, vehicle_name: str = "") -> MultirotorState:
        pose = self.simGetVehiclePose(vehicle_name)
        return MultirotorState(position=pose.position)

    def moveToPositionAsync(self, x: float, y: float, z: float, velocity: float, 
                             vehicle_name: str = ""):
        """
        Translate a 3D target coordinate to a grid move.
        Note: This bridge triggers a partial env.step() internally.
        """
        target_zone = self._xyz_to_zone(x, y)
        drone_id = self.drone_map.get(vehicle_name, "D1")
        
        # Build action for THIS drone
        action = Action(actions=[
            DroneAction(drone_id=drone_id, move_to=target_zone)
        ])
        
        # Execute in env
        # Note: Since OpenEnv is multi-agent, we usually step all at once.
        # But PEDRA legacy loops per agent. We handle this in the bridge runner.
        obs, reward, done, info = self.env.step(action)
        
        # Returns a mock future (not actually used by PEDRA for logic)
        return SimpleNamespace(join=lambda: True)

    def moveByVelocityAsync(self, vx: float, vy: float, vz: float, duration: float, 
                             vehicle_name: str = ""):
        """Translate velocity into a direction vector for grid movement."""
        # Heuristic: Find neighbor zone in the direction of velocity
        current_pose = self.simGetVehiclePose(vehicle_name)
        new_x = current_pose.position.x_val + vx * duration
        new_y = current_pose.position.y_val + vy * duration
        return self.moveToPositionAsync(new_x, new_y, 0, 5, vehicle_name)

    def simSetVehiclePose(self, pose: Pose, ignore_collison: bool = True, vehicle_name: str = ""):
        """Teleport drone in the env."""
        target_zone = self._xyz_to_zone(pose.position.x_val, pose.position.y_val)
        drone_id = self.drone_map.get(vehicle_name, "D1")
        
        for d in self.env.drones:
            if d.id == drone_id:
                d.location = target_zone
        return True

    def simGetImages(self, requests: List[any], vehicle_name: str = ""):
        """
        Mock image acquisition. 
        PEDRA expects RGB and Depth. We'll return dummy black frames.
        """
        # Return something that PedraAgent.get_CustomImage() can process
        # Usually a list of ImageResponse objects with image_data_uint8
        class ImageResponse:
            def __init__(self):
                self.image_data_uint8 = bytes([0] * (64*64*3))
                self.width = 64
                self.height = 64
        return [ImageResponse(), ImageResponse()]
