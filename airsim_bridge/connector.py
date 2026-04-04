"""
AirSim Bridge Connector (PEDRA Stage 4B)
=====================================
Provides the hook into Unreal Engine / PEDRA environments.

If `airsim` is installed and an Unreal environment is running, this binds
the OpenEnv grid moves to physical 3D drone movements.
If not, it acts as a graceful no-op stub.
"""

from typing import Dict, Tuple

try:
    import airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False


class PedraConnector:
    def __init__(self, use_airsim: bool = False, grid_scale: float = 10.0, flight_height: float = -5.0):
        self.enabled = use_airsim and AIRSIM_AVAILABLE
        self.scale = grid_scale
        self.z = flight_height
        self.client = None
        
        if self.enabled:
            print("[PEDRA] Connecting to AirSim Multirotor Client...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print("[PEDRA] Connection established.")
        else:
            if use_airsim and not AIRSIM_AVAILABLE:
                print("[WARN] AirSim not installed. PedraConnector running in stub mode.")
            
    def _zone_to_xyz(self, zone_str: str) -> Tuple[float, float, float]:
        """Convert 'A1' -> (0, 0, Z), 'C2' -> (20, 10, Z), etc."""
        if zone_str == "hover":
            # Handled by caller to keep current pos
            return (0, 0, 0)
            
        char_p = ord(zone_str[0].upper()) - 65
        num_p = int(zone_str[1:]) - 1
        
        return (char_p * self.scale, num_p * self.scale, self.z)

    def init_drones(self, drone_ids: list):
        """Enable API control and takeoff for all drones."""
        if not self.enabled: return
        
        for d_id in drone_ids:
            self.client.enableApiControl(True, vehicle_name=d_id)
            self.client.armDisarm(True, vehicle_name=d_id)
            self.client.takeoffAsync(vehicle_name=d_id)
            
    def move_drone(self, drone_id: str, to_zone: str):
        """Send asynchronous 3D move command."""
        if not self.enabled: return
        
        if to_zone == "hover":
            self.client.moveByVelocityAsync(0, 0, 0, duration=1, vehicle_name=drone_id)
            return
            
        x, y, z = self._zone_to_xyz(to_zone)
        # 5 m/s velocity
        self.client.moveToPositionAsync(x, y, z, velocity=5, vehicle_name=drone_id)
