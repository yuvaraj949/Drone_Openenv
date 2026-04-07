"""
DroneDispatchEnv — Core OpenEnv simulation logic.
Manages drone status, movement validation, battery, and collisions.
"""

from __future__ import annotations
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Action,
    DroneAction,
    DroneState,
    HOVER,
    Observation,
    Reward,
    RewardDetails,
)
from .tasks import get_config

class DroneDispatchEnv:
    """
    OpenEnv-compliant environment for Autonomous Drone Dispatching.
    """

    def __init__(self, task: str = "easy", seed: Optional[int] = None) -> None:
        self.task_name = task
        self.cfg = get_config(task)
        self.graph = self.cfg["graph"]
        self.all_zones = self.cfg["all_zones"]
        self.max_steps = self.cfg["max_steps"]
        self.energy_drain = self.cfg["energy_drain"]
        self.bottlenecks = self.cfg.get("bottlenecks", [])
        self.dynamic_nfz = self.cfg.get("dynamic_nfz", [])
        self.deadline = self.cfg["deadline"]
        
        self._rng = random.Random(seed)
        self._drones: List[DroneState] = []
        self._step: int = 0
        self._collisions: int = 0
        self._total_deliv: int = 0

    async def reset(self) -> Observation:
        """Initialize fresh episode."""
        self._step = 0
        self._collisions = 0
        self._total_deliv = 0
        
        num_drones = self.cfg["num_drones"]
        num_emerg = self.cfg["num_emergencies"]
        
        # Sample unique start/end pairs
        locs = self._rng.sample(self.all_zones, num_drones)
        remaining = [z for z in self.all_zones if z not in locs]
        dests = self._rng.sample(remaining, num_drones) if len(remaining) >= num_drones else \
                [self._rng.choice(self.all_zones) for _ in range(num_drones)]
        
        priorities = [2] * num_emerg + [1] * (num_drones - num_emerg)
        self._rng.shuffle(priorities)
        
        self._drones = [
            DroneState(
                id=f"D{i+1}",
                location=locs[i],
                altitude=self._rng.uniform(10.0, 30.0),
                destination=dests[i],
                battery=100.0,
                priority=priorities[i],
                delivered=False,
                steps_taken=0
            ) for i in range(num_drones)
        ]
        
        return self._build_obs()

    async def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Advance simulation by one step."""
        self._step += 1
        act_map = {a.drone_id: a for a in action.actions}
        
        step_details = RewardDetails()
        nfz = self._get_active_nfz()
        new_locs: Dict[str, List[DroneState]] = defaultdict(list)
        
        for drone in self._drones:
            if drone.delivered:
                continue
            
            drone_act = act_map.get(drone.id)
            req_zone = drone_act.move_to if drone_act and drone_act.move_to != HOVER else drone.location
            climb = drone_act.climb if drone_act else 0.0
            
            # Validation
            old_dist = self._manhattan(drone.location, drone.destination)
            
            # Constraints
            if req_zone in self.graph.get(drone.location, []) or req_zone == drone.location:
                if req_zone not in nfz and drone.battery > 0:
                    drone.location = req_zone
            
            drone.altitude = max(0.0, drone.altitude + climb)
            drone.steps_taken += 1
            
            # Energy
            drain = self.energy_drain * (1.5 if climb > 0 else (0.8 if climb < 0 else 1.0))
            drone.battery = max(0.0, drone.battery - drain)
            
            # Rewards
            new_dist = self._manhattan(drone.location, drone.destination)
            step_details.progress_reward += (old_dist - new_dist) * 0.1
            
            if drone.location == drone.destination:
                drone.delivered = True
                self._total_deliv += 1
                step_details.delivery_bonus += 1.0
                if drone.priority == 2 and drone.steps_taken <= self.deadline:
                    step_details.delivery_bonus += 0.5
            
            if drone.battery < 10.0:
                step_details.energy_penalty -= 0.05
                
            new_locs[drone.location].append(drone)

        # Collisions (3D-lite: same zone + altitude diff < 2m)
        step_colls = 0
        for zone, ds in new_locs.items():
            if len(ds) < 2: continue
            # Check pairwise altitude
            for i in range(len(ds)):
                for j in range(i+1, len(ds)):
                    if abs(ds[i].altitude - ds[j].altitude) < 2.0:
                        step_colls += 1
        
        self._collisions += step_colls
        step_details.collision_penalty -= step_colls * 0.5
        step_details.time_penalty -= 0.01
        
        total_r = sum(step_details.model_dump().values())
        done = self._is_done()
        
        obs = self._build_obs()
        reward = Reward(total=total_r, details=step_details, done=done)
        info = {
            "step": self._step,
            "delivered": self._total_deliv,
            "collisions": self._collisions,
            "nfz_active": list(nfz)
        }
        
        return obs, reward, done, info

    async def state(self) -> Dict[str, Any]:
        return {
            "drones": [d.model_dump() for d in self._drones],
            "collisions": self._collisions,
            "step": self._step,
            "delivered": self._total_deliv
        }

    async def close(self) -> None:
        pass

    def _build_obs(self) -> Observation:
        cong = defaultdict(int)
        for d in self._drones:
            if not d.delivered: cong[d.location] += 1
        return Observation(
            drones=list(self._drones),
            congestion=dict(cong),
            step=self._step,
            collisions=self._collisions,
            graph=self.graph
        )

    def _manhattan(self, z1: str, z2: str) -> int:
        r1, r2 = ord(z1[0]), ord(z2[0])
        c1, c2 = int(z1[1:]), int(z2[1:])
        return abs(r1 - r2) + abs(c1 - c2)

    def _get_active_nfz(self) -> set:
        active = set()
        for start, end, zone in self.dynamic_nfz:
            if start <= self._step <= end:
                active.add(zone)
        return active

    def _is_done(self) -> bool:
        all_d = all(d.delivered for d in self._drones)
        tout = self._step >= self.max_steps
        dead = all(d.battery <= 0 and not d.delivered for d in self._drones)
        return all_d or tout or dead
