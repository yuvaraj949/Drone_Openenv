"""
DroneTrafficEnv - OpenEnv-compatible environment class.

Lifecycle
---------
env = DroneTrafficEnv(task="easy")
obs  = env.reset()
while not done:
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)

The environment models city airspace as a directed graph of discrete zones.
Drones move one zone per step; simultaneous occupation causes collisions.
Battery drains linearly; a drone stranded with 0% battery is counted as lost.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from environment.models import (
    Action,
    DroneAction,
    DroneState,
    HOVER,
    Observation,
    Reward,
    RewardDetails,
)
from environment.tasks import get_task_config


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class DroneTrafficEnv:
    """
    Autonomous Drone Traffic Control environment.

    Parameters
    ----------
    task : str
        One of "easy", "medium", "hard".
    seed : int, optional
        Random seed for reproducibility.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, task: str = "easy", seed: Optional[int] = None) -> None:
        self.task_name = task
        self.cfg = get_task_config(task)
        self.graph: Dict[str, List[str]] = self.cfg["graph"]
        self.all_zones: List[str] = self.cfg["all_zones"]
        self.max_steps: int = self.cfg["max_steps"]
        self.battery_drain: float = self.cfg["battery_drain_per_step"]
        self.bottleneck_zones: List[str] = self.cfg["bottleneck_zones"]
        self.dynamic_obstacles: List[Tuple[int, int, str]] = self.cfg["dynamic_obstacles"]
        self.emergency_deadline: int = self.cfg["emergency_deadline"]
        self._rng = random.Random(seed)

        # mutable episode state - initialised by reset()
        self._drones: List[DroneState] = []
        self._step: int = 0
        self._collisions: int = 0
        self._episode_rewards: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Initialise a fresh episode and return the first observation."""
        self._step = 0
        self._collisions = 0
        self._episode_rewards = []

        num_drones = self.cfg["num_drones"]
        num_emerg = self.cfg["num_emergencies"]
        
        # Max altitude for the tasks (default 15m)
        max_alt = self.cfg.get("max_altitude", 15.0)

        # Sample distinct start/destination pairs
        locations = self._rng.sample(self.all_zones, num_drones)
        remaining_zones = [z for z in self.all_zones if z not in locations]
        destinations = self._rng.sample(remaining_zones, min(num_drones, len(remaining_zones)))
        
        while len(destinations) < num_drones:
            destinations.append(self._rng.choice(self.all_zones))

        priorities = [2] * num_emerg + [1] * (num_drones - num_emerg)
        self._rng.shuffle(priorities)

        self._drones = [
            DroneState(
                id=f"D{i + 1}",
                location=locations[i],
                altitude=self._rng.uniform(0.0, max_alt),
                destination=destinations[i],
                target_altitude=self._rng.uniform(0.0, max_alt),
                battery=100.0,
                priority=priorities[i],
                delivered=False,
                steps_taken=0,
            )
            for i in range(num_drones)
        ]

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if not self._drones:
            raise RuntimeError("Call reset() before step().")

        self._step += 1

        # Build action lookup: drone_id -> (move_to, dz)
        action_map: Dict[str, Tuple[str, float]] = {
            a.drone_id: (a.move_to, a.vertical_command) for a in action.actions
        }

        blocked_zones = self._get_blocked_zones(self._step)
        new_positions: Dict[str, List[DroneState]] = defaultdict(list)
        step_reward_components = RewardDetails()

        for drone in self._drones:
            if drone.delivered:
                continue

            old_dist = self._get_manhattan_dist(drone.location, drone.destination)
            act_target, dz = action_map.get(drone.id, (HOVER, 0.0))
            requested_zone = drone.location if act_target == HOVER else act_target

            # Validate horizontal move
            valid_zone = self._validate_move(drone, requested_zone, blocked_zones)
            drone.location = valid_zone
            
            new_dist = self._get_manhattan_dist(drone.location, drone.destination)
            # Distance reward (Shaping)
            step_reward_components.distance_reward += (old_dist - new_dist) * 2.0
            
            # Update altitude (Option B - Continuous)
            old_alt = drone.altitude
            drone.altitude = max(0.0, drone.altitude + dz)
            drone.steps_taken += 1

            # 3D Battery drain
            # Base drain + climb penalty (1.5x for ascending)
            climb_factor = 2.0 if dz > 0 else (0.5 if dz < 0 else 1.0)
            drain = self.battery_drain * climb_factor
            drone.battery = max(0.0, drone.battery - drain)
            
            if drone.battery < 10.0:
                step_reward_components.battery_penalty -= 0.1

            # Check delivery (needs to be in zone AND roughly at ground/target alt?)
            # Let's say reaching the zone is enough for now, but 3D state is tracked.
            if drone.location == drone.destination and not drone.delivered:
                drone.delivered = True
                step_reward_components.deliveries += 20.0
                if drone.priority == 2 and drone.steps_taken <= self.emergency_deadline:
                    step_reward_components.emergency_bonus += 10.0

            new_positions[drone.location].append(drone)

        # 3D Collision detection
        # Collision only if in same zone AND altitude difference < safety margin
        safety_margin = self.cfg.get("safety_margin", 2.0)
        step_collisions = 0
        for zone, drones_in_zone in new_positions.items():
            if len(drones_in_zone) < 2:
                continue
            
            # Check pairwise altitudes
            for i in range(len(drones_in_zone)):
                for j in range(i + 1, len(drones_in_zone)):
                    if abs(drones_in_zone[i].altitude - drones_in_zone[j].altitude) < safety_margin:
                        step_collisions += 1

        self._collisions += step_collisions
        step_reward_components.collision_penalty = -2.0 * step_collisions
        step_reward_components.step_penalty = -0.5

        total_step_reward = (
            step_reward_components.deliveries
            + step_reward_components.step_penalty
            + step_reward_components.distance_reward
            + step_reward_components.collision_penalty
            + step_reward_components.emergency_bonus
            + step_reward_components.battery_penalty
        )
        self._episode_rewards.append(total_step_reward)

        obs = self._build_observation()
        done = self._check_done()
        reward = Reward(
            total=round(total_step_reward, 4),
            details=step_reward_components,
            done=done,
        )
        info = {
            "step": self._step,
            "cumulative_collisions": self._collisions,
            "delivered": sum(1 for d in self._drones if d.delivered),
            "blocked_zones": list(blocked_zones),
        }

        return obs, reward, done, info

    def close(self) -> None:
        """Cleanup environment resources."""
        pass

    def state(self) -> Dict[str, Any]:
        """Return raw episode state dict (used by graders)."""
        return {
            "drones": [d.model_dump() for d in self._drones],
            "collisions": self._collisions,
            "step": self._step,
        }

    def episode_rewards(self) -> List[float]:
        """Return per-step reward history for the current episode."""
        return list(self._episode_rewards)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_manhattan_dist(self, z1: str, z2: str) -> int:
        r1, r2 = ord(z1[0]) - ord("A"), ord(z2[0]) - ord("A")
        c1, c2 = int(z1[1:]) - 1, int(z2[1:]) - 1
        return abs(r1 - r2) + abs(c1 - c2)

    def _build_observation(self) -> Observation:
        congestion: Dict[str, int] = defaultdict(int)
        for drone in self._drones:
            if not drone.delivered:
                congestion[drone.location] += 1

        return Observation(
            drones=list(self._drones),
            congestion_map=dict(congestion),
            step=self._step,
            collisions=self._collisions,
            graph_edges=self.graph,
        )

    def _validate_move(
        self, drone: DroneState, requested_zone: str, blocked_zones: set
    ) -> str:
        """Return the zone the drone actually moves to after constraint checks."""
        # Drone already delivered -> hover in place
        if drone.delivered:
            return drone.location

        # Battery dead -> cannot move
        if drone.battery <= 0.0:
            return drone.location

        # Requested zone must be adjacent or equal (hover)
        valid_moves = self.graph.get(drone.location, []) + [drone.location]
        if requested_zone not in valid_moves:
            return drone.location  # illegal move -> hover

        # Dynamic obstacle blocks the zone
        if requested_zone in blocked_zones:
            return drone.location  # blocked -> hover

        return requested_zone

    def _is_collision(self, zone: str, drone_ids: List[str]) -> bool:
        """A collision occurs when ->2 non-delivered drones share a zone."""
        if len(drone_ids) < 2:
            return False
        # Bottleneck zones allow only 1 drone
        if zone in self.bottleneck_zones:
            return len(drone_ids) >= 1 and len(drone_ids) > 1
        return len(drone_ids) >= 2

    def _get_blocked_zones(self, step: int) -> set:
        blocked = set()
        for start, end, zone in self.dynamic_obstacles:
            if start <= step <= end:
                blocked.add(zone)
        return blocked

    def _check_done(self) -> bool:
        all_delivered = all(d.delivered for d in self._drones)
        out_of_steps = self._step >= self.max_steps
        all_dead = all(d.battery <= 0.0 and not d.delivered for d in self._drones)
        return all_delivered or out_of_steps or all_dead
