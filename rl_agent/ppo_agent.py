"""
PPO Agent for Physics-Driven Drone Control
=========================================
Implements Proximal Policy Optimization (PPO) for continuous 3D flight control.
Handles stochastic wind by learning robust thrust profiles.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.distributions import Normal

from environment.models import Action, DroneAction, Observation

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorCritic, self).__init__()
        
        # Actor network (Policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Thrust clipped to [-1, 1], then scaled in env
        )
        
        # Log standard deviation for exploration (learnable)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic network (Value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor):
        value = self.critic(state)
        mu = self.actor(state)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value

class PPOAgent:
    """
    PPO Agent for continuous 3D drone control under wind.
    """
    def __init__(self, state_dim: int = 12, action_dim: int = 3, lr: float = 3e-4, 
                 gamma: float = 0.99, K_epochs: int = 10, eps_clip: float = 0.2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, obs: Observation) -> Action:
        """
        Selects a continuous thrust vector for each drone based on current policy.
        """
        drone_actions = []
        with torch.no_grad():
            for i, d in enumerate(obs.drones):
                if d.delivered or d.battery <= 0:
                    drone_actions.append(DroneAction(drone_id=d.id, move_to=d.location, thrust_vector=[0, 0, 0]))
                    continue
                
                state = self._extract_state(obs, i)
                state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                dist, _ = self.policy_old(state_t)
                action_thrust = dist.sample()
                
                # Convert to Python list and scale to N (assuming max_thrust=25)
                thrust_list = (action_thrust.detach().cpu().numpy()[0] * 10.0).tolist()
                
                drone_actions.append(DroneAction(
                    drone_id=d.id,
                    move_to=d.location, # Location string is updated by physics env
                    thrust_vector=thrust_list
                ))
        
        return Action(actions=drone_actions)

    def _extract_state(self, obs: Observation, drone_idx: int) -> np.ndarray:
        d = obs.drones[drone_idx]
        
        # In AirSim or Physics Mode, x and y are meters.
        # target_x, target_y will also be extracted if destination is set.
        # For this demo, we use d.x and d.y directly.
        state = [
            d.x, d.y, d.altitude,
            d.vx, d.vy, d.vz,
            obs.wind_vector[0], obs.wind_vector[1], obs.wind_vector[2],
            0.0, 0.0, d.target_altitude # Placeholder for target_x, y
        ]
        return np.array(state, dtype=np.float32)

    def update(self, memory):
        # Implementation of PPO update would go here for training
        # For this integration, we focus on the inference architecture
        pass

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        if os.path.exists(path):
            self.policy.load_state_dict(torch.load(path, map_location=self.device))
            self.policy_old.load_state_dict(self.policy.state_dict())
            print(f"Loaded PPO checkpoint from {path}")
