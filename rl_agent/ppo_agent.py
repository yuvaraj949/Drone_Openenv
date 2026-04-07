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
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
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
    def __init__(self, state_dim: int = 36, action_dim: int = 3, lr: float = 3e-4, 
                 gamma: float = 0.99, K_epochs: int = 20, eps_clip: float = 0.2):
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
        pos = np.array([d.x, d.y, d.altitude])
        vel = np.array([d.vx, d.vy, d.vz])
        
        # 1. Self State (7)
        # Placeholder for target (ideally passed from environment/task)
        target = np.array([3.0, 3.0, 0.0]) if d.destination == "C3" else np.array([0.0, 0.0, 0.0])
        rel_target = target - pos
        self_state = rel_target.tolist() + vel.tolist() + [d.battery / 100.0]
        
        # 2. Wind State (3)
        wind_state = obs.wind_vector
        
        # 3. Neighbor State (K=3 nearest within radius) (18)
        neighbors = []
        for i, other in enumerate(obs.drones):
            if i == drone_idx or other.delivered or other.battery <= 0:
                continue
            other_pos = np.array([other.x, other.y, other.altitude])
            dist = np.linalg.norm(pos - other_pos)
            if dist < obs.sensing_radius:
                rel_pos = other_pos - pos
                rel_vel = np.array([other.vx, other.vy, other.vz]) - vel
                neighbors.append((dist, rel_pos, rel_vel))
        
        neighbors.sort(key=lambda x: x[0])
        neighbor_feats = []
        for i in range(3):
            if i < len(neighbors):
                neighbor_feats.extend(neighbors[i][1].tolist())
                neighbor_feats.extend(neighbors[i][2].tolist())
            else:
                neighbor_feats.extend([0.0]*6)
                
        # 4. Obstacle State (O=2 nearest) (8)
        obstacles = []
        for ob in obs.stationary_obstacles:
            ob_pos = np.array([ob.x, ob.y, ob.z])
            dist = np.linalg.norm(pos - ob_pos)
            obstacles.append((dist, ob_pos - pos, ob.radius))
            
        obstacles.sort(key=lambda x: x[0])
        obstacle_feats = []
        for i in range(2):
            if i < len(obstacles):
                obstacle_feats.extend(obstacles[i][1].tolist())
                obstacle_feats.append(obstacles[i][2])
            else:
                obstacle_feats.extend([0.0]*4)
                
        full_state = self_state + wind_state + neighbor_feats + obstacle_feats
        return np.array(full_state, dtype=np.float32)

    def update(self, states, actions, logprobs, rewards, dones):
        states = torch.stack(states).to(self.device).detach()
        actions = torch.stack(actions).to(self.device).detach()
        logprobs = torch.stack(logprobs).to(self.device).detach()
        
        # Monte Carlo estimate of state rewards
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        for _ in range(self.K_epochs):
            dist, state_values = self.policy(states)
            entropy = dist.entropy().sum(dim=-1)
            new_logprobs = dist.log_prob(actions).sum(dim=-1)
            
            # Policy Loss
            ratios = torch.exp(new_logprobs - logprobs)
            advantages = returns - state_values.detach().squeeze()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss
            critic_loss = self.MseLoss(state_values.squeeze(), returns)
            
            # Total Loss
            loss = policy_loss + 0.5*critic_loss - 0.01*entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        if os.path.exists(path):
            self.policy.load_state_dict(torch.load(path, map_location=self.device))
            self.policy_old.load_state_dict(self.policy.state_dict())
            print(f"Loaded PPO checkpoint from {path}")
