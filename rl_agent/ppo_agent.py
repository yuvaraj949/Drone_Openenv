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

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

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

    def select_action(self, obs: Observation, memory: Optional[Memory] = None) -> Action:
        """
        Selects a continuous thrust vector for each drone based on current policy.
        """
        drone_actions = []
        for i, d in enumerate(obs.drones):
            if d.delivered or d.battery <= 0:
                drone_actions.append(DroneAction(drone_id=d.id, move_to=d.location, thrust_vector=[0, 0, 0]))
                continue
            
            state = self._extract_state(obs, i)
            state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                dist, _ = self.policy_old(state_t)
                action_thrust = dist.sample()
                action_logprob = dist.log_prob(action_thrust).sum(dim=-1)
            
            if memory is not None:
                memory.states.append(state_t)
                memory.actions.append(action_thrust)
                memory.logprobs.append(action_logprob)
            
            # Scale to Newtons (assuming max_thrust=25)
            # action_thrust is in [-1, 1] range due to Tanh
            thrust_list = (action_thrust.cpu().numpy()[0] * 10.0).tolist()
            
            drone_actions.append(DroneAction(
                drone_id=d.id,
                move_to=d.location,
                thrust_vector=thrust_list
            ))
        
        return Action(actions=drone_actions)

    def _extract_state(self, obs: Observation, drone_idx: int) -> np.ndarray:
        d = obs.drones[drone_idx]
        
        # Target coordinates from destination zone
        # Hardcoding A1 origin for simplicity in mapping
        target_r = ord(d.destination[0]) - ord("A")
        target_c = int(d.destination[1:]) - 1
        
        state = [
            d.x, d.y, d.altitude,
            d.vx, d.vy, d.vz,
            obs.wind_vector[0], obs.wind_vector[1], obs.wind_vector[2],
            float(target_c), float(target_r), d.target_altitude
        ]
        return np.array(state, dtype=np.float32)

    def update(self, memory: Memory):
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states), 1).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(memory.actions), 1).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            dist, state_values = self.policy(old_states)
            
            # Finding the log probability of the old actions in the new distribution
            logprobs = dist.log_prob(old_actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            
            # Computing the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Computing Surrogate Loss
            advantages = rewards - state_values.detach().squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), rewards) - 0.01 * dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        if os.path.exists(path):
            self.policy.load_state_dict(torch.load(path, map_location=self.device))
            self.policy_old.load_state_dict(self.policy.state_dict())
            print(f"Loaded PPO checkpoint from {path}")
