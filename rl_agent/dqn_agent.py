"""
DDQN Agent for Drone Traffic Control
===================================
A PyTorch re-implementation of PEDRA's DeepQLearning.py.
Uses Double DQN + Prioritized Experience Replay (via per_memory.py).
"""

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rl_agent.per_memory import PrioritizedReplayMemory
from environment.models import HOVER, Action, DroneAction, Observation


class QNetwork(nn.Module):
    """
    Standard Feed-Forward Q-Network, inspired by PEDRA's tf fully connected blocks.
    Takes flattened drone state -> outputs Q-values for all possible zones.
    """
    def __init__(self, input_dim: int, hidden_sizes: List[int], output_dim: int, activation_type: str = 'relu'):
        super().__init__()
        layers = []
        last_dim = input_dim

        # Mapping string to PyTorch activation
        act_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU()
        }
        act = act_map.get(activation_type.lower(), nn.ReLU())

        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(act)
            last_dim = h

        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DDQNAgent:
    """
    Double DQN Agent with Prioritized Experience Replay.
    PEDRA compatibility: Maintains separate Q and Target networks, uses PER.
    Action Space: Node IDs + Hover.
    Observation Space: Flattened features (loc, dest, battery, priority) per drone.
    
    Since this is multi-agent, we treat each drone as an independent decision maker
    sharing the same weights (Parameter Sharing MA-DQN).
    """
    
    def __init__(self, cfg: Dict, num_zones: int, zone_names: List[str], graph: Dict[str, List[str]] = None, task_cfg: Dict = None):
        # The agent needs to know the zone list to map indices to zone strings
        self.zone_names = zone_names
        self.graph = graph or {}
        self.task_cfg = task_cfg or {}
        
        # Determine I/O shapes
        # State: [Drone row, Drone col, Dest row, Dest col, Battery (0-1), Priority, Step/Max, 
        #         Congestion_Self, Congestion_Up, Congestion_Down, Congestion_Left, Congestion_Right] = 12 features
        self.state_dim = 12
        self.action_dim = num_zones + 1 # Include 'hover'
        self.HOVER_IDX = num_zones

        # Networking params from config
        net_cfg = cfg['network']
        hidden_sizes = [int(x.strip()) for x in net_cfg['hidden_sizes'].split(',')]
        activation = net_cfg['activation']
        
        # Check GPU
        use_cuda = cfg['general']['device'].lower() == 'cuda' and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        # Build Networks
        self.q_net = QNetwork(self.state_dim, hidden_sizes, self.action_dim, activation).to(self.device)
        self.target_net = QNetwork(self.state_dim, hidden_sizes, self.action_dim, activation).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        # DQ params
        dqn_cfg = cfg['dqn']
        self.gamma = float(dqn_cfg['gamma'])
        self.lr = float(dqn_cfg['learning_rate'])
        self.batch_size = int(dqn_cfg['batch_size'])
        self.update_target_interval = int(dqn_cfg['update_target_interval'])
        self.train_interval = int(dqn_cfg['train_interval'])
        self.wait_before_train = int(dqn_cfg['wait_before_train'])
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.memory = PrioritizedReplayMemory(int(dqn_cfg['buffer_len']))
        
        # Epsilon params
        self.epsilon = float(dqn_cfg['epsilon_start'])
        self.epsilon_start = self.epsilon
        self.epsilon_end = float(dqn_cfg['epsilon_end'])
        self.epsilon_decay_steps = int(dqn_cfg['epsilon_decay_steps'])
        self.eps_decay = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        
        self.step_count = 0
        
        # Tensorboard
        log_cfg = cfg['logging']
        self.writer = SummaryWriter(log_cfg['tensorboard_dir'])
        
        # Fast lookup map to go from zone ID (e.g. A1) to graph coordinates
        self.zone_coords = self._build_zone_coords()
        
    def _build_zone_coords(self) -> Dict[str, Tuple[int, int]]:
        # Helper to convert "A1" -> (0, 0), "C2" -> (2, 1) based on pure string parsing
        coords = {}
        for z in self.zone_names:
            if z == HOVER:
                continue
            char_p = ord(z[0].upper()) - 65 # A=0, B=1...
            num_p = int(z[1:]) - 1 # 1=0, 2=1...
            coords[z] = (char_p, num_p)
        return coords

    def _extract_drone_state(self, obs: Observation, drone_idx: int, step: int = 0) -> np.ndarray:
        """Flattens a specific drone's state into a vector."""
        d = obs.drones[drone_idx]
        
        r1, c1 = self.zone_coords.get(d.location, (0,0))
        r2, c2 = self.zone_coords.get(d.destination, (0,0))
        
        # Grid dimensions for normalization
        rows = self.task_cfg.get("rows", 3)
        cols = self.task_cfg.get("cols", 3)
        
        # Normalize coordinates to [0, 1]
        nr1, nc1 = r1 / max(1, rows - 1), c1 / max(1, cols - 1)
        nr2, nc2 = r2 / max(1, rows - 1), c2 / max(1, cols - 1)
        
        bat = max(0.0, d.battery / 100.0)
        pri = d.priority
        
        # Calculate step progression (0.0 to 1.0)
        max_steps = self.task_cfg.get("max_steps", 30)
        step_prog = min(1.0, step / max_steps)
        
        # Local Congestion (Coordination Senses)
        row_l = d.location[0]
        col_l = int(d.location[1:])
        
        neighbors = [
            d.location,                       # Center
            f"{chr(ord(row_l)-1)}{col_l}",    # Up
            f"{chr(ord(row_l)+1)}{col_l}",    # Down
            f"{row_l}{col_l-1}",              # Left
            f"{row_l}{col_l+1}"               # Right
        ]
        
        # Extract congestion and normalize (max drones in grid ~5-10)
        congestion_features = [obs.congestion_map.get(n, 0) / 10.0 for n in neighbors]
        
        base_features = [nr1, nc1, nr2, nc2, bat, pri, step_prog]
        return np.array(base_features + congestion_features, dtype=np.float32)

    def select_action(self, obs: Observation, training: bool = True, step: int = 0) -> Action:
        """
        Takes environment observation, issues joint action for all drones.
        Uses PEDRA's greedy vs random exploration toggle.
        """
        drone_actions = []
        
        self.q_net.eval()
        
        for i, d in enumerate(obs.drones):
            
            # Dead drones must hover
            if d.battery <= 0.0 or d.delivered:
                drone_actions.append(DroneAction(drone_id=d.id, move_to=HOVER))
                continue
                
            state = self._extract_drone_state(obs, i, step)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if training and random.random() < self.epsilon:
                # Random VALID action (adjacent or hover)
                valid_moves = self.graph.get(d.location, []) + [HOVER]
                move = random.choice(valid_moves)
            else:
                with torch.no_grad():
                    q_values = self.q_net(state_t)
                    action_idx = q_values.argmax(dim=1).item()
                    
                # Convert action_idx back to zone name
                if action_idx == self.HOVER_IDX:
                    move = HOVER
                else:
                    move = self.zone_names[action_idx]
                
            drone_actions.append(DroneAction(drone_id=d.id, move_to=move))
            
        self.q_net.train()
        
        return Action(actions=drone_actions)

    def store_transition(self, 
                         obs: Observation, 
                         action: Action, 
                         reward_val: float, 
                         next_obs: Observation, 
                         done: bool,
                         step: int = 0):
        """
        Stores INDIVIDUAL drone transitions into the PER buffer.
        """
        for i, d in enumerate(obs.drones):
            # Don't store if already dead/delivered before step began
            if d.battery <= 0.0 or d.delivered:
                continue
                
            state = self._extract_drone_state(obs, i, step)
            next_state = self._extract_drone_state(next_obs, i, step + 1)
            
            # Find action index for this drone
            act_str = next(a.move_to for a in action.actions if a.drone_id == d.id)
            if act_str == HOVER:
                act_idx = self.HOVER_IDX
            else:
                # Fallback to hover if action was invalid somehow
                act_idx = self.zone_names.index(act_str) if act_str in self.zone_names else self.HOVER_IDX
                
            # Compute initial TD Error for PER priority
            td_error = self._compute_td_error(state, act_idx, reward_val, next_state, done)
            
            self.memory.add(td_error, (state, act_idx, reward_val, next_state, done))

    def _compute_td_error(self, state, action, reward, next_state, done) -> float:
        """Helper to get Priority (absolute TD error) for a single sample."""
        self.q_net.eval()
        self.target_net.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            sn = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            q = self.q_net(s)[0, action].item()
            
            if done:
                q_target = reward
            else:
                # Double DQN logic
                next_act = self.q_net(sn).argmax(dim=1).item()
                q_target = reward + self.gamma * self.target_net(sn)[0, next_act].item()
                
        self.q_net.train()
        return abs(q_target - q)

    def train_step(self):
        """
        Samples a batch from PER, performs backprop, updates priorities.
        PEDRA equivalent: DeepQLearning.py minibatch_double().
        """
        self.step_count += 1
        
        # Decay Epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.eps_decay
            
        if len(self.memory) < self.wait_before_train:
            return
            
        if self.step_count % self.train_interval != 0:
            return
            
        # Sample PER
        batch, idxs, _ = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions = torch.LongTensor([b[1] for b in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        dones = torch.FloatTensor([float(b[4]) for b in batch]).unsqueeze(1).to(self.device)
        
        # Double DQN loss calculation
        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)
        
        # Target = r + gamma * Q_target(next_s, argmax_a Q(next_s, a))
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_q_target_values = self.target_net(next_states).gather(1, next_actions)
            expected_q_values = rewards + self.gamma * next_q_target_values * (1.0 - dones)
            
        # Update PER priorities based on new TD Errors
        td_errors = torch.abs(expected_q_values - q_values).detach().cpu().numpy()
        for i in range(self.batch_size):
            self.memory.update(idxs[i], td_errors[i][0])
            
        # Loss & Optimize
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Clip max gradients (similar to PEDRA Q_clip handling)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Tensorboard Logging
        if self.step_count % 100 == 0:
            self.writer.add_scalar("Train/Loss", loss.item(), self.step_count)
            self.writer.add_scalar("Train/Epsilon", self.epsilon, self.step_count)
            
        # Target network update
        if self.step_count % self.update_target_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, filepath: str):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon
        }, filepath)
        
    def load(self, filepath: str):
        if not os.path.exists(filepath):
            print(f"[Warn] Checkpoint {filepath} not found. Starting fresh.")
            return
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        try:
            self.q_net.load_state_dict(checkpoint['q_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except (RuntimeError, ValueError) as e:
            print(f"Error loading agent: {e}")
            raise  # Rethrow so the caller knows it failed
        
        self.step_count = checkpoint['step_count']
        self.epsilon = checkpoint['epsilon']
        print(f"Loaded checkpoint from {filepath}")
