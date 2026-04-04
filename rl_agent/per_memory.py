"""
Prioritized Experience Replay Memory
=====================================
Ported from PEDRA (aqeelanwar/PEDRA) — network/Memory.py + network/SumTree.py
Original author: Aqeel Anwar (ICSRL, Georgia Tech)
Adapted for: Drone Traffic Control DDQN agent (PyTorch, drone grid env)

Changes from PEDRA original:
- Combined SumTree and Memory into one module
- Removed TensorFlow dependency
- Added type hints and docstrings
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# SumTree  (direct port of PEDRA network/SumTree.py)
# ---------------------------------------------------------------------------

class SumTree:
    """
    Binary tree where leaf values are transition priorities.
    Enables O(log n) sampling proportional to priority.
    """

    write: int = 0

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return self.tree[0]

    def add(self, p: float, data: object) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity

    def update(self, idx: int, p: float) -> None:
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# ---------------------------------------------------------------------------
# Prioritized Replay Memory (direct port of PEDRA network/Memory.py)
# ---------------------------------------------------------------------------

class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay buffer.

    Transitions with high TD-error are sampled more frequently,
    allowing the agent to learn more from surprising experiences.

    Reference: Schaul et al. 2016 — "Prioritized Experience Replay"
    PEDRA implementation: aqeelanwar/PEDRA/network/Memory.py
    """

    e: float = 0.01   # small constant to ensure all priorities > 0
    a: float = 0.6    # priority exponent (0 = uniform, 1 = full prioritisation)

    def __init__(self, capacity: int) -> None:
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error: float) -> float:
        return (abs(error) + self.e) ** self.a

    def add(self, error: float, sample: object) -> None:
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n: int) -> Tuple[List[object], List[int], List[float]]:
        batch, idxs, priorities = [], [], []
        segment = self.tree.total / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        return batch, idxs, priorities

    def update(self, idx: int, error: float) -> None:
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self) -> int:
        return min(self.tree.write + (self.capacity - 1), self.capacity)
