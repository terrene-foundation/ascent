# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 3: RL Fundamentals with RLTrainer
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Understand MDPs, value functions, and Q-learning by
#   building a simple RL agent, then use RLTrainer for a real
#   environment.
#
# TASKS:
#   1. Define MDP for grid-world (states, actions, transitions, rewards)
#   2. Implement value iteration
#   3. Implement Q-learning from scratch
#   4. Compare with RLTrainer(algorithm="dqn")
#   5. Visualize learning curves
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import random

import polars as pl

from kailash_ml import ModelVisualizer
from kailash_ml.rl.trainer import RLTrainer, RLTrainingConfig, RLTrainingResult

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

# 4x4 grid: agent navigates (0,0) → (3,3). Actions: 0=up 1=right 2=down 3=left
GRID_SIZE = 4
GOAL = (3, 3)
OBSTACLES = [(1, 1), (2, 2)]
ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}


# ══════════════════════════════════════════════════════════════════════
# TASK 1: MDP transition and reward functions.
#   get_next_state: apply delta, stay in bounds, block on obstacles.
#   get_reward: +10 goal, -1 wall/obstacle (state unchanged), -0.1 step.
# ══════════════════════════════════════════════════════════════════════
def get_next_state(state: tuple[int, int], action: int) -> tuple[int, int]:
    ____


def get_reward(state: tuple[int, int], next_state: tuple[int, int]) -> float:
    ____


def get_all_states() -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if (r, c) not in OBSTACLES
    ]


states = get_all_states()
print(f"=== Grid-World MDP: {len(states)} states, {len(ACTIONS)} actions ===")

# ══════════════════════════════════════════════════════════════════════
# TASK 2: Value iteration — sweep until max delta < 1e-6 (gamma=0.99).
#   Bellman optimality: V(s) = max_a [R(s,a,s') + gamma*V(s')].
#   Extract greedy policy. Print iteration count, value grid, policy grid.
# ══════════════════════════════════════════════════════════════════════
____
____
____
____
____

# ══════════════════════════════════════════════════════════════════════
# TASK 3: Q-learning — 500 episodes, epsilon-greedy (start=0.3,
#   decay=0.995/ep, floor=0.01), alpha=0.1, gamma=0.99.
#   Q(s,a) += alpha*[r + gamma*max Q(s',·) - Q(s,a)].
#   Print final 50-ep avg reward and policy match count vs value iteration.
# ══════════════════════════════════════════════════════════════════════
____
____
____
____

# ══════════════════════════════════════════════════════════════════════
# TASK 4: Build RLTrainer() and RLTrainingConfig for DQN
#   (lr=3e-4, gamma=0.99, buffer_size=10000, batch_size=32,
#   total_timesteps=10000). Print config and note Gymnasium requirement.
# ══════════════════════════════════════════════════════════════════════
____
____

# ══════════════════════════════════════════════════════════════════════
# TASK 5: Build polars DataFrame from episode_rewards, add 20-ep rolling
#   mean, call ModelVisualizer().training_history(). Print method
#   comparison: Value Iteration / Q-Learning / RLTrainer DQN.
# ══════════════════════════════════════════════════════════════════════
____
____
____

print(
    "\n✓ Exercise 3 complete — RL fundamentals: MDP, value iteration, Q-learning, DQN"
)
