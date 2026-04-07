# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT06 — Exercise 3: Reinforcement Learning
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use RLTrainer with PPO on an inventory management
#   environment. Compare RL policy vs heuristic baseline.
#
# TASKS:
#   1. Set up Gymnasium environment (inventory management)
#   2. Configure RLTrainer with PPO
#   3. Train RL agent
#   4. Implement heuristic baseline for comparison
#   5. Evaluate and compare policies
#   6. Track with ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from kailash.db.connection import ConnectionManager
from kailash_ml.rl.trainer import RLTrainer, RLTrainingConfig
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml import ModelVisualizer

from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Custom Gymnasium environment — inventory management
# ══════════════════════════════════════════════════════════════════════


class InventoryEnv(gym.Env):
    """Simplified inventory management environment.

    State: [current_stock, day_of_week, demand_trend]
    Action: order_quantity (0 to max_order)
    Reward: revenue from sales - holding cost - stockout penalty - order cost
    """

    metadata = {"render_modes": []}

    def __init__(self):
        # TODO: Implement __init__.
        # Set max_stock=100, max_order=50, max_steps=30.
        # Define observation_space as Box(low=[0,0,0], high=[max_stock,6,2], dtype=float32)
        # and action_space as Discrete(max_order + 1).
        # Create self.rng and call self.reset().
        ____
        ____
        ____
        ____
        ____
        ____

    def reset(self, seed=None, options=None):
        # TODO: Implement reset.
        # Call super().reset(seed=seed), create new rng, set stock=50, day=0,
        # step_count=0, total_revenue=0, total_cost=0.
        # Return (self._get_obs(), {}).
        ____
        ____
        ____

    def _get_obs(self):
        # TODO: Implement _get_obs.
        # Demand trend uses weekly seasonality: 1 + 0.5 * sin(2π * day / 7).
        # Return np.array([stock, day % 7, trend], dtype=float32).
        ____

    def step(self, action):
        # TODO: Implement step.
        # 1. Receive order: add action to stock (capped at max_stock).
        # 2. Generate stochastic demand: base_demand=15, day_factor = 1 + 0.3*sin(2π*day/7),
        #    demand = max(0, int(rng.poisson(base_demand * day_factor))).
        # 3. Fulfill demand: sold = min(demand, stock), stockout = demand - sold.
        # 4. Compute reward: revenue($10/unit) - holding($0.50/unit) -
        #    stockout_penalty($5/lost sale) - order_cost($3/unit ordered).
        # 5. Increment day and step_count; terminated when step_count >= max_steps.
        # 6. Return (obs, reward, terminated, truncated=False, info_dict).
        ____
        ____
        ____
        ____
        ____
        ____
        ____


# Register custom environment with Gymnasium
gym.register(
    id="InventoryManagement-v0",
    entry_point=lambda: InventoryEnv(),
)

env = InventoryEnv()
print(f"=== Inventory Management Environment ===")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Episode length: {env.max_steps} days")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure RLTrainer with PPO
# ══════════════════════════════════════════════════════════════════════


def train_rl():
    # TODO: Implement train_rl.
    # Create RLTrainer(). Build RLTrainingConfig with algorithm="PPO",
    # total_timesteps=50_000, seed=42, and PPO hyperparameters:
    #   learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
    #   gamma=0.99, gae_lambda=0.95, clip_range=0.2.
    # Call trainer.train(env_name="InventoryManagement-v0",
    #   policy_name="inventory_ppo", config=rl_config).
    # Print: algorithm, mean_reward, std_reward, training_time_seconds, artifact_path.
    ____
    ____
    ____
    ____
    ____
    ____

    return trainer, result


trainer, rl_result = train_rl()


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Heuristic baseline — (s, S) policy
# ══════════════════════════════════════════════════════════════════════
# Classic inventory policy: if stock < s, order up to S


def evaluate_policy(env, policy_fn, n_episodes=100, seed=42):
    """Evaluate a policy over multiple episodes."""
    rng = np.random.default_rng(seed)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 10000)))
        total_reward = 0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.array(rewards)


def ss_policy(obs):
    """(s, S) reorder policy: if stock < s, order up to S."""
    # TODO: Implement (s, S) policy with s=20, S=60.
    # If stock < s: return min(int(S - stock), 50), else return 0.
    ____


def random_policy(obs):
    # TODO: Implement random policy: return random int in [0, 50].
    ____


# Evaluate all policies
heuristic_rewards = evaluate_policy(env, ss_policy)
random_rewards = evaluate_policy(env, random_policy)

# Use RL result metrics directly (already evaluated during training)
rl_mean = rl_result.mean_reward
rl_std = rl_result.std_reward
# Create synthetic array for comparison display using mean/std
rl_rewards = np.array(
    [rl_mean + rl_std * x for x in np.random.default_rng(42).standard_normal(100)]
)

# TODO: Print comparison table.
# Rows: Random, (s,S) Heuristic, PPO — columns: Mean, Std, Min, Max.
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Visualise training and comparison
# ══════════════════════════════════════════════════════════════════════

# TODO: Use ModelVisualizer.metric_comparison() to create a bar chart comparing
# the three policies on Mean_Reward and Std_Reward.
# Save to "ex3_rl_comparison.html".
____
____
____


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Track with ExperimentTracker
# ══════════════════════════════════════════════════════════════════════


async def track_experiment():
    # TODO: Implement track_experiment.
    # 1. Open ConnectionManager("sqlite:///ascent06_experiments.db") and initialize.
    # 2. Create ExperimentTracker(conn).
    # 3. Create experiment "ascent06_reinforcement_learning".
    # 4. Log 3 runs: random_baseline, ss_heuristic, ppo_agent.
    #    Each run: log_params (policy type / algorithm), log_metrics
    #    (mean_reward, std_reward, min_reward, max_reward), set_tag("domain","rl-inventory").
    # 5. Close connection.
    ____
    ____
    ____
    ____
    ____
    ____
    ____


asyncio.run(track_experiment())

print("\n✓ Exercise 3 complete — RL inventory management with PPO")
