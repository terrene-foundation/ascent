# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 4: PPO Training for Inventory Management
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Train a PPO agent for inventory management — clipped
#   objective, GAE, reward shaping — using RLTrainer.
#
# TASKS:
#   1. Define inventory environment (state: stock, demand; action: order qty)
#   2. Configure RLTrainer with PPO
#   3. Implement reward function with penalties
#   4. Train and compare vs heuristic baseline
#   5. Analyze policy behavior across demand scenarios
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import random

import polars as pl

from kailash_ml import ModelVisualizer
from kailash_ml.rl.trainer import RLTrainer, RLTrainingConfig, RLTrainingResult

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

loader = ASCENTDataLoader()
demand_data = loader.load("ascent10", "inventory_demand.parquet")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Complete InventoryEnv.step() — apply order (clip to max_order,
#   cap stock at max_stock), sample demand from pattern + N(0,3), compute
#   reward = revenue - holding - stockout_penalty - order_cost, advance
#   self.day. Return (next_state, reward, done).
# ══════════════════════════════════════════════════════════════════════
class InventoryEnv:
    """State: (stock_ratio, day_of_week, avg_demand). Action: order qty."""

    def __init__(
        self,
        demand_pattern: list[float],
        max_stock: int = 100,
        max_order: int = 50,
        holding_cost: float = 0.5,
        stockout_penalty: float = 5.0,
        order_cost: float = 1.0,
        unit_price: float = 10.0,
    ):
        self.demand_pattern = demand_pattern
        self.max_stock = max_stock
        self.max_order = max_order
        self.holding_cost = holding_cost
        self.stockout_penalty = stockout_penalty
        self.order_cost = order_cost
        self.unit_price = unit_price
        self.stock = max_stock // 2
        self.day = 0
        self.total_reward = 0.0

    def reset(self) -> list[float]:
        self.stock = self.max_stock // 2
        self.day = 0
        self.total_reward = 0.0
        return self._get_state()

    def _get_state(self) -> list[float]:
        avg_demand = sum(self.demand_pattern[max(0, self.day - 7) : self.day + 1]) / 7
        return [self.stock / self.max_stock, (self.day % 7) / 6.0, avg_demand / 50.0]

    def step(self, action: int) -> tuple[list[float], float, bool]:
        ____


demand_values = demand_data["demand"].to_list()[:90]
env = InventoryEnv(demand_pattern=demand_values)
print(f"=== Inventory Env: max_stock={env.max_stock}, max_order={env.max_order} ===")

# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build RLTrainingConfig for PPO (lr=3e-4, gamma=0.99,
#   clip_range=0.2, gae_lambda=0.95, batch_size=64, timesteps=50000).
#   Print key params and the PPO clipped-objective formula.
# ══════════════════════════════════════════════════════════════════════
____
____

# ══════════════════════════════════════════════════════════════════════
# TASK 3: Print reward shaping breakdown — revenue, holding cost,
#   stockout penalty, order cost — and explain the newsvendor trade-off.
# ══════════════════════════════════════════════════════════════════════
____


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate heuristic_policy over 10 episodes, compute avg reward.
#   Implement improved_policy (demand-adaptive reorder), evaluate 10 eps,
#   compute improvement %. Print comparison.
# ══════════════════════════════════════════════════════════════════════
def heuristic_policy(state: list[float], target_ratio: float = 0.6) -> int:
    """Simple (s, S) policy: order up to target when stock below 40% of target."""
    current_stock = int(state[0] * env.max_stock)
    target_stock = int(target_ratio * env.max_stock)
    if current_stock < target_stock * 0.4:
        return min(target_stock - current_stock, env.max_order)
    return 0


____
____
____
____
____

# ══════════════════════════════════════════════════════════════════════
# TASK 5: Analyze policy behavior — for stock ratios [0.1, 0.3, 0.5,
#   0.7, 0.9] print what improved_policy orders. Explain the pattern.
# ══════════════════════════════════════════════════════════════════════
____
____

print("\n✓ Exercise 4 complete — PPO inventory management vs heuristic baseline")
