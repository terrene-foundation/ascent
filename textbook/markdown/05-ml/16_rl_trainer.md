# Chapter 16: RLTrainer

## Overview

`RLTrainer` brings reinforcement learning into the Kailash ML lifecycle. It wraps Stable-Baselines3 (SB3) algorithms with two registries -- `PolicyRegistry` for trained policy artifacts and `EnvironmentRegistry` for Gymnasium environment specs -- plus a structured training config and result type. This chapter covers training configuration, policy management, environment registration, training execution, evaluation, and the supported algorithm set.

## Prerequisites

- Python 3.10+ installed
- Kailash ML installed (`pip install kailash-ml`)
- Optional: `pip install kailash-ml[rl]` for live training (requires stable-baselines3 and gymnasium)
- Completed [Chapter 15: ML Agents](15_ml_agents.md)
- Basic understanding of reinforcement learning concepts (agent, environment, policy, reward)

## Concepts

### Concept 1: RLTrainingConfig

`RLTrainingConfig` defines everything needed for a training run: algorithm (PPO, DQN, SAC, A2C, TD3, DDPG), policy type (MlpPolicy, CnnPolicy), total timesteps, evaluation frequency, hyperparameters, seed, and save path. Defaults are sensible for getting started (PPO, MlpPolicy, 100K timesteps).

- **What**: A configuration object that fully specifies a reinforcement learning training run
- **Why**: Separates training configuration from execution, enabling reproducible experiments
- **How**: `RLTrainingConfig(algorithm="PPO", total_timesteps=50000, hyperparameters={"learning_rate": 0.001})`
- **When**: Before every `trainer.train()` call -- one config per experiment

### Concept 2: PolicyRegistry

`PolicyRegistry` manages policy specifications (algorithm, hyperparameters, description) and trained policy versions (artifact paths, reward statistics, timestep counts). It tracks the evolution of a policy across training runs, similar to how ModelRegistry tracks supervised learning models.

- **What**: A versioned registry for RL policy definitions and trained artifacts
- **Why**: RL training is iterative; PolicyRegistry tracks which configuration produced which reward
- **How**: `register_spec()` stores the policy definition; `register_version()` stores a trained artifact with reward stats
- **When**: Before training (register the spec) and after training (register the version)

### Concept 3: EnvironmentRegistry

`EnvironmentRegistry` manages Gymnasium environment specifications. Each `EnvironmentSpec` defines the environment name, entry point, kwargs, max episode steps, and reward threshold. Registration wraps Gymnasium's native registration system with additional metadata.

- **What**: A registry for Gymnasium environment definitions
- **Why**: Custom environments need consistent registration across training runs and evaluation
- **How**: `EnvironmentSpec(name="MyEnv-v0", entry_point="module:Class", ...)` defines the env; `env_reg.register(spec)` registers it
- **When**: When using custom environments; standard Gymnasium envs (CartPole, etc.) are available without registration

### Concept 4: Training and Evaluation

`trainer.train()` creates the environment, instantiates the SB3 algorithm, trains for the configured timesteps, evaluates the policy, saves the artifact, and registers the version in PolicyRegistry. `load_and_evaluate()` loads a trained policy and evaluates it on an environment without further training.

- **What**: End-to-end training and evaluation pipeline
- **Why**: Wraps the multi-step SB3 workflow (create env, create model, train, evaluate, save) in a single call
- **How**: `trainer.train(env_name="CartPole-v1", policy_name="my_policy", config=config)` returns `RLTrainingResult`
- **When**: For every RL training experiment

## Key API

| Class / Method          | Parameters                                                            | Returns               | Description                                  |
| ----------------------- | --------------------------------------------------------------------- | --------------------- | -------------------------------------------- |
| `RLTrainingConfig()`    | `algorithm`, `policy_type`, `total_timesteps`, `hyperparameters`, ... | `RLTrainingConfig`    | Training configuration                       |
| `PolicyRegistry()`      | `root_dir: str`                                                       | `PolicyRegistry`      | Manage policy specs and versions             |
| `register_spec()`       | `spec: PolicySpec`                                                    | `None`                | Register a policy definition                 |
| `register_version()`    | `version: PolicyVersion`                                              | `None`                | Register a trained policy artifact           |
| `get_latest_version()`  | `name: str`                                                           | `PolicyVersion`       | Get the best (latest) trained version        |
| `EnvironmentRegistry()` | --                                                                    | `EnvironmentRegistry` | Manage Gymnasium environment specs           |
| `register()`            | `spec: EnvironmentSpec`                                               | `None`                | Register an environment                      |
| `RLTrainer()`           | `env_registry`, `policy_registry`, `root_dir`                         | `RLTrainer`           | Create the trainer                           |
| `trainer.train()`       | `env_name`, `policy_name`, `config`                                   | `RLTrainingResult`    | Train a policy                               |
| `load_and_evaluate()`   | `policy_name`, `env_name`, `n_episodes`                               | `(float, float)`      | Evaluate a trained policy (mean_reward, std) |

## Code Walkthrough

```python
from __future__ import annotations

import tempfile
from pathlib import Path

from kailash_ml.rl.env_registry import EnvironmentRegistry, EnvironmentSpec
from kailash_ml.rl.policy_registry import PolicyRegistry, PolicySpec, PolicyVersion
from kailash_ml.rl.trainer import RLTrainer, RLTrainingConfig, RLTrainingResult

# ════════════════════════════════════════════════════════════════════
# Part A: RLTrainingConfig
# ════════════════════════════════════════════════════════════════════

# ── 1. RLTrainingConfig — defaults and customization ────────────────

config = RLTrainingConfig()
assert config.algorithm == "PPO"
assert config.policy_type == "MlpPolicy"
assert config.total_timesteps == 100_000
assert config.n_eval_episodes == 10
assert config.seed == 42

# Custom config
custom_config = RLTrainingConfig(
    algorithm="DQN",
    policy_type="MlpPolicy",
    total_timesteps=50_000,
    hyperparameters={"learning_rate": 0.001, "buffer_size": 10000},
    n_eval_episodes=5,
    seed=123,
)

assert custom_config.algorithm == "DQN"
assert custom_config.hyperparameters["learning_rate"] == 0.001

# Serialization
cfg_dict = custom_config.to_dict()
assert cfg_dict["algorithm"] == "DQN"
assert cfg_dict["total_timesteps"] == 50_000

# ════════════════════════════════════════════════════════════════════
# Part B: PolicyRegistry
# ════════════════════════════════════════════════════════════════════

# ── 2. PolicyRegistry — register specs and versions ─────────────────

policy_reg = PolicyRegistry(root_dir=tempfile.mkdtemp())

# Supported algorithms
supported = PolicyRegistry.supported_algorithms()
assert "PPO" in supported
assert "SAC" in supported
assert "DQN" in supported
assert "A2C" in supported
assert "TD3" in supported
assert "DDPG" in supported

# ── 3. Register a policy spec ───────────────────────────────────────

ppo_spec = PolicySpec(
    name="cartpole_ppo",
    algorithm="PPO",
    policy_type="MlpPolicy",
    hyperparameters={"learning_rate": 0.0003},
    description="PPO agent for CartPole-v1",
)

policy_reg.register_spec(ppo_spec)
assert "cartpole_ppo" in policy_reg
assert len(policy_reg) == 1

retrieved_spec = policy_reg.get_spec("cartpole_ppo")
assert retrieved_spec.algorithm == "PPO"

# ── 4. Register policy versions (trained artifacts) ─────────────────

v1 = PolicyVersion(
    name="cartpole_ppo",
    version=1,
    algorithm="PPO",
    artifact_path="/tmp/models/cartpole_ppo_v1",
    mean_reward=200.0,
    std_reward=15.0,
    total_timesteps=50_000,
    metadata={"learning_rate": 0.0003},
)

policy_reg.register_version(v1)

v2 = PolicyVersion(
    name="cartpole_ppo",
    version=2,
    algorithm="PPO",
    artifact_path="/tmp/models/cartpole_ppo_v2",
    mean_reward=450.0,
    std_reward=10.0,
    total_timesteps=100_000,
    metadata={"learning_rate": 0.0001},
)

policy_reg.register_version(v2)

# List and retrieve versions
versions = policy_reg.list_versions("cartpole_ppo")
assert len(versions) == 2

latest = policy_reg.get_latest_version("cartpole_ppo")
assert latest.version == 2
assert latest.mean_reward == 450.0

specific = policy_reg.get_version("cartpole_ppo", 1)
assert specific.mean_reward == 200.0

missing = policy_reg.get_version("cartpole_ppo", 99)
assert missing is None

# ── 5. Edge case: invalid algorithm ─────────────────────────────────

try:
    policy_reg.register_spec(PolicySpec(name="bad", algorithm="INVALID"))
except ValueError as e:
    assert "unknown algorithm" in str(e).lower()

# ════════════════════════════════════════════════════════════════════
# Part C: EnvironmentRegistry
# ════════════════════════════════════════════════════════════════════

# ── 6. EnvironmentSpec — environment configuration ──────────────────

env_spec = EnvironmentSpec(
    name="CustomGrid-v0",
    entry_point="gymnasium.envs.classic_control:CartPoleEnv",
    kwargs={},
    max_episode_steps=500,
    reward_threshold=475.0,
    description="Custom grid environment for testing",
)

assert env_spec.name == "CustomGrid-v0"
assert env_spec.max_episode_steps == 500

spec_dict = env_spec.to_dict()
assert spec_dict["name"] == "CustomGrid-v0"

# ── 7. EnvironmentRegistry — manages environments ──────────────────

try:
    import gymnasium

    env_reg = EnvironmentRegistry()
    env_reg.register(env_spec)
    assert "CustomGrid-v0" in env_reg
    assert len(env_reg) == 1

    envs = env_reg.list_environments()
    assert envs[0].name == "CustomGrid-v0"

    gymnasium_available = True
except ImportError:
    gymnasium_available = False

# ════════════════════════════════════════════════════════════════════
# Part D: RLTrainer
# ════════════════════════════════════════════════════════════════════

# ── 8. RLTrainer — instantiation ────────────────────────────────────

with tempfile.TemporaryDirectory() as rl_dir:
    trainer = RLTrainer(
        env_registry=env_reg if gymnasium_available else None,
        policy_registry=policy_reg,
        root_dir=rl_dir,
    )

    algos = RLTrainer.supported_algorithms()
    assert "PPO" in algos

    # ── 9. RLTrainer.train() — requires SB3 ────────────────────────
    # Full training requires stable-baselines3 and gymnasium.

    try:
        import stable_baselines3
        sb3_available = True
    except ImportError:
        sb3_available = False

    if sb3_available and gymnasium_available:
        train_config = RLTrainingConfig(
            algorithm="PPO",
            policy_type="MlpPolicy",
            total_timesteps=256,  # Minimal for tutorial
            n_eval_episodes=2,
            seed=42,
            verbose=0,
            save_path=Path(rl_dir) / "test_policy",
        )

        result = trainer.train(
            env_name="CartPole-v1",
            policy_name="tutorial_cartpole",
            config=train_config,
        )

        assert isinstance(result, RLTrainingResult)
        assert result.policy_name == "tutorial_cartpole"
        assert result.algorithm == "PPO"
        assert isinstance(result.mean_reward, float)
        assert result.training_time_seconds > 0

        # ── 10. Evaluate a trained model ────────────────────────

        mean_r, std_r = trainer.load_and_evaluate(
            "tutorial_cartpole",
            env_name="CartPole-v1",
            n_episodes=2,
        )
        assert isinstance(mean_r, float)

# ── 11. RLTrainingResult — standalone creation ──────────────────────

manual_result = RLTrainingResult(
    policy_name="demo_policy",
    algorithm="SAC",
    total_timesteps=10_000,
    mean_reward=300.0,
    std_reward=20.0,
    training_time_seconds=45.2,
    artifact_path="/tmp/demo_policy/model",
)

assert manual_result.mean_reward == 300.0
result_dict = manual_result.to_dict()
assert result_dict["algorithm"] == "SAC"

print("PASS: 05-ml/16_rl_trainer")
```

### Step-by-Step Explanation

1. **RLTrainingConfig defaults**: The default config uses PPO with MlpPolicy, 100K timesteps, 10 evaluation episodes, and seed 42. Custom configs override any or all of these. `to_dict()` serializes for logging and reproducibility.

2. **PolicyRegistry setup**: `PolicyRegistry(root_dir=...)` creates a registry that stores policy specs and trained versions. `supported_algorithms()` returns the six supported SB3 algorithms.

3. **Policy spec registration**: `PolicySpec` defines what a policy is (algorithm, hyperparameters, description). `register_spec()` stores it. The spec is retrievable by name and appears in `list_specs()`.

4. **Policy version registration**: `PolicyVersion` captures a trained artifact with its reward statistics, timestep count, and metadata. Versions are numbered and retrievable by name+version or as the latest.

5. **Invalid algorithm guard**: Registering a spec with an unknown algorithm raises `ValueError`, preventing typos from creating unusable policy entries.

6. **EnvironmentSpec**: Defines a Gymnasium environment with name, entry point, kwargs, episode limits, and reward threshold. Serializable via `to_dict()`.

7. **EnvironmentRegistry**: Wraps Gymnasium's registration system. After `register(spec)`, the environment is available for training. Requires Gymnasium to be installed.

8. **RLTrainer creation**: Takes optional environment and policy registries plus a root directory for artifacts. `supported_algorithms()` mirrors PolicyRegistry's list.

9. **Training**: `trainer.train()` runs the full SB3 pipeline: create environment, instantiate algorithm, train, evaluate, save, and register. Returns `RLTrainingResult` with reward statistics and timing.

10. **Evaluation**: `load_and_evaluate()` loads a trained policy and evaluates it on an environment, returning mean and standard deviation of reward across episodes.

11. **Standalone result**: `RLTrainingResult` can be created manually for testing or when wrapping external training runs. Supports `to_dict()` for serialization.

## Common Mistakes

| Mistake                                | Correct Pattern                         | Why                                                                                |
| -------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------- |
| Training without SB3 installed         | `pip install kailash-ml[rl]`            | RLTrainer wraps SB3; without it, training raises `ImportError`                     |
| Using unsupported algorithm names      | Check `supported_algorithms()` first    | Typos in algorithm names raise `ValueError`                                        |
| Setting `total_timesteps` too low      | Use at least 10K for meaningful results | Very short training produces random policies that cannot be evaluated meaningfully |
| Forgetting to register the policy spec | Call `register_spec()` before `train()` | The spec documents the policy configuration for reproducibility                    |

## Exercises

1. Create policy specs for PPO and DQN targeting the same environment. Register both, then create `RLTrainingConfig` objects for each. What hyperparameters would you tune differently for discrete (DQN) vs continuous (PPO) action spaces?

2. Register three policy versions with increasing mean rewards (100, 250, 450). Use `get_latest_version()` to retrieve the best one and verify its reward.

3. If SB3 and Gymnasium are installed, train a PPO agent on CartPole-v1 with 1000 timesteps. Evaluate it and compare the reward to the environment's reward threshold (475). How many more timesteps would be needed to solve the environment?

## Key Takeaways

- RLTrainer wraps Stable-Baselines3 with Kailash's registry and configuration patterns
- `RLTrainingConfig` fully specifies a training run for reproducibility
- PolicyRegistry tracks policy specs (definitions) and versions (trained artifacts with rewards)
- EnvironmentRegistry manages Gymnasium environment definitions
- Six algorithms are supported: PPO, DQN, SAC, A2C, TD3, DDPG
- `train()` handles the full pipeline: environment creation, training, evaluation, saving, and registration
- `load_and_evaluate()` loads a trained policy for evaluation without further training
- SB3 and Gymnasium are optional dependencies -- types and registries work without them

## Next Chapter

This concludes the ML section and the Kailash SDK Textbook. For hands-on practice, see the [ASCENT course modules](../../modules/) which apply these concepts in progressive exercises.
