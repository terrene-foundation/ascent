# Chapter 8: Evaluation

## Overview

After aligning a model, you need to measure whether it improved -- and whether it lost capabilities in the process. The **AlignmentEvaluator** wraps the lm-eval-harness to run standard benchmarks (ARC, HellaSwag, TruthfulQA, MMLU) against your adapted model. **EvalConfig** controls which tasks to run, **TaskResult** captures per-task scores, and **EvalResult** aggregates everything into a single report. This chapter teaches you how to configure evaluations, interpret results, and integrate scores into the adapter registry.

## Prerequisites

- [Chapter 6: Adapter Registry](06_adapter_registry.md)
- Understanding of LLM evaluation benchmarks (helpful)

## Concepts

### Concept 1: Task Presets

Kailash Align ships with two task presets:

| Preset           | Tasks                                                                | Time       |
| ---------------- | -------------------------------------------------------------------- | ---------- |
| `QUICK_TASKS`    | arc_easy, hellaswag, truthfulqa_mc1                                  | ~5 min     |
| `STANDARD_TASKS` | arc_easy, hellaswag, truthfulqa_mc1, arc_challenge, mmlu, winogrande | ~30-60 min |

- **What**: Pre-defined lists of benchmark task names for common evaluation scenarios
- **Why**: Choosing tasks manually is error-prone; presets ensure consistent evaluation across adapters
- **How**: `EvalConfig(tasks=tuple(STANDARD_TASKS))` for thorough evaluation
- **When**: Use QUICK_TASKS during development iterations; STANDARD_TASKS for promotion decisions

### Concept 2: EvalConfig

EvalConfig controls the evaluation run: which tasks, how many samples per task (`limit`), batch size, few-shot count, and device. The `use_adapter` flag determines whether to evaluate the adapter or the base model (useful for before/after comparison).

- **What**: A frozen dataclass controlling evaluation parameters
- **Why**: Reproducible evaluations require fixed configuration
- **How**: `EvalConfig(tasks=("arc_easy", "mmlu"), limit=100, device="cuda")`
- **When**: Create one per evaluation run

### Concept 3: Result Hierarchy

Results are organized in two levels: **TaskResult** captures a single benchmark's metrics (accuracy, normalized accuracy, standard error), and **EvalResult** aggregates multiple TaskResults with adapter metadata and timing.

- **What**: A two-level result hierarchy -- TaskResult per benchmark, EvalResult per evaluation run
- **Why**: Per-task granularity lets you detect capability regressions on specific benchmarks
- **How**: Access via `eval_result.task_results` (list) or `eval_result.summary` (quick accuracy lookup)
- **When**: After every evaluation run, before promotion decisions

### Key API

| Class / Method           | Parameters                                              | Returns      | Description                         |
| ------------------------ | ------------------------------------------------------- | ------------ | ----------------------------------- |
| `EvalConfig()`           | `tasks`, `limit`, `batch_size`, `num_fewshot`, `device` | `EvalConfig` | Configure an evaluation run         |
| `TaskResult()`           | `task_name`, `metrics`, `num_samples`, `task_version`   | `TaskResult` | Single benchmark result             |
| `TaskResult.to_dict()`   | --                                                      | `dict`       | Serialize for storage               |
| `TaskResult.from_dict()` | `data: dict`                                            | `TaskResult` | Deserialize                         |
| `EvalResult()`           | `adapter_name`, `adapter_version`, `task_results`, ...  | `EvalResult` | Aggregated evaluation report        |
| `EvalResult.summary`     | --                                                      | `dict`       | Task name -> primary accuracy score |
| `AlignmentEvaluator()`   | `adapter_registry`                                      | `Evaluator`  | Create evaluator with registry      |

## Code Walkthrough

```python
from __future__ import annotations

from kailash_align import AlignmentEvaluator, EvalConfig, EvalResult, TaskResult
from kailash_align.config import QUICK_TASKS, STANDARD_TASKS
```

### EvalConfig Defaults

```python
default_eval = EvalConfig()

assert default_eval.tasks == ("arc_easy", "hellaswag", "truthfulqa_mc1")
assert default_eval.limit == 100, "Default limit=100 for interactive use"
assert default_eval.batch_size == "auto"
assert default_eval.num_fewshot is None, "Uses task default"
assert default_eval.device is None, "Auto-detect device"
assert default_eval.local_files_only is False
assert default_eval.use_adapter is True
```

The defaults run the three QUICK_TASKS with 100 samples each. This takes about 5 minutes on a single GPU and gives a quick read on model quality.

### EvalConfig Validation

```python
try:
    default_eval.limit = 50  # type: ignore[misc]
    assert False, "Should have raised FrozenInstanceError"
except AttributeError:
    pass  # Expected: frozen dataclass

try:
    EvalConfig(limit=0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # limit must be >= 1
```

EvalConfig is frozen and validates that limit is at least 1.

### Custom EvalConfig

```python
custom_eval = EvalConfig(
    tasks=("arc_easy", "arc_challenge", "mmlu"),
    limit=50,
    batch_size="16",
    num_fewshot=5,
    device="cpu",
    use_adapter=False,
)

assert custom_eval.tasks == ("arc_easy", "arc_challenge", "mmlu")
assert custom_eval.limit == 50
assert custom_eval.batch_size == "16"
assert custom_eval.num_fewshot == 5
assert custom_eval.device == "cpu"
assert custom_eval.use_adapter is False
```

Set `use_adapter=False` to evaluate the base model without the adapter. This is useful for measuring the delta that fine-tuning introduced.

### Task Presets

```python
assert isinstance(QUICK_TASKS, list)
assert len(QUICK_TASKS) == 3
assert "arc_easy" in QUICK_TASKS
assert "hellaswag" in QUICK_TASKS
assert "truthfulqa_mc1" in QUICK_TASKS

assert isinstance(STANDARD_TASKS, list)
assert len(STANDARD_TASKS) == 6
assert "mmlu" in STANDARD_TASKS
assert "winogrande" in STANDARD_TASKS
assert "arc_challenge" in STANDARD_TASKS

# All quick tasks are in standard tasks
for task in QUICK_TASKS:
    assert task in STANDARD_TASKS, f"Quick task {task} should also be in standard"
```

QUICK_TASKS is a subset of STANDARD_TASKS. This means quick evaluations are directly comparable to standard ones on the overlapping benchmarks.

### TaskResult

```python
task_result = TaskResult(
    task_name="arc_easy",
    metrics={"acc": 0.75, "acc_norm": 0.72, "acc_stderr": 0.02},
    num_samples=100,
    task_version="1.0",
)

assert task_result.task_name == "arc_easy"
assert task_result.metrics["acc"] == 0.75
assert task_result.num_samples == 100
assert task_result.task_version == "1.0"
```

TaskResult captures the benchmark name, a metrics dictionary, sample count, and task version. The metrics dictionary follows lm-eval-harness conventions: `acc` (accuracy), `acc_norm` (length-normalized accuracy), and `acc_stderr` (standard error).

### TaskResult Serialization

```python
task_dict = task_result.to_dict()
assert task_dict["task_name"] == "arc_easy"
assert task_dict["metrics"]["acc"] == 0.75

# Round-trip: to_dict -> from_dict
restored = TaskResult.from_dict(task_dict)
assert restored.task_name == task_result.task_name
assert restored.metrics == task_result.metrics
assert restored.num_samples == task_result.num_samples
```

TaskResult supports round-trip serialization for storage in registries and databases.

### EvalResult -- Aggregated Report

```python
eval_result = EvalResult(
    adapter_name="my-adapter",
    adapter_version="3",
    task_results=[
        TaskResult(
            task_name="arc_easy",
            metrics={"acc": 0.75, "acc_norm": 0.72},
            num_samples=100,
        ),
        TaskResult(
            task_name="hellaswag",
            metrics={"acc": 0.62, "acc_norm": 0.58},
            num_samples=100,
        ),
        TaskResult(
            task_name="truthfulqa_mc1",
            metrics={"acc": 0.41, "mc2": 0.55},
            num_samples=100,
        ),
    ],
    eval_config={"tasks": ["arc_easy", "hellaswag", "truthfulqa_mc1"], "limit": 100},
    total_duration_seconds=245.7,
)

assert eval_result.adapter_name == "my-adapter"
assert eval_result.adapter_version == "3"
assert len(eval_result.task_results) == 3
assert eval_result.total_duration_seconds == 245.7
```

EvalResult bundles all task results with adapter identification and timing information.

### The summary Property

```python
summary = eval_result.summary
assert "arc_easy" in summary
assert summary["arc_easy"] == 0.75
assert "hellaswag" in summary
assert summary["hellaswag"] == 0.62
assert "truthfulqa_mc1" in summary
assert summary["truthfulqa_mc1"] == 0.41
```

The `summary` property extracts the first metric containing `"acc"` from each task. This gives you a quick accuracy lookup without digging into individual TaskResults.

```python
# summary gracefully handles tasks with no 'acc' key
no_acc_result = EvalResult(
    adapter_name="test",
    adapter_version="1",
    task_results=[
        TaskResult(
            task_name="custom_task",
            metrics={"f1": 0.85, "precision": 0.9},
            num_samples=50,
        ),
    ],
    eval_config={"tasks": ["custom_task"]},
    total_duration_seconds=10.0,
)
assert no_acc_result.summary == {}, "No 'acc' metrics -> empty summary"
```

If a task has no accuracy metric, it is omitted from the summary rather than causing an error.

### AlignmentEvaluator

```python
evaluator = AlignmentEvaluator()
assert evaluator._registry is None

from kailash_align import AdapterRegistry

registry = AdapterRegistry()
evaluator_with_registry = AlignmentEvaluator(adapter_registry=registry)
assert evaluator_with_registry._registry is registry
```

AlignmentEvaluator optionally takes an AdapterRegistry to automatically store evaluation results. Without a registry, results are returned but not persisted.

## Exercises

1. Create an EvalResult with results for all six STANDARD_TASKS. Use the `summary` property to find the task with the lowest accuracy. Which benchmark would you investigate first?
2. Create two EvalConfigs: one with `use_adapter=True` and one with `use_adapter=False`. Explain how you would use them together to measure the improvement from fine-tuning.
3. Write a function that takes two EvalResult objects (before and after fine-tuning) and returns a dictionary mapping task names to the accuracy delta (after - before). Which tasks improved? Which regressed?

## Key Takeaways

- EvalConfig controls evaluation parameters: tasks, sample limit, batch size, device
- QUICK_TASKS (3 benchmarks, ~5 min) for iteration; STANDARD_TASKS (6 benchmarks, ~60 min) for promotion
- TaskResult captures per-benchmark metrics; EvalResult aggregates into a report
- The `summary` property provides a quick accuracy lookup across all tasks
- Both TaskResult and EvalResult support round-trip serialization
- AlignmentEvaluator optionally integrates with AdapterRegistry for persistent storage

## Next Chapter

[Chapter 9: Serving Configuration](09_serving.md) -- Configure model serving with GGUF export, Ollama deployment, and vLLM backends.
