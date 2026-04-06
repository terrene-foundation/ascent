# Chapter 9: Serving Configuration

## Overview

The final step in the alignment lifecycle is **serving** -- making your fine-tuned model available for inference. Kailash Align supports two deployment targets: **Ollama** (local inference with GGUF) and **vLLM** (high-throughput GPU serving). This chapter teaches you how to configure serving, understand quantization options, set up generation backends, and deploy through the AlignmentServing interface.

## Prerequisites

- [Chapter 7: Adapter Merge](07_merge.md)
- [Chapter 8: Evaluation](08_evaluation.md)
- Familiarity with GGUF format and quantization concepts (helpful)

## Concepts

### Concept 1: ServingConfig

ServingConfig specifies the deployment target (`"ollama"` or `"vllm"`), quantization level, system prompt, and validation settings. It is the main configuration object for the `AlignmentServing` class.

- **What**: A frozen dataclass controlling how and where to deploy a fine-tuned model
- **Why**: Deployment configuration must be explicit and reproducible -- the same config produces the same deployment
- **How**: `ServingConfig(target="ollama", quantization="q4_k_m")` for local Ollama deployment
- **When**: After merging and evaluating, when you are ready to serve the model

### Concept 2: Quantization Types

Quantization reduces model size and inference cost by lowering numerical precision. Kailash Align supports three quantization levels:

| Type     | Description                   | Size Reduction | Quality Impact |
| -------- | ----------------------------- | -------------- | -------------- |
| `f16`    | No quantization (float16)     | 2x vs float32  | None           |
| `q4_k_m` | 4-bit K-quant mixed (default) | ~4x vs float16 | Minimal        |
| `q8_0`   | 8-bit quantization            | ~2x vs float16 | Very small     |

- **What**: Numerical precision reduction for inference
- **Why**: 4-bit quantization makes 8B models run on consumer GPUs (8GB VRAM)
- **How**: `ServingConfig(quantization="q4_k_m")` -- the default
- **When**: Use q4_k_m for most deployments; f16 when quality is critical; q8_0 as a middle ground

### Concept 3: Generation Backends

For online alignment methods (GRPO, RLOO, OnlineDPO), Kailash Align provides two generation backends:

| Backend               | Use Case                   | Speed    |
| --------------------- | -------------------------- | -------- |
| `VLLMBackend`         | Multi-GPU, high-throughput | Fast     |
| `HFGenerationBackend` | Single GPU, simple setup   | Moderate |

Both implement the `GenerationBackend` abstract base class and use lazy model loading.

### Concept 4: Supported Architectures

Not all model architectures support GGUF conversion. The `SUPPORTED_ARCHITECTURES` dictionary maps architecture names to their support level. Llama, Mistral, Phi-3, and Qwen2 are fully supported.

### Key API

| Class / Field             | Parameters                                            | Returns         | Description                    |
| ------------------------- | ----------------------------------------------------- | --------------- | ------------------------------ |
| `ServingConfig()`         | `target`, `quantization`, `system_prompt`, ...        | `ServingConfig` | Configure deployment           |
| `VLLMConfig()`            | `tensor_parallel_size`, `gpu_memory_utilization`, ... | `VLLMConfig`    | vLLM-specific settings         |
| `AlignmentServing()`      | `adapter_registry`, `config`                          | `Serving`       | Main serving interface         |
| `VLLMBackend()`           | `model_id`, `config`                                  | `VLLMBackend`   | vLLM generation backend        |
| `HFGenerationBackend()`   | `model_id`, `device`                                  | `HFBackend`     | HuggingFace generation backend |
| `SUPPORTED_ARCHITECTURES` | --                                                    | `dict`          | Architecture -> support level  |
| `QUANTIZATION_TYPES`      | --                                                    | `dict`          | Quantization name -> config    |

## Code Walkthrough

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kailash_align import (
    AdapterRegistry,
    AlignmentServing,
    GenerationBackend,
    HFGenerationBackend,
    ServingConfig,
    VLLMBackend,
    VLLMConfig,
)
from kailash_align.exceptions import ServingError
from kailash_align.serving import QUANTIZATION_TYPES, SUPPORTED_ARCHITECTURES

model_id = os.environ.get("DEFAULT_LLM_MODEL", "meta-llama/Llama-3.1-8B")
```

### ServingConfig Defaults

```python
default_serving = ServingConfig()

assert default_serving.target == "ollama"
assert default_serving.quantization == "q4_k_m"
assert default_serving.system_prompt is None
assert default_serving.ollama_host == "http://localhost:11434"
assert default_serving.validate_gguf is True
assert default_serving.validation_timeout == 120
```

The default targets Ollama with 4-bit quantization and GGUF validation enabled. The validation timeout gives Ollama 120 seconds to load and validate the model.

### Target Validation

```python
for valid_target in ("ollama", "vllm"):
    cfg = ServingConfig(target=valid_target)
    assert cfg.target == valid_target

try:
    ServingConfig(target="invalid")
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "target" in str(e)
```

Only `"ollama"` and `"vllm"` are valid targets.

### Quantization Validation

```python
for valid_quant in ("f16", "q4_k_m", "q8_0"):
    cfg = ServingConfig(quantization=valid_quant)
    assert cfg.quantization == valid_quant

try:
    ServingConfig(quantization="q2_k")
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "quantization" in str(e)
```

Only three quantization types are supported. Unsupported types like `"q2_k"` are rejected.

### Custom Serving Configs

```python
vllm_serving = ServingConfig(
    target="vllm",
    quantization="f16",
    system_prompt="You are a helpful assistant.",
    validate_gguf=False,
)

assert vllm_serving.target == "vllm"
assert vllm_serving.quantization == "f16"
assert vllm_serving.system_prompt == "You are a helpful assistant."

ollama_serving = ServingConfig(
    target="ollama",
    quantization="q8_0",
    ollama_host="http://my-server:11434",
    validation_timeout=300,
)

assert ollama_serving.ollama_host == "http://my-server:11434"
assert ollama_serving.validation_timeout == 300
```

For vLLM, you typically use f16 (no quantization) since vLLM manages GPU memory efficiently. For Ollama, you can point to a remote host and extend the validation timeout for large models.

### Supported Architectures

```python
assert "LlamaForCausalLM" in SUPPORTED_ARCHITECTURES
assert "MistralForCausalLM" in SUPPORTED_ARCHITECTURES
assert "Phi3ForCausalLM" in SUPPORTED_ARCHITECTURES
assert "Qwen2ForCausalLM" in SUPPORTED_ARCHITECTURES

assert SUPPORTED_ARCHITECTURES["LlamaForCausalLM"] == "fully_supported"
```

Check this dictionary before attempting GGUF conversion to ensure your model architecture is supported.

### Quantization Types

```python
assert "f16" in QUANTIZATION_TYPES
assert "q4_k_m" in QUANTIZATION_TYPES
assert "q8_0" in QUANTIZATION_TYPES
assert QUANTIZATION_TYPES["f16"] is None, "f16 means no quantization step"
```

`f16` maps to `None` because it skips the quantization step entirely -- the GGUF file contains float16 weights.

### AlignmentServing

```python
serving = AlignmentServing()
assert serving._registry is None
assert isinstance(serving._config, ServingConfig)

registry = AdapterRegistry()
serving_with_registry = AlignmentServing(
    adapter_registry=registry,
    config=ServingConfig(target="ollama", quantization="q4_k_m"),
)
assert serving_with_registry._registry is registry
assert serving_with_registry._config.target == "ollama"
```

AlignmentServing is the main interface for deploying adapters. It requires an AdapterRegistry for operations that look up adapter metadata.

```python
import asyncio

async def test_serving_requires_registry() -> None:
    no_registry = AlignmentServing(config=ServingConfig(target="vllm"))
    try:
        await no_registry.deploy("any-adapter")
        assert False, "Should have raised ServingError"
    except ServingError as e:
        assert "AdapterRegistry" in str(e)

asyncio.run(test_serving_requires_registry())
```

Attempting to deploy without a registry raises `ServingError`.

### VLLMConfig

```python
vllm_config = VLLMConfig()

assert vllm_config.tensor_parallel_size == 1
assert vllm_config.gpu_memory_utilization == 0.9
assert vllm_config.max_model_len is None
assert vllm_config.dtype == "auto"
assert vllm_config.seed == 42
```

VLLMConfig controls vLLM-specific settings. `tensor_parallel_size` sets multi-GPU parallelism. `gpu_memory_utilization` controls how much GPU memory vLLM reserves (0.9 = 90%).

```python
# gpu_memory_utilization must be in (0, 1]
try:
    VLLMConfig(gpu_memory_utilization=0.0)
    assert False, "Should have raised ValueError"
except ValueError:
    pass

full_util = VLLMConfig(gpu_memory_utilization=1.0)
assert full_util.gpu_memory_utilization == 1.0, "1.0 is valid (use all GPU memory)"

# tensor_parallel_size must be >= 1
multi_gpu = VLLMConfig(tensor_parallel_size=4)
assert multi_gpu.tensor_parallel_size == 4
```

### Generation Backends

```python
assert issubclass(VLLMBackend, GenerationBackend)
assert issubclass(HFGenerationBackend, GenerationBackend)

# VLLMBackend with lazy loading
vllm_backend = VLLMBackend(model_id=model_id)
assert vllm_backend._model_id == model_id
assert vllm_backend._llm is None, "Model not loaded yet (lazy)"

# HFGenerationBackend with lazy loading
hf_backend = HFGenerationBackend(model_id=model_id)
assert hf_backend._model_id == model_id
assert hf_backend._model is None, "Model not loaded yet (lazy)"

# Custom VLLMBackend
custom_backend = VLLMBackend(
    model_id=model_id,
    config=VLLMConfig(tensor_parallel_size=2, gpu_memory_utilization=0.8),
)
assert custom_backend._config.tensor_parallel_size == 2

# HFGenerationBackend with device override
hf_cpu = HFGenerationBackend(model_id=model_id, device="cpu")
assert hf_cpu._device == "cpu"
```

Both backends use lazy loading -- the model is not loaded until the first generation call. This means instantiation is fast and does not require GPU memory.

### Safe Shutdown

```python
vllm_backend.shutdown()  # No-op: model not loaded
assert vllm_backend._llm is None

hf_backend.shutdown()  # No-op: model not loaded
assert hf_backend._model is None
```

`shutdown()` is safe to call even when the model has not been loaded yet -- it is a no-op in that case.

## Exercises

1. Create a ServingConfig for deploying to a remote Ollama instance at `http://gpu-server:11434` with q8_0 quantization and a 5-minute validation timeout. Which parameters do you change from the defaults?
2. Create a VLLMConfig for a 2-GPU setup that reserves 85% of GPU memory. What is the `tensor_parallel_size` and `gpu_memory_utilization`?
3. Write the complete deployment path: create an AdapterRegistry, register an adapter, set up AlignmentServing with an Ollama config, and call deploy. What error do you get if the adapter has not been merged first?

## Key Takeaways

- ServingConfig targets either Ollama (GGUF + local) or vLLM (GPU + high-throughput)
- Three quantization levels: f16 (no quant), q4_k_m (4-bit, default), q8_0 (8-bit)
- AlignmentServing requires an AdapterRegistry to look up adapter metadata
- VLLMConfig controls multi-GPU parallelism and memory utilization
- Generation backends (VLLMBackend, HFGenerationBackend) use lazy model loading
- Check SUPPORTED_ARCHITECTURES before GGUF conversion

## Next Section

Continue to [Section 8: Integration](../08-integration/01_ml_to_registry.md) -- Learn how Kailash packages work together in production pipelines.
