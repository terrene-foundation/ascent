# Chapter 6: Streaming Chat Agent

## Overview

The **StreamingChatAgent** provides real-time token-by-token output for conversational interfaces. Instead of waiting for the complete response, the consumer receives tokens as they are generated -- creating a "typing" effect. This chapter teaches you the streaming pattern, the dual-mode interface (`.stream()` vs `.run()`), and how to configure chunk sizes for the latency-throughput trade-off.

## Prerequisites

- Python 3.10+ installed
- Kailash Kaizen installed (`pip install kailash-kaizen`)
- Completion of Chapters 1-4 (Delegate, SimpleQA, ReAct, CoT)
- Familiarity with Python async iterators (`async for`)

## Concepts

### Concept 1: Streaming vs Buffered

Traditional agents buffer the entire response and return it as a dict. Streaming agents yield tokens incrementally as the LLM generates them. This reduces perceived latency -- the user sees the first token immediately rather than waiting for the complete response.

- **What**: An async iterator that yields text tokens as they are generated
- **Why**: For chat interfaces, perceived responsiveness is critical -- streaming makes the first token appear in milliseconds instead of seconds
- **How**: `async for token in agent.stream("message"): print(token, end="", flush=True)`
- **When**: Use for chat interfaces, real-time dashboards, and any UI where incremental display improves the experience

### Concept 2: StreamingStrategy

`StreamingStrategy` is the execution strategy that enables token-level streaming. When `streaming=True`, the agent uses this strategy. When `streaming=False`, it falls back to the default `AsyncSingleShotStrategy` and only `.run()` works -- `.stream()` raises `ValueError`.

- **What**: An execution strategy that yields tokens incrementally instead of buffering the full response
- **Why**: The strategy abstraction cleanly separates streaming from non-streaming behavior without conditional logic in the agent
- **How**: Automatically selected when `streaming=True` in the constructor; verified via `isinstance(agent.strategy, StreamingStrategy)`
- **When**: Whenever the `StreamingChatAgent` is instantiated with `streaming=True` (the default)

### Concept 3: Chunk Size

Chunk size controls the latency-throughput trade-off. `chunk_size=1` yields one token at a time (lowest latency, more overhead). Larger values (5, 10, 20) batch tokens for higher throughput at the cost of slightly delayed display.

- **What**: The number of tokens yielded per iteration of the async iterator
- **Why**: Token-by-token streaming has per-yield overhead; batching amortizes this cost
- **How**: Set `chunk_size=N` in the constructor; each `async for` iteration yields N tokens
- **When**: Use `chunk_size=1` for chat UIs, larger values for batch processing or high-throughput pipelines

### Concept 4: Dual-Mode Interface

StreamingChatAgent provides two interfaces: `.stream(message)` for streaming output and `.run(message=...)` for standard dict output. `.run()` always works regardless of the streaming setting. `.stream()` requires `StreamingStrategy` and raises `ValueError` without it.

- **What**: Two methods that provide the same answer in different delivery modes
- **Why**: Some consumers need streaming (chat UIs), others need the full response at once (batch processing, testing)
- **How**: `.stream()` -> `async for token in ...`; `.run()` -> `dict` with `response` key
- **When**: Use `.stream()` for interactive interfaces, `.run()` for programmatic consumption

### Key API

| Class / Method          | Parameters                                         | Returns               | Description                              |
| ----------------------- | -------------------------------------------------- | --------------------- | ---------------------------------------- |
| `StreamingChatAgent()`  | `llm_provider`, `model`, `streaming`, `chunk_size` | `StreamingChatAgent`  | Create a streaming chat agent            |
| `agent.stream()`        | `message: str`                                     | `AsyncIterator[str]`  | Stream tokens incrementally              |
| `agent.run()`           | `message: str`                                     | `dict`                | Get the full response as a dict          |
| `StreamingChatConfig()` | `streaming`, `chunk_size`, `temperature`, ...      | `StreamingChatConfig` | Configuration for streaming behavior     |
| `ChatSignature`         | --                                                 | `Signature`           | Minimal chat I/O (message -> response)   |
| `StreamingStrategy`     | --                                                 | `Strategy`            | Token-level streaming execution strategy |

## Code Walkthrough

### Imports and Setup

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents.agents.specialized.streaming_chat import (
    StreamingChatAgent,
    StreamingChatConfig,
    ChatSignature,
)
from kaizen.core.base_agent import BaseAgent
from kaizen.strategies.streaming import StreamingStrategy

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
```

### ChatSignature -- Minimal Chat I/O

```python
assert "message" in ChatSignature._signature_inputs
assert "response" in ChatSignature._signature_outputs

message_field = ChatSignature._signature_inputs["message"]
assert message_field.required is True
```

The simplest possible signature: one required input (`message`) and one output (`response`). This reflects the natural chat interface -- user sends a message, agent sends a response.

### StreamingChatConfig

```python
config = StreamingChatConfig()

assert config.streaming is True, "Streaming enabled by default"
assert config.chunk_size == 1, "Token-by-token streaming by default"
assert config.temperature == 0.7, "Higher temperature for conversational style"
assert config.max_tokens == 500
```

Notable defaults: `streaming=True` (opt-out, not opt-in), `chunk_size=1` for the smoothest typing effect, and `temperature=0.7` -- higher than factual agents (0.1) because conversational style benefits from some variation.

Chunk size trade-offs:

| `chunk_size` | Behavior       | Best For                    |
| ------------ | -------------- | --------------------------- |
| 1            | Token-by-token | Chat UIs (lowest latency)   |
| 5            | Small batches  | Balanced latency/throughput |
| 20           | Large batches  | High-throughput pipelines   |

### Streaming Mode Agent

```python
agent = StreamingChatAgent(
    llm_provider="mock",
    model=model,
    streaming=True,
    chunk_size=1,
)

assert isinstance(agent, StreamingChatAgent)
assert isinstance(agent, BaseAgent)
assert isinstance(
    agent.strategy, StreamingStrategy
), "Streaming mode uses StreamingStrategy"
assert agent.chat_config.streaming is True
```

With `streaming=True`, the agent uses `StreamingStrategy` and both `.stream()` and `.run()` are available.

### Non-Streaming Mode Agent

```python
sync_agent = StreamingChatAgent(
    llm_provider="mock",
    model=model,
    streaming=False,
)

assert sync_agent.chat_config.streaming is False
assert not isinstance(
    sync_agent.strategy, StreamingStrategy
), "Non-streaming mode uses default strategy"

# stream() requires StreamingStrategy
try:
    import asyncio

    asyncio.run(sync_agent.stream("test").__anext__())
    assert False, "stream() should raise ValueError without StreamingStrategy"
except ValueError as e:
    assert "StreamingStrategy" in str(e)
except StopAsyncIteration:
    pass  # Also acceptable if strategy validates differently
```

With `streaming=False`, the agent falls back to `AsyncSingleShotStrategy`. Calling `.stream()` raises `ValueError` with a clear message. `.run()` still works normally.

### Streaming Consumption Pattern

```python
# The standard pattern for consuming streaming output:
#
#   import asyncio
#
#   agent = StreamingChatAgent()
#
#   async def chat():
#       async for token in agent.stream("What is Python?"):
#           print(token, end="", flush=True)
#       print()  # Newline after streaming completes
#
#   asyncio.run(chat())
#
# Key points:
# - stream() is async (must be awaited in async context)
# - Each yield is a string token or chunk
# - flush=True ensures immediate display
# - The loop ends when the response is complete
```

This is the canonical streaming pattern. `flush=True` is critical -- without it, Python buffers the output and the typing effect is lost. The newline after the loop ensures the terminal prompt appears on a fresh line.

### Custom Chunk Size

```python
batch_agent = StreamingChatAgent(
    llm_provider="mock",
    model=model,
    streaming=True,
    chunk_size=10,
)

assert batch_agent.chat_config.chunk_size == 10
assert isinstance(batch_agent.strategy, StreamingStrategy)
```

Larger chunk sizes reduce the number of yields, trading latency for throughput. A `chunk_size=10` agent yields 10 tokens per iteration.

### Environment Variable Configuration

All settings can be configured via `KAIZEN_*` environment variables:

| Variable              | Effect                   |
| --------------------- | ------------------------ |
| `KAIZEN_STREAMING`    | Enable/disable streaming |
| `KAIZEN_CHUNK_SIZE`   | Tokens per chunk         |
| `KAIZEN_LLM_PROVIDER` | Provider selection       |
| `KAIZEN_MODEL`        | Model selection          |

The config reads these at instantiation time.

## Common Mistakes

| Mistake                                      | Correct Pattern                              | Why                                                                                    |
| -------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------- |
| Forgetting `flush=True` in the print loop    | `print(token, end="", flush=True)`           | Without flush, Python buffers output and the streaming effect is invisible             |
| Calling `.stream()` on a non-streaming agent | Check `agent.chat_config.streaming` first    | `.stream()` raises `ValueError` when `StreamingStrategy` is not active                 |
| Using `chunk_size=1` for batch processing    | Use `chunk_size=10-20` for throughput        | Token-by-token streaming has per-yield overhead that adds up in batch workloads        |
| Setting `temperature=0.1` for chat           | Use `0.7` (default) for conversational style | Very low temperature makes responses robotic and repetitive in conversational contexts |

## Exercises

1. Create two `StreamingChatAgent` instances: one with `streaming=True` and one with `streaming=False`. Verify the strategy type of each using `isinstance()`. Then try calling `.stream()` on the non-streaming agent and catch the expected `ValueError`.
2. Create three streaming agents with `chunk_size` values of 1, 5, and 20. For each, describe the expected user experience in a chat UI. Which would you choose for a mobile app with high latency?
3. Compare `StreamingChatConfig.temperature` (0.7) with `SimpleQAConfig.temperature` (0.1) and `ChainOfThoughtConfig.temperature` (0.1). Explain why chat uses a higher temperature.

## Key Takeaways

- `StreamingChatAgent` yields tokens incrementally for real-time chat interfaces
- Dual-mode interface: `.stream()` for streaming, `.run()` for buffered dict output
- `StreamingStrategy` is used when `streaming=True`; `AsyncSingleShotStrategy` when False
- `chunk_size` controls the latency-throughput trade-off (1 = smoothest, 20 = fastest)
- `.stream()` raises `ValueError` without `StreamingStrategy` -- always check or use `streaming=True`
- Higher temperature (0.7) produces more natural conversational style

## Next Chapter

[Chapter 7: Governed Supervisor](07_governed_supervisor.md) -- Use PACT governance to build budget-governed multi-agent supervisors with audit trails and clearance enforcement.
