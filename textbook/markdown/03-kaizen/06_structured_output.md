# Chapter 6: Structured Output

## Overview

LLM agents need to return more than free-form text -- they need structured, typed responses that downstream code can parse reliably. Kailash Kaizen's `Signature` class lets you declare typed `OutputField` definitions that constrain the agent's response format. This chapter covers typed output fields, multi-field schemas, boolean decisions, structured data extraction, and the Signature-to-Agent pattern.

## Prerequisites

- Python 3.10+ installed
- Kailash Kaizen installed (`pip install kailash-kaizen`)
- Completed [Chapter 1: Signatures](01_signatures.md)
- Understanding of `InputField` and basic Signature definition

## Concepts

### Concept 1: Typed Output Fields

`OutputField` declares a named, typed output that the agent must produce. The type annotation (str, float, bool, etc.) tells the agent's response parser what format to expect. The description guides the LLM on what content to generate.

- **What**: A class-level annotation with `OutputField(description=...)` that declares one structured output
- **Why**: Typed outputs enable machine-parseable responses instead of free-text that requires regex extraction
- **How**: The Signature metaclass collects all `OutputField` annotations into `_signature_outputs`, which the agent uses to format its prompt and parse its response
- **When**: Use whenever you need the agent to return specific, named values -- not just prose

### Concept 2: Multi-Field Structured Output

A single Signature can declare many output fields, creating a rich structured response. Each field is independently typed and described. The agent produces all fields in one LLM call.

- **What**: Multiple `OutputField` declarations in one Signature class
- **Why**: Many tasks require multiple outputs: a sentiment label AND a confidence score AND key phrases
- **How**: Each field becomes a separate key in the agent's structured response object
- **When**: Use for any task with multiple distinct outputs -- entity extraction, classification with explanation, structured data extraction

### Concept 3: Boolean Decision Output

Output fields with `bool` type enable binary decision tasks. The agent produces `True` or `False` plus supporting fields (reason, category, etc.). This pattern is ideal for moderation, approval gates, and classification.

- **What**: An `OutputField` with a `bool` type annotation
- **Why**: Forces the agent to commit to a yes/no decision rather than hedging in prose
- **How**: The response parser interprets the agent's output as a Python boolean
- **When**: Moderation decisions, eligibility checks, feature flag evaluations, approval workflows

### Concept 4: Signature-to-Agent Pattern

In practice, a Signature is attached to an Agent. The agent's LLM call is constrained to produce exactly the declared output fields. The result object has attributes matching the field names, enabling `result.sentiment`, `result.confidence`, etc.

- **What**: The runtime pattern where Signature defines the contract and Agent enforces it
- **Why**: Separates the "what" (Signature) from the "how" (Agent), making schemas reusable across agents
- **How**: `Agent(signature=MySignature, model=model)` binds the schema; `agent.run()` returns a typed result
- **When**: Always -- Signatures are not useful without an Agent to execute them

## Key API

| Class / Method        | Parameters                                | Returns | Description                           |
| --------------------- | ----------------------------------------- | ------- | ------------------------------------- |
| `Signature`           | Class definition with fields              | --      | Base class for structured I/O schemas |
| `InputField()`        | `description: str`, `default: Any = None` | Field   | Declare a named input                 |
| `OutputField()`       | `description: str`                        | Field   | Declare a named, typed output         |
| `._signature_inputs`  | --                                        | `dict`  | All declared input fields             |
| `._signature_outputs` | --                                        | `dict`  | All declared output fields            |

## Code Walkthrough

```python
from __future__ import annotations

from kaizen import Signature, InputField, OutputField

# ── 1. Typed output fields ──────────────────────────────────────────
# OutputField types tell the LLM the expected format. The agent's
# response is parsed to match these types.

class SentimentSignature(Signature):
    """Analyze the sentiment of the given text."""

    text: str = InputField(description="Text to analyze")
    sentiment: str = OutputField(description="One of: positive, negative, neutral")
    confidence: float = OutputField(description="Confidence score 0.0 to 1.0")
    keywords: str = OutputField(description="Comma-separated key phrases")

assert "sentiment" in SentimentSignature._signature_outputs
assert "confidence" in SentimentSignature._signature_outputs
assert "keywords" in SentimentSignature._signature_outputs

# ── 2. Multi-field structured output ────────────────────────────────
# Signatures can request complex structured responses.

class ExtractEntitiesSignature(Signature):
    """Extract named entities from text."""

    text: str = InputField(description="Text to process")
    context: str = InputField(description="Domain context", default="general")
    people: str = OutputField(description="JSON array of person names")
    organizations: str = OutputField(description="JSON array of org names")
    locations: str = OutputField(description="JSON array of locations")
    dates: str = OutputField(description="JSON array of dates mentioned")
    summary: str = OutputField(description="One-sentence summary")

assert len(ExtractEntitiesSignature._signature_outputs) == 5
assert len(ExtractEntitiesSignature._signature_inputs) == 2

# ── 3. Boolean/decision output ──────────────────────────────────────

class ModerationSignature(Signature):
    """Check if content violates guidelines."""

    content: str = InputField(description="Content to moderate")
    guidelines: str = InputField(description="Moderation guidelines")
    is_safe: bool = OutputField(description="True if content is safe")
    reason: str = OutputField(description="Explanation of decision")
    category: str = OutputField(
        description="Violation category if unsafe, else 'none'"
    )

assert "is_safe" in ModerationSignature._signature_outputs

# ── 4. Signature for structured data extraction ─────────────────────

class InvoiceExtractionSignature(Signature):
    """Extract structured data from an invoice image or text."""

    invoice_text: str = InputField(description="Raw invoice text")
    vendor_name: str = OutputField(description="Name of the vendor")
    invoice_number: str = OutputField(description="Invoice number")
    total_amount: str = OutputField(
        description="Total amount as string (e.g., '1234.56')"
    )
    currency: str = OutputField(description="Currency code (e.g., 'SGD', 'USD')")
    line_items: str = OutputField(
        description="JSON array of {description, quantity, price}"
    )

assert len(InvoiceExtractionSignature._signature_outputs) == 5

# ── 5. Pattern: Signature → Agent → Structured response ────────────
# In practice, a Signature is attached to an agent:
#
#   from kaizen_agents import Agent
#   agent = Agent(signature=SentimentSignature, model=model)
#   result = await agent.run(text="Kailash SDK is amazing!")
#   print(result.sentiment)    # "positive"
#   print(result.confidence)   # 0.95
#   print(result.keywords)     # "Kailash SDK, amazing"

print("PASS: 03-kaizen/06_structured_output")
```

### Step-by-Step Explanation

1. **Typed output fields**: `SentimentSignature` declares three outputs: `sentiment` (str), `confidence` (float), and `keywords` (str). The Signature metaclass registers these in `_signature_outputs`. When an Agent uses this Signature, the LLM is prompted to produce exactly these fields.

2. **Multi-field extraction**: `ExtractEntitiesSignature` has five output fields and two input fields (one with a default). The descriptions guide the LLM -- "JSON array of person names" tells it the expected format. The `context` input with `default="general"` is optional.

3. **Boolean decisions**: `ModerationSignature` uses `is_safe: bool` to force a binary decision. The accompanying `reason` and `category` fields provide the explanation. This pattern works for any yes/no task.

4. **Structured data extraction**: `InvoiceExtractionSignature` demonstrates extracting structured business data from unstructured text. Each field maps to one piece of information on the invoice. Using `str` for `total_amount` avoids floating-point parsing issues.

5. **Signature-to-Agent pattern**: The commented code shows the runtime pattern. `Agent(signature=SentimentSignature, model=model)` binds the schema to an agent. `result = await agent.run(text="...")` returns an object where `result.sentiment`, `result.confidence`, and `result.keywords` are typed attributes.

## Common Mistakes

| Mistake                                         | Correct Pattern                                              | Why                                                                                 |
| ----------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| Using `str` for everything                      | Use specific types: `float` for scores, `bool` for decisions | Types guide the parser and the LLM; `str` produces unparsed text                    |
| Vague descriptions                              | Be specific: "One of: positive, negative, neutral"           | The description is the LLM's primary instruction for what to produce                |
| Expecting nested objects from OutputField       | Use `str` with JSON format instructions                      | OutputField types are Python primitives; for complex structures, use JSON-in-string |
| Defining outputs without a Signature base class | Always inherit from `Signature`                              | The metaclass that collects fields only activates on `Signature` subclasses         |

## Exercises

1. Create a `CodeReviewSignature` with inputs for `code` and `language`, and outputs for `quality_score` (float), `issues` (str as JSON array), `suggestion` (str), and `approved` (bool). Verify the field counts.

2. Design a `TranslationSignature` that takes `text` and `target_language` as inputs and produces `translated_text`, `confidence`, and `detected_source_language` as outputs. What types would you choose for each?

3. Compare two approaches to extracting a list of items: (a) `items: str = OutputField(description="JSON array of items")` vs (b) multiple individual fields like `item_1`, `item_2`, `item_3`. What are the trade-offs of each approach?

## Key Takeaways

- `OutputField` declares typed, named outputs that constrain the agent's response format
- Multiple output fields create rich structured responses from a single LLM call
- Boolean output fields force binary decisions, ideal for moderation and approval workflows
- Descriptions are the primary instruction to the LLM -- be specific about format and content
- The Signature-to-Agent pattern separates schema definition from execution
- Use `str` with JSON format instructions for complex nested structures

## Next Chapter

This concludes the Kaizen section. Continue to [Section 04: PACT](../04-pact/) for organizational governance, or jump to [Section 05: ML](../05-ml/) for the machine learning lifecycle engines.
