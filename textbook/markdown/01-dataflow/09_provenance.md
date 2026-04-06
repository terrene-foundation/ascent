# Chapter 9: Data Provenance

## Overview

Knowing **where** a value came from, **how confident** you are in it, and **what it replaced** is critical for auditable data pipelines. DataFlow's provenance module provides `Provenance[T]` -- a generic typed wrapper that pairs any value with metadata about its origin, confidence score, and change history. This chapter teaches you how to create provenance-tracked values, validate confidence scores, serialize provenance for storage, and build audit trails with change tracking.

## Prerequisites

- Python 3.10+ installed
- Kailash DataFlow installed (`pip install kailash-dataflow`)
- Completed Chapters 1-6 (model definition, CRUD, validation, and classification)
- Familiarity with Python generics and dataclasses

## Concepts

### Concept 1: SourceType

The `SourceType` enum classifies where a value came from: `EXCEL_CELL`, `API_QUERY`, `CALCULATED`, `AGENT_DERIVED`, `MANUAL`, `DATABASE`, or `FILE`. Each member is string-backed for JSON serialization. Source types distinguish human-entered data from API-fetched data from AI-generated predictions -- each carries different trust implications.

- **What**: A string-backed enum that categorizes the origin of a data value
- **Why**: A price entered manually by a user carries different trust than a price fetched from an API or predicted by an AI agent. Source type enables downstream systems to apply appropriate trust policies
- **How**: Seven enum members cover the most common data origins. String-backed (`SourceType("manual")` works) for easy serialization
- **When**: Assign a source type to every value that enters your pipeline -- at the point of ingestion, not after the fact

### Concept 2: ProvenanceMetadata

`ProvenanceMetadata` is a dataclass that holds the full origin story of a value: `source_type` (where it came from), `source_detail` (a human-readable description), `confidence` (0.0 to 1.0), `extracted_at` (UTC timestamp), `previous_value` (what this value replaced), and `change_reason` (why it changed).

- **What**: A structured record of a value's origin, confidence, and change history
- **Why**: Metadata enables audit trails, confidence-based filtering, and change tracking without modifying the value itself
- **How**: Constructed with required `source_type` and `confidence`, plus optional fields. Validates that confidence is a finite number between 0.0 and 1.0
- **When**: Create metadata at the point where a value enters or changes in your pipeline

### Concept 3: Confidence Validation

The `confidence` field must be a finite number between 0.0 and 1.0 inclusive. Values outside this range, `NaN`, and `Infinity` are rejected with `ValueError` at construction time. A confidence of 1.0 means absolute certainty (e.g., user-entered data). A confidence of 0.0 means the value is tracked but not trusted (e.g., missing data from a legacy system).

### Concept 4: Provenance[T] -- The Typed Wrapper

`Provenance` wraps a value of any type (`float`, `str`, `int`, `None`, etc.) with its `ProvenanceMetadata`. Access the value via `.value` and the metadata via `.metadata`. The generic type parameter documents what kind of value is wrapped: `Provenance[float]`, `Provenance[str]`, etc.

- **What**: A generic container that pairs a typed value with its provenance metadata
- **Why**: Keeping value and metadata together ensures they travel as a unit through your pipeline. You cannot accidentally discard provenance by passing just the raw value
- **How**: `Provenance(value=485000.0, metadata=ProvenanceMetadata(...))` creates a tracked value. `.to_dict()` and `.from_dict()` handle serialization
- **When**: Wrap values at ingestion and unwrap them at the point of use. Between those points, pass the `Provenance` object, not the raw value

### Concept 5: Change Tracking

When a value is updated, create a new `Provenance` with the new value and set `previous_value` and `change_reason` in the metadata. This builds an audit trail -- each provenance record points back to what it replaced and why.

### Key API

| Class / Method                   | Parameters                                                                                                      | Returns              | Description                            |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------- | -------------------------------------- |
| `SourceType`                     | enum: `EXCEL_CELL`, `API_QUERY`, `CALCULATED`, `AGENT_DERIVED`, `MANUAL`, `DATABASE`, `FILE`                    | --                   | Data origin classification             |
| `ProvenanceMetadata()`           | `source_type`, `confidence`, `source_detail=""`, `previous_value=None`, `change_reason=""`, `extracted_at=None` | `ProvenanceMetadata` | Create origin metadata                 |
| `ProvenanceMetadata.to_dict()`   | --                                                                                                              | `dict`               | Serialize to JSON-compatible dict      |
| `ProvenanceMetadata.from_dict()` | `data: dict`                                                                                                    | `ProvenanceMetadata` | Deserialize from dict                  |
| `Provenance()`                   | `value: T`, `metadata: ProvenanceMetadata`                                                                      | `Provenance[T]`      | Wrap a value with provenance           |
| `Provenance.value`               | --                                                                                                              | `T`                  | The wrapped value                      |
| `Provenance.metadata`            | --                                                                                                              | `ProvenanceMetadata` | The origin metadata                    |
| `Provenance.to_dict()`           | --                                                                                                              | `dict`               | Flatten value + metadata into one dict |
| `Provenance.from_dict()`         | `data: dict`                                                                                                    | `Provenance`         | Restore from flattened dict            |

## Code Walkthrough

```python
from __future__ import annotations

from datetime import UTC, datetime

from dataflow import Provenance, ProvenanceMetadata, SourceType

# -- 1. SourceType enum -- where a field value came from --------------------
# Classification of data origins. All values are str-backed.

assert SourceType.EXCEL_CELL.value == "excel_cell"
assert SourceType.API_QUERY.value == "api_query"
assert SourceType.CALCULATED.value == "calculated"
assert SourceType.AGENT_DERIVED.value == "agent_derived"
assert SourceType.MANUAL.value == "manual"
assert SourceType.DATABASE.value == "database"
assert SourceType.FILE.value == "file"

# Construct from string
assert SourceType("manual") == SourceType.MANUAL

print("SourceType: 7 origin types defined")

# -- 2. ProvenanceMetadata -- origin and confidence -------------------------
# Describes where a value came from, how confident we are, and
# optionally tracks previous values and change reasons.

meta = ProvenanceMetadata(
    source_type=SourceType.API_QUERY,
    source_detail="Singapore Open Data API -- HDB resale prices Q1 2026",
    confidence=0.95,
)

assert meta.source_type == SourceType.API_QUERY
assert meta.source_detail == "Singapore Open Data API -- HDB resale prices Q1 2026"
assert meta.confidence == 0.95
assert meta.previous_value is None
assert meta.change_reason == ""
assert meta.extracted_at is not None, "Defaults to now (UTC)"
assert isinstance(meta.extracted_at, datetime)
print(f"ProvenanceMetadata: source={meta.source_type.value}, confidence={meta.confidence}")

# -- 3. ProvenanceMetadata -- full fields -----------------------------------
# Track previous values and change reasons for audit trails.

meta_updated = ProvenanceMetadata(
    source_type=SourceType.AGENT_DERIVED,
    source_detail="PriceAdjustmentAgent v2.1",
    confidence=0.88,
    previous_value=450000.0,
    change_reason="Agent adjusted for inflation index 2026-Q1",
    extracted_at=datetime(2026, 3, 15, 10, 30, 0, tzinfo=UTC),
)

assert meta_updated.previous_value == 450000.0
assert meta_updated.change_reason == "Agent adjusted for inflation index 2026-Q1"
assert meta_updated.extracted_at.year == 2026
print(f"Updated metadata: previous_value={meta_updated.previous_value}")

# -- 4. Confidence validation -----------------------------------------------
# Confidence must be a finite number between 0.0 and 1.0.

# Valid boundaries
ProvenanceMetadata(source_type=SourceType.MANUAL, confidence=0.0)
ProvenanceMetadata(source_type=SourceType.MANUAL, confidence=1.0)

# Invalid: out of range
try:
    ProvenanceMetadata(source_type=SourceType.MANUAL, confidence=1.5)
    assert False, "Should raise ValueError"
except ValueError as e:
    assert "between 0.0 and 1.0" in str(e)

try:
    ProvenanceMetadata(source_type=SourceType.MANUAL, confidence=-0.1)
    assert False, "Should raise ValueError"
except ValueError as e:
    assert "between 0.0 and 1.0" in str(e)

# Invalid: NaN and infinity
import math

try:
    ProvenanceMetadata(source_type=SourceType.MANUAL, confidence=float("nan"))
    assert False, "Should raise ValueError"
except ValueError:
    pass

try:
    ProvenanceMetadata(source_type=SourceType.MANUAL, confidence=float("inf"))
    assert False, "Should raise ValueError"
except ValueError:
    pass

print("Confidence validation: boundary checks passed")

# -- 5. SourceType coercion -------------------------------------------------
# source_type accepts both enum members and string values.

meta_from_str = ProvenanceMetadata(source_type="database", confidence=1.0)
assert meta_from_str.source_type == SourceType.DATABASE
assert isinstance(meta_from_str.source_type, SourceType)

# -- 6. ProvenanceMetadata serialization ------------------------------------
# to_dict() and from_dict() for JSON-compatible storage.

meta_dict = meta.to_dict()
assert meta_dict["source_type"] == "api_query"
assert meta_dict["confidence"] == 0.95
assert meta_dict["source_detail"] == "Singapore Open Data API -- HDB resale prices Q1 2026"
assert isinstance(meta_dict["extracted_at"], str), "datetime serialized to ISO string"

# Round-trip
meta_restored = ProvenanceMetadata.from_dict(meta_dict)
assert meta_restored.source_type == meta.source_type
assert meta_restored.confidence == meta.confidence
assert meta_restored.source_detail == meta.source_detail
print("ProvenanceMetadata serialization: round-trip verified")

# -- 7. Provenance[T] -- typed value wrapper --------------------------------
# Provenance wraps a value with its metadata. It is generic over the
# value type: Provenance[float], Provenance[str], Provenance[int], etc.

price = Provenance(
    value=485000.0,
    metadata=ProvenanceMetadata(
        source_type=SourceType.API_QUERY,
        source_detail="HDB resale API -- Ang Mo Kio 4-room",
        confidence=0.99,
    ),
)

assert price.value == 485000.0
assert isinstance(price.value, float)
assert price.metadata.source_type == SourceType.API_QUERY
assert price.metadata.confidence == 0.99
print(f"Provenance[float]: value={price.value}, source={price.metadata.source_type.value}")

# String provenance
address = Provenance(
    value="Block 123 Ang Mo Kio Ave 3",
    metadata=ProvenanceMetadata(
        source_type=SourceType.MANUAL,
        source_detail="User input via form",
        confidence=1.0,
    ),
)

assert address.value == "Block 123 Ang Mo Kio Ave 3"
assert address.metadata.source_type == SourceType.MANUAL

# Integer provenance
floor_area = Provenance(
    value=92,
    metadata=ProvenanceMetadata(
        source_type=SourceType.EXCEL_CELL,
        source_detail="HDB_data_2026.xlsx!Sheet1!C42",
        confidence=0.9,
    ),
)

assert floor_area.value == 92
assert floor_area.metadata.source_detail == "HDB_data_2026.xlsx!Sheet1!C42"

# -- 8. Provenance serialization --------------------------------------------
# to_dict() flattens value + metadata into a single dict.

price_dict = price.to_dict()
assert price_dict["value"] == 485000.0
assert price_dict["source_type"] == "api_query"
assert price_dict["confidence"] == 0.99
assert "extracted_at" in price_dict
print(f"Provenance serialized: {list(price_dict.keys())}")

# Round-trip with from_dict()
price_restored = Provenance.from_dict(price_dict)
assert price_restored.value == price.value
assert price_restored.metadata.source_type == price.metadata.source_type
assert price_restored.metadata.confidence == price.metadata.confidence
print("Provenance serialization: round-trip verified")

# -- 9. Tracking value changes ----------------------------------------------
# Use previous_value and change_reason to build an audit trail.

revised_price = Provenance(
    value=492000.0,
    metadata=ProvenanceMetadata(
        source_type=SourceType.CALCULATED,
        source_detail="Inflation adjustment model v3",
        confidence=0.85,
        previous_value=485000.0,
        change_reason="Q1 2026 inflation adjustment (+1.4%)",
    ),
)

assert revised_price.value == 492000.0
assert revised_price.metadata.previous_value == 485000.0
assert revised_price.metadata.change_reason == "Q1 2026 inflation adjustment (+1.4%)"
print(f"Value change: {revised_price.metadata.previous_value} -> {revised_price.value}")

# Serialize and check previous_value survives round-trip
revised_dict = revised_price.to_dict()
assert revised_dict["previous_value"] == 485000.0
assert revised_dict["change_reason"] == "Q1 2026 inflation adjustment (+1.4%)"

restored = Provenance.from_dict(revised_dict)
assert restored.metadata.previous_value == 485000.0

# -- 10. Agent-derived provenance -------------------------------------------
# When an AI agent produces or transforms a value, track it as
# AGENT_DERIVED with appropriate confidence.

prediction = Provenance(
    value=510000.0,
    metadata=ProvenanceMetadata(
        source_type=SourceType.AGENT_DERIVED,
        source_detail="HDBPricePredictionAgent -- XGBoost ensemble",
        confidence=0.72,
        change_reason="12-month forward prediction based on 2025-2026 trends",
    ),
)

assert prediction.metadata.source_type == SourceType.AGENT_DERIVED
assert prediction.metadata.confidence == 0.72
assert prediction.metadata.confidence < 1.0, "Predictions carry uncertainty"
print(f"Agent prediction: {prediction.value} (confidence={prediction.metadata.confidence})")

# -- 11. Edge case: None value ----------------------------------------------
# Provenance can wrap None to indicate a missing-but-tracked field.

missing = Provenance(
    value=None,
    metadata=ProvenanceMetadata(
        source_type=SourceType.DATABASE,
        source_detail="Legacy system -- field not migrated",
        confidence=0.0,
    ),
)

assert missing.value is None
assert missing.metadata.confidence == 0.0
missing_dict = missing.to_dict()
assert missing_dict["value"] is None

restored_missing = Provenance.from_dict(missing_dict)
assert restored_missing.value is None
print("None value: provenance tracks missing data with zero confidence")

print("PASS: 01-dataflow/09_provenance")
```

### Step-by-Step Explanation

1. **SourceType enum**: Seven origin types covering the most common data sources. String-backed for JSON serialization (`SourceType("manual")` works). Use `AGENT_DERIVED` for AI-generated values and `CALCULATED` for formula-derived values.

2. **ProvenanceMetadata basics**: Constructed with `source_type`, `source_detail`, and `confidence`. The `extracted_at` timestamp defaults to the current UTC time. `previous_value` and `change_reason` start as `None` and empty string respectively.

3. **Full metadata**: When a value is updated, populate `previous_value` with the old value and `change_reason` with an explanation. You can also set an explicit `extracted_at` timestamp for historical data imports.

4. **Confidence validation**: Confidence is validated at construction time. Values outside `[0.0, 1.0]`, `NaN`, and `Infinity` all raise `ValueError`. This prevents meaningless confidence scores from entering the pipeline.

5. **SourceType coercion**: The `source_type` parameter accepts both enum members (`SourceType.DATABASE`) and plain strings (`"database"`). Strings are automatically coerced to the enum, making deserialization easier.

6. **Metadata serialization**: `to_dict()` produces a JSON-compatible dictionary with string values for enums and ISO format for datetimes. `from_dict()` restores the object exactly. Round-trip fidelity is guaranteed.

7. **Provenance[T] wrapper**: `Provenance(value=485000.0, metadata=...)` wraps a float with its origin story. The generic type parameter (`Provenance[float]`, `Provenance[str]`) is for documentation and type checking -- the wrapper works with any Python type.

8. **Provenance serialization**: `to_dict()` flattens the value and all metadata fields into a single dictionary. `from_dict()` reconstructs the full `Provenance` object. The flat structure is convenient for database storage and JSON APIs.

9. **Change tracking**: Creating a new `Provenance` with `previous_value` and `change_reason` builds an audit trail. Each record documents what changed, from what, to what, and why. This survives serialization round-trips.

10. **Agent-derived provenance**: AI predictions should use `SourceType.AGENT_DERIVED` with a confidence score below 1.0 to signal uncertainty. The `source_detail` field identifies the specific agent and model version.

11. **None values**: `Provenance` can wrap `None` to represent a missing-but-tracked field. Use confidence 0.0 to indicate that the value is not trusted. This is preferable to omitting the field entirely, because you retain metadata about why it is missing.

## Common Mistakes

| Mistake                                        | Correct Pattern                                               | Why                                                                                                |
| ---------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Setting confidence > 1.0 for "very high" trust | Use 1.0 as the maximum                                        | Confidence is a probability in `[0.0, 1.0]`. Values outside this range raise `ValueError`.         |
| Discarding provenance and passing raw values   | Pass the `Provenance` object through your pipeline            | Raw values lose their origin story. Keep provenance attached until the point of final consumption. |
| Using `MANUAL` for AI-generated values         | Use `AGENT_DERIVED` with appropriate confidence               | Misclassifying the source type undermines trust decisions downstream.                              |
| Forgetting `previous_value` on updates         | Always set `previous_value` and `change_reason` when updating | Without change tracking, you cannot reconstruct the audit trail or explain why a value changed.    |

## Exercises

1. Create a pipeline that ingests three property prices from different sources (API, Excel, manual entry). Wrap each in `Provenance[float]` with appropriate source types and confidence scores. Serialize all three to dicts and print them.
2. Simulate a value being updated three times (original API fetch, agent adjustment, manual correction). Create a chain of `Provenance` objects where each one's `previous_value` points to the prior value. Print the full audit trail.
3. Write a function that takes a list of `Provenance[float]` objects and returns only those with confidence above a threshold. Test it with values at confidence 0.0, 0.5, 0.8, and 1.0.

## Key Takeaways

- `SourceType` classifies data origins with seven string-backed enum members
- `ProvenanceMetadata` captures source type, detail, confidence, timestamp, and change history
- Confidence is validated at construction: must be a finite number in `[0.0, 1.0]`
- `Provenance[T]` wraps any typed value with its metadata -- keep them paired through your pipeline
- `to_dict()` / `from_dict()` provide full round-trip serialization for both metadata and provenance
- `previous_value` and `change_reason` build audit trails across value updates
- `AGENT_DERIVED` with sub-1.0 confidence signals AI-generated values with uncertainty
- `Provenance(value=None, ...)` tracks missing data explicitly rather than omitting it
