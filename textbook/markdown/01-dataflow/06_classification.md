# Chapter 6: Data Classification

## Overview

Data governance requires knowing **what kind of data** each field holds, **how long** it should be retained, and **how** it should be obscured when displayed to unauthorized users. DataFlow's classification module provides enums for sensitivity levels, retention policies, and masking strategies, plus a `@classify` decorator and a `ClassificationPolicy` registry to manage these metadata at runtime. This chapter teaches you how to classify fields, look up their metadata, and enforce retention policies.

## Prerequisites

- Python 3.10+ installed
- Kailash DataFlow installed (`pip install kailash-dataflow`)
- Completed Chapters 1-5 (model definition, CRUD, and validation)
- Familiarity with Python enums and dataclasses

## Concepts

### Concept 1: DataClassification Levels

The `DataClassification` enum defines six sensitivity levels, ordered from least to most sensitive: `PUBLIC`, `INTERNAL`, `SENSITIVE`, `PII`, `GDPR`, and `HIGHLY_CONFIDENTIAL`. Each level is a string-backed enum member, making it JSON-serializable without conversion.

- **What**: An enum of data sensitivity tiers that categorize how carefully a field must be handled
- **Why**: Different fields require different access controls, audit requirements, and handling procedures. A customer's name (PII) requires different treatment than a product category (PUBLIC)
- **How**: String-backed enum (`DataClassification("pii")` works) so classifications can be stored in JSON, databases, and config files without conversion
- **When**: Classify every field that holds personal, sensitive, or regulated data. Default unclassified fields to PUBLIC

### Concept 2: RetentionPolicy

The `RetentionPolicy` enum defines how long data should be retained: `INDEFINITE`, `DAYS_30`, `DAYS_90`, `YEARS_1`, `YEARS_7`, and `UNTIL_CONSENT_REVOKED`. These map to concrete day counts via the `ClassificationPolicy.retention_days_for_policy()` helper.

### Concept 3: MaskingStrategy

The `MaskingStrategy` enum defines how field values are obscured: `NONE` (no masking), `HASH` (one-way hash), `REDACT` (replace with placeholder), `LAST_FOUR` (show only last 4 characters), and `ENCRYPT` (reversible encryption).

### Concept 4: The @classify Decorator

`@classify(field_name, classification, retention, masking)` attaches classification metadata to a field on a class. Stack multiple decorators for multiple fields. Metadata is stored in `__field_classifications__` on the class. `retention` defaults to `INDEFINITE` and `masking` defaults to `NONE` if omitted.

- **What**: A decorator that attaches governance metadata (sensitivity, retention, masking) to a specific field on a class
- **Why**: Centralizing classification on the model definition ensures that governance rules travel with the data schema, not in separate configuration files that drift out of sync
- **How**: Each decorator call stores a `FieldClassification` record in the class's `__field_classifications__` dict
- **When**: Use `@classify` when defining models that contain personal, sensitive, or regulated data. Apply it alongside `@db.model` or `@dataclass`

### Concept 5: ClassificationPolicy

`ClassificationPolicy` is a runtime registry that maps model fields to their classification metadata. You can register decorated models via `policy.register_model(MyClass)` or set fields programmatically via `policy.set_field()`. The policy provides lookup methods for classification level, retention days, and full field metadata.

### Key API

| Class / Function                     | Parameters                                                                              | Returns                | Description                                  |
| ------------------------------------ | --------------------------------------------------------------------------------------- | ---------------------- | -------------------------------------------- |
| `DataClassification`                 | enum: `PUBLIC`, `INTERNAL`, `SENSITIVE`, `PII`, `GDPR`, `HIGHLY_CONFIDENTIAL`           | --                     | Sensitivity level enum                       |
| `RetentionPolicy`                    | enum: `INDEFINITE`, `DAYS_30`, `DAYS_90`, `YEARS_1`, `YEARS_7`, `UNTIL_CONSENT_REVOKED` | --                     | Data retention period enum                   |
| `MaskingStrategy`                    | enum: `NONE`, `HASH`, `REDACT`, `LAST_FOUR`, `ENCRYPT`                                  | --                     | Value obscuring strategy enum                |
| `@classify()`                        | `field`, `classification`, `retention=INDEFINITE`, `masking=NONE`                       | decorator              | Attach classification to a class field       |
| `get_field_classification()`         | `cls`, `field_name: str`                                                                | `FieldClassification`  | Look up a field's classification metadata    |
| `FieldClassification.to_dict()`      | --                                                                                      | `dict`                 | Serialize to JSON-compatible dict            |
| `FieldClassification.from_dict()`    | `data: dict`                                                                            | `FieldClassification`  | Deserialize from dict                        |
| `ClassificationPolicy()`             | --                                                                                      | `ClassificationPolicy` | Create a runtime policy registry             |
| `policy.register_model()`            | `cls`                                                                                   | `None`                 | Register a decorated model's classifications |
| `policy.classify()`                  | `model_name: str`, `field_name: str`                                                    | `str`                  | Look up classification level as string       |
| `policy.set_field()`                 | `model_name`, `field_name`, `classification`, `retention`, `masking`                    | `None`                 | Programmatically classify a field            |
| `policy.get_retention_days()`        | `classification_level: str`                                                             | `int` or `None`        | Default retention days for a level           |
| `policy.retention_days_for_policy()` | `policy: RetentionPolicy`                                                               | `int` or `None`        | Convert retention enum to day count          |

## Code Walkthrough

```python
from __future__ import annotations

from dataclasses import dataclass

from dataflow.classification import (
    ClassificationPolicy,
    DataClassification,
    FieldClassification,
    MaskingStrategy,
    RetentionPolicy,
    classify,
    get_field_classification,
)

# -- 1. DataClassification enum -- sensitivity levels -----------------------
# Ordered from least to most sensitive.
# All enums are str-backed for JSON-friendly serialization.

assert DataClassification.PUBLIC.value == "public"
assert DataClassification.INTERNAL.value == "internal"
assert DataClassification.SENSITIVE.value == "sensitive"
assert DataClassification.PII.value == "pii"
assert DataClassification.GDPR.value == "gdpr"
assert DataClassification.HIGHLY_CONFIDENTIAL.value == "highly_confidential"

# String-backed: can construct from string
assert DataClassification("pii") == DataClassification.PII

print("DataClassification levels: PUBLIC -> HIGHLY_CONFIDENTIAL")

# -- 2. RetentionPolicy enum -- how long data is kept ----------------------

assert RetentionPolicy.INDEFINITE.value == "indefinite"
assert RetentionPolicy.DAYS_30.value == "days_30"
assert RetentionPolicy.DAYS_90.value == "days_90"
assert RetentionPolicy.YEARS_1.value == "years_1"
assert RetentionPolicy.YEARS_7.value == "years_7"
assert RetentionPolicy.UNTIL_CONSENT_REVOKED.value == "until_consent_revoked"

print("RetentionPolicy: INDEFINITE, DAYS_30, DAYS_90, YEARS_1, YEARS_7, UNTIL_CONSENT_REVOKED")

# -- 3. MaskingStrategy enum -- how field values are obscured ---------------

assert MaskingStrategy.NONE.value == "none"
assert MaskingStrategy.HASH.value == "hash"
assert MaskingStrategy.REDACT.value == "redact"
assert MaskingStrategy.LAST_FOUR.value == "last_four"
assert MaskingStrategy.ENCRYPT.value == "encrypt"

print("MaskingStrategy: NONE, HASH, REDACT, LAST_FOUR, ENCRYPT")

# -- 4. @classify decorator -- attach classification to a class -------------
# Stack multiple @classify decorators on a class.
# Each decorator classifies one field with sensitivity, retention,
# and masking metadata.

@classify(
    "email",
    DataClassification.PII,
    RetentionPolicy.UNTIL_CONSENT_REVOKED,
    MaskingStrategy.REDACT,
)
@classify(
    "name",
    DataClassification.PII,
    RetentionPolicy.YEARS_1,
    MaskingStrategy.REDACT,
)
@classify(
    "phone",
    DataClassification.PII,
    RetentionPolicy.YEARS_1,
    MaskingStrategy.LAST_FOUR,
)
@classify(
    "notes",
    DataClassification.INTERNAL,
    RetentionPolicy.INDEFINITE,
    MaskingStrategy.NONE,
)
@dataclass
class Customer:
    name: str = ""
    email: str = ""
    phone: str = ""
    notes: str = ""

# Verify classification metadata is stored on the class
assert hasattr(Customer, "__field_classifications__")
assert len(Customer.__field_classifications__) == 4

# -- 5. get_field_classification() -- look up a field's metadata ------------
# Returns a FieldClassification dataclass or None for unclassified fields.

email_fc = get_field_classification(Customer, "email")
assert email_fc is not None
assert isinstance(email_fc, FieldClassification)
assert email_fc.classification == DataClassification.PII
assert email_fc.retention == RetentionPolicy.UNTIL_CONSENT_REVOKED
assert email_fc.masking == MaskingStrategy.REDACT
print(f"email: classification={email_fc.classification.value}, masking={email_fc.masking.value}")

phone_fc = get_field_classification(Customer, "phone")
assert phone_fc is not None
assert phone_fc.masking == MaskingStrategy.LAST_FOUR

notes_fc = get_field_classification(Customer, "notes")
assert notes_fc is not None
assert notes_fc.classification == DataClassification.INTERNAL
assert notes_fc.masking == MaskingStrategy.NONE

# Unclassified field returns None
unknown_fc = get_field_classification(Customer, "nonexistent_field")
assert unknown_fc is None, "Unclassified field returns None"

# -- 6. FieldClassification -- serialization --------------------------------

fc_dict = email_fc.to_dict()
assert fc_dict == {
    "classification": "pii",
    "retention": "until_consent_revoked",
    "masking": "redact",
}

# Round-trip: dict -> FieldClassification
fc_restored = FieldClassification.from_dict(fc_dict)
assert fc_restored == email_fc, "Round-trip serialization preserves data"
print("FieldClassification serialization: round-trip verified")

# -- 7. ClassificationPolicy -- runtime policy registry ---------------------
# ClassificationPolicy maps model fields to classification metadata.
# Can register models decorated with @classify or set fields manually.

policy = ClassificationPolicy()

# Register a decorated model
policy.register_model(Customer)

# Look up classification level as a string
level = policy.classify("Customer", "email")
assert level == "pii", f"Expected 'pii', got '{level}'"

level = policy.classify("Customer", "notes")
assert level == "internal"

# Unclassified field defaults to "public"
level = policy.classify("Customer", "nonexistent")
assert level == "public", "Unclassified defaults to public"

# Unregistered model also defaults to "public"
level = policy.classify("UnknownModel", "field")
assert level == "public"

print("ClassificationPolicy: classify() lookups verified")

# -- 8. ClassificationPolicy -- programmatic field setting ------------------
# Use set_field() to classify fields without the @classify decorator.

policy.set_field(
    "Order",
    "credit_card",
    DataClassification.HIGHLY_CONFIDENTIAL,
    RetentionPolicy.DAYS_90,
    MaskingStrategy.ENCRYPT,
)

cc_level = policy.classify("Order", "credit_card")
assert cc_level == "highly_confidential"

cc_fc = policy.get_field("Order", "credit_card")
assert cc_fc is not None
assert cc_fc.retention == RetentionPolicy.DAYS_90
assert cc_fc.masking == MaskingStrategy.ENCRYPT

# -- 9. ClassificationPolicy -- get all fields for a model ------------------

customer_fields = policy.get_model_fields("Customer")
assert len(customer_fields) == 4, "Four classified fields"
assert "email" in customer_fields
assert "name" in customer_fields
assert "phone" in customer_fields
assert "notes" in customer_fields

print(f"Customer classified fields: {list(customer_fields.keys())}")

# -- 10. Retention days lookup -----------------------------------------------
# Convert RetentionPolicy enum to concrete days.

days = policy.retention_days_for_policy(RetentionPolicy.DAYS_30)
assert days == 30

days = policy.retention_days_for_policy(RetentionPolicy.YEARS_1)
assert days == 365

days = policy.retention_days_for_policy(RetentionPolicy.YEARS_7)
assert days == 2555

# Indefinite and consent-based return None (no fixed expiry)
days = policy.retention_days_for_policy(RetentionPolicy.INDEFINITE)
assert days is None

days = policy.retention_days_for_policy(RetentionPolicy.UNTIL_CONSENT_REVOKED)
assert days is None

# Default retention days per classification level
days = policy.get_retention_days("sensitive")
assert days == 365, "Sensitive default: 1 year"

days = policy.get_retention_days("highly_confidential")
assert days == 2555, "Highly confidential default: ~7 years"

days = policy.get_retention_days("public")
assert days is None, "Public: no retention limit"

print("Retention days: lookups verified")

# -- 11. Edge case: @classify with defaults ---------------------------------
# retention defaults to INDEFINITE, masking defaults to NONE

@classify("status", DataClassification.INTERNAL)
@dataclass
class MinimalModel:
    status: str = ""

fc = get_field_classification(MinimalModel, "status")
assert fc is not None
assert fc.classification == DataClassification.INTERNAL
assert fc.retention == RetentionPolicy.INDEFINITE, "Default retention"
assert fc.masking == MaskingStrategy.NONE, "Default masking"

print("PASS: 01-dataflow/06_classification")
```

### Step-by-Step Explanation

1. **DataClassification enum**: Six levels from `PUBLIC` to `HIGHLY_CONFIDENTIAL`. All are string-backed, so `DataClassification("pii")` works for deserialization from JSON or databases.

2. **RetentionPolicy enum**: Six options defining how long data should be kept. `INDEFINITE` and `UNTIL_CONSENT_REVOKED` have no fixed day count (they return `None` from the days lookup).

3. **MaskingStrategy enum**: Five strategies for obscuring values. `NONE` means no masking. `LAST_FOUR` is typical for phone numbers and credit cards. `ENCRYPT` is for reversible protection of highly confidential data.

4. **@classify decorator**: Stack decorators to classify each field independently. The decorator takes the field name, classification level, retention policy, and masking strategy. `retention` and `masking` have sensible defaults (`INDEFINITE` and `NONE`).

5. **get_field_classification()**: Returns a `FieldClassification` dataclass for a specific field on a class, or `None` if the field is not classified. This is the direct lookup -- no policy registry needed.

6. **Serialization**: `FieldClassification.to_dict()` produces a JSON-compatible dictionary. `FieldClassification.from_dict()` restores the object. Round-trip fidelity is guaranteed.

7. **ClassificationPolicy**: A runtime registry that aggregates classifications across models. `register_model()` imports all classifications from a decorated class. `classify()` returns the level as a string, defaulting to `"public"` for unknown fields or models.

8. **Programmatic classification**: `policy.set_field()` classifies fields without using the decorator. This is useful for dynamic models, third-party classes, or configuration-driven classification.

9. **Model field listing**: `policy.get_model_fields()` returns all classified fields for a model, useful for building data maps and compliance reports.

10. **Retention days**: `retention_days_for_policy()` converts enum values to concrete day counts. `get_retention_days()` provides default retention by classification level. Both return `None` for policies without a fixed expiry.

11. **Default parameters**: When `@classify` is called with only the field name and classification, retention defaults to `INDEFINITE` and masking defaults to `NONE`.

## Common Mistakes

| Mistake                                                  | Correct Pattern                                                      | Why                                                                                                                    |
| -------------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Classifying in code but not in policy                    | Call `policy.register_model(MyClass)` after decorating               | The `@classify` decorator stores metadata on the class. The policy registry is separate.                               |
| Assuming unclassified fields raise errors                | `policy.classify()` returns `"public"` for unknown fields            | The safe default is public -- no exception is raised. Design your access controls accordingly.                         |
| Using `retention_days_for_policy()` without `None` check | Always handle `None` return for `INDEFINITE`/`UNTIL_CONSENT_REVOKED` | These policies have no fixed day count. Using the result in date arithmetic without a `None` check causes `TypeError`. |
| Hardcoding classification strings                        | Use `DataClassification.PII` (the enum member)                       | Typos in strings (`"Pii"`, `"PII"`) silently create invalid classifications.                                           |

## Exercises

1. Define a `PatientRecord` model with fields `name`, `email`, `diagnosis`, `insurance_id`, and `notes`. Classify each field with an appropriate sensitivity level, retention policy, and masking strategy. Register it with a `ClassificationPolicy` and verify all lookups.
2. Use `policy.set_field()` to add a `"ssn"` field to the `PatientRecord` model classification at runtime (without modifying the class). Classify it as `HIGHLY_CONFIDENTIAL` with `DAYS_90` retention and `ENCRYPT` masking. Verify the classification.
3. Write a function that takes a `ClassificationPolicy` and a model name, and prints a data governance report showing each field's classification, retention in days, and masking strategy.

## Key Takeaways

- `DataClassification`, `RetentionPolicy`, and `MaskingStrategy` are string-backed enums for JSON-safe governance metadata
- `@classify` attaches field-level governance metadata directly on the model class
- `get_field_classification()` provides direct lookups; `ClassificationPolicy` provides a centralized registry
- Unclassified fields default to `"public"` -- no exceptions for missing metadata
- `retention` defaults to `INDEFINITE` and `masking` defaults to `NONE` when omitted from `@classify`
- `FieldClassification` supports full round-trip serialization via `to_dict()` / `from_dict()`
- `ClassificationPolicy` supports both decorator-based and programmatic field classification

## Next Chapter

[Chapter 7: Transaction Management](07_transactions.md) -- Use context-managed transactions for atomic operations with commit, rollback, and isolation level control.
