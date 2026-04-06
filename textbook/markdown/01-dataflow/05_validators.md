# Chapter 5: Field-Level Validators

## Overview

Data integrity starts before records reach the database. DataFlow's validation module provides a library of **field-level validators** -- reusable functions that check whether a value meets a specific constraint (valid email, within a numeric range, matches a regex, etc.). You attach validators to model classes with the `@field_validator` decorator, then run them all at once with `validate_model()`. This chapter teaches you the full validation toolkit: simple validators, factory validators, model decoration, and result inspection.

## Prerequisites

- Python 3.10+ installed
- Kailash DataFlow installed (`pip install kailash-dataflow`)
- Completed Chapters 1-4 (model definition and CRUD)
- Familiarity with Python decorators and dataclasses

## Concepts

### Concept 1: Simple Validators

Simple validators are standalone functions that take a single value and return `True` or `False`. They require no configuration -- just call them directly. Examples include `email_validator`, `url_validator`, `uuid_validator`, and `phone_validator`.

- **What**: Predicate functions `(value) -> bool` that check a single constraint
- **Why**: Common data formats (email, URL, UUID, phone) have well-defined rules. Built-in validators save you from writing and maintaining regex patterns yourself
- **How**: Each validator applies a format-specific check (RFC 5322 for email, scheme check for URL, etc.) and returns a boolean. Non-string inputs return `False` immediately
- **When**: Use simple validators for standard format checks. If you need configurable bounds or patterns, use factory validators instead

### Concept 2: Factory Validators

Factory validators are functions that take configuration parameters and **return** a validator function. `length_validator(min_len=1, max_len=50)` returns a callable that checks string (or sequence) length. `range_validator(min_val=0, max_val=150)` returns a callable that checks numeric bounds.

- **What**: Higher-order functions that produce configured validator callables
- **Why**: Constraints like "between 0 and 150" or "matches pattern `[A-Z]{2}-\d{4}`" vary per field. Factories let you create purpose-built validators from reusable templates
- **How**: The factory captures configuration in a closure and returns a function with the same `(value) -> bool` signature as simple validators
- **When**: Use `range_validator` for numeric bounds, `length_validator` for string/list length, `pattern_validator` for regex, and `one_of_validator` for enum-like membership

### Concept 3: The @field_validator Decorator

`@field_validator(field_name, validator_fn)` attaches a validator to a specific field on a class. Stack multiple decorators to validate multiple fields. The validators are stored in a `__field_validators__` list on the class.

### Concept 4: validate_model() and ValidationResult

`validate_model(instance)` runs all attached validators against an instance's field values. It collects **all** errors (not fail-fast) and returns a `ValidationResult` with a `valid` boolean and an `errors` list of `FieldValidationError` objects.

- **What**: A function that runs every registered validator on a model instance and aggregates the results
- **Why**: Collecting all errors at once lets you report every problem in a single pass, rather than forcing users to fix one error, resubmit, and discover the next
- **How**: Iterates through `__field_validators__`, calls each one with the corresponding field value, and builds a `ValidationResult`
- **When**: Call `validate_model()` before persisting data -- in form handlers, API endpoints, or import pipelines

### Key API

| Function / Class             | Parameters                               | Returns            | Description                            |
| ---------------------------- | ---------------------------------------- | ------------------ | -------------------------------------- |
| `email_validator()`          | `value`                                  | `bool`             | RFC 5322 simplified email check        |
| `url_validator()`            | `value`                                  | `bool`             | HTTP/HTTPS URL check                   |
| `uuid_validator()`           | `value`                                  | `bool`             | UUID format check                      |
| `phone_validator()`          | `value`                                  | `bool`             | E.164 and common phone formats         |
| `length_validator()`         | `min_len: int`, `max_len: int`           | `Callable`         | Factory: string/sequence length bounds |
| `range_validator()`          | `min_val: float`, `max_val: float`       | `Callable`         | Factory: numeric range bounds          |
| `pattern_validator()`        | `pattern: str`                           | `Callable`         | Factory: regex fullmatch               |
| `one_of_validator()`         | `allowed: list`                          | `Callable`         | Factory: set membership                |
| `@field_validator()`         | `field_name: str`, `validator: Callable` | decorator          | Attach a validator to a class field    |
| `validate_model()`           | `instance`                               | `ValidationResult` | Run all validators on an instance      |
| `ValidationResult.valid`     | --                                       | `bool`             | True if no errors                      |
| `ValidationResult.errors`    | --                                       | `list`             | List of `FieldValidationError`         |
| `ValidationResult.to_dict()` | --                                       | `dict`             | Serialize result for JSON responses    |
| `ValidationResult.merge()`   | `other: ValidationResult`                | `None`             | Combine errors from another result     |

## Code Walkthrough

```python
from __future__ import annotations

from dataclasses import dataclass

from dataflow.validation import (
    FieldValidationError,
    ValidationResult,
    email_validator,
    field_validator,
    length_validator,
    one_of_validator,
    pattern_validator,
    phone_validator,
    range_validator,
    url_validator,
    uuid_validator,
    validate_model,
)

# -- 1. Simple validators -- value -> bool ----------------------------------
# These validators take a single value and return True/False.
# They are used directly, not as factories.

# Email validator: RFC 5322 simplified pattern
assert email_validator("alice@example.com") is True
assert email_validator("bob@terrene.org") is True
assert email_validator("not-an-email") is False
assert email_validator("") is False
assert email_validator(42) is False, "Non-string returns False"

# URL validator: requires http/https scheme
assert url_validator("https://terrene.org") is True
assert url_validator("http://example.com/path?q=1") is True
assert url_validator("ftp://files.example.com") is False, "Only http/https"
assert url_validator("example.com") is False, "Scheme required"

# UUID validator: any UUID version
assert uuid_validator("550e8400-e29b-41d4-a716-446655440000") is True
assert uuid_validator("not-a-uuid") is False

# Phone validator: E.164 and common formats
assert phone_validator("+65 6234 5678") is True
assert phone_validator("+1-555-123-4567") is True
assert phone_validator("12345") is False, "Too short"

print("Simple validators: all checks passed")

# -- 2. Factory validators -- return a Callable ------------------------------
# These take configuration parameters and return a validator function.

# length_validator: checks string/sequence length bounds
check_name = length_validator(min_len=1, max_len=50)
assert check_name("Alice") is True
assert check_name("") is False, "Below min_len"
assert check_name("A" * 51) is False, "Above max_len"

# range_validator: checks numeric value bounds
check_age = range_validator(min_val=0, max_val=150)
assert check_age(25) is True
assert check_age(-1) is False, "Below min_val"
assert check_age(200) is False, "Above max_val"
assert check_age("twenty") is False, "Non-numeric returns False"

# Unbounded: only min or only max
check_positive = range_validator(min_val=0)
assert check_positive(0) is True
assert check_positive(999999) is True
assert check_positive(-1) is False

# pattern_validator: checks string against a regex (fullmatch)
check_code = pattern_validator(r"[A-Z]{2}-\d{4}")
assert check_code("SG-1234") is True
assert check_code("sg-1234") is False, "Case-sensitive fullmatch"
assert check_code("SG-12345") is False, "Extra digit fails fullmatch"

# one_of_validator: checks membership in an allowed set
check_tier = one_of_validator(["free", "pro", "enterprise"])
assert check_tier("pro") is True
assert check_tier("premium") is False, "Not in allowed set"

print("Factory validators: all checks passed")

# -- 3. Decorating a model class with @field_validator ----------------------
# Stack multiple @field_validator decorators on a class.
# Validators are stored in __field_validators__ and run by validate_model().

@field_validator("email", email_validator)
@field_validator("name", length_validator(min_len=1, max_len=100))
@field_validator("age", range_validator(min_val=0, max_val=150))
@field_validator("tier", one_of_validator(["free", "pro", "enterprise"]))
@dataclass
class UserProfile:
    name: str = ""
    email: str = ""
    age: int = 0
    tier: str = "free"

# Verify validators were registered on the class
assert hasattr(UserProfile, "__field_validators__"), "Validators attached to class"
assert len(UserProfile.__field_validators__) == 4, "Four validators registered"

# -- 4. validate_model() -- run all validators on an instance ----------------
# Returns a ValidationResult that collects ALL errors (not fail-fast).

# Valid instance
good_user = UserProfile(name="Alice", email="alice@example.com", age=30, tier="pro")
result = validate_model(good_user)

assert isinstance(result, ValidationResult)
assert result.valid is True, "All validators pass"
assert len(result.errors) == 0
print(f"Valid user: valid={result.valid}, errors={len(result.errors)}")

# Invalid instance -- multiple errors collected
bad_user = UserProfile(name="", email="not-an-email", age=-5, tier="premium")
result = validate_model(bad_user)

assert result.valid is False, "Validation fails"
assert len(result.errors) == 4, "All four fields fail validation"
print(f"Invalid user: valid={result.valid}, errors={len(result.errors)}")

# -- 5. Inspecting ValidationResult and FieldValidationError ----------------
# Each error is a FieldValidationError with field, message, validator, value.

for err in result.errors:
    assert isinstance(err, FieldValidationError)
    assert isinstance(err.field, str)
    assert isinstance(err.message, str)
    assert isinstance(err.validator, str)

# Check specific errors
error_fields = {e.field for e in result.errors}
assert "name" in error_fields, "name validation failed"
assert "email" in error_fields, "email validation failed"
assert "age" in error_fields, "age validation failed"
assert "tier" in error_fields, "tier validation failed"

# Serialize to dict
result_dict = result.to_dict()
assert result_dict["valid"] is False
assert len(result_dict["errors"]) == 4
print(f"Serialized result: valid={result_dict['valid']}")

# -- 6. Partial validation -- only some fields fail -------------------------

partial_bad = UserProfile(name="Bob", email="bad", age=25, tier="free")
result = validate_model(partial_bad)

assert result.valid is False
assert len(result.errors) == 1, "Only email fails"
assert result.errors[0].field == "email"
print(f"Partial invalid: {result.errors[0].field} failed")

# -- 7. Merging results -----------------------------------------------------
# ValidationResult.merge() combines errors from multiple validations.

result_a = ValidationResult()
result_a.add_error(
    field_name="field_a",
    message="Field A invalid",
    validator="custom",
    value="bad",
)

result_b = ValidationResult()
result_b.add_error(
    field_name="field_b",
    message="Field B invalid",
    validator="custom",
    value="worse",
)

result_a.merge(result_b)
assert len(result_a.errors) == 2, "Merged errors from both results"
assert result_a.valid is False

# -- 8. Edge cases -----------------------------------------------------------

# range_validator rejects NaN and infinity
import math

check_finite = range_validator(min_val=0, max_val=100)
assert check_finite(float("nan")) is False, "NaN rejected"
assert check_finite(float("inf")) is False, "Infinity rejected"

# length_validator works on lists too
check_items = length_validator(min_len=1, max_len=5)
assert check_items([1, 2, 3]) is True
assert check_items([]) is False, "Empty list below min_len"

# Invalid range_validator construction
try:
    range_validator(min_val=100, max_val=0)
    assert False, "Should raise ValueError"
except ValueError:
    pass  # Expected: min_val > max_val

# Invalid length_validator construction
try:
    length_validator(min_len=10, max_len=5)
    assert False, "Should raise ValueError"
except ValueError:
    pass  # Expected: min_len > max_len

print("Edge cases: all checks passed")

print("PASS: 01-dataflow/05_validators")
```

### Step-by-Step Explanation

1. **Simple validators**: `email_validator`, `url_validator`, `uuid_validator`, and `phone_validator` are ready-to-use predicate functions. Pass a value, get `True` or `False`. They handle type checking internally -- passing a non-string to `email_validator` returns `False`, not an exception.

2. **Factory validators**: `length_validator(min_len=1, max_len=50)` returns a new function that checks length bounds. `range_validator`, `pattern_validator`, and `one_of_validator` work the same way. You can omit one bound (e.g., `range_validator(min_val=0)`) for open-ended checks.

3. **Model decoration**: Stack `@field_validator("field_name", validator_fn)` decorators on a dataclass. Each decorator registers a validator for one field. The validators are stored in `__field_validators__` on the class -- not on instances.

4. **Running validation**: `validate_model(instance)` iterates through all registered validators, calls each one with the corresponding field value, and collects every error. It does not stop at the first failure -- you get the complete picture.

5. **Inspecting errors**: Each `FieldValidationError` has `field` (which field failed), `message` (human-readable explanation), `validator` (which validator caught it), and `value` (the offending value). Use `result.to_dict()` to serialize for API responses.

6. **Partial failures**: When only some fields are invalid, the result contains only those errors. Valid fields do not appear in the error list.

7. **Merging results**: `result_a.merge(result_b)` combines errors from two separate validation passes into one result. Useful when you validate different aspects of a model in different stages.

8. **Edge cases**: `range_validator` correctly rejects `NaN` and `Infinity`. `length_validator` works on any object with `len()` -- strings, lists, tuples. Constructing a factory with inverted bounds (`min > max`) raises `ValueError` immediately.

## Common Mistakes

| Mistake                                                | Correct Pattern                                                      | Why                                                                                           |
| ------------------------------------------------------ | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Calling `validate_model` on the class, not instance    | `validate_model(UserProfile(...))` not `validate_model(UserProfile)` | Validation reads field values from an instance, not the class definition.                     |
| Using `range_validator(0, 100)` as a direct check      | `check = range_validator(min_val=0, max_val=100); check(50)`         | Factory validators return a function -- you must call the returned function, not the factory. |
| Assuming fail-fast behavior                            | All errors are collected in one pass                                 | `validate_model` runs every validator. Do not assume early termination on first error.        |
| Forgetting `@dataclass` under the validator decorators | Put `@dataclass` at the bottom of the decorator stack                | `@field_validator` decorates a class. `@dataclass` must be applied first (closest to class).  |

## Exercises

1. Create a `RegistrationForm` dataclass with `username: str` (3-20 chars), `email: str`, `password: str` (8-128 chars, must match `.*[A-Z].*[0-9].*` or similar), and `role: str` (one of `["user", "admin", "moderator"]`). Attach validators and test with both valid and invalid instances.
2. Write a custom validator function `def even_number(value) -> bool` that returns `True` only for even integers. Attach it to a model field using `@field_validator` and verify it works with `validate_model()`.
3. Use `ValidationResult.merge()` to combine validation results from two different models into a single report. Print all errors with their field names and messages.

## Key Takeaways

- Simple validators (`email_validator`, `url_validator`, etc.) are direct `(value) -> bool` functions
- Factory validators (`range_validator`, `length_validator`, etc.) return configured validator functions
- `@field_validator` attaches validators to class fields; stack multiple decorators for multiple fields
- `validate_model()` collects all errors in one pass -- it never fails fast
- `ValidationResult` has `.valid`, `.errors`, `.to_dict()`, and `.merge()` for complete error handling
- Factory validators reject inverted bounds at construction time (`ValueError`)
- `range_validator` correctly handles `NaN` and `Infinity`

## Next Chapter

[Chapter 6: Data Classification](06_classification.md) -- Apply sensitivity levels, retention policies, and masking strategies to model fields for data governance.
