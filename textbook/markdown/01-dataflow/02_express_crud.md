# Chapter 2: Express CRUD

## Overview

The Express API (`db.express`) is the fastest way to perform single-record Create, Read, Update, and Delete operations in DataFlow. Instead of building workflows for simple operations, Express provides direct async methods that bypass graph construction entirely -- roughly 23x faster than the workflow equivalent. This chapter teaches you all five Express operations: create, read, update, delete, and list.

## Prerequisites

- Python 3.10+ installed
- Kailash DataFlow installed (`pip install kailash-dataflow`)
- Completed Chapter 1 (model definition and initialization)

## Concepts

### Concept 1: The Express API

Express is DataFlow's convenience layer for straightforward CRUD. It lives at `db.express` and provides async methods that operate on a single model at a time. Every method takes the model name as a string and returns plain dictionaries.

- **What**: A set of async methods for direct database operations without workflow overhead
- **Why**: Building a full workflow graph for a single insert or read is wasteful -- Express gives you the same result with a direct function call
- **How**: `db.express` delegates to the database connection pool directly, skipping graph construction, node instantiation, and runtime execution
- **When**: Use Express for all single-record or simple list operations. Switch to WorkflowBuilder only when you need multi-step pipelines with branching or transformations

### Concept 2: Auto-Generated IDs

When you create a record via `express.create()`, DataFlow automatically generates a unique `id` field. You do not declare `id` in your model -- it is added implicitly. The returned dictionary always includes the `id` key.

### Concept 3: Default Values

Fields with defaults in the model definition are applied automatically during creation. If you omit `done` from the create dict and the model declares `done: bool = False`, the record will have `done = False`.

### Key API

| Method             | Parameters                                   | Returns          | Description                 |
| ------------------ | -------------------------------------------- | ---------------- | --------------------------- |
| `express.create()` | `model_name: str`, `data: dict`              | `dict`           | Insert a new record         |
| `express.read()`   | `model_name: str`, `id: str`                 | `dict` or `None` | Fetch a single record by ID |
| `express.update()` | `model_name: str`, `id: str`, `fields: dict` | `dict`           | Modify fields on a record   |
| `express.delete()` | `model_name: str`, `id: str`                 | `bool`           | Remove a record             |
| `express.list()`   | `model_name: str`, `order_by: str = None`    | `list[dict]`     | Fetch all records           |
| `express.count()`  | `model_name: str`                            | `int`            | Count total records         |

## Code Walkthrough

```python
from __future__ import annotations

import asyncio
import os
import tempfile

from dataflow import DataFlow

# -- Setup -------------------------------------------------------------------

db_path = os.path.join(tempfile.gettempdir(), "textbook_dataflow_02.db")
db = DataFlow(database_url=f"sqlite:///{db_path}")

@db.model
class Task:
    title: str
    description: str
    priority: int
    done: bool = False

async def main():
    await db.initialize()

    # -- 1. CREATE -- insert a new record ------------------------------------
    # express.create(model_name, data_dict) returns the created record
    # with an auto-generated id.

    task1 = await db.express.create(
        "Task",
        {
            "title": "Learn DataFlow",
            "description": "Complete the textbook tutorial",
            "priority": 1,
        },
    )

    assert "id" in task1, "Created record gets an auto-generated id"
    assert task1["title"] == "Learn DataFlow"
    assert task1["done"] is False, "Default value applied"
    print(f"Created: {task1}")

    # Create more records
    task2 = await db.express.create(
        "Task",
        {
            "title": "Build ML Pipeline",
            "description": "Use TrainingPipeline engine",
            "priority": 2,
        },
    )
    task3 = await db.express.create(
        "Task",
        {
            "title": "Deploy with Nexus",
            "description": "Multi-channel deployment",
            "priority": 3,
        },
    )

    # -- 2. READ -- fetch a single record by id -----------------------------
    # express.read(model_name, id) returns the record or None.

    fetched = await db.express.read("Task", task1["id"])

    assert fetched is not None, "Record found by id"
    assert fetched["title"] == "Learn DataFlow"
    assert fetched["id"] == task1["id"]
    print(f"Read: {fetched}")

    # -- 3. UPDATE -- modify an existing record ------------------------------
    # express.update(model_name, id, fields_dict) returns updated record.

    updated = await db.express.update(
        "Task",
        task1["id"],
        {"done": True, "priority": 0},
    )

    assert updated["done"] is True, "Field updated"
    assert updated["priority"] == 0, "Priority updated"
    print(f"Updated: {updated}")

    # Verify the update persisted
    refetched = await db.express.read("Task", task1["id"])
    assert refetched["done"] is True

    # -- 4. DELETE -- remove a record ----------------------------------------
    # express.delete(model_name, id) returns True on success.

    deleted = await db.express.delete("Task", task3["id"])
    assert deleted is True, "Delete returns True"

    # Verify deletion
    gone = await db.express.read("Task", task3["id"])
    assert gone is None, "Deleted record returns None on read"

    # -- 5. LIST -- fetch multiple records -----------------------------------
    # express.list(model_name) returns all records.
    # Supports filters, ordering, limit, offset.

    all_tasks = await db.express.list("Task")
    assert len(all_tasks) == 2, "Two tasks remain after delete"

    # List with ordering
    ordered = await db.express.list("Task", order_by="priority")
    assert len(ordered) == 2
    print(f"Listed {len(ordered)} tasks")

    # -- 6. COUNT -- count records -------------------------------------------

    count = await db.express.count("Task")
    assert count == 2, "Two records in total"

    # -- Cleanup -------------------------------------------------------------
    await db.stop()

asyncio.run(main())

try:
    os.unlink(db_path)
except OSError:
    pass

print("PASS: 01-dataflow/02_express_crud")
```

### Step-by-Step Explanation

1. **Setup**: Define a `Task` model with four fields. `done` has a default of `False`, making it optional during creation.

2. **CREATE**: `await db.express.create("Task", {...})` inserts a record and returns it as a dictionary. The returned dict includes an auto-generated `id` field. Default values (like `done: False`) are applied automatically for omitted fields.

3. **READ**: `await db.express.read("Task", id)` fetches a single record by its ID. Returns `None` if the record does not exist -- no exception is raised for missing records.

4. **UPDATE**: `await db.express.update("Task", id, {...})` modifies specific fields on an existing record. Only the fields you pass are changed; other fields remain untouched. Returns the full updated record.

5. **DELETE**: `await db.express.delete("Task", id)` removes a record and returns `True` on success. A subsequent `read()` for the same ID returns `None`.

6. **LIST**: `await db.express.list("Task")` returns all records as a list of dictionaries. The `order_by` parameter sorts results by a field name.

7. **COUNT**: `await db.express.count("Task")` returns the total number of records without fetching them -- useful for pagination calculations.

## Common Mistakes

| Mistake                                    | Correct Pattern                                     | Why                                                                                         |
| ------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Using WorkflowBuilder for simple CRUD      | Use `db.express.create()` instead                   | Express is ~23x faster because it bypasses graph construction and runtime overhead.         |
| Forgetting `await` on Express methods      | `task = await db.express.create("Task", {...})`     | All Express methods are async. Without `await`, you get a coroutine object, not the result. |
| Checking `if fetched` instead of `is None` | `if fetched is not None:`                           | An empty dict `{}` is falsy in Python. Always use `is None` to check for missing records.   |
| Passing nested dicts to create             | `{"name": "test"}` not `{"data": {"name": "test"}}` | Express expects flat field dictionaries, not nested payloads.                               |

## Exercises

1. Create a `Contact` model with `name: str`, `email: str`, and `favorite: bool = False`. Write a script that creates three contacts, updates one to be a favorite, deletes another, and lists the remaining contacts with their count.
2. What does `express.read()` return when you pass an ID that was never created? What about an ID that was created and then deleted? Verify both cases.
3. Create 10 records with sequential priority values. Use `express.list()` with `order_by="priority"` and verify the results are sorted correctly.

## Key Takeaways

- `db.express` is the go-to API for single-record CRUD -- fast, simple, and async
- `create()` returns the full record with an auto-generated `id`
- `read()` returns `None` for missing records (no exceptions)
- `update()` modifies only the fields you pass; everything else is preserved
- `delete()` returns `True` on success
- `list()` and `count()` support ordering for basic queries
- Always use Express for simple CRUD; reserve WorkflowBuilder for multi-step pipelines

## Next Chapter

[Chapter 3: List with Filters](03_list_filters.md) -- Query records using filters, ordering, pagination with limit and offset, and combined query parameters.
