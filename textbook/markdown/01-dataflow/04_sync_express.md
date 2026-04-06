# Chapter 4: SyncExpress CRUD

## Overview

Not every Python program runs in an async context. CLI scripts, synchronous web handlers, and test suites often need database access without `async`/`await`. **SyncExpress** wraps the async Express API with synchronous equivalents, maintaining a persistent event loop in a background thread so that database connections survive across calls. This chapter teaches you how to use SyncExpress for the same CRUD operations you learned in Chapter 2, but without any async boilerplate.

## Prerequisites

- Python 3.10+ installed
- Kailash DataFlow installed (`pip install kailash-dataflow`)
- Completed Chapters 1-2 (model definition and async Express CRUD)

## Concepts

### Concept 1: The SyncExpress Wrapper

SyncExpress is accessed via `db.express_sync`. It provides the same methods as `db.express` (`create`, `read`, `update`, `delete`, `list`, `count`) but as plain synchronous functions -- no `await` needed.

- **What**: A synchronous wrapper around the async Express API that runs operations on a background event loop
- **Why**: Many Python contexts (CLI tools, pytest without asyncio markers, synchronous frameworks) cannot use `await`. SyncExpress makes DataFlow accessible everywhere
- **How**: Internally, SyncExpress maintains a persistent event loop running in a daemon thread. Each sync call submits work to that loop and blocks until completion. The background loop keeps database connections alive across calls
- **When**: Use SyncExpress in CLI scripts, synchronous tests, and any code that cannot use `async`/`await`. For async code, prefer `db.express` directly

### Concept 2: Initialization Is Still Async

Even when using SyncExpress, `db.initialize()` remains async. You must call it once (via `asyncio.run()`) before accessing `db.express_sync`. This is because initialization involves connection pool setup and table creation that benefit from async I/O.

### Concept 3: Instance Caching

The `db.express_sync` property returns a cached SyncExpress instance. Accessing it multiple times returns the same object -- there is no overhead from repeated property access.

### Key API

| Method / Property | Parameters                                   | Returns          | Description                       |
| ----------------- | -------------------------------------------- | ---------------- | --------------------------------- |
| `db.express_sync` | --                                           | `SyncExpress`    | Cached synchronous CRUD interface |
| `sync.create()`   | `model_name: str`, `data: dict`              | `dict`           | Insert a record (synchronous)     |
| `sync.read()`     | `model_name: str`, `id: str`                 | `dict` or `None` | Fetch by ID (synchronous)         |
| `sync.update()`   | `model_name: str`, `id: str`, `fields: dict` | `dict`           | Modify fields (synchronous)       |
| `sync.delete()`   | `model_name: str`, `id: str`                 | `bool`           | Remove a record (synchronous)     |
| `sync.list()`     | `model_name: str`, `order_by: str = None`    | `list[dict]`     | Fetch all records (synchronous)   |
| `sync.count()`    | `model_name: str`                            | `int`            | Count records (synchronous)       |

## Code Walkthrough

```python
from __future__ import annotations

import asyncio
import os
import tempfile

from dataflow import DataFlow

# -- Setup -------------------------------------------------------------------
# SyncExpress wraps the async ExpressDataFlow methods with synchronous
# equivalents. Useful in CLI scripts, synchronous FastAPI handlers,
# pytest without asyncio, and other non-async code.
#
# Internally, SyncExpress maintains a persistent event loop in a
# background daemon thread so that database connections survive
# across multiple sync calls.

db_path = os.path.join(tempfile.gettempdir(), "textbook_dataflow_04.db")
db = DataFlow(database_url=f"sqlite:///{db_path}")

@db.model
class Book:
    title: str
    author: str
    pages: int
    available: bool = True

# initialize() is still async -- call it once before using express_sync
async def setup():
    await db.initialize()

asyncio.run(setup())

# -- 1. ACCESS -- db.express_sync property -----------------------------------
# After initialize(), access SyncExpress through db.express_sync.
# No 'await' needed -- all calls are plain synchronous functions.

sync = db.express_sync
assert sync is not None, "express_sync property returns SyncExpress instance"

# -- 2. CREATE -- insert records synchronously -------------------------------

book1 = sync.create(
    "Book",
    {
        "title": "The Pragmatic Programmer",
        "author": "Hunt & Thomas",
        "pages": 352,
    },
)

assert "id" in book1, "Created record gets an auto-generated id"
assert book1["title"] == "The Pragmatic Programmer"
assert book1["available"] is True, "Default value applied"
print(f"Created: {book1['title']}")

book2 = sync.create(
    "Book",
    {
        "title": "Clean Code",
        "author": "Robert C. Martin",
        "pages": 464,
    },
)
book3 = sync.create(
    "Book",
    {
        "title": "Refactoring",
        "author": "Martin Fowler",
        "pages": 448,
        "available": False,
    },
)

# -- 3. READ -- fetch a single record by id ---------------------------------

fetched = sync.read("Book", book1["id"])

assert fetched is not None, "Record found by id"
assert fetched["title"] == "The Pragmatic Programmer"
print(f"Read: {fetched['title']}")

# Read non-existent record returns None
missing = sync.read("Book", "nonexistent-id-999")
assert missing is None, "Missing record returns None"

# -- 4. UPDATE -- modify an existing record ----------------------------------

updated = sync.update("Book", book3["id"], {"available": True})

assert updated["available"] is True, "Field updated"
print(f"Updated: {updated['title']} is now available")

# Verify persistence
refetched = sync.read("Book", book3["id"])
assert refetched["available"] is True, "Update persisted"

# -- 5. DELETE -- remove a record --------------------------------------------

deleted = sync.delete("Book", book2["id"])
assert deleted is True, "Delete returns True on success"

gone = sync.read("Book", book2["id"])
assert gone is None, "Deleted record returns None on read"

# -- 6. LIST -- query multiple records ---------------------------------------

all_books = sync.list("Book")
assert len(all_books) == 2, "Two books remain after delete"

# List with ordering
ordered = sync.list("Book", order_by="pages")
assert len(ordered) == 2
pages = [b["pages"] for b in ordered]
assert pages == sorted(pages), "Ordered by pages ascending"

# -- 7. COUNT -- count records -----------------------------------------------

total = sync.count("Book")
assert total == 2, "Two records in total"

# -- 8. Same instance reuse -------------------------------------------------
# The SyncExpress instance is cached on the DataFlow object.
# Accessing db.express_sync again returns the same instance.

sync2 = db.express_sync
assert sync2 is sync, "Same SyncExpress instance returned"

# -- Cleanup -----------------------------------------------------------------

async def teardown():
    await db.stop()

asyncio.run(teardown())

try:
    os.unlink(db_path)
except OSError:
    pass

print("PASS: 01-dataflow/04_sync_express")
```

### Step-by-Step Explanation

1. **Async initialization**: Even for sync usage, `db.initialize()` must be called once via `asyncio.run()`. This sets up the database tables and connection pool.

2. **Accessing SyncExpress**: `db.express_sync` returns the SyncExpress instance. Store it in a variable (`sync = db.express_sync`) for convenience. No `await` is needed anywhere after this point.

3. **CREATE**: `sync.create("Book", {...})` is identical in behavior to `await db.express.create(...)` -- it returns a dict with an auto-generated `id` and applies default values. The only difference is the absence of `await`.

4. **READ**: `sync.read("Book", id)` returns the record dict or `None`. The behavior matches the async version exactly.

5. **UPDATE**: `sync.update("Book", id, {...})` modifies fields and returns the updated record. The update persists immediately -- a subsequent `read()` confirms the new values.

6. **DELETE**: `sync.delete("Book", id)` removes the record and returns `True`. A follow-up `read()` returns `None`.

7. **LIST and COUNT**: Both work identically to their async counterparts. `list()` supports `order_by` for sorting.

8. **Instance caching**: `db.express_sync` is a cached property. Accessing it multiple times returns the same SyncExpress object, so there is no performance cost to repeated access.

9. **Teardown**: `db.stop()` is async, so it also requires `asyncio.run()`. This closes database connections gracefully.

## Common Mistakes

| Mistake                                         | Correct Pattern                        | Why                                                                                                      |
| ----------------------------------------------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Using `await` with SyncExpress methods          | `sync.create(...)` (no await)          | SyncExpress methods are synchronous. Adding `await` causes a `TypeError`.                                |
| Calling `db.express_sync` before `initialize()` | Call `asyncio.run(setup())` first      | The database connection pool must be set up before SyncExpress can dispatch operations.                  |
| Creating a new SyncExpress per operation        | Reuse `db.express_sync` (it is cached) | Creating fresh instances wastes resources. The property always returns the same cached instance.         |
| Mixing `db.express` and `db.express_sync`       | Pick one per context                   | Mixing async and sync calls in the same thread can cause event loop conflicts. Pick the right one early. |

## Exercises

1. Write a CLI script (no async) that uses SyncExpress to manage a `Note` model with `title: str` and `body: str`. The script should create a note, list all notes, and delete a note by ID -- all using synchronous calls.
2. Verify that `db.express_sync` returns the exact same object on repeated access using the `is` operator. Why does this matter for connection management?
3. What happens if you try to use `sync.create()` inside an `async def` function that is already running an event loop? Test it and explain the result.

## Key Takeaways

- `db.express_sync` provides synchronous equivalents of all Express CRUD methods
- Initialization (`db.initialize()`) and teardown (`db.stop()`) are still async -- use `asyncio.run()`
- SyncExpress methods have identical signatures and behavior to their async counterparts, minus `await`
- The SyncExpress instance is cached -- `db.express_sync` always returns the same object
- Use SyncExpress for CLI scripts, synchronous tests, and non-async frameworks
- Use `db.express` (async) when you are already in an async context

## Next Chapter

[Chapter 5: Field-Level Validators](05_validators.md) -- Validate model data at runtime using built-in validators for email, range, pattern, length, and more.
