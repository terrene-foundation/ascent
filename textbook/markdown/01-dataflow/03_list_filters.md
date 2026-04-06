# Chapter 3: List with Filters

## Overview

Real applications rarely fetch all records at once -- you need to filter by field values, sort results, and paginate through large datasets. The Express API's `list()` method supports all of these through `filters`, `order_by`, `limit`, and `offset` parameters. This chapter teaches you how to build precise queries using these parameters individually and in combination.

## Prerequisites

- Python 3.10+ installed
- Kailash DataFlow installed (`pip install kailash-dataflow`)
- Completed Chapter 2 (Express CRUD basics)

## Concepts

### Concept 1: Dictionary Filters

Filters are passed as a plain dictionary mapping field names to expected values. `{"department": "Engineering"}` returns only records where the `department` field equals `"Engineering"`. Multiple keys act as AND conditions -- all must match.

- **What**: A dict of field-value pairs that restricts which records are returned
- **Why**: Fetching all records and filtering in Python is wasteful and slow -- database-level filtering uses indexes and returns only matching rows
- **How**: The filter dict is translated into SQL WHERE clauses internally. Each key-value pair becomes an equality check, and multiple pairs are joined with AND
- **When**: Use filters whenever you need a subset of records. For complex queries beyond equality (greater-than, LIKE, OR), use DataFlow's query engine or WorkflowBuilder

### Concept 2: Ordering

The `order_by` parameter accepts a field name as a string. Results are sorted ascending by that field. This is essential for deterministic pagination and for presenting data in a meaningful sequence.

### Concept 3: Pagination with Limit and Offset

`limit` controls how many records to return. `offset` controls how many records to skip before starting. Together, they implement page-based pagination: page 1 is `limit=N, offset=0`, page 2 is `limit=N, offset=N`, and so on.

- **What**: Parameters that control the window of records returned from a query
- **Why**: Returning thousands of records at once wastes memory and network bandwidth. Pagination keeps response sizes bounded
- **How**: `limit` maps to SQL LIMIT, `offset` maps to SQL OFFSET. Combined with `order_by`, you get deterministic pages
- **When**: Always paginate when the dataset could grow unbounded. Use `count()` to calculate total pages

### Key API

| Method            | Parameters                                                                      | Returns      | Description                        |
| ----------------- | ------------------------------------------------------------------------------- | ------------ | ---------------------------------- |
| `express.list()`  | `model_name`, `filters: dict = None`, `order_by: str = None`, `limit`, `offset` | `list[dict]` | Query records with optional params |
| `express.count()` | `model_name`, `filters: dict = None`                                            | `int`        | Count matching records             |

## Code Walkthrough

```python
from __future__ import annotations

import asyncio
import os
import tempfile

from dataflow import DataFlow

# -- Setup -------------------------------------------------------------------

db_path = os.path.join(tempfile.gettempdir(), "textbook_dataflow_03.db")
db = DataFlow(database_url=f"sqlite:///{db_path}")

@db.model
class Employee:
    name: str
    department: str
    salary: float
    active: bool = True

async def main():
    await db.initialize()

    # Seed data
    employees = [
        {"name": "Alice", "department": "Engineering", "salary": 95000.0},
        {"name": "Bob", "department": "Engineering", "salary": 85000.0},
        {"name": "Carol", "department": "Marketing", "salary": 78000.0},
        {"name": "Dave", "department": "Marketing", "salary": 72000.0},
        {"name": "Eve", "department": "Engineering", "salary": 110000.0},
        {"name": "Frank", "department": "Sales", "salary": 68000.0, "active": False},
    ]
    for emp in employees:
        await db.express.create("Employee", emp)

    # -- 1. List all ---------------------------------------------------------

    all_emps = await db.express.list("Employee")
    assert len(all_emps) == 6, "Six employees created"

    # -- 2. Filter by exact value --------------------------------------------
    # Pass filters as a dict: {field: value}

    engineers = await db.express.list(
        "Employee", filters={"department": "Engineering"}
    )
    assert len(engineers) == 3, "Three engineers"

    active_only = await db.express.list("Employee", filters={"active": True})
    assert len(active_only) == 5, "Five active employees"

    # -- 3. Order by field ---------------------------------------------------

    by_salary = await db.express.list("Employee", order_by="salary")
    salaries = [e["salary"] for e in by_salary]
    assert salaries == sorted(salaries), "Ordered by salary ascending"

    # -- 4. Limit and offset (pagination) ------------------------------------

    page1 = await db.express.list(
        "Employee", order_by="name", limit=2, offset=0
    )
    assert len(page1) == 2, "Page 1: 2 records"

    page2 = await db.express.list(
        "Employee", order_by="name", limit=2, offset=2
    )
    assert len(page2) == 2, "Page 2: 2 records"

    # No overlap between pages
    page1_names = {e["name"] for e in page1}
    page2_names = {e["name"] for e in page2}
    assert len(page1_names & page2_names) == 0, "Pages don't overlap"

    # -- 5. Count with filters -----------------------------------------------

    eng_count = await db.express.count(
        "Employee", filters={"department": "Engineering"}
    )
    assert eng_count == 3

    total = await db.express.count("Employee")
    assert total == 6

    # -- 6. Combined: filter + order + limit ---------------------------------

    top_engineers = await db.express.list(
        "Employee",
        filters={"department": "Engineering"},
        order_by="salary",
        limit=2,
    )
    assert len(top_engineers) == 2
    print(f"Top 2 engineers by salary: {[e['name'] for e in top_engineers]}")

    # -- Cleanup -------------------------------------------------------------
    await db.stop()

asyncio.run(main())

try:
    os.unlink(db_path)
except OSError:
    pass

print("PASS: 01-dataflow/03_list_filters")
```

### Step-by-Step Explanation

1. **Seed data**: Six employees across three departments. One employee (`Frank`) has `active=False` to test boolean filtering. This gives us enough data to verify all query patterns.

2. **List all**: `express.list("Employee")` with no parameters returns every record. This is the baseline -- useful for small datasets but avoid it in production with large tables.

3. **Filter by exact value**: `filters={"department": "Engineering"}` returns only the three Engineering employees. `filters={"active": True}` returns the five active employees. The filter dict uses equality matching -- each key-value pair must match exactly.

4. **Ordering**: `order_by="salary"` sorts results by salary in ascending order. The assertion verifies that the returned salaries are in non-decreasing order.

5. **Pagination**: `limit=2, offset=0` returns the first two records (page 1). `limit=2, offset=2` skips the first two and returns the next two (page 2). The set intersection check confirms that no record appears on both pages.

6. **Count with filters**: `express.count("Employee", filters={"department": "Engineering"})` returns `3` without fetching the actual records. This is useful for calculating total pages in a pagination UI.

7. **Combined query**: All parameters work together. `filters` narrows the dataset, `order_by` sorts it, and `limit` caps the result size. This is the pattern you will use most often in real applications.

## Common Mistakes

| Mistake                                      | Correct Pattern                                          | Why                                                                                              |
| -------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Fetching all records and filtering in Python | Use `filters={"field": value}`                           | Database-level filtering is orders of magnitude faster and uses less memory.                     |
| Paginating without `order_by`                | Always pass `order_by` with `limit`/`offset`             | Without ordering, the database makes no guarantees about row order -- pages may overlap or skip. |
| Using offset-based pagination on huge tables | Consider cursor-based pagination for very large datasets | High `offset` values cause the database to scan and skip rows, getting slower as offset grows.   |
| Passing filter values with wrong types       | Match the model's type annotations exactly               | `filters={"active": "true"}` (string) will not match `active: bool = True` (boolean).            |

## Exercises

1. Seed a `Product` model with 20 products across 4 categories. Write queries to: (a) list all products in a specific category, (b) count products per category using `count()` with filters, and (c) paginate through all products 5 at a time.
2. What happens when `offset` exceeds the total number of records? Write a test that verifies the behavior.
3. Combine `filters`, `order_by`, and `limit` to find the 3 highest-salaried active employees. Verify the results are correct.

## Key Takeaways

- `filters` accepts a dict for equality-based field matching (AND logic for multiple fields)
- `order_by` sorts results ascending by a single field -- essential for deterministic pagination
- `limit` and `offset` implement page-based pagination
- `count()` accepts the same `filters` parameter as `list()` -- use it for pagination totals
- All parameters compose: filter first, then order, then paginate
- Always use `order_by` alongside `limit`/`offset` to guarantee non-overlapping pages

## Next Chapter

[Chapter 4: SyncExpress CRUD](04_sync_express.md) -- Perform the same CRUD operations synchronously, without async/await, using the SyncExpress wrapper.
