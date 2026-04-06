# Chapter 1: DataFlow Model Definition

## Overview

Every DataFlow application starts with **models** -- Python classes decorated with `@db.model` that define your database schema. The decorator registers the class, maps Python type annotations to database column types, and auto-generates CRUD nodes. This chapter teaches you how to define models, inspect their fields, and initialize the database.

## Prerequisites

- Python 3.10+ installed
- Kailash DataFlow installed (`pip install kailash-dataflow`)
- Completed the Core SDK chapters (workflow basics)

## Concepts

### Concept 1: The @db.model Decorator

The `@db.model` decorator transforms a plain Python class into a DataFlow model. It registers the class with the DataFlow instance, creates corresponding CRUD workflow nodes, and maps type annotations to database column types.

- **What**: A decorator that converts a Python class into a database-backed model with auto-generated CRUD operations
- **Why**: Eliminates boilerplate for table creation, column mapping, and CRUD node generation -- you declare your schema once and DataFlow handles the rest
- **How**: The decorator reads the class's type annotations and default values, registers the model internally, and generates typed nodes (e.g., `CreateProduct`, `ReadProduct`, `ListProduct`)
- **When**: Use `@db.model` for every entity you want to persist. This is the standard entry point for all DataFlow data operations

### Concept 2: Field Types and Defaults

Model fields use standard Python type annotations: `str`, `int`, `float`, `bool`, and `datetime`. Fields without a default value are **required** -- they must be provided when creating a record. Fields with a default value are **optional** -- the default is applied automatically if the field is omitted.

- **What**: Python type annotations that map to database column types, with optional default values
- **Why**: Type annotations provide both documentation and runtime validation, ensuring data integrity at the model level
- **How**: `str` maps to TEXT, `int` to INTEGER, `float` to REAL, `bool` to BOOLEAN. Defaults are stored in model metadata and applied during record creation
- **When**: Always declare types for every field. Use defaults for fields that have a sensible fallback (e.g., `active: bool = True`)

### Concept 3: Model Registration and Inspection

After decorating a class, you can inspect registered models and their fields using `db.get_models()` and `db.get_model_fields("ModelName")`. This is useful for debugging, documentation generation, and runtime schema introspection.

### Concept 4: Initialization

`db.initialize()` is an async method that creates database tables, runs any pending migrations, and sets up the connection pool. You must call it before performing any CRUD operations. After initialization, all registered models have corresponding tables in the database.

### Key API

| Method / Decorator      | Parameters          | Returns         | Description                              |
| ----------------------- | ------------------- | --------------- | ---------------------------------------- |
| `DataFlow()`            | `database_url: str` | `DataFlow`      | Create a DataFlow instance               |
| `@db.model`             | (class)             | decorated class | Register a class as a database model     |
| `db.get_models()`       | --                  | `dict`          | All registered model names and metadata  |
| `db.get_model_fields()` | `model_name: str`   | `dict`          | Field definitions for a model            |
| `db.initialize()`       | --                  | `bool` (async)  | Create tables and set up connection pool |
| `db.stop()`             | --                  | `None` (async)  | Close connections and clean up           |

## Code Walkthrough

```python
from __future__ import annotations

import asyncio
import os
import tempfile

from dataflow import DataFlow

# -- 1. Create a DataFlow instance ------------------------------------------
# DataFlow connects to a database. SQLite is simplest for tutorials.
# In production, use PostgreSQL via DATABASE_URL environment variable.

db_path = os.path.join(tempfile.gettempdir(), "textbook_dataflow_01.db")
db = DataFlow(database_url=f"sqlite:///{db_path}")

# -- 2. Define a model with @db.model ---------------------------------------
# The decorator registers the class, creates CRUD nodes, and maps
# Python type annotations to database column types.
#
# Supported types: str, int, float, bool, datetime
# Fields with defaults are optional; fields without are required.

@db.model
class Product:
    name: str
    price: float
    category: str
    in_stock: bool = True

# Verify the model was registered
registered_models = db.get_models()
assert "Product" in registered_models, "Product model should be registered"

# Inspect model fields
fields = db.get_model_fields("Product")
assert "name" in fields, "name field registered"
assert "price" in fields, "price field registered"
assert "category" in fields, "category field registered"
assert "in_stock" in fields, "in_stock field registered"

# Required vs optional: fields with defaults are not required
assert fields["name"]["required"] is True, "name has no default -- required"
assert fields["in_stock"]["required"] is False, "in_stock has default -- optional"
assert fields["in_stock"]["default"] is True

print(f"Registered models: {list(registered_models.keys())}")
print(f"Product fields: {list(fields.keys())}")

# -- 3. Multiple models -----------------------------------------------------

@db.model
class Customer:
    email: str
    name: str
    tier: str = "free"

@db.model
class Order:
    customer_email: str
    product_name: str
    quantity: int
    total: float

all_models = db.get_models()
assert len(all_models) >= 3, "Three models registered"
assert "Customer" in all_models
assert "Order" in all_models

# -- 4. Initialize -- create tables in the database -------------------------
# initialize() is async -- it creates tables, runs migrations,
# and sets up the connection pool.

async def main():
    success = await db.initialize()
    assert success, "DataFlow initialization should succeed"

    # After initialize, tables exist and CRUD is available
    print(f"DataFlow initialized with {len(all_models)} models")

    # -- 5. Table name mapping -----------------------------------------------
    # By default, class names are converted to snake_case table names:
    #   Product -> products
    #   Customer -> customers
    #   Order -> orders
    # Override with __tablename__ class attribute if needed.

    # Cleanup
    await db.stop()

asyncio.run(main())

# -- 6. Edge case: custom table name ----------------------------------------

db2 = DataFlow(database_url=f"sqlite:///{db_path}")

@db2.model
class AuditLog:
    __tablename__ = "custom_audit_logs"
    action: str
    details: str

# The table name override is respected
model_info = db2._models.get("AuditLog", {})
assert model_info.get("table_name") == "custom_audit_logs"

# Cleanup temp file
try:
    os.unlink(db_path)
except OSError:
    pass

print("PASS: 01-dataflow/01_dataflow_model")
```

### Step-by-Step Explanation

1. **Import**: `DataFlow` is the primary import from the `dataflow` package. It is the central object that manages models, connections, and CRUD operations.

2. **DataFlow instance**: `DataFlow(database_url=...)` creates a connection manager. For tutorials, SQLite is the simplest backend. In production, use PostgreSQL via the `DATABASE_URL` environment variable.

3. **Defining a model**: The `@db.model` decorator reads the class's type annotations (`name: str`, `price: float`, etc.) and registers the class as a database model. Fields without defaults (`name`, `price`, `category`) are required. Fields with defaults (`in_stock: bool = True`) are optional.

4. **Inspecting models**: `db.get_models()` returns a dictionary of all registered model names. `db.get_model_fields("Product")` returns field metadata including type, required status, and default values. This is useful for debugging and runtime introspection.

5. **Multiple models**: You can register as many models as you need on the same DataFlow instance. Each one gets its own table and CRUD nodes.

6. **Initialization**: `await db.initialize()` creates the actual database tables for all registered models and sets up the connection pool. This is an async operation -- wrap it in `asyncio.run()` for scripts.

7. **Table name mapping**: By default, `Product` becomes the `products` table (snake_case, pluralized). Override this with `__tablename__ = "custom_name"` on the class.

8. **Cleanup**: Always call `await db.stop()` to close database connections gracefully.

## Common Mistakes

| Mistake                                   | Correct Pattern                               | Why                                                                                                |
| ----------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Using `db.express` before `initialize()`  | Call `await db.initialize()` first            | Tables don't exist yet. CRUD operations will fail with a "table not found" error.                  |
| Forgetting `await` on `db.initialize()`   | `success = await db.initialize()`             | `initialize()` is async. Without `await`, you get a coroutine object instead of executing it.      |
| Using `pandas` types as field annotations | Use `str`, `int`, `float`, `bool`, `datetime` | DataFlow uses standard Python types, not pandas dtypes. `pd.Int64Dtype` is not a valid annotation. |
| Hardcoding database paths in production   | `database_url=os.environ["DATABASE_URL"]`     | Hardcoded paths break across environments. Use environment variables for all connection strings.   |

## Exercises

1. Define a `BlogPost` model with fields `title: str`, `content: str`, `author: str`, `published: bool = False`, and `views: int = 0`. Register it, initialize the database, and verify that `published` and `views` are optional with their correct defaults.
2. Create two DataFlow instances pointing at different SQLite files. Register a `User` model on each. Verify that models registered on one instance are not visible on the other.
3. Use `__tablename__` to map a class named `HTTPRequestLog` to a table named `http_logs`. Verify the mapping is correct by inspecting the internal model info.

## Key Takeaways

- `DataFlow(database_url=...)` is the entry point for all database operations
- `@db.model` registers a class as a database model with auto-generated CRUD
- Python type annotations (`str`, `int`, `float`, `bool`) map directly to database column types
- Fields without defaults are required; fields with defaults are optional
- `db.initialize()` must be called (async) before any CRUD operations
- `db.get_models()` and `db.get_model_fields()` provide runtime schema introspection
- Override table names with `__tablename__` when the default snake_case convention does not fit

## Next Chapter

[Chapter 2: Express CRUD](02_express_crud.md) -- Perform Create, Read, Update, and Delete operations using the Express API for fast single-record operations.
