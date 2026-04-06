# Chapter 10: ConnectionManager

## Overview

`ConnectionManager` is the database abstraction layer that underpins all stateful engines in Kailash -- model registries, drift monitors, event stores, and execution history. It provides async database operations with automatic dialect translation, connection pooling, and transaction support across SQLite, PostgreSQL, and MySQL. This chapter covers the full lifecycle: dialect detection, initialization, queries, transactions, index creation, and the infrastructure store ecosystem built on top.

## Prerequisites

- Python 3.10+ installed
- Kailash SDK installed (`pip install kailash`)
- Completed [Chapter 9: Error Handling](09_error_handling.md)
- Basic understanding of Python `asyncio` (async/await)
- Familiarity with SQL (SELECT, INSERT, CREATE TABLE)

## Concepts

### Concept 1: Dialect Detection and Translation

ConnectionManager auto-detects the database type from the connection URL and translates canonical `?` placeholders to the target dialect's format. You write queries once with `?` placeholders, and they work across SQLite (`?`), PostgreSQL (`$1`, `$2`), and MySQL (`%s`).

- **What**: Automatic detection of database type from URL and query placeholder translation
- **Why**: Write-once queries eliminate dialect-specific SQL, making your code portable across databases
- **How**: `detect_dialect("sqlite:///local.db")` returns a dialect object; `dialect.translate_query(sql)` converts placeholders
- **When**: Always -- ConnectionManager handles this transparently on every `execute()` and `fetch()` call

### Concept 2: The Lifecycle (Create, Initialize, Close)

ConnectionManager follows a strict three-phase lifecycle: create the instance with a URL, call `await initialize()` to open the connection pool, and call `await close()` to release resources. Calling `execute()` or `fetch()` before `initialize()` raises `RuntimeError`.

- **What**: A mandatory initialization step before any database operations
- **Why**: Separating creation from initialization allows configuration before pool creation
- **How**: `initialize()` opens the connection pool; `close()` releases all connections
- **When**: Always call `initialize()` before queries and `close()` when done. Double-close is safe (no-op)

### Concept 3: Transactions

`transaction()` returns an async context manager for multi-statement atomic operations. On success, changes are committed. On exception, changes are rolled back. Use transactions whenever multiple writes must succeed or fail together.

- **What**: An async context manager that wraps multiple SQL statements in an atomic unit
- **Why**: Without transactions, a failure between two writes leaves the database in an inconsistent state
- **How**: `async with conn.transaction() as tx:` -- use `tx.execute()` inside; commit on exit, rollback on exception
- **When**: Use for any operation requiring multiple writes: transfers, multi-table updates, cascading inserts

### Concept 4: Infrastructure Stores

ConnectionManager is the foundation for all database-backed infrastructure in Kailash. The infrastructure module provides specialized stores (event sourcing, checkpointing, dead-letter queues, execution history, idempotency) that each take a ConnectionManager and create their own tables.

- **What**: Higher-level stores built on ConnectionManager for workflow infrastructure
- **Why**: Separates concerns -- ConnectionManager handles connections, stores handle domain logic
- **How**: `DBEventStoreBackend(conn)` takes an initialized ConnectionManager and calls its own `initialize()` to create tables
- **When**: Used internally by engines (ModelRegistry, DriftMonitor, etc.) and available for custom infrastructure

## Key API

| Method / Class        | Parameters                               | Returns             | Description                                |
| --------------------- | ---------------------------------------- | ------------------- | ------------------------------------------ |
| `ConnectionManager()` | `url: str`                               | `ConnectionManager` | Create with a database URL                 |
| `initialize()`        | --                                       | `None`              | Open the connection pool                   |
| `close()`             | --                                       | `None`              | Release all resources (safe to call twice) |
| `execute()`           | `query: str`, `*params`                  | `None`              | Run DDL/DML with dialect translation       |
| `fetch()`             | `query: str`, `*params`                  | `list[dict]`        | Query rows, returned as list of dicts      |
| `fetchone()`          | `query: str`, `*params`                  | `dict or None`      | Query single row or None                   |
| `transaction()`       | --                                       | Context manager     | Atomic multi-statement transaction         |
| `create_index()`      | `name: str`, `table: str`, `column: str` | `None`              | Idempotent CREATE INDEX IF NOT EXISTS      |
| `detect_dialect()`    | `url: str`                               | `Dialect`           | Auto-detect database type from URL         |

## Code Walkthrough

```python
from __future__ import annotations

import asyncio

from kailash.db import ConnectionManager, DatabaseType, detect_dialect

# ── 1. Dialect detection ───────────────────────────────────────────

sqlite_dialect = detect_dialect("sqlite:///local.db")
assert sqlite_dialect.database_type == DatabaseType.SQLITE

pg_dialect = detect_dialect("postgresql://user:pass@localhost/mydb")
assert pg_dialect.database_type == DatabaseType.POSTGRESQL

mysql_dialect = detect_dialect("mysql://user:pass@localhost/mydb")
assert mysql_dialect.database_type == DatabaseType.MYSQL


# ── 2. ConnectionManager lifecycle ─────────────────────────────────

async def test_lifecycle():
    conn = ConnectionManager("sqlite:///:memory:")

    # Before initialize(), the pool is None
    assert conn._pool is None
    assert conn.url == "sqlite:///:memory:"
    assert conn.dialect.database_type == DatabaseType.SQLITE

    # Initialize creates the connection pool
    await conn.initialize()
    assert conn._pool is not None

    # Close releases all resources
    await conn.close()
    assert conn._pool is None

asyncio.run(test_lifecycle())


# ── 3. Execute queries ─────────────────────────────────────────────
# Use canonical ? placeholders — ConnectionManager translates them.

async def test_execute():
    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    # DDL: create a table
    await conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, score REAL)"
    )

    # DML: insert rows using ? placeholders
    await conn.execute(
        "INSERT INTO users (id, name, score) VALUES (?, ?, ?)", 1, "Alice", 95.5
    )
    await conn.execute(
        "INSERT INTO users (id, name, score) VALUES (?, ?, ?)", 2, "Bob", 87.0
    )
    await conn.execute(
        "INSERT INTO users (id, name, score) VALUES (?, ?, ?)", 3, "Carol", 92.3
    )

    # fetch() returns a list of dicts
    rows = await conn.fetch("SELECT * FROM users ORDER BY id")
    assert len(rows) == 3
    assert rows[0]["name"] == "Alice"

    # fetch() with parameters
    high_scorers = await conn.fetch("SELECT * FROM users WHERE score > ?", 90.0)
    assert len(high_scorers) == 2

    # fetchone() returns a single dict or None
    alice = await conn.fetchone("SELECT * FROM users WHERE name = ?", "Alice")
    assert alice is not None
    assert alice["score"] == 95.5

    nobody = await conn.fetchone("SELECT * FROM users WHERE name = ?", "Nobody")
    assert nobody is None

    await conn.close()

asyncio.run(test_execute())


# ── 4. Transactions ────────────────────────────────────────────────

async def test_transactions():
    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    await conn.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance REAL)")
    await conn.execute("INSERT INTO accounts (id, balance) VALUES (?, ?)", 1, 100.0)
    await conn.execute("INSERT INTO accounts (id, balance) VALUES (?, ?)", 2, 50.0)

    # Successful transaction: transfer 25 from account 1 to account 2
    async with conn.transaction() as tx:
        await tx.execute(
            "UPDATE accounts SET balance = balance - ? WHERE id = ?", 25.0, 1
        )
        await tx.execute(
            "UPDATE accounts SET balance = balance + ? WHERE id = ?", 25.0, 2
        )

    rows = await conn.fetch("SELECT * FROM accounts ORDER BY id")
    assert rows[0]["balance"] == 75.0
    assert rows[1]["balance"] == 75.0

    # Failed transaction: rollback on error
    try:
        async with conn.transaction() as tx:
            await tx.execute(
                "UPDATE accounts SET balance = balance - ? WHERE id = ?", 999.0, 1
            )
            raise ValueError("Simulated business logic error")
    except ValueError:
        pass  # Transaction rolled back

    rows_after = await conn.fetch("SELECT * FROM accounts ORDER BY id")
    assert rows_after[0]["balance"] == 75.0, "Rollback preserved original balance"

    await conn.close()

asyncio.run(test_transactions())


# ── 5. Uninitialized access raises RuntimeError ────────────────────

async def test_uninitialized():
    conn = ConnectionManager("sqlite:///:memory:")

    try:
        await conn.execute("SELECT 1")
    except RuntimeError as e:
        assert "not initialized" in str(e).lower()

asyncio.run(test_uninitialized())


# ── 6. Index creation helper ───────────────────────────────────────

async def test_create_index():
    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    await conn.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, category TEXT, price REAL)"
    )

    # Idempotent — safe to call multiple times
    await conn.create_index("idx_items_category", "items", "category")
    await conn.create_index("idx_items_category", "items", "category")

    rows = await conn.fetch("SELECT * FROM items WHERE category = ?", "A")
    assert isinstance(rows, list)

    await conn.close()

asyncio.run(test_create_index())


# ── 7. Infrastructure stores — the bigger picture ──────────────────

from kailash.infrastructure import StoreFactory

async def test_store_factory():
    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    from kailash.infrastructure import DBEventStoreBackend, DBExecutionStore

    event_store = DBEventStoreBackend(conn)
    await event_store.initialize()

    exec_store = DBExecutionStore(conn)
    await exec_store.initialize()

    await conn.close()

asyncio.run(test_store_factory())


# ── 8. Placeholder translation ─────────────────────────────────────

from kailash.db import SQLiteDialect, PostgresDialect, MySQLDialect

sqlite_d = SQLiteDialect()
pg_d = PostgresDialect()
mysql_d = MySQLDialect()

query = "SELECT * FROM users WHERE id = ? AND name = ?"

sqlite_translated = sqlite_d.translate_query(query)
pg_translated = pg_d.translate_query(query)
mysql_translated = mysql_d.translate_query(query)

# SQLite keeps ? as-is
assert "?" in sqlite_translated

# PostgreSQL uses $1, $2, ...
assert "$1" in pg_translated and "$2" in pg_translated

# MySQL uses %s
assert "%s" in mysql_translated


# ── 9. Double close is safe ────────────────────────────────────────

async def test_double_close():
    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()
    await conn.close()
    await conn.close()  # No error

asyncio.run(test_double_close())

print("PASS: 00-core/10_connection_manager")
```

### Step-by-Step Explanation

1. **Dialect detection**: `detect_dialect()` parses the URL prefix to determine the database type. This happens automatically inside ConnectionManager, but you can call it directly for inspection.

2. **Lifecycle**: Create with a URL, `initialize()` to open the pool, `close()` to release. The pool is `None` before init and after close. This pattern prevents resource leaks.

3. **Execute and fetch**: `execute()` runs DDL/DML statements. `fetch()` returns rows as a list of dicts. `fetchone()` returns one dict or `None`. All methods translate `?` placeholders to the target dialect automatically.

4. **Transactions**: `async with conn.transaction() as tx:` provides atomic multi-statement execution. Use `tx.execute()` inside the block. On success, changes commit. On any exception, changes roll back -- the database returns to its prior state.

5. **Uninitialized guard**: Calling `execute()`, `fetch()`, or `fetchone()` before `initialize()` raises `RuntimeError` with a clear message. This prevents silent failures from forgotten initialization.

6. **Index creation**: `create_index()` wraps `CREATE INDEX IF NOT EXISTS`, making it idempotent. Safe to call multiple times without error.

7. **Infrastructure stores**: `DBEventStoreBackend`, `DBExecutionStore`, and other stores take an initialized ConnectionManager and create their own tables. This is how engines like ModelRegistry and DriftMonitor get their persistence.

8. **Placeholder translation**: SQLite keeps `?`, PostgreSQL converts to `$1`/`$2`, MySQL converts to `%s`. You never write dialect-specific SQL.

9. **Double close safety**: Calling `close()` on an already-closed ConnectionManager is a no-op, preventing errors in cleanup code.

## Common Mistakes

| Mistake                                       | Correct Pattern                             | Why                                                                                         |
| --------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Forgetting `await conn.initialize()`          | Always initialize before queries            | Operations on an uninitialized manager raise `RuntimeError`                                 |
| Using `$1` or `%s` placeholders directly      | Use canonical `?` placeholders              | Dialect translation handles the conversion; hardcoded placeholders break on other databases |
| Not using transactions for multi-write ops    | `async with conn.transaction() as tx:`      | Without transactions, a failure between writes leaves data inconsistent                     |
| Calling `conn.execute()` inside a transaction | Use `tx.execute()` (the transaction object) | `conn.execute()` runs outside the transaction, bypassing atomicity                          |

## Exercises

1. Create a ConnectionManager with an in-memory SQLite database. Create a `products` table, insert three rows, and use `fetch()` with a `WHERE` clause to query products above a price threshold.

2. Write a transaction that transfers a quantity from one inventory item to another. Simulate a failure mid-transaction and verify the rollback preserves the original quantities.

3. Use `detect_dialect()` to inspect the placeholder translation for all three supported databases. Write a helper function that logs the translated query for debugging purposes.

## Key Takeaways

- ConnectionManager provides async database operations with automatic dialect translation
- The lifecycle is create, initialize, close -- always initialize before queries
- Use canonical `?` placeholders; dialect translation to `$1`/`%s` is automatic
- `fetch()` returns list of dicts; `fetchone()` returns one dict or None
- Transactions provide atomic multi-statement execution with automatic rollback on error
- Infrastructure stores (event sourcing, checkpointing, etc.) are built on ConnectionManager
- Double-close is safe; uninitialized access raises a clear RuntimeError

## Next Chapter

This concludes the Core SDK section. Continue to [Section 01: DataFlow](../01-dataflow/) for zero-config database operations, or jump to [Section 03: Kaizen](../03-kaizen/) for the AI agent framework.
