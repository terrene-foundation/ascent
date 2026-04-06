# Chapter 7: Transaction Management

## Overview

When multiple database operations must succeed or fail as a unit, you need **transactions**. DataFlow's `TransactionManager` provides a context-manager API for atomic operations: if everything inside the block succeeds, the transaction commits; if an exception occurs, it rolls back automatically. This chapter teaches you how to create transactions, control isolation levels, handle rollbacks, nest transactions, and monitor active transaction state.

## Prerequisites

- Python 3.10+ installed
- Kailash DataFlow installed (`pip install kailash-dataflow`)
- Completed Chapters 1-4 (model definition and CRUD)
- Basic understanding of database transaction concepts (ACID)

## Concepts

### Concept 1: TransactionManager

The `TransactionManager` is accessed via `db.transactions`. It provides a `transaction()` context manager that yields a transaction context dictionary. The context tracks the transaction's ID, isolation level, status, and a list of operations performed within it.

- **What**: A manager that creates, tracks, and finalizes database transactions
- **Why**: Without transactions, a failure halfway through a multi-step operation (e.g., debiting one account but failing to credit another) leaves the database in an inconsistent state
- **How**: `transaction()` returns a context manager. On normal exit, the transaction status becomes `"committed"`. On exception, it becomes `"rolled_back"` and the error message is captured
- **When**: Use transactions whenever two or more operations must succeed or fail together -- fund transfers, order placement, inventory updates

### Concept 2: Isolation Levels

Isolation levels control how concurrent transactions see each other's changes. The default is `READ_COMMITTED`, which prevents dirty reads. For stricter guarantees, use `SERIALIZABLE`, which ensures transactions execute as if they were run one at a time.

- **What**: A configuration that controls the visibility of concurrent changes between transactions
- **Why**: Different operations have different consistency requirements. A balance check needs `SERIALIZABLE`; a reporting query can tolerate `READ_COMMITTED`
- **How**: Pass `isolation_level="SERIALIZABLE"` to `transaction()`. The level is recorded in the transaction context dict
- **When**: Use `SERIALIZABLE` for financial operations and data integrity checks. Use `READ_COMMITTED` (the default) for general-purpose operations

### Concept 3: Automatic Rollback

If an exception is raised inside a `transaction()` block, the context manager catches it, sets the transaction status to `"rolled_back"`, records the error message in the context dict, and re-raises the exception. Your code catches the re-raised exception normally.

### Concept 4: Nested Transactions

Transactions created with `transaction()` are independent. A nested `with txn_mgr.transaction()` inside an outer transaction creates a separate transaction context. Rolling back the inner transaction does not affect the outer one.

### Key API

| Method / Property                   | Parameters                                | Returns              | Description                                   |
| ----------------------------------- | ----------------------------------------- | -------------------- | --------------------------------------------- |
| `db.transactions`                   | --                                        | `TransactionManager` | Access the transaction manager                |
| `txn_mgr.transaction()`             | `isolation_level: str = "READ_COMMITTED"` | context manager      | Create a transaction block                    |
| `txn["id"]`                         | --                                        | `str`                | Unique transaction identifier                 |
| `txn["status"]`                     | --                                        | `str`                | `"active"`, `"committed"`, or `"rolled_back"` |
| `txn["isolation_level"]`            | --                                        | `str`                | The isolation level for this transaction      |
| `txn["operations"]`                 | --                                        | `list`               | Operations logged within the transaction      |
| `txn["error"]`                      | --                                        | `str`                | Error message (only after rollback)           |
| `txn_mgr.get_active_transactions()` | --                                        | `list`               | Currently active (uncommitted) transactions   |
| `txn_mgr.rollback_all()`            | --                                        | `dict`               | Emergency rollback of all active transactions |

## Code Walkthrough

```python
from __future__ import annotations

import asyncio
import os
import tempfile

from dataflow import DataFlow
from dataflow.features.transactions import TransactionManager

# -- Setup -------------------------------------------------------------------

db_path = os.path.join(tempfile.gettempdir(), "textbook_dataflow_07.db")
db = DataFlow(database_url=f"sqlite:///{db_path}")

@db.model
class Account:
    owner: str
    balance: float
    currency: str = "SGD"

async def main():
    await db.initialize()

    # Seed data
    await db.express.create(
        "Account", {"owner": "Alice", "balance": 1000.0, "currency": "SGD"}
    )
    await db.express.create(
        "Account", {"owner": "Bob", "balance": 500.0, "currency": "SGD"}
    )

    # -- 1. ACCESS -- db.transactions property -------------------------------
    # TransactionManager is accessed via db.transactions.

    txn_mgr = db.transactions
    assert isinstance(txn_mgr, TransactionManager)
    print("TransactionManager accessible via db.transactions")

    # -- 2. TRANSACTION -- context manager for atomic operations -------------
    # transaction() yields a transaction context dict with:
    #   id, isolation_level, status, operations
    #
    # On normal exit: status becomes "committed"
    # On exception:   status becomes "rolled_back"

    with txn_mgr.transaction() as txn:
        assert txn["status"] == "active"
        assert txn["isolation_level"] == "READ_COMMITTED", "Default isolation"
        assert "id" in txn

        # Track operations inside the transaction
        txn["operations"].append({"type": "debit", "account": "Alice", "amount": 200})
        txn["operations"].append({"type": "credit", "account": "Bob", "amount": 200})

        print(f"Transaction {txn['id']}: {len(txn['operations'])} operations")

    # After the context manager exits normally, status is "committed"
    assert txn["status"] == "committed"
    print("Transaction committed successfully")

    # -- 3. ISOLATION LEVELS -------------------------------------------------
    # Specify isolation level when creating a transaction.

    with txn_mgr.transaction(isolation_level="SERIALIZABLE") as txn:
        assert txn["isolation_level"] == "SERIALIZABLE"
        txn["operations"].append({"type": "audit", "details": "balance check"})

    assert txn["status"] == "committed"
    print("SERIALIZABLE transaction committed")

    # -- 4. ROLLBACK on exception --------------------------------------------
    # If an exception occurs inside the context manager, the transaction
    # is automatically rolled back.

    try:
        with txn_mgr.transaction() as txn:
            txn["operations"].append(
                {"type": "debit", "account": "Alice", "amount": 9999}
            )
            # Simulate a business rule violation
            raise ValueError("Insufficient funds")
    except ValueError:
        pass  # Expected

    assert txn["status"] == "rolled_back"
    assert txn.get("error") == "Insufficient funds"
    print("Transaction rolled back on exception")

    # -- 5. ACTIVE TRANSACTIONS -- monitoring --------------------------------
    # get_active_transactions() returns currently active transactions.
    # After a transaction completes (commit or rollback), it is removed.

    active = txn_mgr.get_active_transactions()
    assert len(active) == 0, "No active transactions after all completed"

    # -- 6. ROLLBACK ALL -- emergency operation ------------------------------
    # rollback_all() rolls back every active transaction at once.
    # Useful for emergency shutdown or cleanup.

    result = txn_mgr.rollback_all()
    assert result["success"] is True
    assert result["count"] == 0, "No active transactions to roll back"
    assert isinstance(result["rolled_back_transactions"], list)
    print(f"Emergency rollback: {result['count']} transactions affected")

    # -- 7. Nested transaction pattern ---------------------------------------
    # Transactions are independent -- nesting creates separate contexts.

    with txn_mgr.transaction() as outer:
        outer["operations"].append({"type": "outer_op"})

        with txn_mgr.transaction() as inner:
            inner["operations"].append({"type": "inner_op"})

        # Inner committed, outer still active
        assert inner["status"] == "committed"

    assert outer["status"] == "committed"
    print("Nested transactions: both committed independently")

    # -- 8. Edge case: exception in nested transaction -----------------------
    # Inner rollback does not affect the outer transaction.

    with txn_mgr.transaction() as outer:
        outer["operations"].append({"type": "outer_op"})

        try:
            with txn_mgr.transaction() as inner:
                inner["operations"].append({"type": "risky_op"})
                raise RuntimeError("Inner failure")
        except RuntimeError:
            pass  # Inner rolled back

        assert inner["status"] == "rolled_back"
        # Outer continues
        outer["operations"].append({"type": "recovery_op"})

    assert outer["status"] == "committed"
    print("Inner rollback did not affect outer transaction")

    # -- 9. Transaction context data -----------------------------------------
    # The transaction context dict can carry arbitrary operation logs.

    with txn_mgr.transaction(isolation_level="READ_COMMITTED") as txn:
        txn["operations"].append(
            {
                "type": "transfer",
                "from": "Alice",
                "to": "Bob",
                "amount": 100.0,
                "currency": "SGD",
            }
        )

    assert len(txn["operations"]) == 1
    assert txn["operations"][0]["type"] == "transfer"
    assert txn["operations"][0]["amount"] == 100.0
    print("Transaction carries structured operation metadata")

    # -- Cleanup -------------------------------------------------------------
    await db.stop()

asyncio.run(main())

try:
    os.unlink(db_path)
except OSError:
    pass

print("PASS: 01-dataflow/07_transactions")
```

### Step-by-Step Explanation

1. **Accessing TransactionManager**: `db.transactions` returns the `TransactionManager` instance. This is your entry point for all transaction operations.

2. **Creating a transaction**: `with txn_mgr.transaction() as txn:` opens a transaction block. The yielded `txn` dict starts with `status: "active"` and an empty `operations` list. You append operation records to track what happens inside the block. On normal exit, the status changes to `"committed"`.

3. **Isolation levels**: Pass `isolation_level="SERIALIZABLE"` to `transaction()` for stricter consistency guarantees. The default `READ_COMMITTED` is sufficient for most operations.

4. **Automatic rollback**: When an exception is raised inside the transaction block, the context manager sets `status` to `"rolled_back"` and stores the error message in `txn["error"]`. The exception is re-raised so your surrounding `try/except` can handle it.

5. **Monitoring active transactions**: `get_active_transactions()` returns a list of transactions that have not yet committed or rolled back. After all transactions in this example complete, the list is empty.

6. **Emergency rollback**: `rollback_all()` is a safety mechanism that rolls back every active transaction at once. It returns a result dict with `success`, `count`, and `rolled_back_transactions`. Useful during application shutdown or error recovery.

7. **Nested transactions**: Each `transaction()` call creates an independent context. The inner transaction can commit or roll back without affecting the outer one. This is not SQL savepoint semantics -- these are truly independent transaction contexts.

8. **Inner rollback, outer commit**: When the inner transaction raises an exception, only the inner context is rolled back. The outer transaction catches the exception and continues to commit successfully. This pattern is useful for "best-effort" inner operations.

9. **Operation metadata**: The `operations` list in the transaction context can hold arbitrary structured data -- operation types, amounts, account references, timestamps. This data is useful for audit logging and debugging.

## Common Mistakes

| Mistake                                          | Correct Pattern                                           | Why                                                                                                         |
| ------------------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Not catching exceptions from failed transactions | Wrap `with txn_mgr.transaction()` in `try/except`         | The context manager rolls back and re-raises. Uncaught exceptions crash the program.                        |
| Assuming nested transactions share state         | Each `transaction()` creates an independent context       | Inner and outer transactions are separate. Rolling back inner does not roll back outer.                     |
| Using transactions for single operations         | Use `db.express.create()` directly for single-record CRUD | Transactions add overhead. A single Express call is already atomic at the database level.                   |
| Checking `txn["status"]` inside the block        | Check status after the `with` block exits                 | Inside the block, status is always `"active"`. It changes to `"committed"` or `"rolled_back"` only on exit. |

## Exercises

1. Implement a fund transfer function that debits one account and credits another inside a transaction. If the source account has insufficient funds, raise an exception and verify the transaction is rolled back.
2. Create three nested transactions where the middle one fails. Verify that the outer and inner transactions are unaffected by the middle one's rollback.
3. Start a transaction, add several operations, then call `get_active_transactions()` while still inside the block. Verify that the active transaction appears in the list. After the block exits, verify it is gone.

## Key Takeaways

- `db.transactions` provides access to the `TransactionManager`
- `transaction()` is a context manager: commit on success, rollback on exception
- The default isolation level is `READ_COMMITTED`; use `SERIALIZABLE` for strict consistency
- Rolled-back transactions capture the error message in `txn["error"]`
- Nested transactions are independent -- inner rollback does not affect outer
- `get_active_transactions()` monitors in-progress transactions
- `rollback_all()` is an emergency mechanism for bulk rollback
- The transaction context dict carries arbitrary operation metadata for audit trails

## Next Chapter

[Chapter 8: Multi-Tenant Context Switching](08_multi_tenant.md) -- Isolate data across tenants using sync and async context managers with automatic context restoration.
