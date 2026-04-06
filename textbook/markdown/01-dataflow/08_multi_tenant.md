# Chapter 8: Multi-Tenant Context Switching

## Overview

Multi-tenant applications serve multiple organizations from a single deployment, and each tenant's data must be isolated. DataFlow's `TenantContextSwitch` provides **sync and async context managers** that set the active tenant for a block of code, automatically restore the previous context on exit (even after exceptions), and enforce tenant registration and lifecycle rules. This chapter teaches you how to register tenants, switch contexts, nest switches, deactivate tenants, and monitor switching statistics.

## Prerequisites

- Python 3.10+ installed
- Kailash DataFlow installed (`pip install kailash-dataflow`)
- Completed Chapters 1-4 (model definition and CRUD)
- Familiarity with Python context managers (`with` / `async with`)

## Concepts

### Concept 1: TenantContextSwitch

The `TenantContextSwitch` is accessed via `db.tenant_context`. It manages a registry of tenants and provides context managers (`switch()` for sync, `aswitch()` for async) that set the active tenant for the duration of a code block.

- **What**: A manager that tracks registered tenants and provides safe context-switching between them
- **Why**: Without explicit tenant context, queries could accidentally return or modify another tenant's data. Context managers guarantee that the tenant is set on entry and restored on exit
- **How**: Uses Python's `contextvars` module to store the current tenant ID. Context managers save the current value, set the new one, and restore the original on exit -- even if an exception occurs
- **When**: Use tenant context switching in any multi-tenant application -- SaaS platforms, shared infrastructure, white-label products

### Concept 2: Tenant Registration

Before switching to a tenant, you must register it with `ctx.register_tenant(id, name, metadata)`. Registration creates a `TenantInfo` object with the tenant's ID, human-readable name, optional metadata dict, and an `active` flag. Duplicate IDs and empty IDs are rejected with `ValueError`.

### Concept 3: Sync and Async Context Managers

`ctx.switch(tenant_id)` is a synchronous context manager (`with` block). `ctx.aswitch(tenant_id)` is an asynchronous context manager (`async with` block). Both yield a `TenantInfo` object and restore the previous context on exit. Use `switch()` in sync code and `aswitch()` in async code.

### Concept 4: Context Restoration and Nesting

Context switches nest correctly. If you switch to tenant A, then switch to tenant B inside a nested block, exiting the inner block restores tenant A (not None). This works because each context manager saves the current value before switching and restores it on exit, forming a stack.

- **What**: Automatic restoration of the previous tenant context when a switch block exits
- **Why**: Without restoration, a function that switches tenants would permanently change the tenant for all subsequent code -- a dangerous side effect
- **How**: Each context manager saves the current `contextvars` token on entry and resets it on exit
- **When**: Always use context managers for switching. Never set the tenant context directly -- the context manager handles save/restore automatically

### Concept 5: Tenant Lifecycle

Tenants can be deactivated with `ctx.deactivate_tenant(id)` and reactivated with `ctx.activate_tenant(id)`. A deactivated tenant remains registered but cannot be switched to -- attempts raise `ValueError`. This is useful for suspending tenants without losing their configuration.

### Key API

| Method / Property            | Parameters                                      | Returns                | Description                                       |
| ---------------------------- | ----------------------------------------------- | ---------------------- | ------------------------------------------------- |
| `db.tenant_context`          | --                                              | `TenantContextSwitch`  | Access the tenant context manager                 |
| `ctx.register_tenant()`      | `id: str`, `name: str`, `metadata: dict = None` | `TenantInfo`           | Register a new tenant                             |
| `ctx.get_tenant()`           | `tenant_id: str`                                | `TenantInfo` or `None` | Look up a tenant by ID                            |
| `ctx.list_tenants()`         | --                                              | `list[TenantInfo]`     | All registered tenants                            |
| `ctx.switch()`               | `tenant_id: str`                                | context manager        | Sync context switch                               |
| `ctx.aswitch()`              | `tenant_id: str`                                | async context manager  | Async context switch                              |
| `ctx.get_current_tenant()`   | --                                              | `str` or `None`        | Current tenant ID (None if no active switch)      |
| `ctx.require_tenant()`       | --                                              | `str`                  | Current tenant ID; raises `RuntimeError` if None  |
| `ctx.deactivate_tenant()`    | `tenant_id: str`                                | `None`                 | Suspend a tenant (cannot switch to it)            |
| `ctx.activate_tenant()`      | `tenant_id: str`                                | `None`                 | Reactivate a suspended tenant                     |
| `ctx.is_tenant_registered()` | `tenant_id: str`                                | `bool`                 | Check if a tenant exists                          |
| `ctx.is_tenant_active()`     | `tenant_id: str`                                | `bool`                 | Check if a tenant is active                       |
| `ctx.unregister_tenant()`    | `tenant_id: str`                                | `None`                 | Remove a tenant entirely                          |
| `ctx.get_stats()`            | --                                              | `dict`                 | Switching statistics (totals, active count, etc.) |
| `get_current_tenant_id()`    | --                                              | `str` or `None`        | Module-level helper for current tenant ID         |

## Code Walkthrough

```python
from __future__ import annotations

import asyncio
import os
import tempfile

from dataflow import DataFlow, TenantContextSwitch, TenantInfo, get_current_tenant_id

# -- Setup -------------------------------------------------------------------

db_path = os.path.join(tempfile.gettempdir(), "textbook_dataflow_08.db")
db = DataFlow(database_url=f"sqlite:///{db_path}")

@db.model
class Project:
    name: str
    status: str = "active"

async def main():
    await db.initialize()

    # -- 1. ACCESS -- db.tenant_context property -----------------------------
    # TenantContextSwitch is accessed via db.tenant_context.
    # It provides sync and async context managers for safely switching
    # between tenant contexts with guaranteed isolation.

    ctx = db.tenant_context
    assert isinstance(ctx, TenantContextSwitch)
    print("TenantContextSwitch accessible via db.tenant_context")

    # -- 2. REGISTER tenants -------------------------------------------------
    # Register tenants before switching to them. Each tenant has an id,
    # a human-readable name, and optional metadata.

    tenant_a = ctx.register_tenant("acme", "Acme Corporation", {"region": "APAC"})
    assert isinstance(tenant_a, TenantInfo)
    assert tenant_a.tenant_id == "acme"
    assert tenant_a.name == "Acme Corporation"
    assert tenant_a.metadata == {"region": "APAC"}
    assert tenant_a.active is True
    print(f"Registered tenant: {tenant_a.name}")

    tenant_b = ctx.register_tenant("globex", "Globex Inc.", {"region": "EU"})
    assert tenant_b.tenant_id == "globex"

    # Duplicate registration raises ValueError
    try:
        ctx.register_tenant("acme", "Acme Again")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "already registered" in str(e)

    # Invalid tenant_id raises ValueError
    try:
        ctx.register_tenant("", "Empty ID")
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    # -- 3. LIST and GET tenants ---------------------------------------------

    all_tenants = ctx.list_tenants()
    assert len(all_tenants) == 2
    assert all(isinstance(t, TenantInfo) for t in all_tenants)

    found = ctx.get_tenant("acme")
    assert found is not None
    assert found.tenant_id == "acme"

    not_found = ctx.get_tenant("nonexistent")
    assert not_found is None

    # -- 4. SWITCH -- sync context manager -----------------------------------
    # with ctx.switch(tenant_id) sets the tenant for the block.
    # Previous context is restored on exit, even on exception.

    # Before any switch, no tenant is active
    assert ctx.get_current_tenant() is None
    assert get_current_tenant_id() is None, "Module-level helper also returns None"

    with ctx.switch("acme") as tenant_info:
        assert isinstance(tenant_info, TenantInfo)
        assert tenant_info.tenant_id == "acme"

        # Inside the context, the current tenant is set
        assert ctx.get_current_tenant() == "acme"
        assert get_current_tenant_id() == "acme", "Module-level helper reads context"
        print(f"Inside sync switch: tenant={ctx.get_current_tenant()}")

    # After the context manager exits, the previous context is restored
    assert ctx.get_current_tenant() is None
    print("After sync switch: tenant restored to None")

    # -- 5. ASWITCH -- async context manager ---------------------------------
    # async with ctx.aswitch(tenant_id) -- same semantics for async code.
    # contextvars automatically propagates context to async tasks.

    async with ctx.aswitch("globex") as tenant_info:
        assert tenant_info.tenant_id == "globex"
        assert ctx.get_current_tenant() == "globex"
        print(f"Inside async switch: tenant={ctx.get_current_tenant()}")

    assert ctx.get_current_tenant() is None
    print("After async switch: tenant restored to None")

    # -- 6. NESTED switches --------------------------------------------------
    # Nested switches properly track and restore the previous context.

    with ctx.switch("acme"):
        assert ctx.get_current_tenant() == "acme"

        with ctx.switch("globex"):
            assert ctx.get_current_tenant() == "globex"

        # Inner switch exited -- restored to "acme"
        assert ctx.get_current_tenant() == "acme"

    # Outer switch exited -- restored to None
    assert ctx.get_current_tenant() is None
    print("Nested switches: proper context restoration verified")

    # -- 7. REQUIRE_TENANT -- strict enforcement -----------------------------
    # require_tenant() raises RuntimeError if no tenant context is active.
    # Useful for operations that must run within a tenant scope.

    try:
        ctx.require_tenant()
        assert False, "Should raise RuntimeError"
    except RuntimeError as e:
        assert "No tenant context is active" in str(e)

    with ctx.switch("acme"):
        tenant_id = ctx.require_tenant()
        assert tenant_id == "acme"

    # -- 8. DEACTIVATE and ACTIVATE tenants ----------------------------------
    # Deactivated tenants remain registered but cannot be switched to.

    ctx.deactivate_tenant("globex")
    assert ctx.is_tenant_registered("globex") is True
    assert ctx.is_tenant_active("globex") is False

    # Switching to a deactivated tenant raises ValueError
    try:
        with ctx.switch("globex"):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "not active" in str(e)

    # Reactivate the tenant
    ctx.activate_tenant("globex")
    assert ctx.is_tenant_active("globex") is True

    # Now switching works again
    with ctx.switch("globex"):
        assert ctx.get_current_tenant() == "globex"
    print("Deactivate/activate: tenant lifecycle verified")

    # -- 9. SWITCH to unregistered tenant ------------------------------------
    # Raises ValueError with a helpful message listing available tenants.

    try:
        with ctx.switch("unknown-corp"):
            pass
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "not registered" in str(e)
        assert "Available tenants" in str(e)

    # -- 10. UNREGISTER a tenant ---------------------------------------------

    ctx.register_tenant("temp", "Temporary Tenant")
    assert ctx.is_tenant_registered("temp") is True

    ctx.unregister_tenant("temp")
    assert ctx.is_tenant_registered("temp") is False

    # Cannot unregister while active context
    with ctx.switch("acme"):
        try:
            ctx.unregister_tenant("acme")
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "active context" in str(e)

    # -- 11. STATS -- context switching statistics ---------------------------

    stats = ctx.get_stats()
    assert isinstance(stats, dict)
    assert stats["total_tenants"] == 2, "Two registered tenants"
    assert stats["active_tenants"] == 2, "Both are active"
    assert stats["total_switches"] > 0, "Switches have been performed"
    assert stats["active_switches"] == 0, "No switch currently active"
    assert stats["current_tenant"] is None, "No current tenant outside switch"
    print(f"Stats: {stats['total_switches']} total switches performed")

    # -- 12. Context restoration on exception --------------------------------
    # Even when an exception occurs, the previous context is restored.

    with ctx.switch("acme"):
        try:
            with ctx.switch("globex"):
                assert ctx.get_current_tenant() == "globex"
                raise RuntimeError("Simulated failure")
        except RuntimeError:
            pass

        # Restored to "acme" despite exception in inner block
        assert ctx.get_current_tenant() == "acme"

    assert ctx.get_current_tenant() is None
    print("Exception safety: context restored after error")

    # -- Cleanup -------------------------------------------------------------
    await db.stop()

asyncio.run(main())

try:
    os.unlink(db_path)
except OSError:
    pass

print("PASS: 01-dataflow/08_multi_tenant")
```

### Step-by-Step Explanation

1. **Accessing TenantContextSwitch**: `db.tenant_context` returns the context switch manager. This is your entry point for all multi-tenant operations.

2. **Registering tenants**: `register_tenant("acme", "Acme Corporation", {"region": "APAC"})` creates a `TenantInfo` with an ID, display name, and optional metadata. The tenant starts as active. Duplicate IDs and empty IDs are rejected.

3. **Listing and getting tenants**: `list_tenants()` returns all registered tenants. `get_tenant(id)` returns a single `TenantInfo` or `None`.

4. **Sync context switch**: `with ctx.switch("acme") as tenant_info:` sets the current tenant to "acme" for the duration of the block. Both `ctx.get_current_tenant()` and the module-level `get_current_tenant_id()` return the active tenant ID. On exit, the previous context (None) is restored.

5. **Async context switch**: `async with ctx.aswitch("globex")` works identically but for async code. Python's `contextvars` module propagates the tenant context to async tasks automatically.

6. **Nested switches**: Switching to "acme", then switching to "globex" inside a nested block, correctly restores "acme" when the inner block exits. The outer block then restores None. This forms a proper stack.

7. **require_tenant()**: A strict enforcement method that raises `RuntimeError` if called outside a tenant context. Use this at the top of functions that must never run without a tenant.

8. **Deactivation and activation**: `deactivate_tenant("globex")` keeps the tenant registered but prevents switching to it. `activate_tenant("globex")` re-enables it. This supports tenant suspension without data loss.

9. **Unregistered tenant**: Switching to a tenant that was never registered raises `ValueError` with a message listing available tenants -- helpful for debugging typos.

10. **Unregistration**: `unregister_tenant("temp")` removes a tenant entirely. You cannot unregister a tenant while it is the active context.

11. **Statistics**: `get_stats()` returns a dictionary with `total_tenants`, `active_tenants`, `total_switches`, `active_switches`, and `current_tenant`. Useful for monitoring and dashboards.

12. **Exception safety**: Even when an exception occurs inside a nested switch, the context manager restores the previous tenant. This guarantees that errors never corrupt the tenant context.

## Common Mistakes

| Mistake                                         | Correct Pattern                                              | Why                                                                                            |
| ----------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Setting tenant context with a global variable   | Use `ctx.switch()` or `ctx.aswitch()` context managers       | Global variables are not restored on exit or exception. Context managers guarantee cleanup.    |
| Switching without registering first             | Call `ctx.register_tenant(id, name)` before `ctx.switch(id)` | Unregistered tenant IDs raise `ValueError`. Registration is required for safety.               |
| Using `switch()` in async code                  | Use `aswitch()` in `async def` functions                     | `switch()` is synchronous. In async code, `aswitch()` integrates correctly with `contextvars`. |
| Unregistering a tenant during an active context | Exit all switch blocks for that tenant first                 | Unregistering the active tenant would leave the context in an invalid state.                   |

## Exercises

1. Register three tenants and write a loop that switches to each tenant in sequence, creates a `Project` record, and prints the current tenant inside each switch. Verify that no tenant context leaks between iterations.
2. Write a function decorated with `require_tenant()` at the top that creates a record. Call it both inside and outside a `switch()` block to verify the enforcement behavior.
3. Deactivate a tenant, attempt to switch to it (catching the error), reactivate it, and switch successfully. Print the stats before and after the full cycle.

## Key Takeaways

- `db.tenant_context` provides the `TenantContextSwitch` for multi-tenant isolation
- Tenants must be registered before they can be switched to
- `switch()` and `aswitch()` are sync/async context managers that guarantee context restoration on exit
- Nested switches form a proper stack -- inner exit restores the outer tenant, not None
- `require_tenant()` enforces that code runs within a tenant context
- Tenants can be deactivated (suspended) and reactivated without losing registration
- `get_stats()` provides monitoring data for tenant switching activity
- Context restoration is exception-safe -- errors never corrupt the tenant state

## Next Chapter

[Chapter 9: Data Provenance](09_provenance.md) -- Track the origin, confidence, and change history of individual field values using typed provenance wrappers.
