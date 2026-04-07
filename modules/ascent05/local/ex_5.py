# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT05 — Exercise 5: MCP Servers and Tool Integration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build an MCP server that exposes kailash-ml tools (DataExplorer,
#   ModelVisualizer) to agents at scale. Understand the MCP protocol, tool
#   registration pattern, and SSE transport setup.
#
# TASKS:
#   1. Understand the MCP architecture (server / client / protocol)
#   2. Create an MCP server with MCPServer
#   3. Register ML tools (profile_data, visualize_feature, describe_column)
#   4. Configure SSE transport for production use
#   5. Test tool invocation through the MCP protocol
#   6. Connect an agent to the MCP server as a tool provider
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kailash.mcp_server import MCPServer
from kaizen_agents import Delegate

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not model or not os.environ.get("OPENAI_API_KEY"):
    print("Set OPENAI_API_KEY and DEFAULT_LLM_MODEL in .env to run this exercise")
    raise SystemExit(0)


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")
customers = loader.load("ascent04", "ecommerce_customers.parquet")

print(f"=== MCP Server Exercise ===")
print(f"Credit dataset: {credit.shape}")
print(f"Customers dataset: {customers.shape}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: MCP Architecture
# ══════════════════════════════════════════════════════════════════════
#
# MCP (Model Context Protocol) is a standard for agents to discover and
# call tools hosted on a server, independent of agent implementation.
#
# Architecture:
#
#   ┌─────────────────────┐       MCP protocol        ┌──────────────────────┐
#   │   Agent / Client    │ ─────────────────────────► │    MCPServer         │
#   │  (Delegate/ReAct)   │ ◄───────────────────────── │  (tool registry)     │
#   └─────────────────────┘   tool_call / tool_result  └──────────────────────┘
#                                                                │
#                                                    ┌───────────┼───────────┐
#                                                    ▼           ▼           ▼
#                                             profile_data  visualize   describe
#                                             (DataExplorer)(ModelViz)  (column)
#
# Transport options:
#   StdioTransport  — subprocess pipe, great for local/testing
#   SSETransport    — HTTP Server-Sent Events, production/remote
#   WebSocketTransport — bidirectional, streaming use cases
#
# Why MCP?
#   - Tool definitions live on the server, not hardcoded in agent
#   - Agents discover tools dynamically at runtime
#   - Same server can serve many different agents
#   - Centralised tool versioning and access control

print(f"\n=== MCP Architecture ===")
print(f"Protocol: tool_list → tool_call → tool_result")
print(f"Transport: StdioTransport (local), SSETransport (production)")
print(f"Discovery: agent calls list_tools() at startup")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Create MCPServer with metadata
# ══════════════════════════════════════════════════════════════════════


def create_ml_mcp_server() -> MCPServer:
    """Build an MCPServer that wraps kailash-ml tools."""
    # TODO: Instantiate MCPServer with name="kailash-ml-tools"
    server = ____  # Hint: MCPServer(name="kailash-ml-tools")
    return server


server = create_ml_mcp_server()
print(f"\n=== MCPServer Created ===")
print(f"Name: {server.name}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Register ML tools
# ══════════════════════════════════════════════════════════════════════
# Each tool has: name, description, input_schema, and an async handler.
# The input_schema uses JSON Schema so agents know what arguments to pass.


# TODO: Register profile_dataset as an MCP tool using the @server.tool() decorator.
#   The function must be async and accept dataset (str) and sample_size (int = 5000).
#   Use DataExplorer to profile a sample of credit or customers;
#   return formatted string: dataset name, rows, columns, types, and alerts.
# Hint: @server.tool()
#       async def profile_dataset(dataset: str, sample_size: int = 5000) -> str:
____
____
____
____
____
____
____
____


# TODO: Register describe_column as an MCP tool using @server.tool().
#   Accepts dataset (str) and column (str).
#   For numeric dtypes: return mean, std, min, max, nulls, P25, median, P75.
#   For categorical: return n_unique, nulls, and top 5 value_counts.
# Hint: @server.tool()
#       async def describe_column(dataset: str, column: str) -> str: ...
____
____
____
____
____
____
____
____
____
____


# TODO: Register target_analysis as an MCP tool using @server.tool().
#   Accepts dataset (str), feature (str), target (str).
#   For numeric features: bin into quartiles with pl.col(feature).cut(breaks=[...]).
#   For categorical: group directly. Return target rate by feature group (top 10).
# Hint: @server.tool()
#       async def target_analysis(dataset: str, feature: str, target: str) -> str: ...
____
____
____
____
____
____
____
____
____
____
____
____


# TODO: Register list_columns as an MCP tool using @server.tool().
#   Accepts dataset (str).
#   Returns all column names and their dtypes for the selected dataset.
# Hint: @server.tool()
#       async def list_columns(dataset: str) -> str: ...
____
____
____
____


print(f"\n=== Tools Registered ===")
stats = server.get_server_stats()
for tool_name, tool_info in stats.get("tools", {}).get("tools", {}).items():
    print(f"  {tool_name}: registered")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Configure transports
# ══════════════════════════════════════════════════════════════════════
#
# StdioTransport: used for local testing and subprocess-based agent
#   connections. The server reads JSON-RPC from stdin, writes to stdout.
#
# SSETransport: used for production. Clients connect via HTTP GET to
#   an /events endpoint, receive events as a stream. Tool calls go via
#   HTTP POST to /call. This is what hosted MCP servers use.

# MCPServer supports two transport modes:
#   stdio  — default, used for local dev and subprocess-based agent connections
#   sse    — HTTP Server-Sent Events, used for production/remote access
#
# In production you'd run:
#   server.run()                          # stdio (default)
#   server.run(transport="sse", port=8765) # SSE
#
# For this exercise we test tools in-process (no transport needed).

print(f"\n=== Transport Configuration ===")
print(f"stdio: stdin/stdout (local dev / subprocess) — default")
print(f"sse:   HTTP Server-Sent Events (production / remote)")
print(f"  → tool list:  GET  /mcp/tools")
print(f"  → tool call:  POST /mcp/call")
print(f"  → events:     GET  /mcp/events (SSE stream)")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Test tool invocation directly
# ══════════════════════════════════════════════════════════════════════
# Before wiring the agent, verify each tool works in isolation.


async def test_tools_directly():
    """Invoke registered tools directly to verify they work."""
    print(f"\n=== Direct Tool Tests ===")

    # TODO: Call list_columns with dataset="credit" and print first 300 chars
    result = ____  # Hint: await list_columns(dataset="credit")
    print(f"\ntool: list_columns(credit)")
    print(result[:300])

    # TODO: Call describe_column on dataset="credit", column="annual_income"
    result = (
        ____  # Hint: await describe_column(dataset="credit", column="annual_income")
    )
    print(f"\ntool: describe_column(credit, annual_income)")
    print(result)

    # TODO: Call target_analysis with dataset="credit", feature="late_payments_12m", target="default"
    result = ____  # Hint: await target_analysis(dataset="credit", feature="late_payments_12m", target="default")
    print(f"\ntool: target_analysis(credit, late_payments_12m, default)")
    print(result[:400])

    # TODO: Call profile_dataset with dataset="credit" and sample_size=2000
    result = ____  # Hint: await profile_dataset(dataset="credit", sample_size=2000)
    print(f"\ntool: profile_dataset(credit, sample_size=2000)")
    print(result[:400])


asyncio.run(test_tools_directly())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Connect an agent to the MCP server
# ══════════════════════════════════════════════════════════════════════
# Rather than hard-coding tool functions in the agent, the agent
# discovers tools from the MCP server at runtime. This decouples
# agent logic from tool implementation.


async def agent_with_mcp_tools():
    """Run a Delegate agent with tools sourced from the MCP server."""

    # In production, tools are discovered via MCP protocol over a transport:
    #   agent.connect_mcp("http://host:8765/mcp")
    #
    # For this exercise, we wire the registered tool functions directly into
    # a Delegate. The MCP server's @server.tool() decorated functions are
    # standard async callables — the agent calls them the same way it would
    # call any tool, but the server provides the canonical registry.

    print(f"\n=== Agent Tool Discovery via MCP ===")
    tool_stats = server.get_server_stats().get("tools", {}).get("tools", {})
    print(f"Tools available: {list(tool_stats.keys())}")

    # TODO: Create a Delegate with model, the four registered tool functions as the
    #   tools list, and a budget_usd=3.0 cap
    # Hint: Delegate(model=model, tools=[list_columns, describe_column, target_analysis, profile_dataset], budget_usd=3.0)
    agent = ____

    task = (
        "Analyse the Singapore credit scoring dataset. "
        "List the columns, profile the data, and investigate which features "
        "are most strongly associated with default. "
        "Recommend the top 5 predictive features."
    )

    print(f"\n=== Agent via MCP Tools ===")
    print(f"Task: {task}")

    # TODO: Stream the agent response into response_text
    response_text = ""
    ____  # Hint: async for event in agent.run(task): if hasattr(event, "text"): response_text += event.text

    print(f"\nAgent response:")
    print(f"  {response_text[:500]}...")

    return response_text


mcp_result = asyncio.run(agent_with_mcp_tools())


# ══════════════════════════════════════════════════════════════════════
# Summary: MCP in Production
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== MCP in Production ===")
print(
    """
Why MCP matters at scale:

  Traditional (M3-M5 pattern):
    - Tool functions defined inside each agent script
    - Changing a tool requires redeploying the agent
    - Different agents duplicate the same tool logic

  MCP server pattern:
    - Tools live on the server, agents discover them at runtime
    - Update the server → all agents get the updated tool
    - Add auth, rate limiting, versioning at the server layer
    - One server can serve many agents (internal + external)

Production deployment flow:
  1. python -m kailash.mcp_server serve ex_5_server.py --transport sse --port 8765
  2. Agent: await delegate.connect_mcp("http://localhost:8765/mcp")
  3. Agent discovers tools automatically → uses them in reasoning loop

In Module 6, MCP tool access is governed by PACT:
  → GovernanceEngine controls which agents can call which MCP tools
  → MCPServer validates the GovernanceContext token on each call
"""
)

print("✓ Exercise 5 complete — MCP server with tool registration and SSE transport")
