# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT05 — Exercise 8: Production Deployment with Nexus
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Deploy an ML model and an agent wrapper through Nexus for
#   multi-channel access (API + CLI + MCP). Add JWT/RBAC authentication,
#   middleware (logging, rate limiting), and DriftMonitor integration
#   for production health monitoring.
#
# TASKS:
#   1. Set up ML model via InferenceServer + ModelRegistry
#   2. Create a Nexus app and register inference endpoints
#   3. Add JWT authentication and RBAC via NexusAuthPlugin
#   4. Register an agent-wrapped endpoint for intelligent queries
#   5. Integrate DriftMonitor for production health monitoring
#   6. Test all three channels: API, CLI, MCP
# ════════════════════════════════════════════════════════════════════════
"""
# NOTE: Do NOT use `from __future__ import annotations` with Nexus —
# it breaks Nexus dependency injection which inspects type annotations at runtime.

import asyncio
import os

import polars as pl

from kailash.db.connection import ConnectionManager
from kailash_ml.engines.model_registry import ModelRegistry
from kailash_ml.engines.inference_server import InferenceServer
from kailash_ml.engines.drift_monitor import DriftMonitor, DriftSpec
from kailash_ml.types import (
    FeatureSchema,
    FeatureField,
    ModelSignature,
    MetricSpec,
)

from kaizen import Signature, InputField, OutputField
from kaizen_agents import Delegate

from nexus import Nexus
from nexus.auth import JWTConfig, AuditConfig
from nexus.auth.plugin import NexusAuthPlugin

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model_name_llm = os.environ.get(
    "DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL")
)
if not model_name_llm or not os.environ.get("OPENAI_API_KEY"):
    print("Set OPENAI_API_KEY and DEFAULT_LLM_MODEL in .env to run this exercise")
    raise SystemExit(0)
# JWT secret must be >= 32 chars for HS256 (JWTConfig validates this)
jwt_secret = os.environ.get(
    "NEXUS_JWT_SECRET", "ascent05-exercise-jwt-secret-key-min-32-chars"
)


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")
# Use a slice as "production reference data" for drift monitoring
reference_data = credit.sample(n=min(5000, credit.height), seed=0)
production_data = credit.sample(n=min(1000, credit.height), seed=42)

print(f"=== Production Deployment Exercise ===")
print(f"Reference data: {reference_data.shape}")
print(f"Simulated production batch: {production_data.shape}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Set up ML model (ModelRegistry + InferenceServer)
# ══════════════════════════════════════════════════════════════════════


async def setup_ml_stack():
    """Register and warm a model, return InferenceServer."""
    # TODO: Implement setup_ml_stack():
    #   1. ConnectionManager("sqlite:///ascent05_nexus_demo.db") → await conn.initialize() → ModelRegistry(conn)
    #   2. Build ModelSignature with FeatureSchema(name="credit_input", entity_id_column="application_id",
    #      features=[5 FeatureFields: annual_income/float64, total_debt/float64,
    #        credit_utilisation/float64, late_payments_12m/int64, account_age_months/int64])
    #      output_columns=["default_probability", "risk_tier"], output_dtypes=["float64", "utf8"], model_type="classifier"
    #   3. await registry.register_model(name="credit_default_v2", artifact=b"model_weights_placeholder",
    #      metrics=[MetricSpec(name="auc_pr", value=0.62), MetricSpec(name="auc_roc", value=0.89)], signature=signature)
    #   4. await registry.promote_model(name="credit_default_v2", version=..., target_stage="production", reason="Exercise 8 deployment")
    #   5. Create InferenceServer(registry, cache_size=5); print model info; return (conn, registry, server)
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
    ____


conn, registry, inference_server = asyncio.run(setup_ml_stack())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Create Nexus app and register inference endpoints
# ══════════════════════════════════════════════════════════════════════
#
# Nexus is a zero-config multi-channel deployment platform.
# app.register() attaches a workflow or server to all channels.
# Channels: REST API (FastAPI), CLI (Click), MCP (tool server).


async def build_nexus_app(server: InferenceServer) -> Nexus:
    # TODO: Implement build_nexus_app():
    #   1. Create Nexus()
    #   2. Register a POST /models/credit_default_v2/predict endpoint using @app.endpoint decorator
    #      Handler returns {"model": "credit_default_v2", "status": "registered"}
    #   3. Print confirmation; return app
    ____
    ____
    ____
    ____
    ____


# ══════════════════════════════════════════════════════════════════════
# TASK 3: JWT authentication + RBAC via NexusAuthPlugin
# ══════════════════════════════════════════════════════════════════════
#
# NexusAuthPlugin is the unified authentication solution for Nexus.
# It combines JWT, RBAC, rate limiting, and audit logging into one plugin.
#
# RBAC (Role-Based Access Control) defines what each role can do.
# JWT tokens carry the user's role in the payload.
#
# Role hierarchy for this deployment:
#   admin        -> full access (predict, monitor, manage models)
#   analyst      -> predict + view metrics (read-only)
#   external_api -> predict only (for third-party integrations)


def configure_auth(app: Nexus) -> Nexus:
    # TODO: Implement configure_auth():
    #   1. Define rbac_roles dict: admin→["*"], analyst→["read:*","predict:*"], external_api→["predict:*"]
    #   2. Create NexusAuthPlugin.basic_auth(
    #        jwt=JWTConfig(secret=jwt_secret, algorithm="HS256", exempt_paths=["/health", "/docs", "/openapi.json"]),
    #        audit=AuditConfig(backend="logging", log_level="INFO", log_request_body=False))
    #   3. app.add_plugin(auth)
    #   4. Print middleware stack summary and role permissions; return app
    ____
    ____
    ____
    ____
    ____
    ____
    ____
    ____


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Agent-wrapped endpoint for intelligent queries
# ══════════════════════════════════════════════════════════════════════
#
# Beyond raw prediction, expose a natural-language endpoint backed by
# a Delegate agent. Users send a question in plain English; the agent
# calls the InferenceServer and returns an explanation.


class CreditAdvice(Signature):
    """Agent-powered credit advice with model prediction."""

    application_details: str = InputField(description="Credit application details")
    risk_assessment: str = OutputField(description="Risk assessment")
    model_prediction_summary: str = OutputField(description="Model prediction summary")
    recommendation: str = OutputField(description="APPROVE / REVIEW / DENY")
    explanation: str = OutputField(
        description="Plain-language explanation for the applicant"
    )


async def build_agent_endpoint(app: Nexus, server: InferenceServer) -> None:
    """Register a /predict/explain endpoint backed by an agent."""

    # TODO: Implement build_agent_endpoint():
    #   1. Create Delegate(model=model_name_llm, budget_usd=1.0)
    #   2. Define async handle_explain_request(payload: dict) -> dict:
    #      a. Extract 5 feature fields + application_id from payload (use .get with defaults)
    #      b. await server.predict(model_name="credit_default_v2", features=features)
    #      c. Build agent_prompt with prediction.prediction, features, and instructions
    #      d. Stream agent.run(agent_prompt) → accumulate response_text
    #      e. Return dict with default_probability, inference_time_ms, model_version, agent_explanation
    #   3. Register @app.endpoint("/predict/explain", methods=["POST"]) calling handle_explain_request
    #   4. Print endpoint registration confirmation
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
    ____


# ══════════════════════════════════════════════════════════════════════
# TASK 5: DriftMonitor integration
# ══════════════════════════════════════════════════════════════════════
#
# DriftMonitor watches for data distribution shifts between a reference
# dataset (training distribution) and production batches.
# PSI > 0.1 = moderate drift, > 0.2 = severe drift.
#
# Production pattern:
#   - Run DriftMonitor on each batch of predictions
#   - If drift detected: alert on-call, trigger retraining review
#   - Expose drift metrics via Nexus /monitor/drift endpoint


async def setup_drift_monitoring(app: Nexus) -> DriftMonitor:
    """Set up DriftMonitor and register a health endpoint."""

    # TODO: Implement setup_drift_monitoring():
    #   1. Select numeric_features: columns of reference_data with Float64/Float32/Int64/Int32 dtype
    #      excluding "default", take first 5
    #   2. Create DriftSpec(feature_columns=numeric_features, psi_threshold=0.1, ks_threshold=0.05, monitor_interval_hours=6)
    #   3. Create DriftMonitor(reference_data=reference_data.select(numeric_features), spec=drift_spec)
    #   4. Print reference/production batch sizes and monitoring features
    #   5. await monitor.check(production_data=production_data.select(numeric_features)); print drift report
    #      (overall_severity, features_drifted, per-feature PSI with ALERT flags)
    #   6. Define async drift_health_handler(payload) that calls monitor.check and returns
    #      {"overall_severity": ..., "features_drifted": ..., "alerts": [...]}
    #   7. Register GET /monitor/drift endpoint calling drift_health_handler
    #   8. Print confirmation; return monitor
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
    ____
    ____
    ____


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Assemble and test all channels
# ══════════════════════════════════════════════════════════════════════


async def assemble_and_test():
    """Wire all components together and run channel tests."""

    # TODO: Build Nexus app, configure auth plugin, add agent endpoint, add drift monitoring
    app = await build_nexus_app(inference_server)
    ____  # Hint: app = configure_auth(app)
    ____  # Hint: await build_agent_endpoint(app, inference_server)
    ____  # Hint: monitor = await setup_drift_monitoring(app)

    # TODO: Create a unified session using app.create_session()
    session = ____  # Hint: app.create_session()

    print(f"\n=== Full Nexus App Assembled ===")
    print(f"Session: {session}")
    print(f"Endpoints:")
    print(f"  POST /models/credit_default_v2/predict  -- ML prediction")
    print(f"  POST /predict/explain                   -- agent explanation")
    print(f"  GET  /monitor/drift                     -- drift health")
    print(f"Auth: NexusAuthPlugin (JWT + audit)")
    print(f"Channels: API (FastAPI) + CLI (Click) + MCP (tool server)")

    # In production, channels are tested via:
    #   API:  httpx.AsyncClient against the FastAPI app
    #   CLI:  app.cli.invoke(["predict", "--model", "credit_default_v2", ...])
    #   MCP:  await mcp_client.call_tool("predict_credit_default_v2", {...})
    #
    # Each channel serves the same endpoints with the same auth middleware.
    # This is the Nexus promise: deploy once, serve everywhere.

    return app, monitor


app, monitor = asyncio.run(assemble_and_test())


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"   PRODUCTION DEPLOYMENT ARCHITECTURE")
print(f"{'=' * 70}")
print(
    """
  REST API ------+
  CLI      ------| Nexus --- NexusAuthPlugin (JWT + RBAC + Rate Limit + Audit)
  MCP      ------+    |
                      +-- POST /models/*/predict  --- InferenceServer
                      +-- POST /predict/explain   --- InferenceServer + Delegate
                      +-- GET  /monitor/drift     --- DriftMonitor

  NexusAuthPlugin stack (applied as single plugin):
    1. AuditConfig    -- structured JSON logs, no PII
    2. JWTConfig      -- validates Bearer token (HS256, >= 32-char secret)
    3. RBAC           -- role-based permissions, fail-closed
    4. RateLimitConfig -- 60 req/min per user, burst=10

  Channel parity: same model, same auth, same middleware, three channels.
  This is the Nexus promise: deploy once, serve everywhere.

  Governance (Module 6 preview):
    -> PACT GovernanceEngine wraps the agent at /predict/explain
    -> Each credit decision is recorded in an AuditChain
    -> DriftMonitor alerts feed into the PACT policy engine
    -> Regulated industries: every prediction is auditable
"""
)

# Clean up
asyncio.run(conn.close())

print(
    "✓ Exercise 8 complete — production Nexus deployment with auth + drift monitoring"
)
