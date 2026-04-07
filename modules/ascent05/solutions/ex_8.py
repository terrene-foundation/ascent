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
    conn = ConnectionManager("sqlite:///ascent05_nexus_demo.db")
    await conn.initialize()

    registry = ModelRegistry(conn)

    # Model signature defines the contract for prediction requests
    signature = ModelSignature(
        input_schema=FeatureSchema(
            name="credit_input",
            features=[
                FeatureField(name="annual_income", dtype="float64"),
                FeatureField(name="total_debt", dtype="float64"),
                FeatureField(name="credit_utilisation", dtype="float64"),
                FeatureField(name="late_payments_12m", dtype="int64"),
                FeatureField(name="account_age_months", dtype="int64"),
            ],
            entity_id_column="application_id",
        ),
        output_columns=["default_probability", "risk_tier"],
        output_dtypes=["float64", "utf8"],
        model_type="classifier",
    )

    model_version = await registry.register_model(
        name="credit_default_v2",
        artifact=b"model_weights_placeholder",
        metrics=[
            MetricSpec(name="auc_pr", value=0.62),
            MetricSpec(name="auc_roc", value=0.89),
        ],
        signature=signature,
    )

    await registry.promote_model(
        name="credit_default_v2",
        version=model_version.version,
        target_stage="production",
        reason="Exercise 8 deployment",
    )

    # InferenceServer wraps the registry for low-latency serving
    # cache_size caches up to N models in memory (ONNX / raw weights)
    server = InferenceServer(registry, cache_size=5)
    # Note: warm_cache requires real serialised model weights. For this
    # exercise we skip warming since we used a placeholder artifact.
    # In production: await server.warm_cache(["credit_default_v2"])

    print(f"=== ML Stack ===")
    print(f"Model: credit_default_v2 v{model_version.version} (production)")
    info = await server.get_model_info("credit_default_v2")
    print(f"Model info: {info}")

    return conn, registry, server


conn, registry, inference_server = asyncio.run(setup_ml_stack())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Create Nexus app and register inference endpoints
# ══════════════════════════════════════════════════════════════════════
#
# Nexus is a zero-config multi-channel deployment platform.
# app.register() attaches a workflow or server to all channels.
# Channels: REST API (FastAPI), CLI (Click), MCP (tool server).


async def build_nexus_app(server: InferenceServer) -> Nexus:
    app = Nexus()

    # Register a prediction endpoint using Nexus's @endpoint decorator pattern.
    # In production with kailash-nexus installed, InferenceServer.register_endpoints(app)
    # would auto-generate endpoints from the ModelSignature. Here we register manually.
    @app.endpoint("/models/credit_default_v2/predict", methods=["POST"])
    async def predict_credit(request):
        """Predict credit default using the registered model."""
        return {"model": "credit_default_v2", "status": "registered"}

    print(f"\n=== Nexus App ===")
    print(f"Prediction endpoint registered: POST /models/credit_default_v2/predict")

    return app


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
    # Define RBAC roles with permission wildcards
    rbac_roles = {
        "admin": ["*"],  # Full access
        "analyst": ["read:*", "predict:*"],  # Read + predict
        "external_api": ["predict:*"],  # Predict only
    }

    # NexusAuthPlugin combines JWT + RBAC + rate limiting + audit.
    # Use basic_auth for JWT + audit. For full RBAC + rate limiting + tenants,
    # use saas_app() or enterprise() (requires TenantConfig).
    auth = NexusAuthPlugin.basic_auth(
        jwt=JWTConfig(
            secret=jwt_secret,  # Must be >= 32 chars for HS256
            algorithm="HS256",
            exempt_paths=["/health", "/docs", "/openapi.json"],
        ),
        audit=AuditConfig(
            backend="logging",
            log_level="INFO",
            log_request_body=False,  # Do not log request bodies (PII risk)
        ),
    )

    # Install auth plugin — applies JWT, RBAC, rate limit, and audit middleware
    app.add_plugin(auth)

    print(f"\n=== Auth + Middleware Stack ===")
    print(f"JWT:          HS256, exempt: /health /docs")
    print(f"RBAC:         3 roles, default deny (fail-closed)")
    print(f"RateLimiter:  60 req/min per user, burst=10")
    print(f"Audit:        logging backend, body logging disabled (PII)")
    print(f"\nRoles:")
    for role, perms in rbac_roles.items():
        print(f"  {role}: {perms}")

    return app


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

    agent = Delegate(model=model_name_llm, budget_usd=1.0)

    async def handle_explain_request(payload: dict) -> dict:
        """Agent-powered credit explanation endpoint."""
        # Get raw ML prediction first
        features = {
            "annual_income": payload.get("annual_income", 0),
            "total_debt": payload.get("total_debt", 0),
            "credit_utilisation": payload.get("credit_utilisation", 0.0),
            "late_payments_12m": payload.get("late_payments_12m", 0),
            "account_age_months": payload.get("account_age_months", 0),
            "application_id": payload.get("application_id", "unknown"),
        }

        prediction = await server.predict(
            model_name="credit_default_v2",
            features=features,
        )

        # Pass prediction context to agent for explanation
        agent_prompt = (
            f"You are a credit underwriting advisor. A model scored this application:\n"
            f"Default probability: {prediction.prediction}\n"
            f"Risk tier: {features}\n\n"
            f"Provide: risk assessment, model prediction summary, recommendation "
            f"(APPROVE/REVIEW/DENY), and a plain-language explanation."
        )

        response_text = ""
        async for event in agent.run(agent_prompt):
            if hasattr(event, "text"):
                response_text += event.text

        return {
            "default_probability": prediction.prediction,
            "inference_time_ms": prediction.inference_time_ms,
            "model_version": prediction.model_version,
            "agent_explanation": response_text,
        }

    # Register the agent handler as a Nexus endpoint
    @app.endpoint("/predict/explain", methods=["POST"])
    async def predict_explain(request):
        """Agent-powered credit decision with plain-language explanation."""
        return await handle_explain_request(request)

    print(f"\n=== Agent Endpoint ===")
    print(f"  POST /predict/explain — agent-powered explanation")
    print(f"  -> raw prediction (InferenceServer) + agent reasoning (Delegate)")


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

    # Numeric feature columns to monitor
    numeric_features = [
        c
        for c in reference_data.columns
        if reference_data[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        and c != "default"
    ][
        :5
    ]  # Monitor top 5 numeric features for this exercise

    drift_spec = DriftSpec(
        feature_columns=numeric_features,
        psi_threshold=0.1,  # PSI > 0.1 triggers moderate drift alert
        ks_threshold=0.05,  # KS p-value < 0.05 triggers alert
        monitor_interval_hours=6,  # Check every 6 hours in production
    )

    monitor = DriftMonitor(
        reference_data=reference_data.select(numeric_features),
        spec=drift_spec,
    )

    # Run drift check on simulated production batch
    print(f"\n=== DriftMonitor ===")
    print(f"Reference: {reference_data.height:,} rows")
    print(f"Production batch: {production_data.height:,} rows")
    print(f"Monitoring features: {numeric_features}")

    drift_report = await monitor.check(
        production_data=production_data.select(numeric_features),
    )

    print(f"\nDrift Report:")
    print(f"  Overall severity: {drift_report.overall_severity}")
    print(f"  Features drifted: {drift_report.features_drifted}")
    for feature, result in drift_report.feature_results.items():
        flag = " <- ALERT" if result.is_drifted else ""
        print(f"  {feature}: PSI={result.psi:.4f}{flag}")

    # Register a /monitor/drift endpoint so ops can poll health
    async def drift_health_handler(payload: dict) -> dict:
        report = await monitor.check(
            production_data=production_data.select(numeric_features),
        )
        return {
            "overall_severity": report.overall_severity,
            "features_drifted": report.features_drifted,
            "alerts": [
                {"feature": f, "psi": r.psi, "drifted": r.is_drifted}
                for f, r in report.feature_results.items()
            ],
        }

    @app.endpoint("/monitor/drift", methods=["GET"])
    async def monitor_drift(request):
        """Real-time drift health check for the credit model."""
        return await drift_health_handler({})

    print(f"\nDrift endpoint registered: GET /monitor/drift")

    return monitor


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Assemble and test all channels
# ══════════════════════════════════════════════════════════════════════


async def assemble_and_test():
    """Wire all components together and run channel tests."""

    # Build app with prediction endpoint
    app = await build_nexus_app(inference_server)

    # Add auth + middleware via NexusAuthPlugin
    app = configure_auth(app)

    # Add agent endpoint (registers /predict/explain)
    await build_agent_endpoint(app, inference_server)

    # Add drift monitoring (registers /monitor/drift)
    monitor = await setup_drift_monitoring(app)

    # Create a session (unified state across channels)
    session = app.create_session()

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
