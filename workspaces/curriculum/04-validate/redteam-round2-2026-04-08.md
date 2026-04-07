# Red Team Round 2 — 2026-04-08

## Scope

Full validation audit across 10 modules (80 exercises, 320 files, 10 decks).

## Agents Deployed

- **analyst** — Spec coverage, scaffolding measurement
- **testing-specialist** — Static test suite (1121 tests)
- **security-reviewer** — Secrets, PCML refs, commercial refs, pandas, stubs
- **reviewer** — Solution quality (imports, API correctness, exception handling)
- **gold-standards-validator** — Deck title consistency (10 decks + speaker notes)

## Results Summary

| Audit                    | Result                             | Action                     |
| ------------------------ | ---------------------------------- | -------------------------- |
| Static tests             | 1121/1121 PASS                     | None needed                |
| Spec coverage            | 320/320 files (100%)               | None needed                |
| Three-format consistency | 10/10 sample PASS                  | None needed                |
| Security                 | CLEAN — 0 critical                 | None needed                |
| Solution quality         | 1 bare except, 2 placeholder names | Fixed                      |
| Deck titles              | 7 mismatches (M1,M4-M7,M9,M10)     | Fixed all                  |
| Scaffolding (M9)         | 82.7% provided vs 25% target       | Re-stripped to ~25%        |
| Scaffolding (M1-M8,M10)  | Over-scaffolded                    | Documented, future work    |
| M10 slow exercises       | 5 exercises >60s                   | Optimized iteration counts |

## Fixes Applied

### Critical — Deck Title Drift (7 decks)

- ascent01: "Data Pipelines & Visualisation Mastery" → "Foundations — Statistics, Probability & Data"
- ascent04: "Supervised ML: The Complete Model Zoo" → "Unsupervised ML, Anomaly Detection & NLP"
- ascent05: "ML Engineering & Production" → "LLMs, AI Agents & RAG Systems"
- ascent06: "Unsupervised ML & Pattern Discovery" → "Alignment, Governance, RL & Production"
- ascent07: "Deep Learning: Architecture-Driven Feature Engineering" → "Deep Learning"
- ascent09: "LLMs, AI Agents & RAG Systems" → "Fine-Tuning, Alignment & RL"
- ascent10: "Alignment, RL & Governance" → "AI Governance, Safety & Enterprise"
- Also fixed cross-references in recap tables and closing slides

### Critical — M9 Scaffolding (8 exercises)

- All 8 M9 exercises re-stripped from ~17% blanked to ~75% blanked
- Blank markers: 8-29/file → 20-74/file
- 16 notebooks regenerated (8 Jupyter + 8 Colab)
- 1121 static tests still pass

### Important — Solution Quality

- ascent04/ex_2.py:252 — bare `except Exception: pass` → numeric fallback
- ascent04/ex_6.py:115,120 — `_placeholder_b` → `_fallback_income`/`_fallback_debt`

### Important — M10 Performance

- ex_2: FedAvg rounds 5→3, sigma_values 5→3, max_iter 200→50
- ex_4: GBM n_estimators 100→30
- ex_6: DriftMonitor reference 1000→200 samples
- ex_7: GBM n_estimators 100→30
- ex_8: GBM 100→30, n_adv 2000→500, n_certify 100→20, GBM cap at 40

## Accepted Deviations

- sklearn in M3 (8), M4 (7), M6 (1), M10 (6) — accepted per SDK gap issues
- ascent05/ex_8: model_weights_placeholder — intentional exercise shortcut
- 5 M6 files without polars — RL/PACT/governance, no data ops
- Scaffolding M1-M8,M10 still over-target (see outstanding)

## Outstanding

1. **Scaffolding M1-M4**: 80-90% provided vs 70→40% target. Flat curve instead of progressive.
2. **Scaffolding M7-M8**: ~56% provided vs 30% target.
3. **Scaffolding M10**: ~54% provided vs 20% target.
4. **12 SDK issues**: kailash-py #341-348, kailash-rs #240-243 still open.
5. **Notebook end-to-end execution**: 240 notebooks regenerated but never executed.

## Convergence Assessment

- Round 2 fixed all CRITICAL findings from this round (deck titles, M9 scaffolding)
- Remaining scaffolding gap across M1-M8,M10 is a multi-session effort (8+ modules × 8 exercises)
- No new CRITICAL or HIGH findings remain
- Recommending convergence for round 2, with scaffolding as tracked future work
