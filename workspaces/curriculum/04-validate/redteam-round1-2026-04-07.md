# Red Team Round 1 — 2026-04-07

## Solution Execution

- **77/80 PASS** (75 non-slow + 2 slow pass)
- **3 TIMEOUT** (capstone exercises >300s: ascent03/ex_8, ascent07/ex_8, ascent08/ex_7)
- **1 BUG FIXED**: ascent04/ex_1 ShapeError (Spectral 10K labels vs 50K DataFrame)
- No code bugs remain — timeouts are compute-bound, not errors

## Setup & Installation

- 138/138 import + dataset tests pass
- kailash-ml upgraded 0.6.0 -> 0.7.0
- All extras install correctly with `uv sync --all-extras`

## CRITICAL Findings Fixed

| #   | File                          | Issue                         | Fix                                  |
| --- | ----------------------------- | ----------------------------- | ------------------------------------ |
| C1  | ascent04/ex_1.py              | ShapeError on Spectral labels | Slice DataFrame to match label count |
| C2  | ascent05/ex_6.py:90           | Hardcoded model="gpt-4o-mini" | Changed to model=model (env var)     |
| C3  | textbook/.../06_presets.py:84 | Hardcoded JWT secret          | Changed to os.environ                |
| C4  | shared/data_loader.py:109     | Unsafe pickle.load()          | Removed pickle support               |
| C5  | textbook/html/index.html      | 6 broken navigation links     | Fixed href values                    |
| C6  | 71 textbook HTML files        | 123 broken .md links          | Global .md -> .html fix              |
| C7  | .gitignore                    | node_modules/ not excluded    | Added entry                          |
| C8  | .env                          | Stale "PCML" reference        | Updated to ASCENT                    |
| C9  | test_imports.py               | kailash-ml min version stale  | Updated to 0.7.0                     |

## CRITICAL Findings In Progress (academic fix agent)

| #   | File           | Issue                                    |
| --- | -------------- | ---------------------------------------- |
| C10 | M4/ex_2.py:243 | Log-likelihood init 0 instead of -inf    |
| C11 | M6/ex_1.py:206 | LoRA param count formula wrong by 2000x  |
| C12 | M7/ex_4.py:323 | Gradient scaling formulas fabricated     |
| C13 | M1/ex_6.py:78+ | .to_pandas() framework boundary comments |

## CRITICAL Findings — Structural (Next Session)

| #   | Issue                                                | Scope                   |
| --- | ---------------------------------------------------- | ----------------------- |
| S1  | 15 solution/local exercise pairs misaligned in M4-M6 | Re-strip from solutions |
| S2  | 8 datasets 5-100x undersized vs curriculum spec      | Source real-world data  |
| S3  | Zero image/audio/video data files                    | Add multimodal datasets |
| S4  | 4 deck titles misaligned with CLAUDE.md              | Content resequencing    |
| S5  | Scaffolding 20-69pp too generous across ALL modules  | Re-strip exercises      |

## HIGH Findings Fixed/In Progress

- M2/ex_4.py: Incorrect lift + wrong Wilcoxon test
- M2/ex_5.py: Expected loss sign error
- M2/ex_4.py: Median SE extra sqrt(n)
- M1/ex_4.py: Correlation interpretation backwards
- M3/ex_3.py: Focal loss Hessian approximation
- M10/ex_6.py: GovernanceEngine constructor type
- M7/ex_7.py: Conv2 multi-channel simplification note
- M8/ex_3.py: Word2Vec stale gradient values
- M6/ex_8.py: Stale exercise count

## Dataset Size Gaps

| Dataset          | Curriculum | Actual  | Gap  |
| ---------------- | ---------- | ------- | ---- |
| HDB Resale       | 15M+       | 150,750 | 100x |
| ICU              | 60K stays  | 1,500   | 40x  |
| SG News          | 50K        | 5,000   | 10x  |
| CC Fraud         | 284K       | 50,000  | 5.7x |
| E-commerce       | 200K       | 50,000  | 4x   |
| Inventory Demand | real-world | 90      | toy  |

## Deck Issues

- 7/10 decks exceed 80-100 slide target (up to 161)
- M01 missing inline speaker notes
- M07/M08 sparse on Kailash bridge slides
- 6 decks missing three-layer text labels
- KaTeX loading inconsistent across decks

## Standards Compliance

| Check                    | Status                                 |
| ------------------------ | -------------------------------------- |
| Headers with border      | PASS (160/160)                         |
| OBJECTIVE line           | PASS (160/160)                         |
| TASKS list               | PASS (160/160)                         |
| TODO markers in local    | PASS (80/80)                           |
| No blanks in solutions   | PASS                                   |
| ASCENTDataLoader         | PASS                                   |
| No pandas in exercises   | PASS                                   |
| No forward module refs   | PASS                                   |
| Copyright + SPDX         | PASS (160/160)                         |
| No hardcoded API keys    | PASS                                   |
| Solution-local alignment | FAIL (15 pairs in M4-M6)               |
| Import consistency       | FAIL (25+ exercises)                   |
| Progressive scaffolding  | FAIL (all modules 20-69pp over target) |
| Hint coverage            | FAIL (M5/M6/M10: 15-42% missing)       |
