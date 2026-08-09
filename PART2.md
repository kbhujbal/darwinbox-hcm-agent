# Part 2 — Self-Healing HR Ops Platform

Extends the Part 1 repo (see [README.md](README.md)) with a Supervisor +
4 sub-agent architecture that proactively scans employee data for
anomalies, proposes corrective actions gated by a hard-veto compliance
engine, escalates uncertain decisions to a human, remembers past
incidents, and — the core requirement — uses a reinforcement-learning
contextual bandit that demonstrably improves its action proposals from
accumulated human feedback, with state persisted to disk.

## Architecture

```
Signal (reactive_nl | scheduled_scan | system_alert)
        │
        ▼
   Supervisor  (src/graph/orchestrator.py, extends Part 1's router)
        │
  ┌─────┴──────────────────────────────────────────────────┐
  │ reactive_nl: classify intent                             │ scheduled_scan /
  │                                                            │ system_alert:
  ▼                                                            │ straight to scan
 policy ──────► Policy Agent (Part 1, unchanged)               │
 action ──────► Action Agent ─► Compliance Agent (veto?) ─► execute/report
 anomaly_query ─► Anomaly Query Agent (report-only scan, no side effects)
                                                                 ▼
                                              Anomaly Detection Agent
                                       (src/anomaly/scoring.py — pure
                                        stats/rules, ZERO LLM cost)
                                                                 │ Anomaly(type, confidence, evidence)
                                                                 ▼
                                              Episodic Memory lookup
                                       (Chroma "incident_memory" — biases
                                        confidence + RL context on repeats)
                                                                 │
                                                                 ▼
                                       RL Bandit (LinUCB) selects 1 of 5:
                                       {auto-correct, escalate-to-manager,
                                        escalate-to-HR, flag-for-audit,
                                        no-action}
                                                                 │
                                                                 ▼
                                       Compliance Agent — hard veto against
                                       data/compliance_rules.yaml, overrides
                                       the bandit's choice if violated
                                                                 │
                                       ┌─────────────────────────┴───────────────┐
                                       ▼                                          ▼
                                 high-confidence, compliant              low-confidence OR vetoed
                                       │                                          │
                                       ▼                                          ▼
                                 Action Agent auto-executes            HITL Queue — human (Streamlit
                                       │                               Approvals tab) or simulated
                                       ▼                               reviewer decides, or times out
                                 Episodic Memory write ◄────────────────────────┘
                                       + reward computed → bandit.update() → persisted to disk
```

Every node communicates only through shared LangGraph state — no direct
agent-to-agent calls, the same hard requirement Part 1's graph already
satisfied, just extended with two new nodes (Anomaly Detection, Compliance)
rather than restructured.

## New Modules

```
data/employees_dataset.json, employees_ground_truth.json, compliance_rules.yaml
src/dataset/generate.py, schema.py       — synthetic 600-employee dataset generator, seeded, ~5%/category injected anomalies
src/anomaly/scoring.py, models.py        — stats/rule-based detectors (zero LLM cost), Anomaly dataclass
src/graph/anomaly_agent.py               — graph node wrapping scoring.scan()
src/graph/anomaly_query_agent.py         — reactive "flag anyone who..." NL queries (report-only)
src/graph/anomaly_pipeline.py            — memory → bandit → compliance → auto-execute/queue, the core pipeline
src/graph/compliance_agent.py            — loads YAML, evaluates + vetoes any proposed action
src/graph/system_alert.py                — third trigger class: mock upstream alert ingestion
src/graph/orchestrator.py, build_graph.py (extended) — +anomaly_query routing, +2 graph nodes
src/graph/action_agent.py (extended)     — apply_leave now gated through the Compliance Agent too
src/rl/bandit.py                         — LinUCB, numpy, persisted to rl_state/bandit_state.npz
src/rl/features.py                       — Anomaly -> 9-dim numeric context vector
src/rl/reward.py                         — HITL + outcome + compliance-veto reward computation
src/rl/episodic_memory.py                — second Chroma collection, reusing Part 1's VectorStore unchanged
src/rl/simulated_reviewer.py             — heuristic batch approver for feedback-cycle volume
src/rl/resolution.py                     — shared "resolve a HITL decision" used by UI and scripts alike
src/hitl/queue.py                        — pending-approval queue, JSON-backed, timeout sweep
src/tools/mock_api.py (extended)         — +correct_payroll_discrepancy, +remind_compliance_training
ui/app.py (extended)                     — "Anomaly Review Queue" tab: scan trigger + approve/reject/modify
scripts/generate_dataset.py, run_scan_cycle.py, run_feedback_cycles.py
evals/cases.py, run_harness.py           — the 15 required evaluation cases
tests/test_anomaly_*.py, test_bandit.py, test_reward.py, test_compliance_agent.py,
  test_hitl_queue.py, test_resolution.py, test_episodic_memory.py, test_features.py,
  test_simulated_reviewer.py, test_system_alert.py — 51 new unit tests (85 total with Part 1's 34)
```

## Setup & Run

Builds on Part 1's existing setup (`pip install -r requirements.txt`, `.env` with `GEMINI_API_KEY`).

```bash
python -m src.dataset.generate           # generates data/employees_dataset.json (~600 employees, seeded)
python evals/run_harness.py              # 15-case evaluation harness, pass/fail + reasoning
python scripts/run_scan_cycle.py --department Engineering   # one on-demand "cron" scan
python scripts/run_feedback_cycles.py --cycles 5 --department HR --alpha 0.3 --reset-memory
streamlit run ui/app.py                  # "Anomaly Review Queue" tab: scan + live approve/reject/modify
```

## Key Design Decisions & Justifications

**1. Anomaly detection is pure statistics/rules — zero LLM cost.** Payroll
outliers use per-(department, grade, month) cohort **median + MAD**
(median absolute deviation), not mean/stdev — a single extreme outlier in
a ~15-20 person cohort drags the mean and inflates the stdev enough to
mask its own z-score ("self-contamination"); median/MAD barely moves under
one contaminated point. This measurably fixed detection recall from ~0.5
to ~1.0 during tuning (see below). Leave abuse uses weekend-adjacency
clustering + a policy-day cap; compliance violations are direct rule
checks against the same YAML the Compliance Agent uses. An LLM (Flash) is
only used for two things: understanding a reactive NL request, and (not
yet wired) an optional human-readable reasoning string.

**2. LinUCB contextual bandit, not PPO/REINFORCE.** The decision is
single-shot — one action per anomaly, not a multi-step sequence — so
there's no credit-assignment problem policy-gradient methods are built
for. LinUCB's per-arm update is closed-form linear algebra (no gradient
descent, no training instability), its state is two small matrices per
arm (trivial to persist as `.npz`), and its learned weights can be
inspected directly to explain *why* behavior shifted — far easier to
narrate in a 10-minute walkthrough than a policy network's weights.

**3. Reward function**, `src/rl/reward.py`: HITL approve = **+1**, reject =
**−1**, modify = a partial score from a hand-defined 5-action ordinal
distance (our adaptation of "edit distance" for a categorical action
space — auto-correct↔escalate-to-manager is a small substitution,
auto-correct↔no-action is the largest); outcome recurrence within N cycles
= **−0.5**; false positive = **−0.5**; **compliance veto = −1.5**, the
largest single penalty (verified in eval case RL2), satisfying the
explicit requirement that vetoed actions get penalized hardest.

**4. Compliance Agent gates Part 1's actions too, not just Part 2's.**
`apply_leave` now runs through the same `compliance_agent.evaluate()` as
anomaly-driven actions (notice period, probation) — a live-verified
example: Diego Fernandez (`E1004`, 45 days tenure, still in probation)
requesting earned leave is correctly blocked with *"Employees within
their first 90 days cannot avail earned leave"*, while an established
employee with adequate notice is approved normally. One ruleset, two
callers.

**5. HITL is async, not a blocking graph node.** A LangGraph `invoke()` is
synchronous; a human might respond hours later. A queued anomaly simply
ends that graph run at a "queued" state; `src/hitl/queue.py` (JSON-backed,
timeout-swept on load) is resolved separately by the Streamlit Approvals
tab or the batch simulator, both funneling through
`src/rl/resolution.py` so reward computation, bandit update, and
episodic-memory write happen identically either way.

**6. Episodic memory reuses Part 1's `VectorStore` unchanged** — a second
Chroma collection (`incident_memory`). **Live-verified example**: a
payroll outlier recorded with `reward=1.0` for `auto-correct`; a near-
duplicate anomaly for a different employee then retrieved it at cosine
distance 0.018 and its confidence was boosted from **0.60 → 0.745** before
the bandit even saw it — directly producing the "faster, more confident
on the 2nd occurrence" behavior the assignment asks to demonstrate.

**7. Trace schema extended additively**, exactly as promised in Part 1's
README: `signal_type`, `rl_action_selected`, `reward`, `compliance_veto`
added to `TraceStep` with defaults, so Part 1's existing call sites didn't
need to change.

## Live Results

**Detection quality** (measured against the deliberately-injected ground
truth in `data/employees_ground_truth.json`, ~5% of 600 employees per
category):

| Anomaly type | Recall | Precision |
|---|---:|---:|
| payroll_outlier | 0.83 | 0.64 |
| leave_abuse | 1.00 | 0.81 |
| compliance_violation | 1.00 | 0.73 |

Tuned deliberately toward high recall — false positives are cheap here
(they route to human review, per the design), missed anomalies are not.

**Evaluation harness**: `python evals/run_harness.py` → **15/15 PASS**
(4 happy path, 4 edge case, 4 adversarial, 3 RL-specific; full reasoning
per case in `traces/evaluation_report.json`). One case (AD1, Policy Agent
refusing an out-of-scope question) requires a live API key and is skipped
gracefully without one.

**Feedback cycles** (`scripts/run_feedback_cycles.py --cycles 5
--department HR --alpha 0.3`, live Gemini API, 21 real anomalies/cycle):
total reward per cycle was **−17.5, −6.5, −11.5, −17.0, −13.5** — a clear
early improvement (cycle 2) followed by continued exploration noise rather
than a clean monotonic climb. A separate 3-cycle run with episodic memory
reset (`--reset-memory`) showed a cleaner **−16.0 → −7.5 → −6.0** before
hitting the Gemini free-tier's *daily* embedding quota (1000/day — used up
by this session's combined Part 1 + Part 2 live testing) mid-cycle-4; see
[Known Limitations](#known-limitations--what-id-change-at-production-scale).
Across both runs, **7-8 of 10** anomalies present in both the first and
last cycle had their proposed action change — the required "action
proposals measurably change" evidence — and `traces/rl_diagnostics.png`
(reward-per-cycle, cumulative reward, and action-distribution-shift plots)
is committed as the artifact for the Loom walkthrough.

**Persistence**: the bandit saves progressively (on every compliance veto
and every HITL resolution, not just at script exit) — verified directly by
loading `rl_state/bandit_state.npz` after an interrupted run and finding
meaningfully non-identity per-arm state (`‖A−I‖` ranging 22–87 across the
5 arms, reflecting genuinely different amounts of accumulated feedback per
action).

## Compliance Rules Engine

13 rules in `data/compliance_rules.yaml` (structured `field`/`operator`/
`value` predicates, no `eval()`), numbers kept consistent with Part 1's
`data/hr_policy.md`: overtime cap, earned/casual leave notice periods,
probation, resignation notice tiers, payroll-correction approval tier,
training-overdue block, bereavement caps, and a confidence floor
specifically for `auto-correct`. A rule only fires if its field is present
in the context being evaluated, which is what lets one ruleset cover both
leave applications and anomaly-driven corrections without duplication.

## Observability & Cost

Same JSONL trace format as Part 1, additively extended
(`signal_type`, `rl_action_selected`, `reward`, `compliance_veto`).
Anomaly detection itself is a **third cost-optimization lever** beyond
Part 1's two (regex routing, templated responses): scanning the full
600-employee dataset is pure Python/numpy, zero LLM calls — the HR scan
above (21 anomalies found scanning ~85 employees) cost nothing beyond the
per-anomaly embedding calls for episodic memory.

## Known Limitations / What I'd Change at Production Scale

- **Free-tier daily embedding quota (1000/day) is a real constraint this
  session hit**, not a hypothetical — production would use a paid tier or
  batch/cache embeddings more aggressively (e.g., only embed on genuinely
  novel anomaly descriptions, not every single pipeline pass).
- **The simulated reviewer's "ideal action" function is hand-tuned**, not
  learned from real human judgment — it's a stand-in for training volume,
  documented as such; production replaces it entirely with the real
  HITL flywheel.
- **LinUCB's exploration constant (`alpha`) is a manual dial**, not
  decayed automatically. Production would anneal it down as the bandit
  accumulates confidence, converging faster than the fixed-alpha runs
  shown here.
- **Episodic memory and the bandit are both single-process, local state**
  (Chroma + a local `.npz` file) — consistent with Part 1's scope
  decision, but a multi-instance deployment needs both centralized.
- **The Compliance Agent's rules are hand-authored**, not derived from
  the actual `data/hr_policy.md` text — keeping the numbers manually
  consistent between the two is a real maintenance burden at scale; a
  production version would generate/validate rules from the policy
  documents directly.
