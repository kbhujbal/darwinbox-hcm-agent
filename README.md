# Darwinbox HCM Assistant — Agentic HCM Workflow Engine + Self-Healing HR Ops Platform

A complete implementation of **both** tracks of the Darwinbox AI Engineering
take-home: **Part 1**, a multi-agent conversational engine for HR operations
(orchestrator → RAG Policy Agent / tool-calling Action Agent, multi-turn
state, cost-optimized), and **Part 2**, a proactive Self-Healing HR Ops
Platform built on top of it (anomaly detection, a hard-veto compliance
engine, a reinforcement-learning contextual bandit that learns from human
feedback, episodic memory, and human-in-the-loop approval) — all in one
running system with one Streamlit UI.

## What's Implemented

**Part 1 — Agentic HCM Workflow Engine**
- ✅ Orchestrator routing to 2 specialized sub-agents (Policy/RAG, Action)
- ✅ Multi-turn state, persisted per employee, survives process restarts
- ✅ RAG: chunked policy doc, embedded, grounded retrieval, refuses rather than hallucinates
- ✅ 3 mock tools (leave balance, leave application, payslip) with OpenAI-style schemas, retry + graceful fallback on failure
- ✅ Structured trace log per step (agent, tool I/O, latency, tokens, cost)
- ✅ Cost optimization measured at **96.8%** vs a naive all-LLM baseline (target was ≥20%)
- ✅ Bonus: Streamlit UI with a live reasoning trace panel

**Part 2 — Self-Healing HR Ops Platform**
- ✅ Supervisor + 4 sub-agents (Policy, Action, Anomaly Detection, Compliance) — all 3 trigger classes: reactive NL, on-demand "scheduled" scan, system-generated alert
- ✅ Anomaly detection over a 600-employee synthetic dataset: payroll outliers, leave abuse, compliance violations, each with a confidence score and a recommended action
- ✅ Contextual bandit (LinUCB) action-selection policy, persisted to disk, verified to actually learn (not just explore) and to survive a crash mid-run
- ✅ Reward signals: HITL approve/reject/modify, outcome recurrence, false positives, and a compliance-veto penalty (the largest single penalty)
- ✅ Human-in-the-loop: Streamlit approval queue (approve/reject/modify) + timeout-to-safe-default handling
- ✅ Compliance rules engine: 13 rules in YAML (not prompts), hard veto — gates **both** Part 1's and Part 2's actions
- ✅ Episodic memory (Chroma): live-verified confidence boost on a repeat anomaly (0.60 → 0.745)
- ✅ 15/15 evaluation harness (happy path, edge cases, adversarial, RL-specific) with pass/fail + reasoning
- ✅ RL diagnostics: reward-per-cycle, cumulative reward, and action-distribution-shift plots

Full technical write-up for Part 2 (architecture, design-decision
justifications, live measured results): **[PART2.md](PART2.md)**.
1-page Part 1 architecture brief: **[architecture_brief.md](architecture_brief.md)**.

## Architecture

### Part 1 — reactive conversation

```
                         ┌─────────────────────┐
   User (CLI/Streamlit)  │   Orchestrator Node  │  LangGraph StateGraph
   ───────────────────►  │  (regex fast-path +  │  + SqliteSaver checkpoint
                         │   Gemini Flash        │  (state persists across
                         │   fallback classifier)│   turns & restarts)
                         └──────────┬────────────┘
                     route: policy  │  route: action
                    ┌────────────────┴───────────────┐
                    ▼                                 ▼
          ┌───────────────────┐              ┌────────────────────┐
          │   Policy Agent     │              │    Action Agent     │
          │  (RAG, grounded)   │              │  (tool calling)     │
          │  Chroma + Gemini   │              │  leave_balance,     │
          │  embeddings        │              │  apply_leave,       │
          └─────────┬──────────┘              │  get_payslip        │
                    │                          │  + Compliance Agent │
                    ▼                          │  veto + retry       │
             hr_policy.md (chunked,            └─────────┬──────────┘
             persisted vector index)                      │
                    │                                      ▼
                    ▼                              Mock HR API layer
          Tracer → traces/*.jsonl (agent, input, output, tool I/O,
                    latency, tokens, cost, RL fields per step)
                    │
                    ▼
          Streamlit UI: chat pane + live trace/cost panel
```

### Part 2 — proactive anomaly handling

```
Signal (reactive_nl | scheduled_scan | system_alert)
        │
        ▼
   Supervisor (extends the orchestrator above)
        │
        ├─ reactive_nl "flag anyone who..." ──► Anomaly Query Agent (report-only, no side effects)
        │
        └─ scheduled_scan / system_alert ──► Anomaly Detection Agent (pure stats/rules, zero LLM cost)
                                                          │ Anomaly(type, confidence, evidence)
                                                          ▼
                                              Episodic Memory (Chroma) biases confidence + RL context
                                                          ▼
                                              RL Bandit (LinUCB) selects an action
                                                          ▼
                                              Compliance Agent — hard veto, overrides the bandit
                                                          │
                                    ┌─────────────────────┴──────────────────┐
                                    ▼                                         ▼
                          high-confidence, compliant                low-confidence OR vetoed
                                    │                                         │
                                    ▼                                         ▼
                          Action Agent auto-executes                HITL Queue (Streamlit / simulated
                                    │                                reviewer / timeout-to-safe-default)
                                    ▼                                         │
                          Episodic Memory write ◄───────────────────────────┘
                                    + reward computed → bandit.update() → persisted to disk
```

Every node in both diagrams communicates only through shared LangGraph
state — there is no direct agent-to-agent calling anywhere in this
codebase. Full diagram + design-decision justifications for Part 2 in
[PART2.md](PART2.md).

## Repository Layout

```
├── data/
│   ├── hr_policy.md                 # 17-section mock HR policy doc
│   ├── compliance_rules.yaml        # 13 structured compliance rules
│   ├── employees_dataset.json       # 600 synthetic employee records (seeded)
│   └── employees_ground_truth.json  # which records were deliberately made anomalous
├── src/
│   ├── config.py                     # model names, Gemini pricing table, thresholds
│   ├── llm/gemini_client.py          # generate()/embed() wrapper with token+cost accounting
│   ├── graph/                        # orchestrator, policy_agent, action_agent, anomaly_agent,
│   │                                 #   anomaly_query_agent, anomaly_pipeline, compliance_agent,
│   │                                 #   system_alert, state, build_graph, thread_registry
│   ├── rag/                          # chunker, Chroma vector store wrapper, ingest script
│   ├── tools/                        # OpenAI-style schemas, mock HR API, retry executor
│   ├── observability/                # JSONL tracer, cost aggregator
│   ├── dataset/                      # synthetic employee data generator
│   ├── anomaly/                      # payroll/leave/compliance detectors (pure stats, zero LLM cost)
│   ├── rl/                           # LinUCB bandit, reward fn, episodic memory, simulated reviewer
│   └── hitl/                         # pending-approval queue + timeout handling
├── ui/app.py                         # Streamlit: Chat tab + Anomaly Review Queue tab
├── scripts/
│   ├── run_cli.py                    # headless terminal chat loop
│   ├── cost_benchmark.py             # naive-vs-optimized cost comparison (real API calls)
│   ├── generate_dataset.py           # regenerate the synthetic employee dataset
│   ├── run_scan_cycle.py             # one on-demand "cron" anomaly scan
│   └── run_feedback_cycles.py        # N simulated RL feedback cycles + diagnostics plot
├── evals/                            # the 15 required evaluation cases + harness runner
└── tests/                            # 85 unit tests across both parts
```

## Setup

Requires Python 3.10+ (built and tested on 3.12) and a Gemini API key.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY=...

# build the policy vector index (run once, or after editing data/hr_policy.md)
python -m src.rag.ingest

# generate the synthetic employee dataset (already committed, but reproducible)
python -m src.dataset.generate
```

### Run it

```bash
# --- Part 1 ---
python scripts/run_cli.py --employee-id E1001      # terminal chat
streamlit run ui/app.py                             # Chat tab + Anomaly Review Queue tab
pytest tests/ -v                                     # all 85 tests, no API key needed
python scripts/cost_benchmark.py                     # naive-vs-optimized cost comparison (needs API key)

# --- Part 2 ---
python evals/run_harness.py                                              # 15-case evaluation harness
python scripts/run_scan_cycle.py --department Engineering                # one on-demand scan
python scripts/run_feedback_cycles.py --cycles 5 --department HR --alpha 0.3 --reset-memory
```

`--thread-id` on `run_cli.py` (or the Streamlit sidebar) lets you resume a
conversation across restarts — state is checkpointed to
`conversation_state.sqlite`, not held in memory. Mock employees:
`E1001`–`E1004` (see `src/tools/mock_api.py:MOCK_EMPLOYEES`) — `E1004` is
deliberately still in probation, useful for exercising the compliance gate.

## Key Design Decisions

**1. Hybrid orchestrator routing is the primary cost lever.** A regex/keyword
classifier (`src/graph/orchestrator.py`) handles clearly-worded requests —
"apply for leave", "leave balance", "payslip", "maternity", "policy",
"flag anyone who..." — with **zero LLM calls**. Only ambiguous input falls
back to a single Gemini **Flash** classification call. The naive baseline in
`scripts/cost_benchmark.py` instead routes every request through a **Pro**
model call with no shortcut, to give a fair, measured comparison.

**2. Grounded RAG, refuse rather than hallucinate.** `data/hr_policy.md` is
chunked one clause per `## ` heading (17 chunks, ~80–150 tokens each — small
enough that no clause is truncated), embedded with Gemini
`gemini-embedding-001`, and stored in a persistent local Chroma collection.
Retrieval is top-3 by cosine distance; any answer whose best-matching chunks
fall outside `RAG_DISTANCE_FLOOR` gets an explicit "I don't know, contact
HR" response instead of a guess. Every policy answer's trace records which
section(s) grounded it.

**3. Single-call slot extraction, templated responses.** The Action Agent
makes **one** Flash call to both pick a tool and extract its arguments,
rather than separate "which tool" and "which arguments" round trips. On a
successful tool call, the user-facing message is templated in Python from
the structured tool output — **no second LLM call** is spent turning JSON
into prose. This is the second cost lever, alongside routing; anomaly
detection scanning the full employee dataset with zero LLM calls is the
third (see [PART2.md](PART2.md)).

**4. Employee identity comes from session state, not the model.** `employee_id`
is injected by the graph from the authenticated session, never asked of the
LLM or the user. The Streamlit UI takes this further with
`src/graph/thread_registry.py`: each mock employee has their own persisted
"active" conversation thread, so switching the sidebar dropdown switches to
*that employee's* isolated conversation, and "Start new conversation" keeps
the old thread listed under "Previous conversations" instead of discarding it.

**5. Multi-turn state via LangGraph + SqliteSaver.** Conversation history and
any in-progress slot-filling are checkpointed per `thread_id` to
`conversation_state.sqlite`. Restarting `run_cli.py`/Streamlit with the same
thread id resumes the exact conversation state — this isn't just an
in-memory dict.

**6. Tool errors are retried, then gracefully degraded.** `src/tools/mock_api.py`
injects a configurable ~15% failure rate; `src/tools/executor.py` retries
with exponential backoff (3 attempts) before returning a structured fallback
the Action Agent turns into a user-facing "please try again / contact HR"
message instead of crashing the turn. `src/llm/gemini_client.py` separately
retries on Gemini API rate limits (a real, live-encountered `429`), honoring
the API's suggested backoff.

**7. Observability is structured JSONL, not print statements**, additively
extended for Part 2 (`signal_type`, `rl_action_selected`, `reward`,
`compliance_veto`) so Part 1's existing call sites didn't need to change.
The Streamlit trace panel and both cost/eval scripts read this same format.

**8. The Compliance Agent gates both parts' actions through one ruleset.**
`apply_leave` (Part 1) and anomaly-driven corrections (Part 2) both run
through the same `compliance_agent.evaluate()` against
`data/compliance_rules.yaml` — a rule only fires if its field is present in
the context being checked, which is what lets one 13-rule file cover both
without duplication. Full Part 2 design decisions (why LinUCB, the reward
function, episodic memory, HITL) are in [PART2.md](PART2.md).

## Measured Results

**Cost optimization** (`python scripts/cost_benchmark.py`, live Gemini API,
6 representative requests): **96.8% cost reduction** vs a naive all-LLM
baseline ($0.017205 → $0.000553; 15 LLM calls → 9) — well above the ≥20%
target. The naive baseline forces every request through the larger of the
two models this account had quota for, an LLM call for routing on every
request, the entire policy document stuffed into context, and a second
big-model call to turn tool output into prose; the optimized pipeline is
this project's actual default (regex routing, Flash, top-3 retrieved
chunks, templated responses). Raw numbers: `traces/cost_benchmark_result.json`.

**Anomaly detection quality** (measured against deliberately-injected ground
truth, ~5% of 600 employees per category):

| Anomaly type | Recall | Precision |
|---|---:|---:|
| payroll_outlier | 0.83 | 0.64 |
| leave_abuse | 1.00 | 0.81 |
| compliance_violation | 1.00 | 0.73 |

**Evaluation harness**: **15/15 PASS** (`python evals/run_harness.py`),
covering happy path, edge cases, adversarial inputs, and RL-specific checks.

**RL learning demonstrated live**: a 5-cycle run (21 real anomalies/cycle,
live Gemini API) showed 7-8 of 10 repeated anomalies changing their
proposed action between the first and last cycle, and `traces/rl_diagnostics.png`
plots the reward and action-distribution trend. Full numbers and discussion
of the (noisy but net-improving) reward trajectory in [PART2.md](PART2.md#live-results).

## Example Conversation (live run)

```
you> What is our maternity leave policy?
assistant> Based on Meridian Industries' policy, here are the details regarding Maternity Leave:
* Eligibility and Duration: Employees who are the birthing parent are entitled to 26 weeks of
  fully paid Maternity Leave for the first two children, and 12 weeks for the third child onward.
* Timing: Maternity Leave may begin up to 8 weeks before the expected delivery date.
* Notice and Documentation: A minimum of 15 calendar days' written notice is required, along
  with a physician's certificate confirming the expected delivery date.
Source: Section 5. Maternity Leave

you> How many sick leave days do I have?
assistant> You have 9 day(s) of sick leave available.

you> Apply for leave
assistant> To proceed, could you tell me which leave type (earned, casual, or sick leave) and
the start date (YYYY-MM-DD) and how many days?

you> earned leave
assistant> To proceed, could you tell me the start date (YYYY-MM-DD) and how many days?

you> 3 days starting 2026-06-15
assistant> Leave application submitted and approved for 3 day(s) starting 2026-06-15.

you> What is the capital of France?
assistant> The provided policy excerpts do not contain information about the capital of France.
Please contact HR for assistance with questions outside of Meridian Industries' company policies.
```

That last exchange is the refuse-rather-than-hallucinate behavior from
design decision #2. And, live-verified with the Part 2 compliance gate:
asking **E1004** (still in probation) to apply for earned leave is blocked
with *"Employees within their first 90 days cannot avail earned leave"*,
while the same request from an established employee with adequate notice
goes through normally.

## Observability

Every run writes `traces/{thread_id}.jsonl` — one line per graph step:
`agent_name`, `input`, `output`, `tool_calls`, `latency_ms`,
`tokens_in`/`tokens_out`, `cost_usd`, `model`, and (Part 2) `signal_type`,
`rl_action_selected`, `reward`, `compliance_veto`. The Streamlit UI's Chat
tab renders this live; the Anomaly Review Queue tab shows the bandit's
current learned action preferences as a chart.

## Testing

`pytest tests/` — **85 tests**, all offline (no API key needed except where
NL understanding is inherently tested, which gracefully requires a key):
Part 1 covers the chunker, regex router, and tool executor; Part 2 adds
anomaly detection precision/recall fixtures, the compliance rule evaluator,
the LinUCB bandit's actual learning behavior (not just its API), the reward
function, the HITL queue's timeout handling, episodic memory's confidence
boost, and the simulated reviewer. Plus the separate 15-case
`evals/` harness for the assignment's specific evaluation requirement.

## Known Limitations / What I'd Change at Production Scale

- The mock HR API is single-employee-record, in-process, and has no real
  auth — a production version needs a real identity/session layer.
- The regex router is hand-tuned against this demo's sample intents; at
  scale it would need a larger curated pattern set or a small fine-tuned
  classifier to stay cheap without becoming brittle.
- `RAG_DISTANCE_FLOOR` and the anomaly detection thresholds are tuned by
  inspection, not calibrated against labeled production query/incident logs.
- SQLite checkpointing and local Chroma/`.npz` bandit state are fine for a
  single-process demo; a multi-instance deployment needs both centralized
  (Postgres-backed checkpointer, shared vector store, shared RL state).
- The Gemini free tier's **daily** embedding quota (1000/day) is a real
  constraint this session hit during live testing — production needs a
  paid tier or more aggressive embedding caching.
- The compliance rules are hand-authored, not derived from
  `data/hr_policy.md` directly — keeping the two consistent is a manual
  burden at scale.
- The RL bandit's exploration constant (`alpha`) is a manual dial, not
  annealed automatically; production would decay it as confidence accumulates.

Full Part 2 write-up, live results, and additional limitations:
**[PART2.md](PART2.md)**.
