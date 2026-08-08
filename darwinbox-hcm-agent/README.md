# Darwinbox HCM Assistant (Part 1 — Agentic HCM Workflow Engine)

A multi-agent conversational engine for HR operations: an orchestrator routes
natural-language requests to a RAG-grounded **Policy Agent** or a
tool-calling **Action Agent**, with multi-turn state that survives process
restarts, structured per-step tracing, and a measured LLM cost-optimization
strategy.

Built for the Darwinbox AI Engineering take-home — Part 1 scope only (no RL,
compliance engine, or anomaly detection; see [Part 2 readiness](#part-2-readiness)).

## Architecture

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
          │  text-embedding-004│              │  apply_leave,       │
          └─────────┬──────────┘              │  get_payslip        │
                    │                          │  + retry/fallback   │
                    ▼                          └─────────┬──────────┘
             hr_policy.md (chunked,                       │
             persisted vector index)                      ▼
                                                    Mock HR API layer
                    │                                (simulated latency
                    ▼                                 + ~15% failures)
          Tracer → traces/*.jsonl (agent, input, output, tool I/O,
                    latency, tokens, cost per step)
                    │
                    ▼
          Streamlit UI: chat pane + live trace/cost panel
```

Both sub-agents are LangGraph nodes reached only through the shared graph
state — there is no direct agent-to-agent calling anywhere in this codebase.

## Repository Layout

```
darwinbox-hcm-agent/
├── data/hr_policy.md          # 17-section mock HR policy doc (leave, payroll, compliance)
├── src/
│   ├── config.py               # model names, Gemini pricing table, thresholds
│   ├── llm/gemini_client.py    # generate()/embed() wrapper with token+cost accounting
│   ├── graph/                  # orchestrator, policy_agent, action_agent, state, build_graph
│   ├── rag/                    # chunker, Chroma vector store wrapper, ingest script
│   ├── tools/                  # OpenAI-style schemas, mock HR API, retry executor
│   └── observability/          # JSONL tracer, cost aggregator
├── ui/app.py                   # Streamlit chat + live trace/cost panel
├── scripts/
│   ├── run_cli.py               # headless terminal chat loop
│   └── cost_benchmark.py        # naive-vs-optimized cost comparison (real API calls)
└── tests/                      # chunker, router, tool retry/fallback unit tests
```

## Setup

Requires Python 3.10+ (built and tested on 3.12) and a Gemini API key.

```bash
cd darwinbox-hcm-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY=...

# build the policy vector index (run once, or after editing data/hr_policy.md)
python -m src.rag.ingest
```

### Run it

```bash
# terminal chat
python scripts/run_cli.py --employee-id E1001

# Streamlit UI (chat + live trace panel)
streamlit run ui/app.py

# tests (no API key needed — pure logic: chunker, regex router, tool retry)
pytest tests/ -v

# cost benchmark (needs API key — makes real, small Gemini calls)
python scripts/cost_benchmark.py
```

`--thread-id` on `run_cli.py` (or the Streamlit sidebar's thread id) lets you
resume a conversation across restarts — state is checkpointed to
`conversation_state.sqlite`, not held in memory.

Mock employees available in the demo: `E1001`–`E1004` (see
`src/tools/mock_api.py:MOCK_EMPLOYEES`).

## Key Design Decisions

**1. Hybrid orchestrator routing is the primary cost lever.** A regex/keyword
classifier (`src/graph/orchestrator.py`) handles clearly-worded requests —
"apply for leave", "leave balance", "payslip", "maternity", "policy" — with
**zero LLM calls**. Only ambiguous input falls back to a single Gemini
**Flash** classification call. The naive baseline in
`scripts/cost_benchmark.py` instead routes every request through a **Pro**
model call with no shortcut, to give a fair, measured comparison.

**2. Grounded RAG, refuse rather than hallucinate.** `data/hr_policy.md` is
chunked one clause per `## ` heading (17 chunks, ~80–150 tokens each — small
enough that no clause is truncated), embedded with Gemini
`gemini-embedding-001`, and stored in a persistent local Chroma collection.
Retrieval is top-3 by cosine distance; any answer whose best-matching chunks
fall outside `RAG_DISTANCE_FLOOR` gets an explicit "I don't know, contact
HR" response instead of a guess — directly targeting the hallucination
failure signal called out in the brief. Every policy answer's trace records
which section(s) grounded it.

**3. Single-call slot extraction, templated responses.** The Action Agent
makes **one** Flash call to both pick a tool and extract its arguments
(`src/graph/action_agent.py`), rather than separate "which tool" and "which
arguments" round trips. On a successful tool call, the user-facing message
is templated in Python from the structured tool output — **no second LLM
call** is spent turning JSON into prose. This is the second cost lever,
alongside routing.

**4. Employee identity comes from session state, not the model.** `employee_id`
is injected by the graph from the authenticated session (simulated via
`--employee-id` / the Streamlit sidebar), never asked of the LLM or the user
— mirroring how a real Darwinbox session would already know who's logged in.

**5. Multi-turn state via LangGraph + SqliteSaver.** Conversation history and
any in-progress slot-filling (e.g., an `apply_leave` call missing
`start_date`) are checkpointed per `thread_id` to `conversation_state.sqlite`.
Restarting `run_cli.py`/Streamlit with the same thread id resumes the exact
conversation state — this isn't just an in-memory dict.

**6. Tool errors are retried, then gracefully degraded.** `src/tools/mock_api.py`
injects a configurable ~15% failure rate; `src/tools/executor.py` retries with
exponential backoff (3 attempts) before returning a structured
`{"status": "error", ...}` fallback the Action Agent turns into a
user-facing "please try again / contact HR" message instead of crashing the
turn.

**7. Observability is structured JSONL, not print statements.** Every graph
node writes one record to `traces/{thread_id}.jsonl` — agent name, input,
output, tool calls (with their own latency/attempts), latency, tokens
in/out, and cost. The Streamlit trace panel and `cost_benchmark.py` both
read this same format; nothing is UI-only.

**8. LLM calls also retry on rate limits, not just the mock tool layer.**
Live testing surfaced free-tier `429 RESOURCE_EXHAUSTED` responses from the
Gemini API itself (a real production concern, distinct from the *simulated*
mock-API failures in point 6). `src/llm/gemini_client.py` retries on 429s,
honoring the API's suggested `RetryInfo` backoff when present.

## Cost Optimization — Measured Result

Run `python scripts/cost_benchmark.py` (requires `GEMINI_API_KEY`) to
reproduce. It sends the same 6 representative HR requests (3 policy, 3
action) through:

- **naive**: every step forced through the larger of the two models this
  account has usable quota for (`PRO_MODEL`), an LLM call for routing on
  every request (no regex shortcut), the *entire* policy document stuffed
  into context for policy questions, and a second big-model call to turn
  tool output into prose for action requests.
- **optimized** (this project's default): regex fast-path routing,
  `FLASH_MODEL`, top-3 retrieved chunks only, and a templated
  (zero-LLM-cost) final response for action requests.

**Actual measured result (2026-08-08, live Gemini API):**

| | naive | optimized |
|---|---:|---:|
| LLM calls | 15 | 9 |
| tokens in | 8,884 | 3,678 |
| tokens out | 610 | 458 |
| cost (USD) | $0.017205 | $0.000553 |

**Savings: 96.8%** vs the naive baseline — well above the ≥20% target. The
gap is this large because most sample requests hit the zero-cost regex path
entirely (no orchestrator LLM call at all), and the naive baseline pays for
both a big-model router call *and* a separate big-model "prose-ify the JSON"
call that the optimized pipeline skips outright. Raw numbers:
`traces/cost_benchmark_result.json`.

> **Note on model names:** `FLASH_MODEL`/`PRO_MODEL` in `src/config.py` are
> set to whichever models this test account's API key actually had non-zero
> free-tier quota for (`gemini-flash-lite-latest` / `gemini-flash-latest`) —
> every genuine Gemini Pro-tier model returned `429 RESOURCE_EXHAUSTED` (0
> free quota) on this key. `PRO_MODEL` is therefore the best available
> stand-in for "a larger, non-cost-optimized model" rather than true Gemini
> Pro; swap in your account's actual models in `src/config.py` if they
> differ. The *savings percentage* is unaffected by this substitution since
> both pipelines were measured on the same live account.

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

That last exchange is the refuse-rather-than-hallucinate behavior from design
decision #2 — an out-of-scope question routed to the Policy Agent got an
explicit "I don't know" instead of an invented answer.

## Observability

Every run writes `traces/{thread_id}.jsonl` — one line per graph step:
`agent_name`, `input`, `output`, `tool_calls` (name, arguments, output,
success, attempts, latency), `latency_ms`, `tokens_in`/`tokens_out`,
`cost_usd`, `model`. The Streamlit UI's right-hand panel renders this live,
newest step first, with per-step token/cost/latency shown inline.

## Testing

`pytest tests/` covers the chunker (one chunk per policy clause, overlap
behavior on long sections), the regex router (action vs. policy vs.
ambiguous vs. mid-slot-filling continuation), and the tool executor (success,
retry-then-fallback, unknown tool, over-balance rejection, deterministic
payslip generation) — all offline, no API key required.

## Part 2 Readiness

Not built here (would be premature for a Part 1 submission), but the
architecture was deliberately kept extensible for it:

- LangGraph's "no direct agent-to-agent calls" is already how every node
  communicates — adding Part 2's Anomaly Detection and Compliance agents
  means new nodes/edges in `build_graph.py`, not a restructure.
- `src/rag/vector_store.py` is a generic persistent-Chroma wrapper keyed by
  collection name; Part 2's episodic memory is a second collection through
  the same class.
- `src/tools/schemas.py` + `executor.py` already separate "what a tool looks
  like" from "how it's dispatched with retry," so new corrective-action
  tools and a compliance veto check plug into the same pattern.
- The trace schema (`src/observability/tracer.py`) is additive — Part 2-only
  fields (`rl_action_selected`, `reward`, `compliance_veto`) extend the same
  JSONL record rather than replacing it.

## Known Limitations / What I'd Change at Production Scale

- The mock HR API is single-employee-record, in-process, and has no real
  auth — a production version needs a real identity/session layer feeding
  `employee_id` into the graph.
- The regex router is hand-tuned against the 10 sample intents in this demo;
  at scale it would need either a larger curated pattern set or a small
  fine-tuned classifier to stay cheap without becoming brittle as intent
  variety grows.
- `RAG_DISTANCE_FLOOR` is a fixed threshold tuned by inspection, not
  calibrated against a labeled retrieval-quality set — worth revisiting with
  real query logs.
- SQLite checkpointing is fine for a single-process demo; a multi-instance
  deployment needs a shared checkpoint store (Postgres-backed LangGraph
  checkpointer) so any instance can resume any thread.
