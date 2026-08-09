"""Minimal Streamlit UI: chat pane + a live agent reasoning/trace panel.

Run with:
    streamlit run ui/app.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from src.graph import thread_registry
from src.graph.anomaly_agent import run_scan
from src.graph.anomaly_pipeline import process_anomaly
from src.graph.build_graph import build_graph, run_turn
from src.hitl import queue as hitl_queue
from src.observability.cost import summarize
from src.observability.tracer import Tracer
from src.rl.bandit import ACTIONS, DEFAULT_STATE_PATH, LinUCBBandit
from src.rl.resolution import resolve_decision
from src.tools.mock_api import MOCK_EMPLOYEES

st.set_page_config(page_title="Darwinbox HCM Assistant", layout="wide")

DEFAULT_EMPLOYEE = "E1001"


def _thread_label(thread_id: str, created_at: float) -> str:
    when = datetime.fromtimestamp(created_at).strftime("%b %d, %H:%M")
    steps = Tracer.read_run(thread_id)
    first_input = steps[0].get("input", {}).get("user_input") if steps else None
    if first_input:
        snippet = first_input if len(first_input) <= 40 else first_input[:37] + "..."
        return f"{when} — {snippet}"
    return f"{when} — (empty)"


def _activate_thread(thread_id: str) -> None:
    st.session_state.thread_id = thread_id
    st.session_state.tracer = Tracer(run_id=thread_id)
    st.session_state.app = build_graph(st.session_state.tracer)


def _sync_active_thread() -> None:
    """Make sure the loaded thread matches the currently selected employee."""
    active_thread = thread_registry.get_active_thread(st.session_state.employee_id)
    if st.session_state.get("thread_id") != active_thread:
        _activate_thread(active_thread)


def _init_session() -> None:
    if "employee_id" not in st.session_state:
        st.session_state.employee_id = DEFAULT_EMPLOYEE
    _sync_active_thread()


_init_session()

with st.sidebar:
    st.header("Session")
    employee_ids = list(MOCK_EMPLOYEES.keys())
    selected_employee = st.selectbox(
        "Logged in as",
        options=employee_ids,
        format_func=lambda eid: f"{eid} — {MOCK_EMPLOYEES[eid]['name']}",
        index=employee_ids.index(st.session_state.employee_id),
    )
    if selected_employee != st.session_state.employee_id:
        st.session_state.employee_id = selected_employee
        _sync_active_thread()
        st.rerun()

    st.caption(f"Thread: `{st.session_state.thread_id}`")

    if st.button("Start new conversation"):
        new_thread = thread_registry.start_new_thread(st.session_state.employee_id)
        _activate_thread(new_thread)
        st.rerun()

    past_threads = thread_registry.list_threads(st.session_state.employee_id)
    if len(past_threads) > 1:
        st.subheader("Previous conversations")
        options = [t["thread_id"] for t in past_threads]
        labels = {t["thread_id"]: _thread_label(t["thread_id"], t["created_at"]) for t in past_threads}
        resume_choice = st.selectbox(
            "Resume a conversation",
            options=options,
            format_func=lambda tid: labels[tid],
            index=options.index(st.session_state.thread_id)
            if st.session_state.thread_id in options
            else 0,
        )
        if resume_choice != st.session_state.thread_id:
            thread_registry.set_active_thread(st.session_state.employee_id, resume_choice)
            _activate_thread(resume_choice)
            st.rerun()

    st.divider()
    trace_steps = Tracer.read_run(st.session_state.thread_id)
    cost_summary = summarize(trace_steps)
    st.subheader("Session cost")
    st.metric("Total LLM/embedding calls", cost_summary.llm_calls)
    st.metric("Total tokens", cost_summary.total_tokens_in + cost_summary.total_tokens_out)
    st.metric("Total cost (USD)", f"${cost_summary.total_cost_usd:.6f}")

st.title("Darwinbox HCM Assistant")

chat_tab, approvals_tab = st.tabs(["Chat", "Anomaly Review Queue"])

with chat_tab:
    st.caption(
        "Orchestrator routes to a RAG Policy Agent or a tool-calling Action Agent. "
        "Try: 'What is our maternity leave policy?' or 'Apply for 3 days leave starting June 15'."
    )

    chat_col, trace_col = st.columns([3, 2])

    with chat_col:
        st.subheader("Conversation")
        snapshot = st.session_state.app.get_state(
            {"configurable": {"thread_id": st.session_state.thread_id}}
        )
        history = (snapshot.values or {}).get("history", []) if snapshot else []

        for turn in history:
            with st.chat_message(turn["role"]):
                if turn.get("agent"):
                    st.caption(f"via {turn['agent']}")
                st.write(turn["content"])

        user_text = st.chat_input("Ask about HR policy or request an action...")
        if user_text:
            with st.chat_message("user"):
                st.write(user_text)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = run_turn(
                        st.session_state.app,
                        st.session_state.thread_id,
                        st.session_state.employee_id,
                        user_text,
                    )
                st.write(response)
            st.rerun()

    with trace_col:
        st.subheader("Live reasoning trace")
        steps = Tracer.read_run(st.session_state.thread_id)
        if not steps:
            st.info("No steps yet — send a message to see the trace.")
        for step in reversed(steps):
            title = f"turn {step['turn_id']} · {step['agent_name']} · {step['latency_ms']:.0f}ms"
            with st.expander(title, expanded=False):
                st.markdown(f"**Model:** `{step.get('model') or 'n/a (regex / template)'}`")
                st.markdown(
                    f"**Tokens:** {step.get('tokens_in', 0)} in / {step.get('tokens_out', 0)} out "
                    f"· **Cost:** ${step.get('cost_usd', 0.0):.6f}"
                )
                st.markdown("**Input**")
                st.json(step.get("input", {}))
                st.markdown("**Output**")
                st.json(step.get("output", {}))
                if step.get("tool_calls"):
                    st.markdown("**Tool calls**")
                    st.json(step["tool_calls"])

with approvals_tab:
    st.caption(
        "Scan the employee dataset for anomalies (payroll outliers, leave abuse, compliance "
        "violations). High-confidence, compliant findings auto-execute; everything else — and "
        "anything a hard compliance rule vetoes — lands here for a human decision."
    )

    if "bandit" not in st.session_state:
        st.session_state.bandit = LinUCBBandit.load(DEFAULT_STATE_PATH)
    if "ops_tracer" not in st.session_state:
        st.session_state.ops_tracer = Tracer(run_id="ops")

    dept_options = ["All departments"] + [
        "Engineering", "Sales", "Finance", "HR", "Operations", "Marketing", "Customer Support",
    ]
    scan_col1, scan_col2 = st.columns([2, 1])
    with scan_col1:
        dept_choice = st.selectbox("Scan scope", dept_options)
    with scan_col2:
        st.write("")
        st.write("")
        run_scan_clicked = st.button("Run scan cycle", type="primary")

    if run_scan_clicked:
        dept_filter = None if dept_choice == "All departments" else dept_choice
        with st.spinner("Scanning employee dataset..."):
            anomalies = run_scan(turn_id=1, tracer=st.session_state.ops_tracer, department_filter=dept_filter)
            outcomes = {"auto_executed": 0, "queued_for_review": 0, "vetoed_queued": 0}
            for i, anomaly in enumerate(anomalies):
                result = process_anomaly(
                    anomaly, st.session_state.bandit, turn_id=i + 2, tracer=st.session_state.ops_tracer
                )
                outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
        st.success(
            f"Scanned {len(anomalies)} anomalies — {outcomes['auto_executed']} auto-executed, "
            f"{outcomes['queued_for_review']} queued for review, {outcomes['vetoed_queued']} "
            "vetoed by compliance and queued."
        )
        st.rerun()

    st.divider()
    pending = hitl_queue.list_pending()
    st.subheader(f"Pending review ({len(pending)})")

    if not pending:
        st.info("Nothing pending. Run a scan cycle above to populate the queue.")

    for item in pending:
        with st.container(border=True):
            st.markdown(
                f"**{item['anomaly_type'].replace('_', ' ').title()}** — {item['employee_id']} "
                f"· confidence {item['confidence']:.2f}"
            )
            st.caption(item["reasoning"])
            st.markdown(f"Proposed action: **{item['proposed_action']}**")
            with st.expander("Evidence"):
                st.json(item["evidence"])

            approve_col, modify_col, modify_select_col, reject_col = st.columns([1, 1, 2, 1])
            if approve_col.button("Approve", key=f"approve-{item['item_id']}"):
                resolve_decision(item["item_id"], "approve", st.session_state.bandit)
                st.rerun()

            modified = modify_select_col.selectbox(
                "modify to",
                options=[a for a in ACTIONS if a != item["proposed_action"]],
                key=f"modify-select-{item['item_id']}",
                label_visibility="collapsed",
            )
            if modify_col.button("Modify", key=f"modify-{item['item_id']}"):
                resolve_decision(
                    item["item_id"], "modify", st.session_state.bandit, modified_action=modified
                )
                st.rerun()

            if reject_col.button("Reject", key=f"reject-{item['item_id']}"):
                resolve_decision(
                    item["item_id"], "reject", st.session_state.bandit,
                    rejection_reason="reviewer judged this not accurate",
                )
                st.rerun()

    st.divider()
    st.subheader("Bandit action preference")
    st.caption("Relative learned weight per action (||theta||, normalized) — shifts as feedback accumulates.")
    dist = st.session_state.bandit.action_distribution()
    dist_df = pd.DataFrame({"action": list(dist.keys()), "weight": list(dist.values())})
    chart = (
        alt.Chart(dist_df)
        .mark_bar()
        .encode(
            x=alt.X("weight:Q", title="Relative weight"),
            y=alt.Y("action:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
        )
        .properties(height=220, padding={"left": 20})
    )
    st.altair_chart(chart, use_container_width=True)
