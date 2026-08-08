"""Minimal Streamlit UI: chat pane + a live agent reasoning/trace panel.

Run with:
    streamlit run ui/app.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.graph import thread_registry
from src.graph.build_graph import build_graph, run_turn
from src.observability.cost import summarize
from src.observability.tracer import Tracer
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
st.caption(
    "Orchestrator routes to a RAG Policy Agent or a tool-calling Action Agent. "
    "Try: 'What is our maternity leave policy?' or 'Apply for 3 days leave starting June 15'."
)

chat_col, trace_col = st.columns([3, 2])

with chat_col:
    st.subheader("Conversation")
    snapshot = st.session_state.app.get_state({"configurable": {"thread_id": st.session_state.thread_id}})
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
