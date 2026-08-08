"""Minimal Streamlit UI: chat pane + a live agent reasoning/trace panel.

Run with:
    streamlit run ui/app.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.graph.build_graph import build_graph, run_turn
from src.observability.cost import summarize
from src.observability.tracer import Tracer
from src.tools.mock_api import MOCK_EMPLOYEES

st.set_page_config(page_title="Darwinbox HCM Assistant", layout="wide")


def _init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    if "employee_id" not in st.session_state:
        st.session_state.employee_id = "E1001"
    if "tracer" not in st.session_state or st.session_state.get("_tracer_thread") != st.session_state.thread_id:
        st.session_state.tracer = Tracer(run_id=st.session_state.thread_id)
        st.session_state._tracer_thread = st.session_state.thread_id
    if "app" not in st.session_state:
        st.session_state.app = build_graph(st.session_state.tracer)


def _reset_conversation() -> None:
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    st.session_state.tracer = Tracer(run_id=st.session_state.thread_id)
    st.session_state._tracer_thread = st.session_state.thread_id
    st.session_state.app = build_graph(st.session_state.tracer)


_init_session()

with st.sidebar:
    st.header("Session")
    employee_label = st.selectbox(
        "Logged in as",
        options=list(MOCK_EMPLOYEES.keys()),
        format_func=lambda eid: f"{eid} — {MOCK_EMPLOYEES[eid]['name']}",
        index=list(MOCK_EMPLOYEES.keys()).index(st.session_state.employee_id),
    )
    st.session_state.employee_id = employee_label

    st.caption(f"Thread: `{st.session_state.thread_id}`")
    if st.button("Start new conversation"):
        _reset_conversation()
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
