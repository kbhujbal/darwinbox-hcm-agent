"""Wires the orchestrator + 2 sub-agents into a LangGraph StateGraph with a
SQLite checkpointer, so conversation state survives process restarts.
"""
from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src import config
from src.graph.action_agent import handle_action
from src.graph.orchestrator import route_request
from src.graph.policy_agent import answer_policy_question
from src.graph.state import HCMState
from src.observability.tracer import Tracer


def _clarify_node(state: HCMState, tracer: Tracer) -> dict:
    response = (
        "I'm not sure whether that's a policy question or an HR action request — "
        "could you rephrase, e.g. 'What is our maternity leave policy?' or "
        "'Apply for 3 days leave starting June 15'?"
    )
    tracer.log_step(
        turn_id=state["turn_id"],
        agent_name="orchestrator",
        input={"user_input": state["user_input"]},
        output={"response": response},
    )
    return {
        "final_response": response,
        "history": [{"role": "assistant", "content": response, "agent": "orchestrator"}],
    }


def build_graph(tracer: Tracer, checkpointer=None):
    graph = StateGraph(HCMState)

    graph.add_node("orchestrator", lambda state: route_request(state, tracer))
    graph.add_node("policy_agent", lambda state: answer_policy_question(state, tracer))
    graph.add_node("action_agent", lambda state: handle_action(state, tracer))
    graph.add_node("clarify", lambda state: _clarify_node(state, tracer))

    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        lambda state: state["route"],
        {"policy": "policy_agent", "action": "action_agent", "clarify": "clarify"},
    )
    graph.add_edge("policy_agent", END)
    graph.add_edge("action_agent", END)
    graph.add_edge("clarify", END)

    if checkpointer is None:
        conn = sqlite3.connect(str(config.CHECKPOINT_DB), check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    return graph.compile(checkpointer=checkpointer)


def run_turn(app, thread_id: str, employee_id: str, user_text: str) -> str:
    """Runs one conversational turn, deriving turn_id from persisted history
    so restarting the process mid-conversation resumes correctly."""
    run_config = {"configurable": {"thread_id": thread_id}}
    snapshot = app.get_state(run_config)
    current_history = (snapshot.values or {}).get("history", []) if snapshot else []
    turn_id = len(current_history) // 2 + 1

    input_state = {
        "thread_id": thread_id,
        "employee_id": employee_id,
        "turn_id": turn_id,
        "user_input": user_text,
        "history": [{"role": "user", "content": user_text, "agent": None}],
    }
    result = app.invoke(input_state, config=run_config)
    return result["final_response"]
