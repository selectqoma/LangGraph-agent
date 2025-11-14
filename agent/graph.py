from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    generate_answer,
    guard_query,
    passthrough,
    reject_request,
    retrieve_offline,
    retrieve_online,
    router,
)
from agent.state import AgentState


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("guard", guard_query)
    graph.add_node("route", passthrough)
    graph.add_node("offline_retrieve", retrieve_offline)
    graph.add_node("online_retrieve", retrieve_online)
    graph.add_node("generate", generate_answer)
    graph.add_node("reject", reject_request)

    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", lambda s: s["guard"], {"ok": "route", "reject": "reject"})
    graph.add_conditional_edges(
        "route", router, {"offline": "offline_retrieve", "online": "online_retrieve"}
    )
    graph.add_edge("offline_retrieve", "generate")
    graph.add_edge("online_retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("reject", END)

    return graph
