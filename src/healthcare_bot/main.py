from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from healthcare_bot.models import State, RouteDecision
from healthcare_bot.llm import llm
from healthcare_bot.prompts import (
    APPOINTMENT_SYSTEM_PROMPT,
    SUPPORT_SYSTEM_PROMPT,
    PHARMACY_SYSTEM_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
)


router = llm.with_structured_output(RouteDecision)


def appointment_bot(state: State) -> dict[str, str]:
    messages = [
        SystemMessage(content=APPOINTMENT_SYSTEM_PROMPT),
        HumanMessage(content=state["input"]),
    ]
    result = llm.invoke(messages)
    return {"output": result.content}


def customer_support_bot(state: State) -> dict[str, str]:
    messages = [
        SystemMessage(content=SUPPORT_SYSTEM_PROMPT),
        HumanMessage(content=state["input"]),
    ]
    result = llm.invoke(messages)
    return {"output": result.content}


def pharmacy_bot(state: State) -> dict[str, str]:
    messages = [
        SystemMessage(content=PHARMACY_SYSTEM_PROMPT),
        HumanMessage(content=state["input"]),
    ]
    result = llm.invoke(messages)
    return {"output": result.content}


def orchestrator(state: State) -> dict[str, str]:
    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=state["input"]),
    ]
    decision = router.invoke(messages)
    return {"decision": decision.route, "reason": decision.reason}


def route_decision(state: State):
    if state["decision"] == "appointment_bot":
        return "appointment_bot"
    elif state["decision"] == "customer_support_bot":
        return "customer_support_bot"
    elif state["decision"] == "pharmacy_bot":
        return "pharmacy_bot"
    else:
        raise ValueError(f"Unknown decision: {state['decision']}")


graph = StateGraph(State)

graph.add_node("appointment_bot", appointment_bot)
graph.add_node("customer_support_bot", customer_support_bot)
graph.add_node("pharmacy_bot", pharmacy_bot)
graph.add_node("orchestrator", orchestrator)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges(
    "orchestrator",
    route_decision,
    {
        "appointment_bot": "appointment_bot",
        "customer_support_bot": "customer_support_bot",
        "pharmacy_bot": "pharmacy_bot",
    },
)
graph.add_edge("appointment_bot", END)
graph.add_edge("customer_support_bot", END)
graph.add_edge("pharmacy_bot", END)

graph_workflow = graph.compile()

if __name__ == "__main__":
    demo = graph_workflow.invoke({"input": "I need to change my check-up appointment."})
    print(demo["decision"])
    print(demo["output"])
