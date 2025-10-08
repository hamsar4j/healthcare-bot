from typing import Literal
from typing_extensions import TypedDict
from pydantic import BaseModel

from healthcare_bot.config import settings

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


llm = init_chat_model(
    model=settings.llm,
    model_provider=settings.model_provider,
    api_key=settings.api_key,
)


class State(TypedDict):
    input: str
    decision: str
    output: str


APPOINTMENT_SYSTEM_PROMPT = (
    "You are an empathetic healthcare scheduler. Help patients pick suitable "
    "appointment times, gather required information, and confirm next steps."
)

SUPPORT_SYSTEM_PROMPT = (
    "You are a friendly healthcare customer support assistant. Answer "
    "questions about services, billing, insurance, and clinic policies."
)


class RouteDecision(BaseModel):
    route: Literal["appointment_bot", "customer_support_bot"]
    reason: str


router = llm.with_structured_output(RouteDecision)


def appointment_bot(state: State) -> dict[str, str]:
    result = llm.invoke(state["input"])
    return {"output": result.content}


def customer_support_bot(state: State) -> dict[str, str]:
    result = llm.invoke(state["input"])
    return {"output": result.content}


def orchestrator(state: State) -> dict[str, str]:
    decision = router.invoke(
        [
            SystemMessage(
                content=(
                    "Choose the best worker for the latest user need. "
                    "Respond with route and a brief reason."
                )
            ),
            HumanMessage(content=state["input"]),
        ]
    )
    return {"decision": decision.route, "reason": decision.reason}


def route_decision(state: State):
    if state["decision"] == "appointment_bot":
        return "appointment_bot"
    elif state["decision"] == "customer_support_bot":
        return "customer_support_bot"
    else:
        raise ValueError(f"Unknown decision: {state['decision']}")


graph = StateGraph(State)

graph.add_node("appointment_bot", appointment_bot)
graph.add_node("customer_support_bot", customer_support_bot)
graph.add_node("orchestrator", orchestrator)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges(
    "orchestrator",
    route_decision,
    {
        "appointment_bot": "appointment_bot",
        "customer_support_bot": "customer_support_bot",
    },
)
graph.add_edge("appointment_bot", END)
graph.add_edge("customer_support_bot", END)


graph_workflow = graph.compile()


if __name__ == "__main__":
    demo = graph_workflow.invoke({"input": "I need to change my check-up appointment."})
    print(demo["output"])
