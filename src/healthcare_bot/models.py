from typing_extensions import TypedDict
from typing import Literal
from pydantic import BaseModel


class State(TypedDict):
    input: str
    decision: str
    output: str


class RouteDecision(BaseModel):
    route: Literal["appointment_bot", "customer_support_bot", "pharmacy_bot"]
    reason: str
