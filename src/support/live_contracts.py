"""Live serving decisions, separate from frozen model and evaluation contracts."""
from typing import Literal

from pydantic import model_validator

from support.contracts import AdminResponse, ClientResponse, ClientStatus, ServerRoute


class LiveAdminResponse(AdminResponse):
    server_route: ServerRoute | Literal["server_clarification"]

    @model_validator(mode="after")
    def validate_live_clarification(self):
        if self.server_route == "server_clarification":
            if self.raw_model_result is None or self.raw_model_result.human_escalation:
                raise ValueError("server clarification requires a non-escalating model result")
            if not any(event.node == "response_guard" and event.status == "ok"
                       and event.error == "repeated_previous_response"
                       for event in self.trace_events):
                raise ValueError("server clarification requires reply guard trace evidence")
        return self


class LiveClientResponse(ClientResponse):
    status: ClientStatus | Literal["server_clarification"]

    @model_validator(mode="after")
    def validate_live_clarification(self):
        if self.status == "server_clarification":
            if self.escalation or self.handoff_id is not None or self.analysis is None:
                raise ValueError("server clarification requires analysis and no escalation or handoff")
            if self.analysis.recommended_action == "escalate_human":
                raise ValueError("server clarification cannot override required model handoff")
        return self
