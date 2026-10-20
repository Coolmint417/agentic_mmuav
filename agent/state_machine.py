"""Conversation lifecycle for the agent, independent of model and tools."""
from enum import Enum


class AgentState(str, Enum):
    THINK = "think"
    EXECUTE = "execute"
    OBSERVATION = "observation"
    END = "end"


class ThinkOutcome(str, Enum):
    TOOL = "tool"
    FINAL = "final"


class InvalidAgentTransition(ValueError):
    """Raised when an event is submitted in the wrong agent state."""


class AgentStateMachine:
    """Manage one conversation turn at a time.

    Initial END means ready for the first turn. THINK includes generation,
    JSON parsing and selection of a tool call or final answer. Actual tool
    execution belongs to EXECUTE; collecting its feedback belongs to
    OBSERVATION. A parse failure stays in THINK so the caller can retry.

    This class only manages state. It does not invoke the model or tools,
    collect observations, or forcibly interrupt running operations.
    """

    def __init__(self):
        self._state = AgentState.END
        self._termination_requested = False
        self._termination_reason: str | None = None
        self._turn_id = 0
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool, reason: str = "manual control") -> None:
        """Gate all work on control authority; model cancellation is a runner hook."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        if not enabled and self._enabled:
            self.terminate(reason)
        self._enabled = enabled

    @property
    def turn_id(self) -> int:
        """Monotonic turn identity so late tool results cannot finish a new turn."""
        return self._turn_id

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def can_start_turn(self) -> bool:
        """Whether a new user message may begin a conversation turn."""
        return self._enabled and self._state is AgentState.END

    @property
    def termination_requested(self) -> bool:
        """Whether this turn was ended through terminate()."""
        return self._termination_requested

    @property
    def termination_reason(self) -> str | None:
        return self._termination_reason

    def _require_state(self, expected: AgentState) -> None:
        if not self._enabled:
            raise InvalidAgentTransition("Agent is disabled in manual control mode")
        if self._state is not expected:
            raise InvalidAgentTransition(
                f"Expected {expected.value}, got {self._state.value}"
            )

    def start_turn(self) -> AgentState:
        """Accept a new turn only after the previous turn has reached END."""
        self._require_state(AgentState.END)
        self._turn_id += 1
        self._termination_requested = False
        self._termination_reason = None
        self._state = AgentState.THINK
        return self._state

    def complete_think(self, outcome: ThinkOutcome) -> AgentState:
        """Route a successfully parsed TOOL or FINAL decision from THINK."""
        self._require_state(AgentState.THINK)
        if not isinstance(outcome, ThinkOutcome):
            raise TypeError("outcome must be a ThinkOutcome enum member")
        self._state = (
            AgentState.EXECUTE if outcome is ThinkOutcome.TOOL else AgentState.END
        )
        return self._state

    def complete_execute(self) -> AgentState:
        """Execution has finished; its result or error is ready to observe."""
        self._require_state(AgentState.EXECUTE)
        self._state = AgentState.OBSERVATION
        return self._state

    def complete_observation(self) -> AgentState:
        """Feedback has been collected and can be passed to the next THINK."""
        self._require_state(AgentState.OBSERVATION)
        self._state = AgentState.THINK
        return self._state

    def terminate(self, reason: str | None = None) -> AgentState:
        """Request termination from any state and end the current turn.

        A future runner must stop or cancel its active generation/tool work
        before accepting a new turn. This method only records the request
        and changes the logical state; it does not stop running code.
        Repeated termination requests preserve the first reason.
        """
        if reason is not None and not isinstance(reason, str):
            raise TypeError("termination reason must be a string or None")
        if not self._termination_requested:
            self._termination_requested = True
            self._termination_reason = reason
        self._state = AgentState.END
        return self._state
