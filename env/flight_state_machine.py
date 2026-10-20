"""Flight states and control authority shared by manual input and agent tools."""
from enum import Enum


class FlightState(str, Enum):
    ON_GROUND = "on_ground"  # 在地面
    HOVERING = "hovering"  # 悬停
    FLY_AGENT = "fly_agent"  # agent 轨迹飞行


class ControlMode(str, Enum):
    MANUAL = "manual"
    AGENT = "agent"


class InvalidFlightTransition(ValueError):
    """Raised when a requested transition is not allowed."""


class FlightStateMachine:
    """Manage ON_GROUND <-> HOVERING <-> FLY_AGENT transitions.

    Defaults to ON_GROUND under MANUAL control. Repeating a state is a no-op.
    Manual stick motion stays HOVERING; FLY_AGENT is reserved for agent
    trajectory execution. The environment owns physical control and updates
    this shared instance when automatic takeoff/landing completes.
    """

    _TRANSITIONS = {
        FlightState.ON_GROUND: frozenset({FlightState.HOVERING}),
        FlightState.HOVERING: frozenset({FlightState.ON_GROUND, FlightState.FLY_AGENT}),
        FlightState.FLY_AGENT: frozenset({FlightState.HOVERING}),
    }

    def __init__(self, initial_state: FlightState = FlightState.ON_GROUND,
                 *, control_mode: ControlMode = ControlMode.MANUAL):
        self._validate_state(initial_state)
        if not isinstance(control_mode, ControlMode):
            raise TypeError("control_mode must be a ControlMode enum member")
        if initial_state is FlightState.FLY_AGENT and control_mode is ControlMode.MANUAL:
            raise InvalidFlightTransition("FLY_AGENT requires agent authority")
        self._control_mode = control_mode
        self._state = initial_state

    @property
    def control_mode(self) -> ControlMode:
        return self._control_mode

    @property
    def agent_authorized(self) -> bool:
        return self._control_mode is ControlMode.AGENT

    def set_agent_authorized(self, enabled: bool) -> bool:
        """Return whether authority changed; revocation always enters HOVERING."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        mode = ControlMode.AGENT if enabled else ControlMode.MANUAL
        if mode is self._control_mode:
            return False
        self._control_mode = mode
        if not enabled:
            self.transition_to(FlightState.HOVERING)
        return True

    @staticmethod
    def _validate_state(state: FlightState) -> None:
        if not isinstance(state, FlightState):
            raise TypeError("state must be a FlightState enum member")

    @property
    def state(self) -> FlightState:
        """Current state; changes must go through transition_to()."""
        return self._state

    @property
    def allowed_transitions(self) -> frozenset[FlightState]:
        """Other states directly reachable from the current state."""
        states = self._TRANSITIONS[self._state]
        return states if self.agent_authorized else states - {FlightState.FLY_AGENT}

    def can_transition_to(self, target: FlightState) -> bool:
        self._validate_state(target)
        return target == self._state or target in self.allowed_transitions

    def transition_to(self, target: FlightState) -> FlightState:
        """Apply a valid transition; rejected requests leave state unchanged."""
        if not self.can_transition_to(target):
            raise InvalidFlightTransition(
                f"Cannot transition from {self._state.value} to {target.value}; "
                "FLY_AGENT requires agent authority and transitions through hovering"
            )
        self._state = target
        return self._state
