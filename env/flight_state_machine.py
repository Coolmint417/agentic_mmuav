"""Flight states and control authority shared by manual input and agent tools."""
from enum import Enum
import math


class FlightState(str, Enum):
    ON_GROUND = "on_ground"  # 表面支撑（地面、斜面或墙面）
    HOVERING = "hovering"  # 悬停
    FLY_AGENT = "fly_agent"  # agent 轨迹飞行
    CRAWLING = "crawling"  # 爬行（包括停止收尾）


# Physical actuator order: front-left, rear-left, rear-right, front-right.
SUCKER_NAMES = ("sucker_upleft", "sucker_bottomleft",
                "sucker_bottomright", "sucker_upright")


class ControlMode(str, Enum):
    MANUAL = "manual"
    AGENT = "agent"


class InvalidFlightTransition(ValueError):
    """Raised when a requested transition is not allowed."""


class FlightStateMachine:
    """Manage flight and ground locomotion transitions.

    Suction commands are an independent four-boolean vector in SUCKER_NAMES
    order; they can change in any flight state or authority mode. ON_GROUND
    denotes surface-supported rest, including walls after confirmed perching.

    Defaults to ON_GROUND under MANUAL control. Repeating a state is a no-op.
    Manual stick motion stays HOVERING; FLY_AGENT is reserved for agent
    trajectory execution. The environment owns physical control and updates
    this shared instance when automatic takeoff/landing completes.
    """

    _TRANSITIONS = {
        FlightState.ON_GROUND: frozenset({FlightState.HOVERING, FlightState.CRAWLING}),
        FlightState.HOVERING: frozenset({FlightState.ON_GROUND, FlightState.FLY_AGENT}),
        FlightState.FLY_AGENT: frozenset({FlightState.HOVERING, FlightState.ON_GROUND}),
        FlightState.CRAWLING: frozenset({FlightState.ON_GROUND}),
    }

    def __init__(self, initial_state: FlightState = FlightState.ON_GROUND,
                 *, control_mode: ControlMode = ControlMode.MANUAL,
                 sucker_states=(False, False, False, False)):
        self._validate_state(initial_state)
        if not isinstance(control_mode, ControlMode):
            raise TypeError("control_mode must be a ControlMode enum member")
        if initial_state is FlightState.FLY_AGENT and control_mode is ControlMode.MANUAL:
            raise InvalidFlightTransition("FLY_AGENT requires agent authority")
        self._control_mode = control_mode
        self._state = initial_state
        self._nominal_specific_thrust = None
        self.set_sucker_states(sucker_states)

    @property
    def nominal_specific_thrust(self):
        """Last successful perch's optimized f/m in m/s²; None after reset."""
        return self._nominal_specific_thrust

    def set_nominal_specific_thrust(self, value):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0
        ):
            raise ValueError("nominal_specific_thrust must be positive finite m/s² or None")
        self._nominal_specific_thrust = None if value is None else float(value)

    @property
    def sucker_states(self) -> tuple[bool, bool, bool, bool]:
        """Adhesion enable commands, not measurements of physical contact."""
        return self._sucker_states

    def set_sucker_states(self, states) -> tuple[bool, bool, bool, bool]:
        """Atomically replace all four commands, independent of flight/mode."""
        if not isinstance(states, (list, tuple)) or len(states) != 4:
            raise ValueError("sucker_states must contain four booleans in SUCKER_NAMES order")
        if any(type(value) is not bool for value in states):
            raise ValueError("each sucker state must be a bool")
        self._sucker_states = tuple(states)
        return self._sucker_states

    def set_sucker_state(self, index: int, enabled: bool) -> tuple[bool, bool, bool, bool]:
        if type(index) is not int or not 0 <= index < 4:
            raise ValueError("sucker index must be an integer in [0,3]")
        states = list(self._sucker_states)
        states[index] = enabled
        return self.set_sucker_states(states)

    @property
    def control_mode(self) -> ControlMode:
        return self._control_mode

    @property
    def agent_authorized(self) -> bool:
        return self._control_mode is ControlMode.AGENT

    def set_agent_authorized(self, enabled: bool) -> bool:
        """Revocation holds airborne flight; preserve supported rest and suction."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        mode = ControlMode.AGENT if enabled else ControlMode.MANUAL
        if mode is self._control_mode:
            return False
        self._control_mode = mode
        if not enabled and self._state is FlightState.FLY_AGENT:
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
                "ground locomotion must transition through on_ground; "
                "FLY_AGENT requires agent authority and transitions through hovering"
            )
        self._state = target
        return self._state
