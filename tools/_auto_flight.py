"""Shared lifecycle for no-argument tools using the environment's auto control."""
from dataclasses import dataclass
import json
import math
import re

from agent.state_machine import AgentState
from .tasks import execute_synchronously
from env.flight_state_machine import FlightState
from env.mmuav_env import POSITION_AUTO_NONE


@dataclass(frozen=True)
class AutoFlightConfig:
    timeout_seconds: float = 60.0

    def __post_init__(self):
        value = self.timeout_seconds
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")


def parse_no_argument_call(command, tool_name):
    if isinstance(command, str):
        payload = command.strip()
        match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", payload, flags=re.DOTALL)
        command = json.loads(match.group(1) if match else payload)
    if (
        not isinstance(command, dict)
        or set(command) != {"tool"}
        or command["tool"] != tool_name
    ):
        raise ValueError(
            f'Expected only {{"tool": "{tool_name}"}}; this tool accepts no parameters'
        )


class AutoFlightTool:
    """Reuse auto takeoff/landing and its completion criteria without duplicating them.

    Keep the initial flight state while the operation is pending. On success,
    latch a position target and switch both state machines. On interruption,
    install a hold command and report HOVERING if that command succeeds.
    """

    def __init__(self, environment, agent_state_machine, config=None):
        self.environment = environment
        self.agent = agent_state_machine
        self.flight = environment.flight_state_machine
        self.environment.bind_agent_state_machine(self.agent)
        self.config = config if config is not None else AutoFlightConfig()

    def _still_executing(self, turn_id):
        return (
            self.environment.agent_control_enabled
            and self.agent.turn_id == turn_id
            and self.agent.state is AgentState.EXECUTE
            and not self.agent.termination_requested
        )

    def _abort(self, result):
        if not self.environment.agent_control_enabled:
            return result
        try:
            self.environment.require_agent_control()
            self.environment.hold_position()
            self.flight.transition_to(FlightState.HOVERING)
        except Exception as exc:
            result["hold_error"] = str(exc)
        return result

    def execute(self, command):
        return execute_synchronously(self.iter_execute(command), self.environment)

    def iter_execute(self, command):
        if self.agent.state is not AgentState.EXECUTE:
            raise ValueError(f"{self.name} requires agent EXECUTE state")
        turn_id = self.agent.turn_id
        started = False
        try:
            parse_no_argument_call(command, self.name)
            if self.flight.state is not self.required_state:
                raise ValueError(
                    f"{self.name} requires UAV {self.required_state.name} state"
                )
            start_time = self.environment.observe()["time"]
            previous_time = start_time
            started = True
            target = self._start()
            while True:
                if not self._still_executing(turn_id):
                    return self._abort({"tool": self.name, "status": "terminated"})
                yield None
                if not self._still_executing(turn_id):
                    return self._abort({"tool": self.name, "status": "terminated"})
                observation = self.environment.observe()
                if (
                    not math.isfinite(observation["time"])
                    or observation["time"] <= previous_time
                ):
                    raise RuntimeError("simulation time did not advance")
                previous_time = observation["time"]
                elapsed = observation["time"] - start_time
                if self.environment.get_position_auto_mode() == POSITION_AUTO_NONE:
                    # Takeoff keeps its altitude target; landing captures the
                    # actual resting position, including elevated platforms.
                    self.environment.hold_position(target, observation["yaw"])
                    self.flight.transition_to(self.completed_state)
                    return {
                        "tool": self.name,
                        "status": "success",
                        "elapsed_seconds": elapsed,
                        "position": observation["position"].tolist(),
                        "yaw": float(observation["yaw"]),
                        "flight_state": self.flight.state.value,
                    }
                if elapsed >= self.config.timeout_seconds:
                    return self._abort(
                        {
                            "tool": self.name,
                            "status": "timeout",
                            "elapsed_seconds": elapsed,
                        }
                    )
        except Exception as exc:
            result = {"tool": self.name, "status": "error", "error": str(exc)}
            return self._abort(result) if started else result
        finally:
            if self._still_executing(turn_id):
                self.agent.complete_execute()
