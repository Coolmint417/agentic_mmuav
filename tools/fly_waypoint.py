"""Execute model-specified waypoints through the MuJoCo SE3 control loop."""
from dataclasses import dataclass
import json
import math
import re

import numpy as np

from agent.state_machine import AgentState
from .tasks import PlanningRequest, execute_synchronously
from env.flight_state_machine import FlightState
from env.trajectory_optimizer import PolynomialTrajectoryOptimizer


@dataclass(frozen=True)
class FlyWaypointConfig:
    hover_seconds: float = 1.0
    position_tolerance: float = 0.08
    velocity_tolerance: float = 0.08
    yaw_tolerance: float = 0.08
    angular_velocity_tolerance: float = 0.15
    settle_timeout: float = 15.0

    def __post_init__(self):
        for name, value in vars(self).items():
            _number(value, name)
            if value < 0 or (name != "hover_seconds" and value == 0):
                raise ValueError(
                    f"{name} must be {'nonnegative' if name == 'hover_seconds' else 'positive'}"
                )


def _number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def parse_fly_waypoint(command):
    """Require {tool: fly_waypoint, waypoint: [{time, position, yaw}, ...]}."""
    if isinstance(command, str):
        payload = command.strip()
        match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", payload, flags=re.DOTALL)
        command = json.loads(match.group(1) if match else payload)
    if not isinstance(command, dict) or set(command) != {"tool", "waypoint"}:
        raise ValueError("command must contain exactly tool and waypoint")
    if command["tool"] != "fly_waypoint":
        raise ValueError("tool must be fly_waypoint")
    waypoints = command["waypoint"]
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError("waypoint must be an explicit nonempty array")
    times, positions, yaws = [], [], []
    previous_time = 0.0
    for index, point in enumerate(waypoints):
        if not isinstance(point, dict) or set(point) != {"time", "position", "yaw"}:
            raise ValueError(
                f"waypoint[{index}] must contain exactly time, position and yaw"
            )
        time = _number(point["time"], f"waypoint[{index}].time")
        if time <= previous_time:
            raise ValueError("waypoint times must be positive and strictly increasing")
        position = point["position"]
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("each position must be an [x, y, z] array")
        positions.append([_number(v, "position coordinate") for v in position])
        yaws.append(_number(point["yaw"], "yaw"))
        times.append(time)
        previous_time = time
    return np.asarray(times), np.asarray(positions), np.asarray(yaws)


class FlyWaypointTool:
    """Cooperative flight task with a synchronous headless compatibility driver.

    The caller must already have moved the agent from THINK to EXECUTE.
    Only the last waypoint requires a continuous hover interval. All timing
    uses simulation seconds. EnvironmentRuntime drives iter_execute() and
    continuously steps, including while a planning request is in progress.
    execute() is reserved for standalone/headless callers without a runtime.
    """

    name = "fly_waypoint"

    def __init__(self, environment, agent_state_machine, config=None, optimizer=None):
        self.environment = environment
        self.agent = agent_state_machine
        self.flight = environment.flight_state_machine
        self.environment.bind_agent_state_machine(self.agent)
        self.config = config if config is not None else FlyWaypointConfig()
        self.optimizer = (
            optimizer if optimizer is not None else PolynomialTrajectoryOptimizer()
        )

    def _still_executing(self, turn_id):
        return (
            self.environment.agent_control_enabled
            and self.agent.turn_id == turn_id
            and self.agent.state is AgentState.EXECUTE
            and not self.agent.termination_requested
        )

    def execute(self, command):
        return execute_synchronously(self.iter_execute(command), self.environment)

    def iter_execute(self, command):
        if self.agent.state is not AgentState.EXECUTE:
            raise ValueError("fly_waypoint requires agent EXECUTE state")
        turn_id = self.agent.turn_id
        started_flying = False
        hover_command_installed = False
        result = None
        try:
            self.environment.require_agent_control()
            if self.flight.state is not FlightState.HOVERING:
                raise ValueError("fly_waypoint requires UAV HOVERING state")
            times, positions, yaws = parse_fly_waypoint(command)
            observation = self.environment.observe()
            # Include the measured starting knot at t=0. Internal derivatives
            # remain free; the final knot has zero velocity, acceleration/yaw rate.
            knot_count = len(times) + 1
            velocities = np.full((knot_count, 3), np.nan)
            velocities[0] = observation["velocity"]
            velocities[-1] = 0.0
            accelerations = np.full((knot_count, 3), np.nan)
            accelerations[[0, -1]] = 0.0
            yaw_rates = np.full(knot_count, np.nan)
            yaw_rates[0] = observation["angular_velocity"][2]
            yaw_rates[-1] = 0.0
            trajectory = yield PlanningRequest(
                self.optimizer,
                dict(
                    positions=np.vstack([observation["position"], positions]),
                    yaws=np.unwrap(np.r_[observation["yaw"], yaws]),
                    waypoint_times=np.r_[0.0, times],
                    velocities=velocities,
                    accelerations=accelerations,
                    yaw_rates=yaw_rates,
                ),
            )
            if not self._still_executing(turn_id):
                return {"tool": self.name, "status": "terminated"}
            final = trajectory.evaluate(trajectory.total_time)
            start_time = self.environment.observe()["time"]
            self.environment.follow_trajectory(trajectory)
            self.flight.transition_to(FlightState.FLY_AGENT)
            started_flying = True
            stable_since = None
            previous_time = start_time
            deadline = (
                trajectory.total_time
                + self.config.settle_timeout
                + self.config.hover_seconds
            )
            while True:
                if not self._still_executing(turn_id):
                    if self.environment.agent_control_enabled:
                        self.environment.hold_position()
                        hover_command_installed = True
                    result = {"tool": self.name, "status": "terminated"}
                    break
                yield None
                if not self._still_executing(turn_id):
                    if self.environment.agent_control_enabled:
                        self.environment.hold_position()
                        hover_command_installed = True
                    result = {"tool": self.name, "status": "terminated"}
                    break
                current = self.environment.observe()
                if current["time"] <= previous_time:
                    raise RuntimeError("simulation time did not advance")
                previous_time = current["time"]
                elapsed = current["time"] - start_time
                position_error = float(
                    np.linalg.norm(current["position"] - final.position)
                )
                yaw_error = abs(
                    math.atan2(
                        math.sin(current["yaw"] - final.yaw),
                        math.cos(current["yaw"] - final.yaw),
                    )
                )
                stable = (
                    elapsed >= trajectory.total_time
                    and position_error <= self.config.position_tolerance
                    and np.linalg.norm(current["velocity"])
                    <= self.config.velocity_tolerance
                    and yaw_error <= self.config.yaw_tolerance
                    and np.linalg.norm(current["angular_velocity"])
                    <= self.config.angular_velocity_tolerance
                )
                if not stable:
                    stable_since = None
                elif stable_since is None:
                    stable_since = current["time"]
                held = 0.0 if stable_since is None else current["time"] - stable_since
                if stable and held + 1e-9 >= self.config.hover_seconds:
                    self.environment.hold_position(final.position, final.yaw)
                    hover_command_installed = True
                    result = {
                        "tool": self.name,
                        "status": "success",
                        "elapsed_seconds": elapsed,
                        "hover_seconds": held,
                        "position_error": position_error,
                        "yaw_error": yaw_error,
                        "position": current["position"].tolist(),
                        "yaw": float(current["yaw"]),
                        "waypoint_count": len(times),
                    }
                    break
                if elapsed >= deadline:
                    self.environment.hold_position()
                    hover_command_installed = True
                    result = {
                        "tool": self.name,
                        "status": "timeout",
                        "elapsed_seconds": elapsed,
                        "position_error": position_error,
                        "yaw_error": yaw_error,
                    }
                    break
        except Exception as exc:
            result = {"tool": self.name, "status": "error", "error": str(exc)}
            if started_flying and self.environment.agent_control_enabled:
                try:
                    self.environment.hold_position()
                    hover_command_installed = True
                except Exception as hold_error:
                    result["hold_error"] = str(hold_error)
        finally:
            if (
                started_flying
                and hover_command_installed
                and self.flight.state is FlightState.FLY_AGENT
            ):
                self.flight.transition_to(FlightState.HOVERING)
            if self._still_executing(turn_id):
                self.agent.complete_execute()
        return result
