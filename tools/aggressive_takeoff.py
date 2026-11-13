"""Detach from an inclined surface using a planned, attitude-consistent launch."""
from dataclasses import dataclass
import math

import numpy as np

from agent.state_machine import AgentState
from env.flight_state_machine import FlightState
from env.mmuav_env import POSITION_AUTO_NONE
from env.trajectory_optimizer import InitialThrustConstraint, PolynomialTrajectoryOptimizer
from ._auto_flight import parse_no_argument_call
from .tasks import PlanningRequest, execute_synchronously


@dataclass(frozen=True)
class AggressiveTakeoffConfig:
    duration_seconds: float = 2.0
    specific_thrust_min: float = 8.0
    specific_thrust_max: float = 20.0
    minimum_outward_acceleration: float = 0.5
    minimum_height: float = 0.25
    position_gain: float = 40.0
    velocity_gain: float = 12.0
    attitude_gain: float = 12.0
    hover_seconds: float = 0.5
    settle_timeout: float = 12.0
    position_tolerance: float = 0.08
    velocity_tolerance: float = 0.08
    attitude_tolerance: float = 0.08
    angular_velocity_tolerance: float = 0.15

    def __post_init__(self):
        for name, value in vars(self).items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
        if self.specific_thrust_min >= self.specific_thrust_max:
            raise ValueError("specific thrust bounds must satisfy min < max")


class AggressiveTakeoffTool:
    name = "aggressive_takeoff"

    def __init__(self, environment, agent_state_machine, config=None, optimizer=None):
        self.environment = environment
        self.agent = agent_state_machine
        self.flight = environment.flight_state_machine
        environment.bind_agent_state_machine(self.agent)
        self.config = config or AggressiveTakeoffConfig()
        self.optimizer = optimizer or PolynomialTrajectoryOptimizer()

    def _executing(self, turn):
        return (self.environment.agent_control_enabled and self.agent.turn_id == turn
                and self.agent.state is AgentState.EXECUTE and not self.agent.termination_requested)

    def execute(self, command):
        return execute_synchronously(self.iter_execute(command), self.environment)

    def iter_execute(self, command):
        if self.agent.state is not AgentState.EXECUTE:
            raise ValueError("aggressive_takeoff requires agent EXECUTE state")
        turn = self.agent.turn_id
        started = False
        planning = result = None
        try:
            parse_no_argument_call(command, self.name)
            self.environment.require_agent_control()
            if self.flight.state is not FlightState.ON_GROUND:
                raise ValueError("aggressive_takeoff requires UAV ON_GROUND state")
            if self.environment.get_position_auto_mode() != POSITION_AUTO_NONE:
                raise ValueError("aggressive_takeoff cannot start during automatic takeoff/landing")
            nominal = self.flight.nominal_specific_thrust
            if nominal is None:
                raise ValueError("No nominal thrust recorded; aggressive_takeoff requires a successful perch first")
            current = self.environment.observe()
            rotation = self.environment.takeoff_geometry()
            # Magnitude of Euler pitch, robust at 90 degrees and either sign.
            pitch_deg = math.degrees(math.atan2(abs(rotation[2, 0]),
                                                math.hypot(rotation[0, 0], rotation[1, 0])))
            if not 15. - 1e-6 <= pitch_deg <= 90. + 1e-6:
                raise ValueError("aggressive_takeoff requires absolute pitch in [15,90] degrees; use takeoff on level ground")
            pitch_deg = float(np.clip(pitch_deg, 15., 90.))
            normal = rotation[:, 2].copy()
            if normal[2] < -0.05:
                raise ValueError("body Z must not point downwards")
            height = 1 + math.sqrt((pitch_deg - 15) / 75)
            goal = current["position"] + height * normal
            # Body Y remains well-conditioned at vertical pitch, unlike yaw
            # extracted from the nearly vertical body X axis.
            yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
            geometry = self.environment.perch_geometry()
            lower = max(self.config.specific_thrust_min,
                        -float(np.dot(geometry["gravity"], normal))
                        + self.config.minimum_outward_acceleration)
            upper = min(self.config.specific_thrust_max, geometry["max_thrust"] / geometry["mass"])
            initial_thrust = InitialThrustConstraint(
                normal, geometry["gravity"], lower, upper,
                initial_guess=float(np.clip(nominal, lower, upper)))
            trajectory = yield PlanningRequest(self.optimizer, dict(
                positions=np.vstack([current["position"], goal]),
                yaws=np.array([yaw, yaw]),
                waypoint_times=np.array([0., self.config.duration_seconds]),
                velocities=np.zeros((2, 3)),
                accelerations=np.vstack([np.full(3, np.nan), np.zeros(3)]),
                initial_thrust=initial_thrust,
                jerks=np.zeros((2, 3)),
                yaw_rates=np.zeros(2),
                z_bounds=(self.config.minimum_height, None),
            ))
            if not self._executing(turn):
                result = {"tool": self.name, "status": "terminated"}
                return result
            thrusts = [geometry["mass"] * np.linalg.norm(
                trajectory.evaluate(t).acceleration - geometry["gravity"])
                for t in np.linspace(0, trajectory.total_time, 201)]
            planning = {
                "pitch_deg": pitch_deg, "height_m": height, "body_z_world": normal.tolist(),
                "start_position": current["position"].tolist(), "target_position": goal.tolist(),
                "nominal_specific_thrust_m_s2": nominal,
                "initial_specific_thrust_guess_m_s2": initial_thrust.initial_guess,
                "initial_specific_thrust_m_s2": trajectory.initial_specific_thrust,
                "initial_acceleration_m_s2": trajectory.evaluate(0.).acceleration.tolist(),
                "specific_thrust_bounds": [lower, upper],
                "duration_seconds": trajectory.total_time,
                "minimum_planned_z": trajectory.axis_extrema(2)[1],
                "maximum_planned_thrust_N": float(max(thrusts)),
            }
            if not np.isfinite(thrusts).all() or max(thrusts) > geometry["max_thrust"]:
                raise ValueError("planned thrust exceeds motor limits")
            # The solver runs while ON_GROUND, cups and motors untouched. Only
            # after a feasible plan is ready do we start flight and release cups.
            self.environment.follow_aggressive_takeoff(
                trajectory, gains=(self.config.position_gain, self.config.velocity_gain,
                                   self.config.attitude_gain))
            started = True
            start = previous = self.environment.observe()["time"]
            stable_since = None
            while True:
                if not self._executing(turn):
                    result = {"tool": self.name, "status": "terminated"}
                    return result
                yield None
                if not self._executing(turn):
                    result = {"tool": self.name, "status": "terminated"}
                    return result
                observed = self.environment.observe()
                now = observed["time"]
                if not math.isfinite(now) or now <= previous:
                    raise RuntimeError("simulation time did not advance")
                previous = now
                elapsed = now - start
                error = float(np.linalg.norm(observed["position"] - goal))
                body_z = self.environment.takeoff_geometry()[:, 2]
                attitude_error = math.acos(float(np.clip(body_z[2], -1, 1)))
                yaw_error = abs(math.atan2(math.sin(observed["yaw"] - yaw),
                                           math.cos(observed["yaw"] - yaw)))
                stable = (elapsed >= trajectory.total_time
                          and error <= self.config.position_tolerance
                          and attitude_error <= self.config.attitude_tolerance
                          and yaw_error <= self.config.attitude_tolerance
                          and np.linalg.norm(observed["velocity"]) <= self.config.velocity_tolerance
                          and np.linalg.norm(observed["angular_velocity"]) <= self.config.angular_velocity_tolerance)
                if not stable:
                    stable_since = None
                elif stable_since is None:
                    stable_since = now
                held = 0. if stable_since is None else now - stable_since
                if stable and held >= self.config.hover_seconds:
                    self.environment.hold_position(goal, yaw)
                    self.flight.transition_to(FlightState.HOVERING)
                    result = {"tool": self.name, "status": "success",
                              "elapsed_seconds": elapsed, "hover_seconds": held,
                              "position": observed["position"].tolist(), "position_error": error,
                              "attitude_error_rad": attitude_error, "yaw_error": yaw_error,
                              "flight_state": self.flight.state.value,
                              "sucker_states": list(self.flight.sucker_states)}
                    return result
                if elapsed >= trajectory.total_time + self.config.settle_timeout + self.config.hover_seconds:
                    result = {"tool": self.name, "status": "timeout", "elapsed_seconds": elapsed,
                              "position_error": error, "attitude_error_rad": attitude_error,
                              "error": "Takeoff trajectory ended without stable hover"}
                    return result
        except Exception as exc:
            result = {"tool": self.name, "status": "error", "error": str(exc)}
            return result
        finally:
            if result is not None and planning is not None:
                result["planning"] = planning
            if (started and self.environment.agent_control_enabled
                    and self.agent.turn_id == turn):
                try:
                    if self.flight.state is FlightState.FLY_AGENT:
                        self.environment.hold_position()
                        self.flight.transition_to(FlightState.HOVERING)
                except Exception as exc:
                    if result is not None:
                        result["cleanup_error"] = str(exc)
            if self._executing(turn):
                self.agent.complete_execute()
