"""Minimum-snap surface approach with terminal attitude/velocity constraints."""
from dataclasses import dataclass
import json
import math
import re

import numpy as np

from agent.state_machine import AgentState
from env.flight_state_machine import FlightState
from env.trajectory_optimizer import PolynomialTrajectoryOptimizer, TerminalThrustConstraint
from .tasks import PlanningRequest, execute_synchronously


@dataclass(frozen=True)
class AggressivePerchConfig:
    specific_thrust_min: float = 8.0
    specific_thrust_max: float = 20.0  # optimized f/m, m/s²
    approach_speed: float = 0.4   # toward surface, within paper's 0.2--0.8 m/s
    minimum_approach_speed: float = 1.0  # average normal clearance / duration
    minimum_height: float = 0.25  # world Z, body origin
    contact_preload: float = 0.02
    position_gain: float = 40.0
    velocity_gain: float = 12.0
    attitude_gain: float = 12.0
    contact_timeout: float = 1.0
    attach_seconds: float = 0.5
    alignment_tolerance: float = math.radians(15)  # early motor idle only, not success
    angular_velocity_tolerance: float = 0.5

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.specific_thrust_min >= self.specific_thrust_max:
            raise ValueError("specific_thrust_min must be below specific_thrust_max")
        if not .2 <= self.approach_speed <= .8:
            raise ValueError("approach_speed must be in [0.2, 0.8] m/s")


def parse_aggressive_perch(command):
    if isinstance(command, str):
        text = command.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        command = json.loads(fenced.group(1) if fenced else text)
    if (not isinstance(command, dict)
            or set(command) != {"tool", "position", "yaw", "wall_angle_deg", "time"}
            or command["tool"] != "aggressive_perch"):
        raise ValueError("aggressive_perch requires exactly tool, position, yaw, wall_angle_deg and time")

    def number(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        return float(value)

    position = command["position"]
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError("position must be one world [x,y,z] surface contact point")
    position = np.array([number(v, "position coordinate") for v in position])
    yaw = number(command["yaw"], "yaw")
    angle = number(command["wall_angle_deg"], "wall_angle_deg")
    duration = number(command["time"], "time")
    if not 30 <= angle <= 90 or not 0 < duration <= 2.0:
        raise ValueError("wall_angle_deg must be in [30,90] degrees and time must be in (0,2] seconds")
    return position, yaw, angle, duration


def perch_frame(yaw, angle_deg):
    """Outward normal = Rz(yaw) Ry(-angle) e3; feet face opposite body Z."""
    c, s = math.cos(yaw), math.sin(yaw)
    a = math.radians(angle_deg)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]) @ np.array([
        [math.cos(a), 0, -math.sin(a)], [0, 1., 0], [math.sin(a), 0, math.cos(a)]])
    return rotation, rotation[:, 2].copy()


class AggressivePerchTool:
    name = "aggressive_perch"

    def __init__(self, environment, agent_state_machine, config=None, optimizer=None):
        self.environment = environment
        self.agent = agent_state_machine
        self.flight = environment.flight_state_machine
        environment.bind_agent_state_machine(self.agent)
        self.config = config or AggressivePerchConfig()
        self.optimizer = optimizer or PolynomialTrajectoryOptimizer()

    def _executing(self, turn_id):
        return (self.environment.agent_control_enabled and self.agent.turn_id == turn_id
                and self.agent.state is AgentState.EXECUTE and not self.agent.termination_requested)

    def execute(self, command):
        return execute_synchronously(self.iter_execute(command), self.environment)

    def iter_execute(self, command):
        if self.agent.state is not AgentState.EXECUTE:
            raise ValueError("aggressive_perch requires agent EXECUTE state")
        turn = self.agent.turn_id
        prepared = completed = False
        result = None
        planning = None
        try:
            point, yaw, angle, duration = parse_aggressive_perch(command)
            self.environment.require_agent_control()
            if self.flight.state is not FlightState.HOVERING:
                raise ValueError("aggressive_perch requires UAV HOVERING state")
            self.environment.prepare_perch()  # All cups enabled BEFORE optimization.
            prepared = True
            current = self.environment.observe()
            geometry = self.environment.perch_geometry()
            rotation, normal = perch_frame(yaw, angle)
            goal = point - rotation @ geometry["foot_offset"] - self.config.contact_preload * normal
            clearance = float(np.dot(current["position"] - point, normal))
            if clearance <= 0:
                raise ValueError("UAV must start on the outward-normal side of the target plane; check yaw/angle")
            if clearance / duration < self.config.minimum_approach_speed:
                raise ValueError(f"average normal approach speed must be >= {self.config.minimum_approach_speed:g} m/s; reduce time or adjust start position")
            thrust_constraint = TerminalThrustConstraint(
                normal, geometry["gravity"], self.config.specific_thrust_min,
                min(self.config.specific_thrust_max, geometry["max_thrust"] / geometry["mass"]))
            terminal_velocity = -self.config.approach_speed * normal
            trajectory = yield PlanningRequest(self.optimizer, dict(
                positions=np.vstack([current["position"], goal]),
                yaws=np.unwrap([current["yaw"], yaw]),
                waypoint_times=np.array([0., duration]),
                velocities=np.vstack([current["velocity"], terminal_velocity]),
                accelerations=np.vstack([np.zeros(3), np.full(3, np.nan)]),
                terminal_thrust=thrust_constraint,
                z_bounds=(self.config.minimum_height, None),
                jerks=np.zeros((2, 3)),
                yaw_rates=np.array([0., 0.]),
            ))
            if not self._executing(turn):
                result = {"tool": self.name, "status": "terminated"}
                return result
            terminal_acceleration = trajectory.evaluate(duration).acceleration
            optimized_specific_thrust = trajectory.terminal_specific_thrust
            thrusts = [geometry["mass"] * np.linalg.norm(
                trajectory.evaluate(t).acceleration - geometry["gravity"])
                for t in np.linspace(0, duration, 201)]
            planning = {
                "specific_thrust_m_s2": optimized_specific_thrust,
                "specific_thrust_bounds": [thrust_constraint.lower, thrust_constraint.upper],
                "minimum_planned_z": trajectory.axis_extrema(2)[1],
                "minimum_allowed_z": self.config.minimum_height,
                "maximum_planned_thrust_N": float(max(thrusts)),
                "average_normal_approach_speed_m_s": float(np.dot(current["position"] - point, normal) / duration),
                "duration_seconds": duration,
            }
            if not np.isfinite(thrusts).all() or max(thrusts) > geometry["max_thrust"]:
                raise ValueError("planned thrust exceeds motor limits; adjust start position or perch time")
            self.environment.follow_perch_trajectory(
                trajectory, gains=(self.config.position_gain, self.config.velocity_gain,
                                   self.config.attitude_gain))
            self.flight.transition_to(FlightState.FLY_AGENT)
            start = previous = self.environment.observe()["time"]
            stable_since = None
            latched = False
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
                contact = self.environment.perch_contact_status()
                alignment = math.acos(float(np.clip(np.dot(contact["body_z"], normal), -1, 1)))
                error = float(np.linalg.norm(observed["position"] - goal))
                four_cup_contact = contact["contact_count"] == 4
                angular_speed = float(np.linalg.norm(observed["angular_velocity"]))
                # Preserve the early two-cup motor shutdown that lets the
                # remaining feet settle onto the surface. No position gate.
                if not latched and (four_cup_contact or (
                        contact["contact_count"] >= 2
                        and alignment <= self.config.alignment_tolerance)):
                    self.environment.idle_perch_motors()
                    latched = True
                stable = (latched and four_cup_contact
                          and angular_speed <= self.config.angular_velocity_tolerance)
                if not stable:
                    stable_since = None
                elif stable_since is None:
                    stable_since = now
                if stable_since is not None and now - stable_since >= self.config.attach_seconds:
                    self.environment.complete_perch()
                    self.flight.set_nominal_specific_thrust(optimized_specific_thrust)
                    completed = True
                    result = {"tool": self.name, "status": "success", "flight_state": self.flight.state.value,
                              "sucker_states": list(self.flight.sucker_states),
                              "position": observed["position"].tolist(), "target_surface_point": point.tolist(),
                              "body_target": goal.tolist(), "normal": normal.tolist(),
                              "terminal_velocity": terminal_velocity.tolist(),
                              "terminal_acceleration": terminal_acceleration.tolist(),
                              "specific_thrust": optimized_specific_thrust,
                              "stable_seconds": now - stable_since,
                              "angular_speed_rad_s": angular_speed,
                              "elapsed_seconds": elapsed, "alignment_error_rad": alignment,
                              "contact_cups": contact["contact_cups"], "position_error": error}
                    return result
                # Reserve the full confirmation interval in addition to the
                # approach/contact wait; increasing attach_seconds must not
                # consume the time available for the cups to settle.
                if elapsed >= duration + self.config.contact_timeout + self.config.attach_seconds:
                    result = {"tool": self.name, "status": "timeout", "elapsed_seconds": elapsed,
                              "contact_count": contact["contact_count"], "alignment_error_rad": alignment,
                              "stable_seconds": 0. if stable_since is None else now - stable_since,
                              "angular_speed_rad_s": angular_speed,
                              "position_error": error, "error": "No stable four-cup adhesion confirmed"}
                    return result
        except Exception as exc:
            result = {"tool": self.name, "status": "error", "error": str(exc)}
            return result
        finally:
            if result is not None and planning is not None:
                result["planning"] = planning
            if prepared and not completed and self.environment.agent_control_enabled and self.agent.turn_id == turn:
                try:
                    self.environment.hold_position()  # Also releases perching adhesion.
                    if self.flight.state is FlightState.FLY_AGENT:
                        self.flight.transition_to(FlightState.HOVERING)
                except Exception as exc:
                    if result is not None:
                        result["cleanup_error"] = str(exc)
            if self._executing(turn):
                self.agent.complete_execute()
