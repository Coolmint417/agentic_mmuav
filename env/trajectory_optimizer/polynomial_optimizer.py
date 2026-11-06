from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import casadi as ca
import numpy as np

from .config_trajectory_optimizer import TrajectoryOptimizerConfig


@dataclass(frozen=True)
class TerminalThrustConstraint:
    """Optimize alpha in a(T)=gravity+alpha*normal, with alpha in m/s²."""
    normal: np.ndarray
    gravity: np.ndarray
    lower: float = 8.0
    upper: float = 20.0

    def __post_init__(self):
        for name in ("normal", "gravity"):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite 3-vector")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.isclose(np.linalg.norm(self.normal), 1., atol=1e-6):
            raise ValueError("terminal thrust normal must be a unit vector")
        if (any(isinstance(v, bool) or not isinstance(v, (int, float))
                or not np.isfinite(v) for v in (self.lower, self.upper))
                or not 0 < self.lower < self.upper):
            raise ValueError("specific thrust bounds must satisfy 0 < lower < upper")


@dataclass
class TrajectoryState:
    t: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    snap: np.ndarray
    yaw: float
    yaw_rate: float
    yaw_acceleration: float
    yaw_jerk: float
    yaw_snap: float


class PiecewisePolynomialTrajectory:
    """Piecewise polynomial trajectory with increasing-power coefficients."""

    def __init__(
        self,
        pos_coeffs: np.ndarray,
        yaw_coeffs: np.ndarray,
        segment_durations: np.ndarray,
        config: TrajectoryOptimizerConfig | None = None,
        terminal_specific_thrust: float | None = None,
    ) -> None:
        self.pos_coeffs = np.asarray(pos_coeffs, dtype=np.float64)
        self.yaw_coeffs = np.asarray(yaw_coeffs, dtype=np.float64)
        self.segment_durations = np.asarray(segment_durations, dtype=np.float64)
        self.config = config or TrajectoryOptimizerConfig()
        self.terminal_specific_thrust = terminal_specific_thrust
        self.knot_times = np.concatenate(
            [np.zeros(1), np.cumsum(self.segment_durations)]
        )

        if self.pos_coeffs.ndim != 3 or self.pos_coeffs.shape[2] != 3:
            raise ValueError("pos_coeffs must have shape [S, k_pos + 1, 3]")
        if self.yaw_coeffs.ndim != 2:
            raise ValueError("yaw_coeffs must have shape [S, k_yaw + 1]")
        if self.pos_coeffs.shape[0] != self.segment_durations.shape[0]:
            raise ValueError("position coefficient segment count mismatch")
        if self.yaw_coeffs.shape[0] != self.segment_durations.shape[0]:
            raise ValueError("yaw coefficient segment count mismatch")

    @property
    def total_time(self) -> float:
        return float(self.knot_times[-1])

    def evaluate(self, t: float) -> TrajectoryState:
        segment, tau, query_t = self._locate_segment(t)
        return TrajectoryState(
            t=query_t,
            position=self._eval_pos(segment, tau, derivative=0),
            velocity=self._eval_pos(segment, tau, derivative=1),
            acceleration=self._eval_pos(segment, tau, derivative=2),
            jerk=self._eval_pos(segment, tau, derivative=3),
            snap=self._eval_pos(segment, tau, derivative=4),
            yaw=float(self._eval_yaw(segment, tau, derivative=0)),
            yaw_rate=float(self._eval_yaw(segment, tau, derivative=1)),
            yaw_acceleration=float(self._eval_yaw(segment, tau, derivative=2)),
            yaw_jerk=float(self._eval_yaw(segment, tau, derivative=3)),
            yaw_snap=float(self._eval_yaw(segment, tau, derivative=4)),
        )

    __call__ = evaluate

    def axis_extrema(self, axis: int) -> tuple[float, float, float, float]:
        """Return (min_time, min_value, max_time, max_value), including all stationary points."""
        if type(axis) is not int or not 0 <= axis < 3:
            raise ValueError("axis must be 0, 1 or 2")
        candidates = []
        for segment, duration in enumerate(self.segment_durations):
            # Roots in normalized time avoid poor scaling of long/short segments.
            coefficients = self.pos_coeffs[segment, :, axis] * duration ** np.arange(self.pos_coeffs.shape[1])
            roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coefficients))
            times = [0., 1.] + [float(root.real) for root in roots
                                if abs(root.imag) < 1e-7 and 0 < root.real < 1]
            candidates.extend((float(self.knot_times[segment] + u * duration),
                               float(np.polynomial.polynomial.polyval(u, coefficients))) for u in times)
        minimum = min(candidates, key=lambda pair: pair[1])
        maximum = max(candidates, key=lambda pair: pair[1])
        return *minimum, *maximum

    def _locate_segment(self, t: float) -> tuple[int, float, float]:
        query_t = float(t)
        if self.config.clamp_query_time:
            query_t = float(np.clip(query_t, 0.0, self.total_time))
        elif query_t < 0.0 or query_t > self.total_time:
            raise ValueError(f"t={t} is outside [0, {self.total_time}]")

        if query_t >= self.total_time:
            segment = self.segment_durations.shape[0] - 1
        else:
            segment = int(np.searchsorted(self.knot_times, query_t, side="right") - 1)
            segment = max(0, min(segment, self.segment_durations.shape[0] - 1))
        tau = query_t - float(self.knot_times[segment])
        return segment, tau, query_t

    def _eval_pos(self, segment: int, tau: float, derivative: int) -> np.ndarray:
        coeffs = self.pos_coeffs[segment]
        return np.array(
            [
                _eval_poly_np(coeffs[:, axis], tau, derivative)
                for axis in range(3)
            ],
            dtype=np.float64,
        )

    def _eval_yaw(self, segment: int, tau: float, derivative: int) -> float:
        return _eval_poly_np(self.yaw_coeffs[segment], tau, derivative)


class PolynomialTrajectoryOptimizer:
    def __init__(self, config: TrajectoryOptimizerConfig | None = None) -> None:
        self.config = config or TrajectoryOptimizerConfig()
        self._validate_config()

    def optimize(
        self,
        positions: np.ndarray,
        yaws: np.ndarray,
        segment_durations: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
        accelerations: np.ndarray | None = None,
        yaw_rates: np.ndarray | None = None,
        *,
        waypoint_times: np.ndarray | None = None,
        jerks: np.ndarray | None = None,
        terminal_thrust: TerminalThrustConstraint | None = None,
        z_bounds: tuple[float | None, float | None] | None = None,
    ) -> PiecewisePolynomialTrajectory:
        positions = _as_2d(positions, width=3, name="positions")
        yaws = np.asarray(yaws, dtype=np.float64).reshape(-1)
        if positions.shape[0] != yaws.shape[0]:
            raise ValueError("positions and yaws must have the same knot count")
        if positions.shape[0] < 2:
            raise ValueError("at least two knots are required")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(yaws)):
            raise ValueError("positions and yaws must be finite")
        if waypoint_times is not None:
            if segment_durations is not None:
                raise ValueError("provide waypoint_times or segment_durations, not both")
            times = np.asarray(waypoint_times, dtype=np.float64)
            if times.shape != (positions.shape[0],) or not np.all(np.isfinite(times)):
                raise ValueError("waypoint_times must contain one finite time per knot")
            if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
                raise ValueError("waypoint_times must start at zero and strictly increase")
            segment_durations = np.diff(times)

        segment_durations = self._segment_durations(
            knot_count=positions.shape[0],
            segment_durations=segment_durations,
        )
        velocities, velocity_mask = _optional_masked_2d(
            velocities,
            positions.shape[0],
            3,
            "velocities",
        )
        accelerations, acceleration_mask = _optional_masked_2d(
            accelerations,
            positions.shape[0],
            3,
            "accelerations",
        )
        yaw_rates, yaw_rate_mask = _optional_masked_1d(
            yaw_rates,
            positions.shape[0],
            "yaw_rates",
        )

        jerks, jerk_mask = _optional_masked_2d(jerks, positions.shape[0], 3, "jerks")
        if z_bounds is not None:
            if not isinstance(z_bounds, (list, tuple)) or len(z_bounds) != 2:
                raise ValueError("z_bounds must be (lower, upper), with None for an open bound")
            if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                                     or not np.isfinite(v)) for v in z_bounds):
                raise ValueError("z bounds must be finite numbers or None")
            lower, upper = z_bounds
            if ((lower is None and upper is None)
                    or (lower is not None and upper is not None and lower >= upper)):
                raise ValueError("z_bounds must specify a nonempty interval")
            if ((lower is not None and np.any(positions[:, 2] < lower))
                    or (upper is not None and np.any(positions[:, 2] > upper))):
                raise ValueError("waypoint height lies outside z_bounds")
        if terminal_thrust is not None:
            if not isinstance(terminal_thrust, TerminalThrustConstraint):
                raise TypeError("terminal_thrust must be TerminalThrustConstraint")
            if acceleration_mask[-1]:
                raise ValueError("terminal thrust cannot also fix terminal acceleration")
        opti = ca.Opti()
        pos_vars = [
            opti.variable(segment_durations.shape[0], self.config.k_pos + 1)
            for _ in range(3)
        ]
        yaw_var = opti.variable(segment_durations.shape[0], self.config.k_yaw + 1)

        objective = 0
        for axis in range(3):
            objective += self.config.snap_weight_pos * _derivative_integral_ca(
                pos_vars[axis],
                segment_durations,
                self.config.k_pos,
                self.config.objective_derivative_order_pos,
            )
        objective += self.config.snap_weight_yaw * _derivative_integral_ca(
            yaw_var,
            segment_durations,
            self.config.k_yaw,
            self.config.objective_derivative_order_yaw,
        )
        opti.minimize(objective)

        self._add_position_constraints(
            opti,
            pos_vars,
            positions,
            segment_durations,
            velocities,
            velocity_mask,
            accelerations,
            acceleration_mask,
            jerks,
            jerk_mask,
        )
        self._add_yaw_constraints(
            opti,
            yaw_var,
            yaws,
            segment_durations,
            yaw_rates,
            yaw_rate_mask,
        )
        specific_thrust = None
        if terminal_thrust is not None:
            specific_thrust = opti.variable()
            opti.subject_to(opti.bounded(terminal_thrust.lower, specific_thrust, terminal_thrust.upper))
            opti.set_initial(specific_thrust, (terminal_thrust.lower + terminal_thrust.upper) / 2)
            last = len(segment_durations) - 1
            for axis in range(3):
                opti.subject_to(
                    _poly_ca(pos_vars[axis], last, float(segment_durations[-1]), 2)
                    == terminal_thrust.gravity[axis] + specific_thrust * terminal_thrust.normal[axis])
        if z_bounds is not None:
            for segment, duration in enumerate(segment_durations):
                for time in np.linspace(0., float(duration), 101):
                    height = _poly_ca(pos_vars[2], segment, float(time), 0)
                    if z_bounds[0] is not None:
                        opti.subject_to(height >= z_bounds[0])
                    if z_bounds[1] is not None:
                        opti.subject_to(height <= z_bounds[1])
        self._set_initial_guess(
            opti,
            pos_vars,
            yaw_var,
            positions,
            yaws,
            segment_durations,
        )

        opti.solver(
            "ipopt",
            {"expand": True, "print_time": self.config.print_time},
            {
                "print_level": self.config.ipopt_print_level,
                "max_iter": self.config.max_iter,
            },
        )
        for refinement in range(8):
            solution = opti.solve()
            pos_coeffs = np.zeros(
                (segment_durations.shape[0], self.config.k_pos + 1, 3),
                dtype=np.float64,
            )
            for axis in range(3):
                pos_coeffs[:, :, axis] = solution.value(pos_vars[axis])
            yaw_coeffs = np.asarray(solution.value(yaw_var)).reshape(
                segment_durations.shape[0], self.config.k_yaw + 1
            )
            trajectory = PiecewisePolynomialTrajectory(
                pos_coeffs=pos_coeffs,
                yaw_coeffs=yaw_coeffs,
                segment_durations=segment_durations,
                config=self.config,
                terminal_specific_thrust=(float(solution.value(specific_thrust))
                                          if specific_thrust is not None else None),
            )

            if z_bounds is None:
                return trajectory
            min_time, minimum, max_time, maximum = trajectory.axis_extrema(2)
            violations = []
            if z_bounds[0] is not None and minimum < z_bounds[0] - 1e-6:
                violations.append((min_time, z_bounds[0], True))
            if z_bounds[1] is not None and maximum > z_bounds[1] + 1e-6:
                violations.append((max_time, z_bounds[1], False))
            if not violations:
                return trajectory
            # Add constraints at the actual extrema between collocation points.
            # Re-solve until the entire polynomial passes the extrema check.
            for time, bound, is_lower in violations:
                segment, tau, _ = trajectory._locate_segment(time)
                height = _poly_ca(pos_vars[2], segment, tau, 0)
                opti.subject_to(height >= bound if is_lower else height <= bound)
            opti.set_initial(opti.x, solution.value(opti.x))
        raise RuntimeError("trajectory failed continuous z bound verification")

    def _segment_durations(
        self,
        knot_count: int,
        segment_durations: np.ndarray | None,
    ) -> np.ndarray:
        if segment_durations is None:
            durations = np.full(knot_count - 1, self.config.waypoint_dt, dtype=np.float64)
        else:
            durations = np.asarray(segment_durations, dtype=np.float64).reshape(-1)
        if durations.shape[0] != knot_count - 1:
            raise ValueError("segment_durations must have length knot_count - 1")
        if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
            raise ValueError("all segment durations must be positive")
        return durations

    def _add_position_constraints(
        self,
        opti: ca.Opti,
        pos_vars: list[ca.MX],
        positions: np.ndarray,
        durations: np.ndarray,
        velocities: np.ndarray | None,
        velocity_mask: np.ndarray,
        accelerations: np.ndarray | None,
        acceleration_mask: np.ndarray,
        jerks: np.ndarray | None,
        jerk_mask: np.ndarray,
    ) -> None:
        num_segments = durations.shape[0]
        for axis, coeff in enumerate(pos_vars):
            for segment in range(num_segments):
                duration = float(durations[segment])
                opti.subject_to(_poly_ca(coeff, segment, 0.0, 0) == positions[segment, axis])
                opti.subject_to(
                    _poly_ca(coeff, segment, duration, 0)
                    == positions[segment + 1, axis]
                )
                if velocity_mask[segment]:
                    opti.subject_to(
                        _poly_ca(coeff, segment, 0.0, 1) == velocities[segment, axis]
                    )
                if velocity_mask[segment + 1]:
                    opti.subject_to(
                        _poly_ca(coeff, segment, duration, 1)
                        == velocities[segment + 1, axis]
                    )
                if acceleration_mask[segment]:
                    opti.subject_to(
                        _poly_ca(coeff, segment, 0.0, 2)
                        == accelerations[segment, axis]
                    )
                if acceleration_mask[segment + 1]:
                    opti.subject_to(
                        _poly_ca(coeff, segment, duration, 2)
                        == accelerations[segment + 1, axis]
                    )

                if jerk_mask[segment]:
                    opti.subject_to(_poly_ca(coeff, segment, 0.0, 3) == jerks[segment, axis])
                if jerk_mask[segment + 1]:
                    opti.subject_to(_poly_ca(coeff, segment, duration, 3) == jerks[segment + 1, axis])

            for knot in range(1, num_segments):
                left_segment = knot - 1
                right_segment = knot
                left_duration = float(durations[left_segment])
                continuity_orders = set(
                    range(3, self.config.continuity_order_pos + 1)
                )
                if jerk_mask[knot]:
                    continuity_orders.discard(3)
                if self.config.continuity_order_pos >= 1 and not velocity_mask[knot]:
                    continuity_orders.add(1)
                if self.config.continuity_order_pos >= 2 and not acceleration_mask[knot]:
                    continuity_orders.add(2)
                for derivative in sorted(continuity_orders):
                    opti.subject_to(
                        _poly_ca(coeff, left_segment, left_duration, derivative)
                        == _poly_ca(coeff, right_segment, 0.0, derivative)
                    )

    def _add_yaw_constraints(
        self,
        opti: ca.Opti,
        yaw_var: ca.MX,
        yaws: np.ndarray,
        durations: np.ndarray,
        yaw_rates: np.ndarray | None,
        yaw_rate_mask: np.ndarray,
    ) -> None:
        num_segments = durations.shape[0]
        for segment in range(num_segments):
            duration = float(durations[segment])
            opti.subject_to(_poly_ca(yaw_var, segment, 0.0, 0) == yaws[segment])
            opti.subject_to(
                _poly_ca(yaw_var, segment, duration, 0) == yaws[segment + 1]
            )
            if yaw_rate_mask[segment]:
                opti.subject_to(
                    _poly_ca(yaw_var, segment, 0.0, 1) == yaw_rates[segment]
                )
            if yaw_rate_mask[segment + 1]:
                opti.subject_to(
                    _poly_ca(yaw_var, segment, duration, 1) == yaw_rates[segment + 1]
                )

        for knot in range(1, num_segments):
            left_segment = knot - 1
            right_segment = knot
            left_duration = float(durations[left_segment])
            continuity_orders = set(range(2, self.config.continuity_order_yaw + 1))
            if self.config.continuity_order_yaw >= 1 and not yaw_rate_mask[knot]:
                continuity_orders.add(1)
            for derivative in sorted(continuity_orders):
                opti.subject_to(
                    _poly_ca(yaw_var, left_segment, left_duration, derivative)
                    == _poly_ca(yaw_var, right_segment, 0.0, derivative)
                )

    def _set_initial_guess(
        self,
        opti: ca.Opti,
        pos_vars: list[ca.MX],
        yaw_var: ca.MX,
        positions: np.ndarray,
        yaws: np.ndarray,
        durations: np.ndarray,
    ) -> None:
        for segment, duration in enumerate(durations):
            for axis, coeff in enumerate(pos_vars):
                guess = np.zeros(self.config.k_pos + 1)
                guess[0] = positions[segment, axis]
                guess[1] = (
                    positions[segment + 1, axis] - positions[segment, axis]
                ) / duration
                opti.set_initial(coeff[segment, :], guess)

            yaw_guess = np.zeros(self.config.k_yaw + 1)
            yaw_guess[0] = yaws[segment]
            yaw_guess[1] = (yaws[segment + 1] - yaws[segment]) / duration
            opti.set_initial(yaw_var[segment, :], yaw_guess)

    def _validate_config(self) -> None:
        if self.config.continuity_order_pos < 0:
            raise ValueError("continuity_order_pos must be non-negative")
        if self.config.continuity_order_yaw < 0:
            raise ValueError("continuity_order_yaw must be non-negative")
        if self.config.objective_derivative_order_pos < 0:
            raise ValueError("objective_derivative_order_pos must be non-negative")
        if self.config.objective_derivative_order_yaw < 0:
            raise ValueError("objective_derivative_order_yaw must be non-negative")
        if self.config.continuity_order_pos > self.config.k_pos:
            raise ValueError("continuity_order_pos cannot exceed k_pos")
        if self.config.continuity_order_yaw > self.config.k_yaw:
            raise ValueError("continuity_order_yaw cannot exceed k_yaw")
        if self.config.objective_derivative_order_pos > self.config.k_pos:
            raise ValueError("objective_derivative_order_pos cannot exceed k_pos")
        if self.config.objective_derivative_order_yaw > self.config.k_yaw:
            raise ValueError("objective_derivative_order_yaw cannot exceed k_yaw")


def _as_2d(value: np.ndarray, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape [N, {width}]")
    return array


def _optional_masked_2d(
    value: object | None,
    length: int,
    width: int,
    name: str,
) -> tuple[np.ndarray | None, np.ndarray]:
    mask = np.zeros(length, dtype=bool)
    if value is None:
        return None, mask

    object_array = np.asarray(value, dtype=object)
    if object_array.ndim == 2 and object_array.shape == (length, width):
        array = np.asarray(value, dtype=np.float64)
        row_is_nan = np.any(np.isnan(array), axis=1)
        mask = ~row_is_nan
        array[row_is_nan] = 0.0
        return array, mask

    if object_array.ndim != 1 or object_array.shape[0] != length:
        raise ValueError(f"{name} must have length {length}")

    array = np.zeros((length, width), dtype=np.float64)
    for index, item in enumerate(object_array):
        if _is_missing(item):
            continue
        row = np.asarray(item, dtype=np.float64).reshape(-1)
        if row.shape[0] != width:
            raise ValueError(f"{name}[{index}] must have length {width} or be None")
        if np.any(np.isnan(row)):
            continue
        array[index] = row
        mask[index] = True
    return array, mask


def _optional_masked_1d(
    value: object | None,
    length: int,
    name: str,
) -> tuple[np.ndarray | None, np.ndarray]:
    mask = np.zeros(length, dtype=bool)
    if value is None:
        return None, mask

    object_array = np.asarray(value, dtype=object).reshape(-1)
    if object_array.shape[0] != length:
        raise ValueError(f"{name} must have length {length}")

    array = np.zeros(length, dtype=np.float64)
    for index, item in enumerate(object_array):
        if _is_missing(item):
            continue
        scalar = float(item)
        if np.isnan(scalar):
            continue
        array[index] = scalar
        mask[index] = True
    return array, mask


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _poly_ca(coeff: ca.MX, segment: int, tau: float, derivative: int) -> ca.MX:
    degree = coeff.shape[1] - 1
    expr = 0
    for power in range(derivative, degree + 1):
        scale = factorial(power) / factorial(power - derivative)
        expr += coeff[segment, power] * scale * (tau ** (power - derivative))
    return expr


def _derivative_integral_ca(
    coeff: ca.MX,
    durations: np.ndarray,
    degree: int,
    derivative_order: int,
) -> ca.MX:
    derivative_order = int(derivative_order)
    if derivative_order < 0:
        raise ValueError("derivative_order must be non-negative")
    if derivative_order > degree:
        return 0

    objective = 0
    for segment, duration in enumerate(durations):
        for i in range(derivative_order, degree + 1):
            scale_i = factorial(i) / factorial(i - derivative_order)
            for j in range(derivative_order, degree + 1):
                scale_j = factorial(j) / factorial(j - derivative_order)
                power = i + j - 2 * derivative_order + 1
                objective += (
                    coeff[segment, i]
                    * coeff[segment, j]
                    * scale_i
                    * scale_j
                    * (float(duration) ** power)
                    / power
                )
    return objective


def _eval_poly_np(coeffs: np.ndarray, tau: float, derivative: int) -> float:
    degree = coeffs.shape[0] - 1
    value = 0.0
    for power in range(derivative, degree + 1):
        scale = factorial(power) / factorial(power - derivative)
        value += coeffs[power] * scale * (tau ** (power - derivative))
    return float(value)
