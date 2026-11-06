"""Optimized terminal thrust and bounded-height minimum-snap trajectories."""
import unittest

import numpy as np

from env.trajectory_optimizer import (
    PiecewisePolynomialTrajectory, PolynomialTrajectoryOptimizer, TerminalThrustConstraint,
)


def arguments(start_x=-1.7, duration=1.5):
    return dict(positions=np.array([[start_x, 1., 1.], [.36894445, 1., .98919003]]),
                yaws=np.zeros(2), waypoint_times=np.array([0., duration]),
                velocities=np.array([[0., 0., 0.], [.4, 0., 0.]]),
                accelerations=np.array([[0., 0., 0.], [np.nan] * 3]),
                jerks=np.zeros((2, 3)), yaw_rates=np.zeros(2))


def constraint():
    return TerminalThrustConstraint([-1., 0., 0.], [0., 0., -9.8])


def snap_cost(trajectory):
    cost = 0.
    for segment, duration in enumerate(trajectory.segment_durations):
        for axis in range(3):
            derivative = np.polynomial.polynomial.polyder(trajectory.pos_coeffs[segment, :, axis], 4)
            integral = np.polynomial.polynomial.polyint(np.polynomial.polynomial.polymul(derivative, derivative))
            cost += np.polynomial.polynomial.polyval(duration, integral)
    return cost


class PerchOptimizerTests(unittest.TestCase):
    def test_specific_thrust_is_solved_jointly_and_improves_snap_cost(self):
        optimizer = PolynomialTrajectoryOptimizer()
        values = []
        for start in (-1.7, -2.4):
            params = arguments(start)
            variable = optimizer.optimize(**params, terminal_thrust=constraint())
            value = variable.terminal_specific_thrust
            self.assertGreaterEqual(value, 8. - 1e-6)
            self.assertLessEqual(value, 20. + 1e-6)
            np.testing.assert_allclose(variable.evaluate(1.5).acceleration,
                                       [-value, 0., -9.8], atol=1e-6)
            params = arguments(start)
            params["accelerations"][-1] = [-12., 0., -9.8]
            fixed = optimizer.optimize(**params)
            self.assertLess(snap_cost(variable), snap_cost(fixed))
            self.assertIsNone(fixed.terminal_specific_thrust)
            values.append(value)
        self.assertGreater(abs(values[0] - values[1]), 1.)

    def test_two_seconds_still_requires_z_bounds(self):
        optimizer = PolynomialTrajectoryOptimizer()
        unbounded = optimizer.optimize(**arguments(duration=2.), terminal_thrust=constraint())
        self.assertLess(unbounded.axis_extrema(2)[1], 0.)
        bounded = optimizer.optimize(**arguments(duration=2.), terminal_thrust=constraint(),
                                      z_bounds=(.25, 1.05))
        self.assertGreaterEqual(bounded.axis_extrema(2)[1], .25 - 1e-6)
        self.assertLessEqual(bounded.axis_extrema(2)[3], 1.05 + 1e-6)
        final = bounded.evaluate(2.)
        np.testing.assert_allclose(final.position, arguments()["positions"][-1], atol=1e-6)
        np.testing.assert_allclose(final.velocity, [.4, 0., 0.], atol=1e-6)
        np.testing.assert_allclose(final.acceleration, [-bounded.terminal_specific_thrust, 0, -9.8], atol=1e-6)
        np.testing.assert_allclose(final.jerk, 0., atol=1e-6)

    def test_extrema_include_stationary_points_between_endpoints(self):
        coefficients = np.zeros((1, 3, 3))
        coefficients[0, :, 2] = [1., -4., 4.]
        trajectory = PiecewisePolynomialTrajectory(coefficients, np.zeros((1, 1)), [1.])
        time, minimum, _, maximum = trajectory.axis_extrema(2)
        self.assertAlmostEqual(time, .5)
        self.assertAlmostEqual(minimum, 0.)
        self.assertAlmostEqual(maximum, 1.)

    def test_invalid_and_conflicting_constraints_are_rejected(self):
        for params in (([0, 0, 0], [0, 0, -9.8], 6., 20.),
                       ([-1, 0, 0], [0, 0, -9.8], 20., 6.),
                       ([-1, 0, 0], [0, 0, -9.8], 0., 20.)):
            with self.assertRaises(ValueError):
                TerminalThrustConstraint(*params)
        optimizer = PolynomialTrajectoryOptimizer()
        params = arguments()
        params["accelerations"][-1] = [0, 0, 0]
        with self.assertRaises(ValueError):
            optimizer.optimize(**params, terminal_thrust=constraint())
        for bounds in ((None, None), (1, 0), (.999, 2), (float("nan"), None)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                optimizer.optimize(**arguments(), z_bounds=bounds)


if __name__ == "__main__":
    unittest.main()
