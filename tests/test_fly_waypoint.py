"""Tool lifecycle tests plus real optimizer/MuJoCo control integration."""
import json
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from agent.state_machine import AgentState, AgentStateMachine, ThinkOutcome
from env.flight_state_machine import ControlMode, FlightState, FlightStateMachine
from tests.support import FakeJoystick, AuthorizedEnvironment
from env.mmuav_env import MMUAVEnv
from env.trajectory_optimizer import PolynomialTrajectoryOptimizer
from tools.fly_waypoint import FlyWaypointConfig, FlyWaypointTool, parse_fly_waypoint


def executing_agent():
    agent = AgentStateMachine()
    agent.start_turn()
    agent.complete_think(ThinkOutcome.TOOL)
    return agent


def command(times=(1.0, 3.0)):
    return {"tool": "fly_waypoint", "waypoint": [
        {"time": t, "position": [0.1 * (i + 1), 0.0, 1.0], "yaw": 0.0}
        for i, t in enumerate(times)
    ]}


class FakeOptimizer:
    def optimize(self, **kwargs):
        self.arguments = kwargs
        final = SimpleNamespace(position=kwargs["positions"][-1], yaw=kwargs["yaws"][-1])
        return SimpleNamespace(total_time=kwargs["waypoint_times"][-1], evaluate=lambda t: final)


class FakeEnv(AuthorizedEnvironment):
    def __init__(self):
        self.flight_state_machine = FlightStateMachine(FlightState.HOVERING, control_mode=ControlMode.AGENT)
        self.time = 0.0
        self.position = np.array([0., 0., 1.])
        self.trajectory = None
        self.steps = 0
        self.on_step = lambda: None
        self.disturbance = False
        self.hold_count = 0

    def observe(self):
        disturbed = self.disturbance and 3.25 <= self.time <= 3.35
        return {"time": self.time, "position": self.position.copy(),
                "velocity": np.array([1., 0., 0.]) if disturbed else np.zeros(3),
                "yaw": 0.0, "angular_velocity": np.zeros(3)}

    def follow_trajectory(self, trajectory):
        self.trajectory = trajectory

    def hold_position(self, position=None, yaw=None):
        self.hold_count += 1
        if position is not None:
            self.position = position.copy()

    def step(self):
        self.time += 0.05
        self.steps += 1
        if self.trajectory:
            self.position = self.trajectory.evaluate(self.time).position.copy()
        self.on_step()


class ToolLifecycleTests(unittest.TestCase):
    def test_unequal_times_and_continuous_hover(self):
        env, agent, optimizer = FakeEnv(), executing_agent(), FakeOptimizer()
        env.disturbance = True
        result = FlyWaypointTool(env, agent, FlyWaypointConfig(hover_seconds=0.5), optimizer).execute(command())
        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(result["elapsed_seconds"], 3.8)
        np.testing.assert_equal(optimizer.arguments["waypoint_times"], [0., 1., 3.])
        self.assertTrue(np.isnan(optimizer.arguments["velocities"][1:-1]).all())
        np.testing.assert_equal(optimizer.arguments["velocities"][-1], np.zeros(3))
        self.assertEqual(agent.state, AgentState.OBSERVATION)
        self.assertEqual(env.flight_state_machine.state, FlightState.HOVERING)

    def test_ground_and_flying_rejected_without_execution(self):
        for state in (FlightState.ON_GROUND, FlightState.FLY_AGENT):
            with self.subTest(state=state):
                env, agent = FakeEnv(), executing_agent()
                env.flight_state_machine = FlightStateMachine(state, control_mode=ControlMode.AGENT)
                result = FlyWaypointTool(env, agent, optimizer=FakeOptimizer()).execute(command())
                self.assertEqual(result["status"], "error")
                self.assertEqual(env.steps, 0)
                self.assertEqual(env.flight_state_machine.state, state)
                self.assertEqual(agent.state, AgentState.OBSERVATION)

    def test_bad_commands(self):
        bad = [None, {"tool": "fly_waypoint"}, {"tool": "other", "waypoint": []},
               command(()), command((0.,)), command((2., 1.)), command((1., 1.)),
               command((float("nan"),)), command((True,))]
        point = command((1.,))
        point["waypoint"][0]["position"] = [0, 1]
        bad.append(point)
        for value in bad:
            with self.subTest(command=value):
                env, agent = FakeEnv(), executing_agent()
                result = FlyWaypointTool(env, agent, optimizer=FakeOptimizer()).execute(value)
                self.assertEqual(result["status"], "error")
                self.assertEqual(env.steps, 0)
                self.assertEqual(agent.state, AgentState.OBSERVATION)
        times, _, _ = parse_fly_waypoint("```json\n" + json.dumps(command()) + "\n```")
        np.testing.assert_equal(times, [1., 3.])

    def test_timeout(self):
        env, agent = FakeEnv(), executing_agent()
        env.disturbance = True
        original_observe = env.observe
        def observe():
            value = original_observe()
            value["velocity"] = np.ones(3)
            return value
        env.observe = observe
        result = FlyWaypointTool(env, agent, FlyWaypointConfig(hover_seconds=.2, settle_timeout=.3), FakeOptimizer()).execute(command((1.,)))
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(agent.state, AgentState.OBSERVATION)
        self.assertEqual(env.flight_state_machine.state, FlightState.HOVERING)
        self.assertEqual(env.hold_count, 1)

    def test_termination_and_late_result(self):
        for new_turn in (False, True):
            with self.subTest(new_turn=new_turn):
                env, agent = FakeEnv(), executing_agent()
                def stop():
                    agent.terminate("stop")
                    if new_turn:
                        agent.start_turn()
                        agent.complete_think(ThinkOutcome.TOOL)
                env.on_step = stop
                result = FlyWaypointTool(env, agent, optimizer=FakeOptimizer()).execute(command())
                self.assertEqual(result["status"], "terminated")
                self.assertEqual(agent.state, AgentState.EXECUTE if new_turn else AgentState.END)
                self.assertEqual(env.flight_state_machine.state, FlightState.HOVERING)
                self.assertEqual(env.hold_count, 1)

    def test_step_failure_and_optimizer_failure(self):
        env, agent = FakeEnv(), executing_agent()
        def fail():
            raise RuntimeError("stepping failed")
        env.on_step = fail
        result = FlyWaypointTool(env, agent, optimizer=FakeOptimizer()).execute(command())
        self.assertEqual(result["status"], "error")
        self.assertEqual(agent.state, AgentState.OBSERVATION)
        self.assertEqual(env.hold_count, 1)
        env, agent = FakeEnv(), executing_agent()
        optimizer = FakeOptimizer()
        optimizer.optimize = lambda **kwargs: fail()
        result = FlyWaypointTool(env, agent, optimizer=optimizer).execute(command())
        self.assertEqual(result["status"], "error")
        self.assertEqual(env.flight_state_machine.state, FlightState.HOVERING)
        self.assertEqual(env.steps, 0)

    def test_failed_hold_does_not_claim_hover(self):
        env, agent = FakeEnv(), executing_agent()
        def fail():
            raise RuntimeError("control failed")
        env.on_step = fail
        env.hold_position = lambda *args, **kwargs: fail()
        result = FlyWaypointTool(env, agent, optimizer=FakeOptimizer()).execute(command())
        self.assertEqual(result["status"], "error")
        self.assertIn("hold_error", result)
        self.assertEqual(env.flight_state_machine.state, FlightState.FLY_AGENT)
        self.assertEqual(agent.state, AgentState.OBSERVATION)


class OptimizerTests(unittest.TestCase):
    def test_variable_times_endpoints_and_continuity(self):
        optimizer = PolynomialTrajectoryOptimizer()
        positions = np.array([[0., 0., 1.], [.1, .2, 1.1], [.4, 0., 1.]])
        trajectory = optimizer.optimize(positions, [0., .1, .2], waypoint_times=[0., 1.5, 4.],
                                        velocities=[np.zeros(3), None, np.zeros(3)],
                                        accelerations=[np.zeros(3), None, np.zeros(3)],
                                        yaw_rates=[0., None, 0.])
        np.testing.assert_allclose(trajectory.segment_durations, [1.5, 2.5])
        for t, position, yaw in zip([0., 1.5, 4.], positions, [0., .1, .2]):
            state = trajectory.evaluate(t)
            np.testing.assert_allclose(state.position, position, atol=1e-7)
            self.assertAlmostEqual(state.yaw, yaw, places=7)
        for order in range(5):
            np.testing.assert_allclose(trajectory._eval_pos(0, 1.5, order),
                                       trajectory._eval_pos(1, 0., order), atol=1e-6)
        np.testing.assert_allclose(trajectory.evaluate(4.).velocity, 0., atol=1e-7)

    def test_single_future_waypoint_and_bad_times(self):
        optimizer = PolynomialTrajectoryOptimizer()
        positions = [[0., 0., 1.], [.2, 0., 1.]]
        trajectory = optimizer.optimize(positions, [0., .1], waypoint_times=[0., 2.])
        self.assertEqual(trajectory.yaw_coeffs.shape, (1, 6))
        for times in ([1., 2.], [0., 0.], [0., float("inf")], [0.]):
            with self.assertRaises(ValueError):
                optimizer.optimize(positions, [0., 0.], waypoint_times=times)
        with self.assertRaises(ValueError):
            optimizer.optimize(positions, [0., 0.], segment_durations=[2.], waypoint_times=[0., 2.])
        with self.assertRaises(ValueError):
            optimizer.optimize(positions, [0., 0.], segment_durations=[float("nan")])


class MuJoCoIntegrationTests(unittest.TestCase):
    def test_real_waypoints_hover_and_persistent_hold(self):
        from env import mmuav_env as controller
        previous_mass = controller.position_controller.vehicle_mass
        with MMUAVEnv(FlightStateMachine(FlightState.HOVERING), joystick_reader=FakeJoystick()) as env:
            # Test fixture starts airborne; this is not a takeoff tool.
            env.data.qpos[:3] = [-2., 2., 1.5]
            mujoco.mj_forward(env.model, env.data)
            env.hold_position()
            for _ in range(2500):
                env.step()
            agent = executing_agent()
            tool = FlyWaypointTool(env, agent, FlyWaypointConfig(hover_seconds=.5))
            result = tool.execute(json.dumps({"tool": "fly_waypoint", "waypoint": [
                {"time": 2., "position": [-1.8, 2., 1.5], "yaw": .1},
                {"time": 5., "position": [-1.5, 1.8, 1.5], "yaw": .2}]}))
            self.assertEqual(result["status"], "success", result)
            self.assertGreaterEqual(result["elapsed_seconds"], 5.5 - 1e-7)
            self.assertLess(result["position_error"], .08)
            self.assertEqual(agent.state, AgentState.OBSERVATION)
            self.assertEqual(env.flight_state_machine.state, FlightState.HOVERING)
            for _ in range(500):
                env.step()
            self.assertLess(np.linalg.norm(env.observe()["position"] - [-1.5, 1.8, 1.5]), .08)
            agent.complete_observation()
            agent.complete_think(ThinkOutcome.TOOL)
            result = tool.execute({"tool": "fly_waypoint", "waypoint": [
                {"time": 2., "position": [-1.3, 1.8, 1.5], "yaw": -.1}]})
            self.assertEqual(result["status"], "success", result)
        self.assertEqual(controller.position_controller.vehicle_mass, previous_mass)
        self.assertIsNone(mujoco.get_mjcb_control())


if __name__ == "__main__":
    unittest.main()
