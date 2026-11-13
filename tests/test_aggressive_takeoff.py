"""Inclined launch optimization, safe detachment and real perching/takeoff."""
import json
import math
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from agent.protocol import parse_decision
from agent.state_machine import AgentState, ThinkOutcome
from env import mmuav_env as control
from env.flight_state_machine import FlightState, FlightStateMachine
from env.trajectory_optimizer import InitialThrustConstraint, PolynomialTrajectoryOptimizer
from tests.support import FakeJoystick
from tests.test_aggressive_perch import command, executing_tool, perch_frame, surface_fixture
from tools import AggressiveTakeoffTool


CALL = {"tool": "aggressive_takeoff"}


def launch_tool(env):
    agent = env.agent_state_machine
    if agent.state is AgentState.OBSERVATION:
        agent.complete_observation()
    elif agent.state is AgentState.END:
        agent.start_turn()
    agent.complete_think(ThinkOutcome.TOOL)
    return AggressiveTakeoffTool(env, agent)


class AggressiveTakeoffTests(unittest.TestCase):
    def test_protocol_and_nominal_thrust_validation(self):
        self.assertEqual(parse_decision(json.dumps(CALL)), CALL)
        with self.assertRaises(ValueError):
            parse_decision(json.dumps({**CALL, "h": 2}))
        machine = FlightStateMachine()
        self.assertIsNone(machine.nominal_specific_thrust)
        for value in (False, 0, -1, float("nan"), float("inf"), "12"):
            with self.assertRaises(ValueError):
                machine.set_nominal_specific_thrust(value)
        machine.set_nominal_specific_thrust(12)
        self.assertEqual(machine.nominal_specific_thrust, 12.)
        machine.set_nominal_specific_thrust(None)
        self.assertIsNone(machine.nominal_specific_thrust)

    def test_initial_acceleration_is_optimized_and_attitude_consistent(self):
        gravity = np.array([0., 0., -9.8])
        _, normal = perch_frame(.8, 60.)
        positions = np.vstack([[0, 0, 3], [0, 0, 3] + 1.8 * normal])
        constraint = InitialThrustConstraint(normal, gravity, 8., 20., initial_guess=16.)
        trajectory = PolynomialTrajectoryOptimizer().optimize(
            positions, np.array([.8, .8]), waypoint_times=np.array([0., 2.]),
            velocities=np.zeros((2, 3)),
            accelerations=np.vstack([np.full(3, np.nan), np.zeros(3)]),
            initial_thrust=constraint, jerks=np.zeros((2, 3)), yaw_rates=np.zeros(2))
        initial, final = trajectory.evaluate(0), trajectory.evaluate(2)
        self.assertGreaterEqual(trajectory.initial_specific_thrust, 8. - 1e-6)
        self.assertLess(trajectory.initial_specific_thrust, 16. - 1.)
        np.testing.assert_allclose(initial.acceleration - gravity,
                                   trajectory.initial_specific_thrust * normal, atol=1e-6)
        self.assertGreater(np.linalg.norm(initial.acceleration), 1.)
        for value in (initial.velocity, final.velocity, final.acceleration):
            np.testing.assert_allclose(value, 0., atol=1e-6)
        np.testing.assert_allclose(final.position, positions[-1], atol=1e-6)
        with self.assertRaises(ValueError):
            PolynomialTrajectoryOptimizer().optimize(
                positions, np.zeros(2), accelerations=np.zeros((2, 3)), initial_thrust=constraint)

    def test_planning_failure_and_missing_nominal_preserve_supported_state(self):
        with control.MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            env.set_sucker_states((True,) * 4)
            result = launch_tool(env).execute(CALL)
            self.assertEqual(result["status"], "error")
            self.assertIn("nominal", result["error"])
            self.assertEqual(env.flight_state_machine.sucker_states, (True,) * 4)
            self.assertIs(env.flight_state_machine.state, FlightState.ON_GROUND)
        with control.MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            surface_fixture(env, 60)
            self.assertEqual(executing_tool(env).execute(command(60))["status"], "success")
            nominal = env.flight_state_machine.nominal_specific_thrust
            task = launch_tool(env).iter_execute(CALL)
            next(task)
            self.assertEqual(env.flight_state_machine.sucker_states, (True,) * 4)
            np.testing.assert_array_equal(env.data.ctrl[5:9], 0.)
            with self.assertRaises(StopIteration) as done:
                task.throw(RuntimeError("solver failed"))
            self.assertEqual(done.exception.value["status"], "error")
            self.assertEqual(env.flight_state_machine.sucker_states, (True,) * 4)
            self.assertIs(env.flight_state_machine.state, FlightState.ON_GROUND)
            self.assertEqual(env.flight_state_machine.nominal_specific_thrust, nominal)

    def test_real_perch_then_takeoff_at_multiple_angles_and_yaws(self):
        for angle, yaw in ((30, 0.), (60, .7), (90, 0.), (90, 1.1)):
            with self.subTest(angle=angle, yaw=yaw), control.MMUAVEnv(joystick_reader=FakeJoystick()) as env:
                surface_fixture(env, angle, yaw)
                perch = executing_tool(env).execute(command(angle, yaw))
                self.assertEqual(perch["status"], "success", perch)
                self.assertEqual(env.flight_state_machine.nominal_specific_thrust, perch["specific_thrust"])
                previous_gains = (control.position_controller.kx, control.position_controller.kv,
                                  control.position_controller.kR)
                result = launch_tool(env).execute(CALL)
                self.assertEqual(result["status"], "success", result)
                self.assertIs(env.agent_state_machine.state, AgentState.OBSERVATION)
                self.assertIs(env.flight_state_machine.state, FlightState.HOVERING)
                self.assertEqual(env.flight_state_machine.sucker_states, (False,) * 4)
                np.testing.assert_array_equal(env.data.ctrl[1:5], 0.)
                planning = result["planning"]
                expected_h = 1 + math.sqrt((planning["pitch_deg"] - 15) / 75)
                self.assertAlmostEqual(planning["height_m"], expected_h)
                np.testing.assert_allclose(
                    np.array(planning["target_position"]) - planning["start_position"],
                    expected_h * np.array(planning["body_z_world"]), atol=1e-8)
                self.assertEqual(previous_gains, (control.position_controller.kx,
                                                 control.position_controller.kv,
                                                 control.position_controller.kR))
                self.assertIsNone(control.takeoff_saved_gains)
                env.reload_scene("map1")
                self.assertIsNone(env.flight_state_machine.nominal_specific_thrust)

    def test_fifteen_degree_supported_launch(self):
        # Initial-condition fixture only: gravity and adhesion settle the cups;
        # the entire takeoff afterward uses normal controls and physics.
        with control.MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            point, normal = surface_fixture(env, 15.)
            rotation, _ = perch_frame(0., 15.)
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, rotation.ravel())
            env.data.qpos[:3] = point - rotation @ env.perch_geometry()["foot_offset"] - .002 * normal
            env.data.qpos[3:7] = quaternion
            env.data.qvel[:] = 0.
            env.flight_state_machine.transition_to(FlightState.ON_GROUND)
            env.set_sucker_states((True,) * 4)
            env.flight_state_machine.set_nominal_specific_thrust(10.)
            mujoco.mj_forward(env.model, env.data)
            for _ in range(300):
                env.step()
            # Ensure the boundary case starts at exactly the requested angle;
            # small contact wobble can otherwise fall just below 15 degrees.
            env.data.qpos[3:7] = quaternion
            env.data.qvel[:] = 0.
            result = launch_tool(env).execute(CALL)
            self.assertEqual(result["status"], "success", result)
            self.assertAlmostEqual(result["planning"]["height_m"], 1., places=5)

    def test_low_wall_launch_preserves_floor_clearance(self):
        with control.MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            env.flight_state_machine.transition_to(FlightState.HOVERING)
            env.data.qpos[:3] = [-1.7, 1., 1.]
            mujoco.mj_forward(env.model, env.data)
            env.hold_position([-1.7, 1., 1.], 0.)
            for _ in range(500):
                env.step()
            perch = {"tool": "aggressive_perch", "position": [.38, 1., 1.],
                     "yaw": 0., "wall_angle_deg": 90., "time": 1.5}
            self.assertEqual(executing_tool(env).execute(perch)["status"], "success")
            heights = []
            original_step = env.step

            def step():
                original_step()
                heights.append(env.observe()["position"][2])

            with patch.object(env, "step", step):
                result = launch_tool(env).execute(CALL)
            self.assertEqual(result["status"], "success", result)
            self.assertGreaterEqual(result["planning"]["minimum_planned_z"], .25 - 1e-6)
            self.assertGreater(min(heights), .15)

    def test_cancel_during_planning_and_after_detachment(self):
        for detach in (False, True):
            remote = FakeJoystick()
            with self.subTest(detach=detach), control.MMUAVEnv(joystick_reader=remote) as env:
                point, _ = surface_fixture(env, 90.)
                self.assertEqual(executing_tool(env).execute(command())["status"], "success")
                task = launch_tool(env).iter_execute(CALL)
                request = next(task)
                if detach:
                    task.send(request.compute())
                    for _ in range(30):
                        env.step()
                        next(task)
                remote.axes[4] = 0.
                env.step()
                with self.assertRaises(StopIteration) as done:
                    if detach:
                        next(task)
                    else:
                        task.send(request.compute())
                self.assertEqual(done.exception.value["status"], "terminated")
                self.assertIsNone(control.takeoff_saved_gains)
                self.assertIs(env.agent_state_machine.state, AgentState.END)
                self.assertEqual(env.flight_state_machine.sucker_states, (not detach,) * 4)

    def test_real_runtime_registry_observation_and_model_decision(self):
        from agent.runner import AgentRunner
        from runtime import EnvironmentRuntime, RuntimeConfig
        from tests.test_runner import OwnedEnvironment, ScriptedModel, wait_until

        def factory():
            env = OwnedEnvironment(joystick_reader=FakeJoystick())
            surface_fixture(env, 60.)
            return env

        runtime = EnvironmentRuntime(RuntimeConfig(realtime=False, preview_enabled=False),
                                     environment_factory=factory)
        runtime.start()
        runner = None
        try:
            runtime.wait_ready()
            model = ScriptedModel([json.dumps(command(60.)), json.dumps(CALL), '{"final":"已起飞"}'])
            runner = AgentRunner(runtime, backend_factory=lambda spec: model)
            runner.load_model()
            wait_until(lambda: not runner.get_status()["busy"])
            runner.submit("吸附后从斜面起飞")
            wait_until(lambda: not runner.get_status()["busy"], timeout=30)
            self.assertEqual(len(model.calls), 3)
            feedback = json.loads(model.calls[-1]["messages"][-1]["content"])
            self.assertEqual(feedback["tool_result"]["status"], "success", feedback)
            self.assertIsNotNone(feedback["observation"]["nominal_specific_thrust_m_s2"])
            self.assertEqual(runtime.get_status()["flight_state"], "hovering")
        finally:
            if runner is not None:
                runner.close()
            runtime.close()
