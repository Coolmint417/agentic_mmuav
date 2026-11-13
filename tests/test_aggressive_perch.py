"""Terminal flatness constraints and actual multi-cup MuJoCo perching."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from agent.protocol import parse_decision
from agent.state_machine import AgentState, ThinkOutcome
from env import mmuav_env as control
from env.flight_state_machine import FlightState
from env.trajectory_optimizer import PolynomialTrajectoryOptimizer
from env.controller.se3_controller import SE3Controller, VehicleState
from tests.support import FakeJoystick
from tools import AggressivePerchTool
from tools.aggressive_perch import perch_frame


def command(angle=90, yaw=0., duration=2.):
    return {"tool": "aggressive_perch", "position": [-2., 0., 3.],
            "yaw": yaw, "wall_angle_deg": angle, "time": duration}


def surface_fixture(env, angle, yaw=0.):
    root = ET.parse(Path(control.__file__).with_name("mmuav.xml")).getroot()
    root.find("compiler").set("meshdir", str(Path(control.__file__).parent / "assets"))
    rotation, normal = perch_frame(yaw, angle)
    point = np.array(command()["position"])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    ET.SubElement(root.find("worldbody"), "geom", {
        "name": "perch_surface", "type": "box", "size": "3 3 .1",
        "pos": " ".join(map(str, point - .1 * normal)),
        "quat": " ".join(map(str, quat)), "friction": "1 0.01 0.01"})
    ET.SubElement(root.find("worldbody"), "camera", {
        "name": "world_camera", "pos": "-4 4 5", "euler": "0 0 0"})
    body = root.find('.//body[@name="Body"]')
    body.set("pos", " ".join(map(str, point + 2.1 * normal)))
    body.set("quat", f"{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}")
    callback = mujoco.get_mjcb_control()
    try:
        mujoco.set_mjcb_control(None)
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    finally:
        mujoco.set_mjcb_control(callback)
    env.reload_scene("map1", scene_model=model)
    env.flight_state_machine.transition_to(FlightState.HOVERING)
    env.hold_position()
    for _ in range(300):
        env.step()
    return point, normal


def executing_tool(env):
    env.agent_state_machine.start_turn()
    env.agent_state_machine.complete_think(ThinkOutcome.TOOL)
    return AggressivePerchTool(env, env.agent_state_machine)


class AggressivePerchTests(unittest.TestCase):
    def test_map6_estimated_plane_offset_no_longer_rejects_real_adhesion(self):
        with control.MMUAVEnv(joystick_reader=FakeJoystick(), scene_id="map6") as env:
            env.flight_state_machine.transition_to(FlightState.HOVERING)
            env.data.qpos[:3] = [1.05, 2., 3.]
            mujoco.mj_forward(env.model, env.data)
            env.hold_position([1.05, 2., 3.], 0.)
            for _ in range(500):
                env.step()
            call = {"tool": "aggressive_perch", "position": [2.95, 2., 2.],
                    "yaw": 0., "wall_angle_deg": 45., "time": 1.8}
            result = executing_tool(env).execute(call)
            self.assertEqual(result["status"], "success", result)
            self.assertGreaterEqual(result["stable_seconds"], .5)
            self.assertEqual(len(result["contact_cups"]), 4)
            self.assertEqual(env.flight_state_machine.sucker_states, (True,) * 4)
            np.testing.assert_array_equal(env.data.ctrl[5:9], 0.)
            for _ in range(500):
                env.step()
            self.assertEqual(env.perch_contact_status()["contact_count"], 4)

    def test_stability_timer_restarts_after_contact_loss_or_angular_motion(self):
        for interrupt in ("contact", "angular"):
            with self.subTest(interrupt=interrupt), control.MMUAVEnv(joystick_reader=FakeJoystick()) as env:
                surface_fixture(env, 60.)
                latched = []
                idle = env.idle_perch_motors
                observe = env.observe
                contacts = env.perch_contact_status

                def latch():
                    latched.append(env.data.time)
                    idle()

                def disturbed():
                    return latched and .20 <= env.data.time - latched[0] < .30

                def sample():
                    value = observe()
                    if interrupt == "angular" and disturbed():
                        value["angular_velocity"] = np.array([1., 0., 0.])
                    return value

                def contact_sample(*args):
                    value = contacts(*args)
                    if interrupt == "contact" and disturbed():
                        value["contact_count"] = 3
                        value["contact_cups"] = value["contact_cups"][:3]
                    return value

                with patch.object(env, "idle_perch_motors", latch), \
                     patch.object(env, "observe", sample), \
                     patch.object(env, "perch_contact_status", contact_sample):
                    result = executing_tool(env).execute(command(60.))
                self.assertEqual(result["status"], "success", result)
                self.assertGreaterEqual(result["stable_seconds"], .5)
                self.assertGreaterEqual(env.data.time - latched[0], .80 - 1e-6)

    def test_logged_low_altitude_approaches_perch_without_touching_floor(self):
        for start_x in (-1.7, -2.4):
            with self.subTest(start_x=start_x), control.MMUAVEnv(joystick_reader=FakeJoystick()) as env:
                env.flight_state_machine.transition_to(FlightState.HOVERING)
                env.data.qpos[:3] = [start_x, 1., 1.]
                mujoco.mj_forward(env.model, env.data)
                env.hold_position([start_x, 1., 1.], 0.)
                for _ in range(500):
                    env.step()
                call = {"tool": "aggressive_perch", "position": [.38, 1., 1.],
                        "yaw": 0., "wall_angle_deg": 90., "time": 1.5}
                result = executing_tool(env).execute(call)
                self.assertEqual(result["status"], "success", result)
                self.assertGreaterEqual(result["planning"]["minimum_planned_z"], .25 - 1e-6)
                self.assertGreaterEqual(result["planning"]["average_normal_approach_speed_m_s"], 1.)
                self.assertEqual(len(result["contact_cups"]), 4)
                self.assertGreater(env.observe()["position"][2], .8)

    def test_slow_approach_is_rejected_before_planning(self):
        with control.MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            point, normal = surface_fixture(env, 90)
            env.data.qpos[:3] = point + normal
            result = executing_tool(env).execute(command())
            self.assertEqual(result["status"], "error")
            self.assertIn("average normal approach speed", result["error"])
            self.assertIs(env.flight_state_machine.state, FlightState.HOVERING)
            self.assertEqual(env.flight_state_machine.sucker_states, (False,) * 4)

    def test_controller_rotation_remains_orthonormal_through_vertical_pitch(self):
        controller = SE3Controller()
        controller.set_parameters(40, 12, 12)
        for angle in (30, 60, 89.99, 90, 90.01):
            rotation, normal = perch_frame(.7, angle)
            controller.set_current_state(VehicleState(rotation=rotation))
            controller.set_goal_state(VehicleState())
            acceleration = 12 * normal + np.array([0, 0, -controller.gravity])
            output = controller.control_update(.7, acceleration,
                       preferred_body_y=np.array([-np.sin(.7), np.cos(.7), 0.]))
            np.testing.assert_allclose(controller.command_rotation, rotation, atol=1e-10)
            np.testing.assert_allclose(controller.command_rotation.T @ controller.command_rotation,
                                       np.eye(3), atol=1e-10)
            np.testing.assert_allclose(output[1:], 0., atol=1e-10)

    def test_runtime_registry_and_runner_observation(self):
        from agent.runner import AgentRunner
        from runtime import EnvironmentRuntime, RuntimeConfig
        from tests.test_runner import OwnedEnvironment, ScriptedModel, wait_until
        remote = FakeJoystick()

        def factory():
            env = OwnedEnvironment(joystick_reader=remote)
            try:
                surface_fixture(env, 90)
                return env
            except Exception:
                env.close()
                raise

        runtime = EnvironmentRuntime(RuntimeConfig(realtime=False, preview_enabled=False),
                                     environment_factory=factory)
        runtime.start()
        runner = None
        try:
            runtime.wait_ready()
            model = ScriptedModel([json.dumps(command()), '{"final":"已吸附到墙面"}'])
            runner = AgentRunner(runtime, backend_factory=lambda spec: model)
            runner.load_model()
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertTrue(runner.submit("吸附到墙面"))
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(len(model.calls), 2)
            feedback = json.loads(model.calls[-1]["messages"][-1]["content"])
            self.assertEqual(feedback["tool_result"]["status"], "success", feedback)
            self.assertEqual(feedback["observation"]["flight_state"], "on_ground")
            self.assertEqual(runtime.get_status()["agent_state"], "end")
        finally:
            if runner is not None:
                runner.close()
            runtime.close()

    def test_absent_surface_times_out_and_restores_hover(self):
        with control.MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            surface_fixture(env, 90)
            env.model.geom("perch_surface").contype = 0
            env.model.geom("perch_surface").conaffinity = 0
            result = executing_tool(env).execute(command())
            self.assertEqual(result["status"], "timeout", result)
            self.assertIs(env.flight_state_machine.state, FlightState.HOVERING)
            self.assertIs(env.agent_state_machine.state, AgentState.OBSERVATION)
            np.testing.assert_array_equal(env.data.ctrl[1:5], 0.)
            self.assertFalse(control.perch_active)
    def test_protocol_rejects_invalid_parameters(self):
        self.assertEqual(parse_decision(json.dumps(command())), command())
        self.assertEqual(parse_decision('```json\n' + json.dumps(command()) + '\n```'), command())
        for update in ({"position": [[0, 0, 1], [0, 0, 2]]}, {"wall_angle_deg": 29},
                       {"wall_angle_deg": 91}, {"time": 0}, {"time": 2.01}, {"time": True},
                       {"yaw": float("nan")}, {"position": [0, False, 1]}, {"extra": 1}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                parse_decision(json.dumps(command() | update))

    def test_terminal_acceleration_velocity_and_zero_jerk(self):
        for angle in (30, 60, 90):
            rotation, normal = perch_frame(1.1, angle)
            acceleration = 12 * normal + np.array([0., 0., -9.8])
            velocity = -.4 * normal
            goal = np.array([0., 0., 3.])
            trajectory = PolynomialTrajectoryOptimizer().optimize(
                positions=np.vstack([goal + normal, goal]), yaws=np.array([1.1, 1.1]),
                waypoint_times=np.array([0., 2.]), velocities=np.vstack([np.zeros(3), velocity]),
                accelerations=np.vstack([np.zeros(3), acceleration]),
                jerks=np.zeros((2, 3)), yaw_rates=np.zeros(2))
            final = trajectory.evaluate(2.)
            np.testing.assert_allclose(final.position, goal, atol=1e-6)
            np.testing.assert_allclose(final.acceleration, acceleration, atol=1e-6)
            np.testing.assert_allclose(final.velocity, velocity, atol=1e-6)
            np.testing.assert_allclose(final.jerk, 0., atol=1e-6)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            np.testing.assert_allclose((final.acceleration + [0, 0, 9.8]) / 12, rotation[:, 2], atol=1e-6)

    def test_real_inclined_and_vertical_surface_adhesion(self):
        for angle, yaw in ((30, 0.), (60, 0.), (75, 0.), (90, 0.), (90, 1.1)):
            with self.subTest(angle=angle, yaw=yaw), control.MMUAVEnv(joystick_reader=FakeJoystick()) as env:
                point, normal = surface_fixture(env, angle, yaw)
                original_gains = (control.position_controller.kx, control.position_controller.kv,
                                  control.position_controller.kR)
                result = executing_tool(env).execute(command(angle, yaw))
                self.assertEqual(result["status"], "success", result)
                self.assertIs(env.flight_state_machine.state, FlightState.ON_GROUND)
                self.assertIs(env.agent_state_machine.state, AgentState.OBSERVATION)
                np.testing.assert_array_equal(env.data.ctrl[1:5], 1.)
                self.assertEqual(env.flight_state_machine.sucker_states, (True,) * 4)
                np.testing.assert_array_equal(env.data.ctrl[5:9], 0.)
                self.assertIsNone(control.planner_target_provider)
                self.assertFalse(control.perch_active)
                self.assertEqual(original_gains, (control.position_controller.kx, control.position_controller.kv,
                                                 control.position_controller.kR))
                # Confirm attachment persists after the tool has returned,
                # without the flight controller or a welded pose override.
                for _ in range(500):
                    env.step()
                self.assertEqual(env.perch_contact_status(normal, point)["contact_count"], 4)
                self.assertLess(np.linalg.norm(env.observe()["velocity"]), .02)
                control.joystick.axes[4] = 0.
                env.step()
                self.assertIs(env.flight_state_machine.state, FlightState.ON_GROUND)
                self.assertEqual(env.flight_state_machine.sucker_states, (True,) * 4)
                np.testing.assert_array_equal(env.data.ctrl[5:9], 0.)

    def test_wrong_state_and_planning_failure_do_not_leave_cups_enabled(self):
        with control.MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            result = executing_tool(env).execute(command())
            self.assertEqual(result["status"], "error")
            np.testing.assert_array_equal(env.data.ctrl[1:5], 0.)
        with control.MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            surface_fixture(env, 90)
            task = executing_tool(env).iter_execute(command())
            request = next(task)
            self.assertTrue(control.perch_active)
            np.testing.assert_array_equal(env.data.ctrl[1:5], 1.)  # before solve
            with self.assertRaises(StopIteration) as done:
                task.throw(RuntimeError("optimizer failure"))
            self.assertEqual(done.exception.value["status"], "error")
            self.assertFalse(control.perch_active)
            np.testing.assert_array_equal(env.data.ctrl[1:5], 0.)
            self.assertIs(env.flight_state_machine.state, FlightState.HOVERING)

    def test_axis4_revocation_and_generator_close(self):
        for revoke in (True, False):
            remote = FakeJoystick()
            with self.subTest(revoke=revoke), control.MMUAVEnv(joystick_reader=remote) as env:
                surface_fixture(env, 90)
                task = executing_tool(env).iter_execute(command())
                request = next(task)
                task.send(request.compute())
                for _ in range(30):
                    env.step()
                    next(task)
                if revoke:
                    remote.axes[4] = 0.
                    env.step()
                    with self.assertRaises(StopIteration) as done:
                        next(task)
                    self.assertEqual(done.exception.value["status"], "terminated")
                else:
                    env.agent_state_machine.terminate("user stop")
                    task.close()
                self.assertIs(env.agent_state_machine.state, AgentState.END)
                self.assertIs(env.flight_state_machine.state, FlightState.HOVERING)
                np.testing.assert_array_equal(env.data.ctrl[1:5], 0.)
                if revoke:
                    self.assertIsNone(control.planner_target_provider)
                else:
                    self.assertIsNotNone(control.planner_target_provider)
                self.assertFalse(control.perch_active)


if __name__ == "__main__":
    unittest.main()
