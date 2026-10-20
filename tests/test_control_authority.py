"""Manual/agent authority, remote masking and physical handover regressions."""
import unittest

import mujoco
import numpy as np

from agent.state_machine import AgentState, AgentStateMachine, InvalidAgentTransition, ThinkOutcome
from env import mmuav_env as control
from env.flight_state_machine import ControlMode, FlightState, FlightStateMachine, InvalidFlightTransition
from env.mmuav_env import MMUAVEnv, POSITION_AUTO_NONE, POSITION_AUTO_LAND
from tests.support import FakeJoystick
from tools import FlyWaypointTool, TakeoffTool, LandTool


class FlightMachineTests(unittest.TestCase):
    def test_transition_matrix_with_authority(self):
        for mode in ControlMode:
            for source in FlightState:
                if source is FlightState.FLY_AGENT and mode is ControlMode.MANUAL:
                    with self.assertRaises(InvalidFlightTransition):
                        FlightStateMachine(source, control_mode=mode)
                    continue
                for target in FlightState:
                    machine = FlightStateMachine(source, control_mode=mode)
                    allowed = not (
                        {source, target} == {FlightState.ON_GROUND, FlightState.FLY_AGENT}
                        or (mode is ControlMode.MANUAL and target is FlightState.FLY_AGENT)
                    )
                    self.assertEqual(machine.can_transition_to(target), allowed)
                    if allowed:
                        self.assertEqual(machine.transition_to(target), target)
                    else:
                        with self.assertRaises(InvalidFlightTransition):
                            machine.transition_to(target)
                        self.assertEqual(machine.state, source)

    def test_revoke_grant_and_disabled_agent(self):
        machine = FlightStateMachine()
        self.assertTrue(machine.set_agent_authorized(True))
        self.assertEqual(machine.state, FlightState.ON_GROUND)
        self.assertTrue(machine.set_agent_authorized(False))
        self.assertEqual(machine.state, FlightState.HOVERING)
        agent = AgentStateMachine()
        agent.start_turn()
        agent.set_enabled(False)
        self.assertEqual(agent.state, AgentState.END)
        self.assertFalse(agent.can_start_turn)
        with self.assertRaises(InvalidAgentTransition):
            agent.start_turn()
        agent.set_enabled(True)
        self.assertEqual(agent.state, AgentState.END)
        agent.start_turn()
        self.assertEqual(agent.state, AgentState.THINK)


class AuthorityIntegrationTests(unittest.TestCase):
    def test_agent_masks_every_other_axis(self):
        remote = FakeJoystick()
        remote.axes.update({0: 1., 1: 1., 2: 1., 3: 1., 5: 1., 7: 1.})
        with MMUAVEnv(FlightStateMachine(FlightState.HOVERING), joystick_reader=remote) as env:
            self.assertIs(env.flight_state_machine, control.flight_state_machine)
            target = control.position_target.copy()
            remote.reads.clear()
            for _ in range(10):
                env.step()
            self.assertEqual(set(remote.reads), {4})
            np.testing.assert_equal(control.position_target, target)
            self.assertEqual(env.get_position_auto_mode(), POSITION_AUTO_NONE)

    def test_revoke_during_each_active_agent_state_and_resume(self):
        for state in (AgentState.THINK, AgentState.EXECUTE, AgentState.OBSERVATION):
            with self.subTest(state=state):
                remote = FakeJoystick()
                with MMUAVEnv(FlightStateMachine(FlightState.HOVERING), joystick_reader=remote) as env:
                    agent = env.agent_state_machine
                    agent.start_turn()
                    if state is not AgentState.THINK:
                        agent.complete_think(ThinkOutcome.TOOL)
                    if state is AgentState.OBSERVATION:
                        agent.complete_execute()
                    stopped = []
                    env.set_model_stop_callback(lambda: stopped.append(True))
                    remote.axes.update({0: 1., 1: 1., 2: 1., 3: 1., 7: 1., 4: 0.})
                    initial = env.observe()["position"]
                    env.step()
                    self.assertEqual(stopped, [True])
                    self.assertEqual(agent.state, AgentState.END)
                    self.assertFalse(agent.enabled)
                    self.assertTrue(agent.termination_requested)
                    self.assertEqual(env.flight_state_machine.state, FlightState.HOVERING)
                    self.assertEqual(env.flight_state_machine.control_mode, ControlMode.MANUAL)
                    self.assertIsNone(control.planner_target_provider)
                    np.testing.assert_equal(control.position_target, initial)
                    with self.assertRaises(InvalidAgentTransition):
                        agent.start_turn()
                    # Unmask movement next step, but do not replay the Axis7
                    # change that occurred during agent control.
                    env.step()
                    self.assertGreater(np.linalg.norm(control.position_target - initial), 0.)
                    self.assertEqual(env.get_position_auto_mode(), POSITION_AUTO_NONE)
                    self.assertEqual(stopped, [True])
                    remote.axes[4] = 1.
                    env.step()
                    self.assertTrue(agent.enabled)
                    self.assertEqual(agent.state, AgentState.END)
                    agent.start_turn()
                    self.assertEqual(agent.state, AgentState.THINK)

    def test_manual_axis7_takeoff_motion_and_land(self):
        remote = FakeJoystick(agent_enabled=False)
        remote.axes[5] = -1.  # The removed mode switch must have no effect.
        with MMUAVEnv(joystick_reader=remote) as env:
            self.assertFalse(env.agent_state_machine.enabled)
            remote.axes[7] = 1.
            for _ in range(2500):
                env.step()
                self.assertNotEqual(env.flight_state_machine.state, FlightState.FLY_AGENT)
                if env.flight_state_machine.state is FlightState.HOVERING:
                    break
            self.assertEqual(env.flight_state_machine.state, FlightState.HOVERING)
            initial_target = control.position_target.copy()
            remote.axes[1] = .2
            for _ in range(100):
                env.step()
            self.assertGreater(control.position_target[0], initial_target[0])
            self.assertEqual(env.flight_state_machine.state, FlightState.HOVERING)
            remote.axes[1] = 0.
            remote.axes[7] = -1.
            env.step()
            self.assertEqual(env.get_position_auto_mode(), POSITION_AUTO_LAND)
            for _ in range(6000):
                env.step()
                if env.flight_state_machine.state is FlightState.ON_GROUND:
                    break
            self.assertEqual(env.flight_state_machine.state, FlightState.ON_GROUND)
            self.assertLess(env.observe()["position"][2], .1)
            self.assertNotIn(5, remote.reads)
            self.assertEqual(env.agent_state_machine.state, AgentState.END)

    def test_axis4_interrupts_each_tool_without_stale_hold(self):
        for Tool in (TakeoffTool, LandTool, FlyWaypointTool):
            with self.subTest(tool=Tool.name):
                remote = FakeJoystick()
                state = FlightState.ON_GROUND if Tool is TakeoffTool else FlightState.HOVERING
                with MMUAVEnv(FlightStateMachine(state), joystick_reader=remote) as env:
                    if state is FlightState.HOVERING:
                        env.data.qpos[:3] = [-2., 2., 1.5]
                        mujoco.mj_forward(env.model, env.data)
                        env.hold_position()
                    agent = env.agent_state_machine
                    agent.start_turn()
                    agent.complete_think(ThinkOutcome.TOOL)
                    stop_calls = []
                    env.set_model_stop_callback(lambda: stop_calls.append(True))
                    def revoke():
                        if env.data.time >= .02:
                            remote.axes[4] = 0.
                    remote.on_update = revoke
                    original_hold = env.hold_position
                    def authorized_hold(*args, **kwargs):
                        self.assertTrue(env.agent_control_enabled, "stale tool overwrote manual hold")
                        return original_hold(*args, **kwargs)
                    env.hold_position = authorized_hold
                    call = {"tool": Tool.name}
                    if Tool is FlyWaypointTool:
                        call["waypoint"] = [{"time": 2., "position": [-1.8, 2., 1.5], "yaw": 0.}]
                    result = Tool(env, agent).execute(call)
                    self.assertEqual(result["status"], "terminated", result)
                    self.assertNotIn("hold_error", result)
                    self.assertEqual(agent.state, AgentState.END)
                    self.assertEqual(env.flight_state_machine.state, FlightState.HOVERING)
                    self.assertEqual(env.flight_state_machine.control_mode, ControlMode.MANUAL)
                    self.assertEqual(env.get_position_auto_mode(), POSITION_AUTO_NONE)
                    self.assertIsNone(control.planner_target_provider)
                    self.assertEqual(stop_calls, [True])

    def test_manual_mode_rejects_tools_and_trajectory(self):
        remote = FakeJoystick(agent_enabled=False)
        with MMUAVEnv(joystick_reader=remote) as env:
            agent = env.agent_state_machine
            for Tool in (TakeoffTool, LandTool, FlyWaypointTool):
                tool = Tool(env, agent)
                with self.assertRaises(ValueError):
                    tool.execute({"tool": Tool.name})
                self.assertEqual(agent.state, AgentState.END)
            with self.assertRaises(PermissionError):
                env.follow_trajectory(None)
            self.assertEqual(env.flight_state_machine.state, FlightState.ON_GROUND)


if __name__ == "__main__":
    unittest.main()
