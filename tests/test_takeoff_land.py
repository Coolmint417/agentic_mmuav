"""No-argument tool validation, cancellation and real auto-control tests."""
import json
import unittest

import mujoco
import numpy as np

from agent.state_machine import AgentState, AgentStateMachine, ThinkOutcome
from env.flight_state_machine import ControlMode, FlightState, FlightStateMachine
from tests.support import FakeJoystick, AuthorizedEnvironment
from env.mmuav_env import MMUAVEnv, POSITION_AUTO_NONE, POSITION_AUTO_TAKEOFF, POSITION_AUTO_LAND
from tools import AutoFlightConfig, TakeoffTool, LandTool


def executing_agent():
    agent = AgentStateMachine()
    agent.start_turn()
    agent.complete_think(ThinkOutcome.TOOL)
    return agent


class FakeEnv(AuthorizedEnvironment):
    def __init__(self, state):
        self.flight_state_machine = FlightStateMachine(state, control_mode=ControlMode.AGENT)
        self.time = 0.0
        self.mode = POSITION_AUTO_NONE
        self.steps = 0
        self.complete = True
        self.position = np.array([0., 0., .04])
        self.on_step = lambda: None
        self.hold_calls = []

    def observe(self):
        return {"time": self.time, "position": self.position.copy(), "yaw": 0.0}

    def start_auto_takeoff(self):
        self.mode = POSITION_AUTO_TAKEOFF
        return np.array([0., 0., 1.])

    def start_auto_land(self):
        self.mode = POSITION_AUTO_LAND

    def get_position_auto_mode(self):
        return self.mode

    def hold_position(self, position=None, yaw=None):
        self.hold_calls.append(position)
        self.mode = POSITION_AUTO_NONE

    def step(self):
        self.time += .1
        self.steps += 1
        if self.complete and self.steps == 3:
            self.mode = POSITION_AUTO_NONE
        self.on_step()


class ToolTests(unittest.TestCase):
    def test_success_states_and_no_parameters(self):
        for Tool, initial, completed in ((TakeoffTool, FlightState.ON_GROUND, FlightState.HOVERING),
                                         (LandTool, FlightState.HOVERING, FlightState.ON_GROUND)):
            with self.subTest(tool=Tool.name):
                env, agent = FakeEnv(initial), executing_agent()
                def check_pending():
                    self.assertEqual(agent.state, AgentState.EXECUTE)
                    self.assertEqual(env.flight_state_machine.state, initial)
                env.on_step = check_pending
                result = Tool(env, agent).execute(json.dumps({"tool": Tool.name}))
                self.assertEqual(result["status"], "success")
                self.assertEqual(agent.state, AgentState.OBSERVATION)
                self.assertEqual(env.flight_state_machine.state, completed)
                self.assertEqual(env.steps, 3)
                self.assertEqual(env.mode, POSITION_AUTO_NONE)

    def test_invalid_calls_and_wrong_flight_state(self):
        for Tool in (TakeoffTool, LandTool):
            for call in ({"tool": Tool.name, "height": 1}, {"tool": Tool.name, "arguments": {}},
                         {"tool": "other"}, {}, None):
                with self.subTest(tool=Tool.name, call=call):
                    env, agent = FakeEnv(Tool.required_state), executing_agent()
                    result = Tool(env, agent).execute(call)
                    self.assertEqual(result["status"], "error")
                    self.assertEqual(agent.state, AgentState.OBSERVATION)
                    self.assertEqual(env.flight_state_machine.state, Tool.required_state)
                    self.assertEqual(env.steps, 0)
            for initial in FlightState:
                if initial is Tool.required_state:
                    continue
                env, agent = FakeEnv(initial), executing_agent()
                result = Tool(env, agent).execute({"tool": Tool.name})
                self.assertEqual(result["status"], "error")
                self.assertEqual(env.flight_state_machine.state, initial)
                self.assertEqual(env.steps, 0)

    def test_requires_execute(self):
        for Tool in (TakeoffTool, LandTool):
            agent = AgentStateMachine()
            with self.assertRaises(ValueError):
                Tool(FakeEnv(Tool.required_state), agent).execute({"tool": Tool.name})
            self.assertEqual(agent.state, AgentState.END)

    def test_timeout_and_failed_hold(self):
        for Tool in (TakeoffTool, LandTool):
            env, agent = FakeEnv(Tool.required_state), executing_agent()
            env.complete = False
            result = Tool(env, agent, AutoFlightConfig(timeout_seconds=.2)).execute({"tool": Tool.name})
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(agent.state, AgentState.OBSERVATION)
            self.assertEqual(env.flight_state_machine.state, FlightState.HOVERING)
        env, agent = FakeEnv(FlightState.ON_GROUND), executing_agent()
        env.complete = False
        def fail(*args):
            raise RuntimeError("hold failed")
        env.hold_position = fail
        result = TakeoffTool(env, agent, AutoFlightConfig(timeout_seconds=.1)).execute({"tool": "takeoff"})
        self.assertIn("hold_error", result)
        self.assertEqual(env.flight_state_machine.state, FlightState.ON_GROUND)

    def test_termination_and_new_turn_isolation(self):
        for Tool in (TakeoffTool, LandTool):
            for new_turn in (False, True):
                env, agent = FakeEnv(Tool.required_state), executing_agent()
                def stop():
                    agent.terminate("stop")
                    if new_turn:
                        agent.start_turn()
                        agent.complete_think(ThinkOutcome.TOOL)
                env.on_step = stop
                result = Tool(env, agent).execute({"tool": Tool.name})
                self.assertEqual(result["status"], "terminated")
                self.assertEqual(agent.state, AgentState.EXECUTE if new_turn else AgentState.END)
                self.assertEqual(env.flight_state_machine.state, FlightState.HOVERING)

    def test_step_error(self):
        for Tool in (TakeoffTool, LandTool):
            env, agent = FakeEnv(Tool.required_state), executing_agent()
            def fail():
                raise RuntimeError("step failed")
            env.on_step = fail
            result = Tool(env, agent).execute({"tool": Tool.name})
            self.assertEqual(result["status"], "error")
            self.assertEqual(agent.state, AgentState.OBSERVATION)
            self.assertEqual(env.flight_state_machine.state, FlightState.HOVERING)


class RealControlTests(unittest.TestCase):
    def test_ground_takeoff_land_takeoff(self):
        with MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            agent = executing_agent()
            for Tool in (TakeoffTool, LandTool, TakeoffTool):
                result = Tool(env, agent).execute({"tool": Tool.name})
                self.assertEqual(result["status"], "success", result)
                self.assertEqual(agent.state, AgentState.OBSERVATION)
                self.assertEqual(env.flight_state_machine.state, Tool.completed_state)
                self.assertEqual(env.get_position_auto_mode(), POSITION_AUTO_NONE)
                if Tool is TakeoffTool:
                    self.assertLess(abs(result["position"][2] - 1.), .1)
                else:
                    self.assertLess(result["position"][2], .1)
                agent.complete_observation()
                agent.complete_think(ThinkOutcome.TOOL)

    def test_elevated_platform_land_and_takeoff(self):
        with MMUAVEnv(FlightStateMachine(FlightState.HOVERING), joystick_reader=FakeJoystick()) as env:
            # Fixture above the existing brick platform (top z=1.6).
            env.data.qpos[:3] = [1.6, 1.6, 2.]
            mujoco.mj_forward(env.model, env.data)
            env.hold_position()
            for _ in range(1500):
                env.step()
            agent = executing_agent()
            landed = LandTool(env, agent).execute({"tool": "land"})
            self.assertEqual(landed["status"], "success", landed)
            self.assertGreater(landed["position"][2], 1.6)
            self.assertLess(landed["position"][2], 1.8)
            initial_z = env.observe()["position"][2]
            agent.complete_observation()
            agent.complete_think(ThinkOutcome.TOOL)
            took_off = TakeoffTool(env, agent).execute({"tool": "takeoff"})
            self.assertEqual(took_off["status"], "success", took_off)
            self.assertLess(abs(took_off["position"][2] - (initial_z + .8)), .1)
            self.assertEqual(env.flight_state_machine.state, FlightState.HOVERING)
            self.assertEqual(agent.state, AgentState.OBSERVATION)


if __name__ == "__main__":
    unittest.main()
