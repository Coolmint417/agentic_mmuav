"""Counted agent crawling, schema, rejection and physical interruption."""
import json
import unittest

import numpy as np

from agent.protocol import parse_decision
from agent.state_machine import AgentState, ThinkOutcome
from env import mmuav_env as control
from env.flight_state_machine import FlightState
from tests.support import FakeJoystick
from tools import CrawlTool


class CrawlToolTests(unittest.TestCase):
    def executing_tool(self, env):
        env.agent_state_machine.start_turn()
        env.agent_state_machine.complete_think(ThinkOutcome.TOOL)
        return CrawlTool(env, env.agent_state_machine)

    def test_protocol(self):
        valid = {"tool": "crawl", "steps": 2, "attached": True}
        self.assertEqual(parse_decision(json.dumps(valid)), valid)
        self.assertEqual(parse_decision('```json\n' + json.dumps(valid) + '\n```'), valid)
        for changes in ({"steps": True}, {"steps": 0}, {"steps": -1},
                        {"steps": 1.5}, {"attached": 1}, {"attached": "false"},
                        {"extra": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse_decision(json.dumps(valid | changes))
        for missing in ("steps", "attached"):
            with self.assertRaises(ValueError):
                parse_decision(json.dumps({k: v for k, v in valid.items() if k != missing}))

    def test_count_state_and_repeated_call_with_suction_enabled(self):
        remote = FakeJoystick()
        with control.MMUAVEnv(joystick_reader=remote) as env:
            for steps, attached in ((1, True), (2, False)):
                tool = self.executing_tool(env)
                remote.reads.clear()
                result = tool.execute({"tool": "crawl", "steps": steps, "attached": attached})
                self.assertEqual(result["status"], "success", result)
                self.assertEqual(result["steps_completed"], steps)
                self.assertEqual(set(remote.reads), {4})
                self.assertIs(env.agent_state_machine.state, AgentState.OBSERVATION)
                self.assertIs(env.flight_state_machine.state,
                              FlightState.ON_GROUND)
                np.testing.assert_array_equal(env.data.ctrl[1:5], float(attached))
                self.assertEqual(env.flight_state_machine.sucker_states, (attached,) * 4)
                np.testing.assert_array_equal(env.data.ctrl[5:9], 0.0)
                self.assertEqual(env.data.ctrl[0], 0.0)
                self.assertLess(abs(float(env.data.joint("joint_upper_link").qpos[0])), .03)
                for name in control.SUCKER_GEOMS:
                    self.assertEqual(env.model.geom(name).contype, 1)
                    self.assertEqual(env.model.geom(name).conaffinity, 1)
                if attached:
                    env.agent_state_machine.terminate("next independent task")

    def test_reject_other_states_without_actuation(self):
        for state in (FlightState.HOVERING, FlightState.FLY_AGENT, FlightState.CRAWLING):
            with self.subTest(state=state), control.MMUAVEnv(joystick_reader=FakeJoystick()) as env:
                # Establish the state after authorization, without stepping an
                # unrelated physical operation while testing rejection.
                env.flight_state_machine.transition_to(
                    FlightState.CRAWLING if state is FlightState.CRAWLING else FlightState.HOVERING)
                if state is FlightState.FLY_AGENT:
                    env.flight_state_machine.transition_to(state)
                tool = self.executing_tool(env)
                before = env.data.ctrl.copy()
                result = tool.execute({"tool": "crawl", "steps": 1, "attached": False})
                self.assertEqual(result["status"], "error", result)
                self.assertIs(env.flight_state_machine.state, state)
                self.assertIs(env.agent_state_machine.state, AgentState.OBSERVATION)
                np.testing.assert_array_equal(env.data.ctrl, before)

    def test_axis4_revocation_terminates_and_cleanup_finishes_on_owner_loop(self):
        remote = FakeJoystick()
        with control.MMUAVEnv(joystick_reader=remote) as env:
            tool = self.executing_tool(env)
            remote.on_update = lambda: remote.axes.update({4: 0.0}) if env.data.time > 1 else None
            result = tool.execute({"tool": "crawl", "steps": 10, "attached": False})
            self.assertEqual(result["status"], "terminated", result)
            self.assertIs(env.agent_state_machine.state, AgentState.END)
            self.assertFalse(env.agent_control_enabled)
            deadline = env.data.time + 7
            while env.flight_state_machine.state is FlightState.CRAWLING and env.data.time < deadline:
                env.step()
            self.assertIs(env.flight_state_machine.state, FlightState.ON_GROUND)
            np.testing.assert_array_equal(env.data.ctrl[1:5], 1.0)
            np.testing.assert_array_equal(env.data.ctrl[5:9], 0.0)

    def test_generator_close_requests_cleanup(self):
        with control.MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            task = self.executing_tool(env).iter_execute(
                {"tool": "crawl", "steps": 10, "attached": False})
            next(task)
            env.agent_state_machine.terminate("user cancelled")
            task.close()
            self.assertTrue(control.crawl_stop_requested)
            self.assertIs(env.agent_state_machine.state, AgentState.END)
            for _ in range(4000):
                env.step()
                if env.flight_state_machine.state is not FlightState.CRAWLING:
                    break
            self.assertIs(env.flight_state_machine.state, FlightState.ON_GROUND)


if __name__ == "__main__":
    unittest.main()
