"""Desktop startup fallback and normal physical Axis4 behavior."""
import unittest
from unittest.mock import patch

from agent.runner import AgentRunner
from env import mmuav_env as control
from env.flight_state_machine import ControlMode, FlightState
from runtime import EnvironmentRuntime, RuntimeConfig
from runtime.events import EventBus
from tests.test_runner import ScriptedModel, wait_until


def reader(connected=False):
    # Real reader type/state, without opening or reading an OS device in tests.
    result = object.__new__(control.JoystickReader)
    result.fd = 42 if connected else None
    result.axes = {4: 0.0, 7: -1.0}
    result.update = lambda: None
    return result


class NoJoystickTests(unittest.TestCase):
    def test_missing_device_fallback_is_opt_in_and_restores_reader(self):
        missing = reader()
        with patch.object(control, "joystick", missing):
            with control.MMUAVEnv() as env:
                self.assertFalse(env.agent_control_enabled)
            with control.MMUAVEnv(auto_agent_without_joystick=True) as env:
                self.assertTrue(env.agent_control_enabled)
                self.assertTrue(env.agent_state_machine.enabled)
                self.assertEqual(env.control_authority_source, "no_joystick")
                self.assertEqual(
                    env.flight_state_machine.control_mode, ControlMode.AGENT
                )
                self.assertEqual(env.flight_state_machine.state, FlightState.ON_GROUND)
                self.assertEqual(env.agent_state_machine.state.value, "end")
                env.require_agent_control()
                for _ in range(10):
                    env.step()
                self.assertTrue(env.agent_control_enabled)
                self.assertEqual(
                    env.get_position_auto_mode(), control.POSITION_AUTO_NONE
                )
            self.assertIs(control.joystick, missing)

    def test_connected_device_still_requires_axis4_and_can_revoke(self):
        connected = reader(True)
        with control.MMUAVEnv(
            joystick_reader=connected, auto_agent_without_joystick=True
        ) as env:
            self.assertFalse(env.agent_control_enabled)
            self.assertEqual(env.control_authority_source, "axis4")
            connected.axes[4] = 1.0
            env.poll_control_authority()
            self.assertTrue(env.agent_control_enabled)
            connected.axes[4] = 0.0
            env.poll_control_authority()
            self.assertFalse(env.agent_control_enabled)

    def test_runner_can_takeoff_finish_stop_and_start_again_without_joystick(self):
        events = EventBus()
        model = ScriptedModel(
            ['{"tool":"takeoff"}', '{"final":"已起飞"}', '{"final":"仍在悬停"}']
        )
        runtime = EnvironmentRuntime(
            RuntimeConfig(preview_enabled=False, realtime=False),
            events=events,
            environment_factory=lambda: control.MMUAVEnv(
                joystick_reader=reader(), auto_agent_without_joystick=True
            ),
        )
        runtime.start()
        runtime.wait_ready()
        runner = AgentRunner(runtime, events=events, backend_factory=lambda path: model)
        try:
            self.assertEqual(runtime.get_status()["authority_source"], "no_joystick")
            runner.load_model()
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertTrue(runner.submit("起飞"))
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(runtime.get_status()["flight_state"], "hovering")
            self.assertEqual(len(model.calls), 2)
            runner.cancel()
            wait_until(lambda: runtime.get_status()["agent_state"] == "end")
            self.assertTrue(runtime.get_status()["authorized"])
            self.assertTrue(runner.submit("现在状态？"))
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(len(model.calls), 3)
        finally:
            runner.close()
            runtime.close()
