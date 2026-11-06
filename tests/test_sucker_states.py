"""Adhesion command state is independent of flight state and authority."""
import unittest

import numpy as np

from env import mmuav_env as control
from env.flight_state_machine import ControlMode, FlightState, FlightStateMachine, SUCKER_NAMES
from observations import MuJoCoObservationProvider
from tests.support import FakeJoystick


class SuckerStateTests(unittest.TestCase):
    def test_runtime_accepts_commands_without_an_agent_turn_or_authority(self):
        from tests.test_runner import ScriptedModel, running_system
        model = ScriptedModel([])
        with running_system(model, authorized=False) as (runtime, runner, remote, events):
            values = [True, False, True, False]
            result = runtime.set_sucker_states(values).result(timeout=5)
            self.assertEqual(result, tuple(values))
            self.assertEqual(runtime.get_status()["sucker_states"], values)
            self.assertFalse(runtime.get_status()["authorized"])
            self.assertEqual(runtime.get_status()["flight_state"], "on_ground")
            self.assertEqual(model.calls, [])

    def test_independent_commands_in_every_flight_state_and_mode(self):
        for mode in ControlMode:
            for state in FlightState:
                if mode is ControlMode.MANUAL and state is FlightState.FLY_AGENT:
                    continue
                with self.subTest(state=state, mode=mode):
                    machine = FlightStateMachine(state, control_mode=mode)
                    values = [True, False, True, False]
                    self.assertEqual(machine.set_sucker_states(values), tuple(values))
                    values[0] = False
                    self.assertEqual(machine.sucker_states, (True, False, True, False))
                    machine.set_sucker_state(1, True)
                    self.assertEqual(machine.sucker_states, (True, True, True, False))
                    self.assertIs(machine.state, state)
                    self.assertIs(machine.control_mode, mode)
                    with self.assertRaises(TypeError):
                        machine.sucker_states[0] = False

    def test_validation_is_atomic_and_authority_does_not_reset_commands(self):
        machine = FlightStateMachine(sucker_states=[True] * 4)
        for bad in (None, [], [True] * 3, [True] * 5, [1, 0, 1, 0], "1111"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                machine.set_sucker_states(bad)
            self.assertEqual(machine.sucker_states, (True,) * 4)
        for index, enabled in ((-1, True), (4, True), (True, True), (0, 1)):
            with self.assertRaises(ValueError):
                machine.set_sucker_state(index, enabled)
            self.assertEqual(machine.sucker_states, (True,) * 4)
        machine.set_agent_authorized(True)
        machine.set_agent_authorized(False)
        self.assertEqual(machine.sucker_states, (True,) * 4)
        self.assertIs(machine.state, FlightState.ON_GROUND)
        self.assertFalse(hasattr(FlightState, "ADHERED"))

    def test_direct_fsm_and_environment_commands_reach_actuators_and_observations(self):
        remote = FakeJoystick(agent_enabled=False)
        with control.MMUAVEnv(joystick_reader=remote) as env:
            values = (True, False, False, True)
            env.flight_state_machine.set_sucker_states(values)
            env.step()  # No tool or authorization is needed for adhesion commands.
            np.testing.assert_array_equal(env.data.ctrl[1:5], values)
            self.assertIs(env.flight_state_machine.state, FlightState.ON_GROUND)
            with MuJoCoObservationProvider(env) as provider:
                packet = provider.collect()
                self.assertEqual(packet.data["sucker_states"], list(values))
                self.assertEqual(packet.data["sucker_order"], list(SUCKER_NAMES))
            env.flight_state_machine.transition_to(FlightState.HOVERING)
            env.set_sucker_states((False, True, True, False))
            self.assertEqual(packet.data["sucker_states"], list(values))
            np.testing.assert_array_equal(env.data.ctrl[1:5], (False, True, True, False))
            env.flight_state_machine.set_sucker_state(0, True)
            env.step()
            np.testing.assert_array_equal(env.data.ctrl[1:5], (True, True, True, False))
            remote.axes[4] = 1.
            env.step()
            self.assertTrue(env.agent_control_enabled)
            env.set_sucker_states((False,) * 4)
            np.testing.assert_array_equal(env.data.ctrl[1:5], 0.)
            self.assertIs(env.flight_state_machine.state, FlightState.HOVERING)

    def test_axis6_gait_updates_the_same_command_vector(self):
        remote = FakeJoystick(agent_enabled=False)
        with control.MMUAVEnv(joystick_reader=remote) as env:
            remote.axes[6] = 1.
            seen = set()
            for _ in range(1300):
                env.step()
                values = env.flight_state_machine.sucker_states
                np.testing.assert_array_equal(env.data.ctrl[1:5], values)
                seen.add(values)
            self.assertIn((True, False, True, False), seen)
            self.assertIn((False, True, False, True), seen)
            remote.axes[6] = 0.
            for _ in range(4000):
                env.step()
                if env.flight_state_machine.state is FlightState.ON_GROUND:
                    break
            self.assertIs(env.flight_state_machine.state, FlightState.ON_GROUND)
            self.assertEqual(env.flight_state_machine.sucker_states, (True,) * 4)
            remote.axes[7] = 1.
            env.step()
            self.assertEqual(env.flight_state_machine.sucker_states, (False,) * 4)


if __name__ == "__main__":
    unittest.main()
