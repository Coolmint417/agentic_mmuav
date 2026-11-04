"""Manual crawling edges, stopping choreography and authority handover."""
import unittest

import numpy as np

from env import mmuav_env as control
from env.flight_state_machine import FlightState, FlightStateMachine
from env.mmuav_env import MMUAVEnv
from tests.support import FakeJoystick


class CrawlTests(unittest.TestCase):
    def test_contact_friction_follows_attachment_across_diagonal_switch(self):
        remote = FakeJoystick(agent_enabled=False)
        with MMUAVEnv(joystick_reader=remote) as env:
            remote.axes[6] = 1.0
            seen = set()
            for duration in (.2, .8, .8):
                self.advance(env, duration)
                for contact in env.data.contact:
                    for cup_id, surface_id in ((contact.geom1, contact.geom2),
                                               (contact.geom2, contact.geom1)):
                        for index, name in enumerate(control.SUCKER_GEOMS, start=1):
                            if (cup_id != env.model.geom(name).id
                                    or env.model.geom_bodyid[surface_id] != 0):
                                continue
                            attached = bool(env.data.ctrl[index])
                            expected = (control.CRAWL_ATTACHED_FRICTION if attached
                                        else control.CRAWL_RELEASE_FRICTION)
                            self.assertAlmostEqual(float(contact.friction[0]), expected)
                            seen.add(attached)
            self.assertEqual(seen, {False, True})

    def advance(self, env, seconds):
        until = env.data.time + seconds
        while env.data.time < until:
            env.step()

    def stop_and_check(self, env):
        deadline = env.data.time + 6.0
        while env.flight_state_machine.state is FlightState.CRAWLING and env.data.time < deadline:
            env.step()
            np.testing.assert_array_equal(env.data.ctrl[5:9], 0.0)
        self.assertIs(env.flight_state_machine.state, FlightState.ON_GROUND)
        self.assertEqual(env.data.ctrl[0], 0.0)
        self.assertLess(abs(float(env.data.joint("joint_upper_link").qpos[0])), .03)
        np.testing.assert_array_equal(env.data.ctrl[1:5], 1.0)
        for name in control.SUCKER_GEOMS:
            self.assertEqual(env.model.geom(name).contype, 1)
            self.assertEqual(env.model.geom(name).conaffinity, 1)
        self.assertTrue(np.isfinite(env.data.qpos).all())

    def test_stop_in_each_stage_and_restart_after_idle(self):
        for stage in range(4):
            with self.subTest(stage=stage):
                remote = FakeJoystick(agent_enabled=False)
                with MMUAVEnv(joystick_reader=remote) as env:
                    self.advance(env, 1.0)  # Start time belongs to this crawl, not app startup.
                    remote.axes[6] = 1.0
                    env.step()
                    self.assertIs(env.flight_state_machine.state, FlightState.CRAWLING)
                    self.advance(env, stage * control.CRAWL_STAGE_DURATION + .1)
                    self.assertEqual(control.crawl_stage, stage)
                    remote.axes[6] = 0.0
                    env.step()
                    self.assertIs(env.flight_state_machine.state, FlightState.CRAWLING)
                    remote.axes[7] = 1.0  # No takeoff while stopping.
                    self.stop_and_check(env)
                    self.assertEqual(env.get_position_auto_mode(), control.POSITION_AUTO_NONE)
                    self.advance(env, .1)
                    self.assertIs(env.flight_state_machine.state, FlightState.ON_GROUND)
                    remote.axes[6] = 1.0
                    env.step()
                    self.assertEqual(control.crawl_stage, 0)
                    self.assertFalse(control.crawl_stop_requested)
                    remote.axes[6] = -1.0
                    self.stop_and_check(env)
                    remote.axes[7] = -1.0
                    env.step()
                    remote.axes[7] = 1.0
                    env.step()
                    self.assertEqual(env.get_position_auto_mode(), control.POSITION_AUTO_TAKEOFF)
                    np.testing.assert_array_equal(env.data.ctrl[1:5], 0.0)

    def test_axis4_waits_for_stopping_and_masks_axis6_after_grant(self):
        remote = FakeJoystick(agent_enabled=False)
        with MMUAVEnv(joystick_reader=remote) as env:
            remote.axes[6] = 1.0
            env.step()
            self.advance(env, 1.0)
            remote.axes[4] = 1.0
            env.poll_control_authority()
            self.assertFalse(env.agent_control_enabled)
            self.assertFalse(env.agent_state_machine.enabled)
            self.stop_and_check(env)
            env.step()
            self.assertTrue(env.agent_control_enabled)
            self.assertTrue(env.agent_state_machine.enabled)
            remote.reads.clear()
            self.advance(env, .1)
            self.assertEqual(set(remote.reads), {4})
            remote.axes[4] = 0.0
            env.step()
            # Revocation follows existing hover policy. A held Axis6 must not
            # become a new rising edge after returning to ground.
            env.flight_state_machine.transition_to(FlightState.ON_GROUND)
            env.step()
            self.assertIs(env.flight_state_machine.state, FlightState.ON_GROUND)
            remote.axes[6] = 0.0
            env.step()
            remote.axes[6] = .2
            env.step()
            self.assertIs(env.flight_state_machine.state, FlightState.CRAWLING)

    def test_held_switch_in_hover_does_not_start_after_landing(self):
        remote = FakeJoystick(agent_enabled=False)
        with MMUAVEnv(FlightStateMachine(FlightState.HOVERING), joystick_reader=remote) as env:
            remote.axes[6] = 1.0
            env.step()
            self.assertIs(env.flight_state_machine.state, FlightState.HOVERING)
            env.flight_state_machine.transition_to(FlightState.ON_GROUND)
            env.step()
            self.assertIs(env.flight_state_machine.state, FlightState.ON_GROUND)


if __name__ == "__main__":
    unittest.main()
