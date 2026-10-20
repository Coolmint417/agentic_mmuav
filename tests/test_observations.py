"""Observation serialization, provider replacement, rendering and lifecycle."""
import base64
from io import BytesIO
import json
import math
import unittest

import mujoco
import numpy as np
from PIL import Image

from agent.state_machine import AgentState, ThinkOutcome
from env.flight_state_machine import FlightState, FlightStateMachine
from env.mmuav_env import MMUAVEnv
from observations import (
    MuJoCoObservationProvider,
    ObservationCancelled,
    ObservationConfig,
    ObservationPacket,
    ObservationProvider,
    collect_for_agent,
)
from tests.support import FakeJoystick


class ReplacementProvider(ObservationProvider):
    def __init__(self, callback=None):
        self.callback = callback or (lambda: None)

    def collect(self):
        self.callback()
        return ObservationPacket(
            {"type": "test_observation", "point_cloud": [[1.0, 2.0, 3.0]]}
        )


def begin_observation(env):
    agent = env.agent_state_machine
    agent.start_turn()
    agent.complete_think(ThinkOutcome.TOOL)
    agent.complete_execute()
    return agent


class PacketTests(unittest.TestCase):
    def test_json_png_round_trip_and_model_inputs(self):
        source = Image.new("RGB", (16, 12), (12, 34, 56))
        data = {"position_m": [1.0, 2.0, 3.0], "rgb": {"camera": "fpv_camera"}}
        packet = ObservationPacket(data, {"rgb": source})
        source.putpixel((0, 0), (0, 0, 0))
        data["position_m"][0] = 99
        payload = json.loads(packet.to_json())
        self.assertEqual(payload["position_m"], [1.0, 2.0, 3.0])
        rgb = payload["rgb"]
        with Image.open(
            BytesIO(base64.b64decode(rgb["data"], validate=True))
        ) as decoded:
            self.assertEqual(decoded.size, (16, 12))
            self.assertEqual(decoded.mode, "RGB")
            self.assertEqual(decoded.getpixel((0, 0)), (12, 34, 56))
        model_inputs = packet.to_model_inputs()
        prompt = json.loads(model_inputs["prompt"])
        self.assertNotIn("data", prompt["rgb"])
        self.assertEqual(prompt["rgb"]["encoding"], "attachment")
        self.assertEqual(model_inputs["image"].getpixel((0, 0)), (12, 34, 56))
        model_inputs["image"].putpixel((0, 0), (1, 1, 1))
        self.assertEqual(packet.images["rgb"].getpixel((0, 0)), (12, 34, 56))

    def test_invalid_json_data_and_multi_image_model_adapter(self):
        for value in (float("nan"), float("inf"), np.zeros(3)):
            with self.assertRaises((ValueError, TypeError)):
                ObservationPacket({"sensor": value})
        packet = ObservationPacket(
            {}, {"a": Image.new("RGB", (2, 2)), "b": Image.new("RGB", (2, 2))}
        )
        self.assertEqual(len(packet.to_dict()), 2)
        inputs = packet.to_model_inputs()
        self.assertEqual(inputs["image"].size, (2, 2))
        self.assertEqual(list(inputs["extra_images"]), ["b"])


class MuJoCoObservationTests(unittest.TestCase):
    def test_default_rgb_pose_time_state_and_no_simulation_side_effect(self):
        remote = FakeJoystick()
        with MMUAVEnv(
            FlightStateMachine(FlightState.HOVERING), joystick_reader=remote
        ) as env:
            env.data.qpos[:3] = [-2.0, 2.0, 1.5]
            yaw = math.pi / 2
            env.data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            # Deliberately do not mj_forward: RGB must use current qpos, not
            # stale camera caches, and must not invoke the control callback.
            env.data.time = 12.5
            original = {
                key: getattr(env.data, key).copy()
                for key in ("qpos", "qvel", "ctrl", "sensordata")
            }
            remote.reads.clear()
            with MuJoCoObservationProvider(env) as observer:
                packet = observer.collect()
                payload = packet.to_dict(embed_images=False)
                self.assertEqual(payload["position_m"], [-2.0, 2.0, 1.5])
                self.assertEqual(payload["flight_state"], "hovering")
                self.assertEqual(payload["sim_time_seconds"], 12.5)
                self.assertEqual(payload["coordinate_frame"], "NWU")
                np.testing.assert_allclose(
                    payload["attitude"]["quaternion_wxyz"], env.data.qpos[3:7]
                )
                np.testing.assert_allclose(
                    payload["attitude"]["euler_rpy_rad"], [0.0, 0.0, yaw], atol=1e-10
                )
                self.assertEqual(packet.images["rgb"].size, (224, 224))
                self.assertEqual(packet.images["rgb"].mode, "RGB")
                self.assertGreater(np.asarray(packet.images["rgb"]).std(), 1.0)
                # The rendered image must match an independently refreshed
                # position cache for the same pose and camera.
                fresh = mujoco.MjData(env.model)
                mujoco.mj_copyData(fresh, env.model, env.data)
                mujoco.mj_fwdPosition(env.model, fresh)
                with mujoco.Renderer(env.model, height=224, width=224) as renderer:
                    renderer.update_scene(fresh, camera="fpv_camera")
                    np.testing.assert_equal(
                        np.asarray(packet.images["rgb"]), renderer.render()
                    )
            self.assertEqual(remote.reads, [])
            self.assertEqual(env.data.time, 12.5)
            for key, value in original.items():
                np.testing.assert_equal(getattr(env.data, key), value)
            with self.assertRaises(RuntimeError):
                observer.collect()

    def test_configurable_size_extensions_and_clock_since_env_start(self):
        with MMUAVEnv(joystick_reader=FakeJoystick(False)) as env:
            for _ in range(3):
                env.step()
            config = ObservationConfig(width=320, height=512)
            with MuJoCoObservationProvider(
                env,
                config,
                extra_sensors={
                    "point_cloud": lambda model, data: {
                        "frame": "NWU",
                        "points_m": data.qpos[:3].reshape(1, 3).tolist(),
                    }
                },
            ) as observer:
                packet = observer.collect()
                payload = json.loads(packet.to_json())
                self.assertEqual(packet.images["rgb"].size, (320, 512))
                self.assertEqual(payload["flight_state"], "on_ground")
                self.assertAlmostEqual(
                    payload["sim_time_seconds"], 3 * env.model.opt.timestep
                )
                self.assertEqual(
                    payload["extras"]["point_cloud"]["points_m"],
                    [payload["position_m"]],
                )
                env.step()
                self.assertLess(
                    packet.data["sim_time_seconds"],
                    observer.collect().data["sim_time_seconds"],
                )

    def test_invalid_config_camera_extension_and_closed_environment(self):
        for kwargs in (
            {"width": 0},
            {"height": -1},
            {"width": True},
            {"height": 2.5},
            {"camera_name": ""},
        ):
            with self.assertRaises(ValueError):
                ObservationConfig(**kwargs)
        with MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            with self.assertRaises(ValueError):
                MuJoCoObservationProvider(env, ObservationConfig(camera_name="missing"))
            with self.assertRaises(ValueError):
                MuJoCoObservationProvider(env, extra_sensors={"bad": 1})
            provider = MuJoCoObservationProvider(env)
        with self.assertRaises(RuntimeError):
            provider.collect()
        provider.close()


class ObservationLifecycleTests(unittest.TestCase):
    def test_replaceable_provider_advances_observation_only_after_success(self):
        with MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            agent = begin_observation(env)
            packet = collect_for_agent(ReplacementProvider(), env)
            self.assertEqual(agent.state, AgentState.THINK)
            self.assertEqual(
                json.loads(packet.to_json())["point_cloud"], [[1.0, 2.0, 3.0]]
            )
            self.assertEqual(set(packet.to_model_inputs()), {"prompt"})
            with self.assertRaises(ValueError):
                collect_for_agent(ReplacementProvider(), env)

    def test_sensor_failure_preserves_observation(self):
        with MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            agent = begin_observation(env)

            def fail():
                raise RuntimeError("sensor unavailable")

            with self.assertRaises(RuntimeError):
                collect_for_agent(ReplacementProvider(fail), env)
            self.assertEqual(agent.state, AgentState.OBSERVATION)

    def test_axis4_cancellation_discards_packet(self):
        remote = FakeJoystick()
        with MMUAVEnv(joystick_reader=remote) as env:
            agent = begin_observation(env)

            def revoke():
                remote.axes[4] = 0.0

            with self.assertRaises(ObservationCancelled):
                collect_for_agent(ReplacementProvider(revoke), env)
            self.assertEqual(agent.state, AgentState.END)
            self.assertFalse(agent.enabled)
            with self.assertRaises(PermissionError):
                collect_for_agent(ReplacementProvider(), env)

    def test_old_packet_cannot_advance_new_turn(self):
        with MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            agent = begin_observation(env)

            def new_turn():
                agent.terminate("new turn")
                begin_observation(env)

            with self.assertRaises(ObservationCancelled):
                collect_for_agent(ReplacementProvider(new_turn), env)
            self.assertEqual(agent.state, AgentState.OBSERVATION)


if __name__ == "__main__":
    unittest.main()
