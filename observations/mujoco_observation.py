"""MuJoCo position, attitude, flight state, simulation clock and FPV RGB."""
from dataclasses import dataclass
import math
import uuid
from typing import Any, Callable, Mapping

import mujoco
import numpy as np
from PIL import Image
from env.flight_state_machine import SUCKER_NAMES

from .base import ObservationPacket, ObservationProvider
from .depth import depth_image, depth_metadata
from .camera import CameraSnapshot, camera_intrinsics


@dataclass(frozen=True)
class ObservationConfig:
    width: int = 224
    height: int = 224
    camera_name: str = "fpv_camera"
    include_depth: bool = False
    depth_size: int = 56

    def __post_init__(self):
        for name in ("width", "height", "depth_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.include_depth, bool) or not 8 <= self.depth_size <= 1024:
            raise ValueError(
                "Depth must be enabled/disabled and depth_size must be in [8, 1024]"
            )
        if not isinstance(self.camera_name, str) or not self.camera_name:
            raise ValueError("camera_name must be a nonempty string")


def _attitude(quaternion):
    """MuJoCo free-joint quaternion: body-to-world, wxyz, ZYX Euler angles."""
    quaternion = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(quaternion)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or norm == 0:
        raise ValueError("attitude must be a finite nonzero wxyz quaternion")
    quaternion = quaternion / norm
    w, x, y, z = quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return {
        "quaternion_wxyz": quaternion.tolist(),
        "euler_rpy_rad": [roll, pitch, yaw],
        "rotation_convention": "body_to_world; Rz(yaw) Ry(pitch) Rx(roll)",
    }


class MuJoCoObservationProvider(ObservationProvider):
    """Capture on a private MjData copy; never run the live control callback.

    Extension sensors receive (model, snapshot_data) and return finite JSON
    values. They must treat both inputs as read-only. The provider, not the
    environment, owns its renderer and must be closed before the environment.
    """

    def __init__(
        self,
        environment,
        config: ObservationConfig | None = None,
        *,
        extra_sensors: Mapping[str, Callable[[Any, Any], Any]] | None = None,
    ):
        self.environment = environment
        self.config = config if config is not None else ObservationConfig()
        self.extra_sensors = dict(extra_sensors or {})
        if any(
            not isinstance(name, str) or not name or not callable(sensor)
            for name, sensor in self.extra_sensors.items()
        ):
            raise ValueError("extra_sensors must map nonempty names to callables")
        # Observe checks the environment lifecycle without advancing it.
        environment.observe()
        self._model = environment.model
        if (
            mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera_name
            )
            < 0
        ):
            raise ValueError(f"Camera not found: {self.config.camera_name}")
        self._snapshot = mujoco.MjData(self._model)
        self._renderer = None
        self._depth_renderer = None
        self._geometry_renderer = None
        self._geometry_size = None
        self._closed = False

    def _get_renderer(self):
        if self._renderer is None:
            # Larger configurable frames need a large enough offscreen buffer.
            visual = self._model.vis.global_
            visual.offwidth = max(visual.offwidth, self.config.width)
            visual.offheight = max(visual.offheight, self.config.height)
            self._renderer = mujoco.Renderer(
                self._model, height=self.config.height, width=self.config.width
            )
        return self._renderer

    def _copy_snapshot(self):
        if self._closed:
            raise RuntimeError("Observation provider is closed")
        self.environment.observe()  # Reject a closed environment.
        if self.environment.model is not self._model:
            raise RuntimeError(
                "Environment model changed; create a new observation provider"
            )
        mujoco.mj_copyData(self._snapshot, self._model, self.environment.data)
        data = self._snapshot
        # mj_step integrates qpos after calculating camera/body transforms.
        # Refresh only position-dependent caches on the copy, without mj_forward
        # (which would invoke the global control callback and consume RC input).
        mujoco.mj_fwdPosition(self._model, data)
        return data

    def collect(self, *, include_depth=None) -> ObservationPacket:
        return self._collect_data(self._copy_snapshot(), include_depth=include_depth)

    def _collect_data(self, data, *, include_depth=None, rgb_image=None):
        payload = {
            "type": "uav_observation",
            "observation_id": "obs_" + uuid.uuid4().hex,
            "schema_version": 1,
            "sim_time_seconds": float(data.time),
            "coordinate_frame": "NWU",
            "position_m": data.qpos[:3].tolist(),
            "velocity_m_s": data.qvel[:3].tolist(),
            "angular_velocity_rad_s_body": data.qvel[3:6].tolist(),
            "attitude": _attitude(data.qpos[3:7]),
            "flight_state": self.environment.flight_state_machine.state.value,
            "sucker_states": list(self.environment.flight_state_machine.sucker_states),
            "nominal_specific_thrust_m_s2": self.environment.flight_state_machine.nominal_specific_thrust,
            "sucker_order": list(SUCKER_NAMES),
            "rgb": {"camera": self.config.camera_name},
        }
        if rgb_image is None:
            renderer = self._get_renderer()
            renderer.update_scene(data, camera=self.config.camera_name)
            image = Image.fromarray(renderer.render().copy())
        else:
            image = rgb_image
        images = {"rgb": image}
        want_depth = self.config.include_depth if include_depth is None else include_depth
        if want_depth:
            if self._depth_renderer is None:
                size = self.config.depth_size
                self._model.vis.global_.offwidth = max(
                    self._model.vis.global_.offwidth, size
                )
                self._model.vis.global_.offheight = max(
                    self._model.vis.global_.offheight, size
                )
                self._depth_renderer = mujoco.Renderer(
                    self._model, height=size, width=size
                )
                self._depth_renderer.enable_depth_rendering()
            self._depth_renderer.update_scene(data, camera=self.config.camera_name)
            depth = self._depth_renderer.render().copy()
            payload["depth"] = depth_metadata(
                depth, self._model, self.config.camera_name
            )
            camera_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera_name
            )
            payload["depth"]["fovy_deg"] = float(self._model.cam_fovy[camera_id])
            payload["image_order"] = ["rgb", "depth"]
            images["depth"] = depth_image(depth)
        if self.extra_sensors:
            payload["extras"] = {
                name: sensor(self._model, data)
                for name, sensor in self.extra_sensors.items()
            }
        return ObservationPacket(payload, images)

    def capture_rgbd(self, width=640, height=480):
        """Owner-only metric RGB-D capture, independent of model depth settings.

        Capture RGB, Z-depth, camera pose and vehicle sensors from one copy.
        A 4:3 sensor image is fitted inside the model input bounds without
        changing aspect ratio. No raw depth image is added to this feedback.
        """
        if any(type(v) is not int or not 16 <= v <= 2048 for v in (width, height)):
            raise ValueError("plane capture width/height must be in [16,2048]")
        data = self._copy_snapshot()
        camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera_name)
        if (self._model.cam_projection[camera_id] != mujoco.mjtProjection.mjPROJ_PERSPECTIVE
                or np.any(self._model.cam_sensorsize[camera_id])):
            raise ValueError("plane capture currently requires a perspective fovy camera without sensorsize overrides")
        if self._geometry_size != (width, height):
            if self._geometry_renderer is not None:
                self._geometry_renderer.close()
                self._geometry_renderer = None
            self._model.vis.global_.offwidth = max(self._model.vis.global_.offwidth, width)
            self._model.vis.global_.offheight = max(self._model.vis.global_.offheight, height)
            self._geometry_renderer = mujoco.Renderer(self._model, width=width, height=height)
            self._geometry_size = (width, height)
        renderer = self._geometry_renderer
        renderer.update_scene(data, camera=self.config.camera_name)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        try:
            depth = renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
        image = Image.fromarray(rgb)
        image.thumbnail((self.config.width, self.config.height), Image.Resampling.LANCZOS)
        packet = self._collect_data(data, include_depth=False, rgb_image=image)
        T = np.eye(4)
        # MuJoCo camera: right/up/back. Optical: right/down/forward.
        T[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
        T[:3, 3] = data.cam_xpos[camera_id]
        extent = self._model.stat.extent
        snapshot = CameraSnapshot(
            packet.data["observation_id"], float(data.time), rgb, depth,
            camera_intrinsics(width, height, float(self._model.cam_fovy[camera_id])), T,
            float(self._model.vis.map.znear * extent), float(self._model.vis.map.zfar * extent))
        packet.data["rgb"].update(source_width=width, source_height=height,
                                    resizing="aspect_preserving_fit")
        return snapshot, packet

    def close(self):
        if not self._closed:
            if self._geometry_renderer is not None:
                self._geometry_renderer.close()
                self._geometry_renderer = None
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
            if self._depth_renderer is not None:
                self._depth_renderer.close()
                self._depth_renderer = None
            self._closed = True
