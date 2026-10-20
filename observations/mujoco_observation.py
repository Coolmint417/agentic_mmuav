"""MuJoCo position, attitude, flight state, simulation clock and FPV RGB."""
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import mujoco
import numpy as np
from PIL import Image

from .base import ObservationPacket, ObservationProvider
from .depth import depth_image, depth_metadata


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

    def collect(self) -> ObservationPacket:
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
        payload = {
            "type": "uav_observation",
            "schema_version": 1,
            "sim_time_seconds": float(data.time),
            "coordinate_frame": "NWU",
            "position_m": data.qpos[:3].tolist(),
            "attitude": _attitude(data.qpos[3:7]),
            "flight_state": self.environment.flight_state_machine.state.value,
            "rgb": {"camera": self.config.camera_name},
        }
        renderer = self._get_renderer()
        renderer.update_scene(data, camera=self.config.camera_name)
        image = Image.fromarray(renderer.render().copy())
        images = {"rgb": image}
        if self.config.include_depth:
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

    def close(self):
        if not self._closed:
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
            if self._depth_renderer is not None:
                self._depth_renderer.close()
                self._depth_renderer = None
            self._closed = True
