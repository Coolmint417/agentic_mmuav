"""Owner-thread desktop rendering with a private interactive world camera."""
import math

import mujoco
import numpy as np
from PIL import Image
from observations.depth import depth_image


class WorldCamera:
    actions = {
        "rotate": mujoco.mjtMouse.mjMOUSE_ROTATE_V,
        "rotate_horizontal": mujoco.mjtMouse.mjMOUSE_ROTATE_H,
        "move": mujoco.mjtMouse.mjMOUSE_MOVE_V,
        "move_horizontal": mujoco.mjtMouse.mjMOUSE_MOVE_H,
        "zoom": mujoco.mjtMouse.mjMOUSE_ZOOM,
    }

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.camera = mujoco.MjvCamera()
        self._scene = mujoco.MjvScene(model, maxgeom=1)
        snapshot = mujoco.MjData(model)
        mujoco.mj_copyData(snapshot, model, data)
        mujoco.mj_fwdPosition(model, snapshot)
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "world_camera")
        if index < 0:
            raise ValueError("Camera not found: world_camera")
        eye = snapshot.cam_xpos[index].copy()
        forward = -snapshot.cam_xmat[index].reshape(3, 3)[:, 2]
        # Use the ground intersection as the orbit center when looking down.
        distance = (
            eye[2] / -forward[2]
            if eye[2] > 0 and forward[2] < -0.01
            else max(1.0, float(model.stat.extent))
        )
        self._initial = {
            "lookat": eye + distance * forward,
            "distance": distance,
            "azimuth": math.degrees(math.atan2(forward[1], forward[0])),
            "elevation": math.degrees(math.asin(float(np.clip(forward[2], -1, 1)))),
            "fov": float(model.cam_fovy[index]),
        }
        self.reset()

    def reset(self):
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = self._initial["lookat"]
        for key in ("distance", "azimuth", "elevation"):
            setattr(self.camera, key, self._initial[key])
        self.fov = self._initial["fov"]

    def move(self, action, dx, dy):
        if action not in self.actions or not all(math.isfinite(v) for v in (dx, dy)):
            raise ValueError("Invalid camera movement")
        # MuJoCo's own viewer movement, independent of the rendering context.
        mujoco.mjv_updateCamera(self.model, self.data, self.camera, self._scene)
        mujoco.mjv_moveCamera(
            self.model,
            self.actions[action],
            float(np.clip(dx, -2, 2)),
            float(np.clip(dy, -2, 2)),
            self._scene,
            self.camera,
        )
        self.camera.distance = float(np.clip(self.camera.distance, 0.05, 200))
        self.camera.elevation = float(np.clip(self.camera.elevation, -89.9, 89.9))

    def set_fov(self, value):
        if not math.isfinite(value) or not 5 <= value <= 150:
            raise ValueError("Camera FOV must be between 5 and 150 degrees")
        self.fov = float(value)

    def state(self):
        azimuth, elevation = np.radians([self.camera.azimuth, self.camera.elevation])
        forward = np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ]
        )
        return {
            "position": (self.camera.lookat - self.camera.distance * forward).tolist(),
            "lookat": self.camera.lookat.tolist(),
            "distance": self.camera.distance,
            "azimuth": self.camera.azimuth,
            "elevation": self.camera.elevation,
            "fov": self.fov,
        }


class PreviewRenderer:
    def __init__(self, environment, world_camera, width, height):
        self.environment, self.world_camera = environment, world_camera
        model = environment.model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self._snapshot = mujoco.MjData(model)
        self._renderer = mujoco.Renderer(model, width=width, height=height)

    def render(self, camera_name):
        model = self.environment.model
        mujoco.mj_copyData(self._snapshot, model, self.environment.data)
        mujoco.mj_fwdPosition(model, self._snapshot)
        world = camera_name == "world_camera"
        depth = camera_name == "depth_camera"
        self._renderer.update_scene(
            self._snapshot,
            camera=self.world_camera.camera
            if world
            else ("fpv_camera" if depth else camera_name),
        )
        if world:
            # Override only this renderer's frustum: never change model FOV/FPV.
            for camera in self._renderer.scene.camera:
                half_height = camera.frustum_near * math.tan(
                    math.radians(self.world_camera.fov) / 2
                )
                camera.frustum_top = half_height
                camera.frustum_bottom = -half_height
                camera.frustum_width = 0
                camera.frustum_center = 0
        if depth:
            self._renderer.enable_depth_rendering()
            try:
                return depth_image(self._renderer.render().copy())
            finally:
                self._renderer.disable_depth_rendering()
        return Image.fromarray(self._renderer.render().copy())

    def close(self):
        self._renderer.close()
