"""Scene catalog shared by CLI, desktop UI and simulation runtime."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scene:
    id: str
    label: str
    filename: str

    @property
    def path(self):
        return Path(__file__).parent / self.filename


SCENES = (
    Scene("map1", "地图 1 · 砖墙转角", "scene_map1.xml"),
    Scene("map2", "地图 2 · 混合材质柱阵", "scene_map2.xml"),
    Scene("map3", "地图 3 · 错位走廊", "scene_map3.xml"),
    Scene("map4", "地图 4 · 双门洞", "scene_map4.xml"),
    Scene("map5", "地图 5 · 平台庭院", "scene_map5.xml"),
)


def get_scene(scene_id):
    for scene in SCENES:
        if scene.id == scene_id:
            return scene
    raise ValueError(f"未知地图：{scene_id}")


def load_scene_model(scene_id):
    """Compile before replacing the live environment; restore its callback on error."""
    import mujoco

    path = get_scene(scene_id).path
    callback = mujoco.get_mjcb_control()
    try:
        mujoco.set_mjcb_control(None)
        return mujoco.MjModel.from_xml_path(str(path))
    finally:
        mujoco.set_mjcb_control(callback)
