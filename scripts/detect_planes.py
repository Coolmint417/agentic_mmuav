"""Offline example: python -m scripts.detect_planes --map map6."""
import argparse
import json
import os
from pathlib import Path
import threading

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mujoco
import numpy as np
from PIL import Image

from env.mmuav_env import MMUAVEnv
from env.scenes import SCENES
from observations import MuJoCoObservationProvider
from observations.camera import CameraSnapshot
from tools.detect_planes import DetectPlanesConfig, analyze_snapshot


def main():
    parser = argparse.ArgumentParser(description="同步 RGB-D 单帧平面检测；无需加载模型")
    parser.add_argument("--map", default="map6", choices=[s.id for s in SCENES])
    parser.add_argument("--position", nargs=3, type=float, default=[1.05, 2., 3.])
    parser.add_argument("--yaw", type=float, default=0., help="弧度")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--snapshot", help="重算已有 *_snapshot.npz，无需初始化 MuJoCo")
    parser.add_argument("--output", default="artifacts/plane_demo")
    args = parser.parse_args()
    config = DetectPlanesConfig(args.width, args.height, output_directory=args.output)
    if args.snapshot:
        with np.load(args.snapshot, allow_pickle=False) as data:
            snapshot = CameraSnapshot(
                str(data["observation_id"].item()), float(data["sim_time_seconds"]),
                data["rgb"], data["depth_m"], data["K"], data["T_world_optical"],
                float(data["near_m"]), float(data["far_m"]))
    else:
        with MMUAVEnv(auto_agent_without_joystick=True, scene_id=args.map) as env:
            # Offline fixture setup, not a flight tool: pose once, then render
            # without advancing time or invoking an agent/control action.
            env.data.qpos[:3] = args.position
            env.data.qpos[3:7] = [np.cos(args.yaw/2), 0, 0, np.sin(args.yaw/2)]
            env.data.qvel[:] = 0
            mujoco.mj_fwdPosition(env.model, env.data)
            with MuJoCoObservationProvider(env) as provider:
                snapshot, _ = provider.capture_rgbd(args.width, args.height)
    _, result = analyze_snapshot(snapshot, config, threading.Event())
    image_path = Path(result["artifacts"]["annotated_rgb"])
    stem = image_path.with_suffix("")
    Image.fromarray(snapshot.rgb).save(stem.with_name(stem.name + "_rgb").with_suffix(".png"))
    np.savez_compressed(stem.with_name(stem.name + "_snapshot").with_suffix(".npz"),
                        observation_id=snapshot.observation_id,
                        sim_time_seconds=snapshot.sim_time_seconds,
                        rgb=snapshot.rgb, depth_m=snapshot.depth_m, K=snapshot.K,
                        T_world_optical=snapshot.T_world_optical,
                        near_m=snapshot.near_m, far_m=snapshot.far_m)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
