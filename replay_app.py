"""Standalone log replay desktop; never loads a model or calls a model API."""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    from env.scenes import SCENES
    parser = argparse.ArgumentParser(description="MMUAV 日志重放")
    parser.add_argument("log", nargs="?", default="", help="logs/*.jsonl 路径")
    parser.add_argument("--map", dest="scene_id", default="map1",
                        choices=[scene.id for scene in SCENES])
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--preview-height", type=int, default=960)
    args = parser.parse_args()
    if args.preview_width < 1 or args.preview_height < 1:
        parser.error("预览尺寸必须为正整数")
    from PySide6.QtWidgets import QApplication
    from agent.replay_runner import ReplayRunner
    from runtime import EnvironmentRuntime, RuntimeConfig
    from runtime.events import EventBus
    from env.mmuav_env import MMUAVEnv
    from ui.replay_window import ReplayWindow

    application = QApplication(sys.argv[:1])
    events = EventBus()
    runtime = EnvironmentRuntime(
        RuntimeConfig(preview_enabled=not args.no_preview,
                      preview_width=args.preview_width, preview_height=args.preview_height),
        events=events, environment_factory=lambda: MMUAVEnv(
            auto_agent_without_joystick=True, scene_id=args.scene_id))
    runner = ReplayRunner(runtime, events=events)
    window = ReplayWindow(runtime, runner, events, log_path=args.log, scene_id=args.scene_id)
    runtime.start()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
