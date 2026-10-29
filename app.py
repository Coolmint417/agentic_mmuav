"""Start the local desktop application: conda run -n mm_planner python app.py."""
import argparse
import os
import sys

# Must be set before importing MuJoCo or Transformers.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main():
    from env.scenes import SCENES
    parser = argparse.ArgumentParser(description="MMUAV PySide6 desktop")
    parser.add_argument(
        "--map", dest="scene_id", choices=[scene.id for scene in SCENES],
        default="map1", help="初始地图",
    )
    parser.add_argument("--model-path", help="本地模型路径/缓存 ID，或 DeepSeek 模型名")
    parser.add_argument(
        "--backend", choices=("ministral", "deepseek"), default="ministral"
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--reasoning-effort", choices=("low", "high", "max"), default="high"
    )
    parser.add_argument("--api-timeout", type=int, default=180)
    parser.add_argument(
        "--no-preview", action="store_true", help="Disable rendering for diagnosis"
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--observation-size", type=int, default=224)
    parser.add_argument(
        "--depth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="向模型添加第一视角深度图",
    )
    parser.add_argument(
        "--depth-size", type=int, default=56, help="模型深度图的正方形边长（8～1024）"
    )
    parser.add_argument(
        "--preview-width", type=int, default=1280, help="界面预览宽度，不影响模型图像"
    )
    parser.add_argument(
        "--preview-height", type=int, default=960, help="界面预览高度，不影响模型图像"
    )
    parser.add_argument(
        "--preview-fps", type=float, default=30, help="界面预览目标帧率，范围 (0, 60]"
    )
    args = parser.parse_args()
    if not 8 <= args.depth_size <= 1024:
        parser.error("depth-size must be in [8, 1024]")
    if args.max_new_tokens is None:
        args.max_new_tokens = 4096 if args.backend == "deepseek" else 384
    if args.max_new_tokens < 1 or args.observation_size < 1 or args.api_timeout < 1:
        parser.error(
            "max-new-tokens, observation-size and api-timeout must be positive"
        )
    if (
        args.preview_width < 1
        or args.preview_height < 1
        or not 0 < args.preview_fps <= 60
    ):
        parser.error(
            "preview dimensions must be positive and preview-fps must be in (0, 60]"
        )
    from PySide6.QtWidgets import QApplication
    from agent.runner import AgentRunner, RunnerConfig
    from runtime import EnvironmentRuntime, RuntimeConfig
    from runtime.events import EventBus
    from env.mmuav_env import MMUAVEnv
    from ui.main_window import AgentWindow

    application = QApplication(sys.argv[:1])
    events = EventBus()
    runtime = EnvironmentRuntime(
        RuntimeConfig(
            preview_enabled=not args.no_preview,
            preview_width=args.preview_width,
            preview_height=args.preview_height,
            preview_fps=args.preview_fps,
            observation_width=args.observation_size,
            observation_height=args.observation_size,
        ),
        events=events,
        environment_factory=lambda: MMUAVEnv(
            auto_agent_without_joystick=True, scene_id=args.scene_id
        ),
    )
    runner = AgentRunner(
        runtime,
        events=events,
        config=RunnerConfig(
            max_new_tokens=args.max_new_tokens,
            include_depth=args.depth,
            depth_size=args.depth_size,
        ),
    )
    window = AgentWindow(runtime, runner, events)
    window.backend_selector.setCurrentIndex(1 if args.backend == "deepseek" else 0)
    window.model_path.setText(
        args.model_path
        or (
            "deepseek-flash"
            if args.backend == "deepseek"
            else "mistralai/Ministral-3-3B-Instruct-2512"
        )
    )
    window.api_url.setText(args.api_base_url)
    window.api_thinking.setChecked(args.thinking)
    window.api_effort.setCurrentText(args.reasoning_effort)
    window.api_timeout.setValue(args.api_timeout)
    # Backend selection may replace the local token default; explicit CLI wins.
    runner.configure(
        RunnerConfig(
            max_new_tokens=args.max_new_tokens,
            include_depth=args.depth,
            depth_size=args.depth_size,
        )
    )
    window._show_config(runner.get_config())
    runtime.start()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
# 起飞后，根据rgb和深度图像，自主规划路径观察前方障碍物，并大致推断出这个障碍物的形状
