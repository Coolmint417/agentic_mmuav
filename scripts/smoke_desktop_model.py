"""Real cached-model runner smoke test, with explicitly simulated Axis4 authorization."""
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.runner import AgentRunner, RunnerConfig
from env.mmuav_env import MMUAVEnv
from runtime import EnvironmentRuntime, RuntimeConfig
from runtime.events import EventBus
from tests.support import FakeJoystick


def wait(condition, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise TimeoutError("Desktop model smoke test timed out")


def main():
    events, captured = EventBus(), []

    def record(event):
        if event.kind in {"chat", "model_output"}:
            captured.append({"kind": event.kind, "data": event.data})
            print(event.kind, event.data, flush=True)

    events.subscribe(record)
    runtime = EnvironmentRuntime(
        RuntimeConfig(preview_enabled=False),
        events=events,
        environment_factory=lambda: MMUAVEnv(joystick_reader=FakeJoystick(True)),
    )
    runtime.start()
    runtime.wait_ready()
    runner = AgentRunner(
        runtime, events=events, config=RunnerConfig(max_new_tokens=128, max_decisions=3)
    )
    output = Path("artifacts/desktop_model_smoke.json")
    report = {"simulated_remote": True, "events": captured}
    try:
        runner.load_model()
        wait(lambda: not runner.get_status()["busy"], 180)
        if runner.get_status()["model_state"] != "ready":
            raise RuntimeError("Real model failed to load")
        start_time = runtime.get_status()["sim_time"]
        runner.submit("不要调用任何工具。请通过 final 字段回复“模型已就绪”，并简要报告观测中的当前位置。")
        wait(lambda: not runner.get_status()["busy"], 240)
        report["status"] = runtime.get_status()
        report["simulation_advanced_during_inference"] = (
            report["status"]["sim_time"] > start_time
        )
        report["passed"] = (
            any(
                e["kind"] == "chat" and e["data"]["role"] == "assistant"
                for e in captured
            )
            and not any(
                e["kind"] == "chat" and e["data"]["role"] == "error" for e in captured
            )
            and report["status"]["agent_state"] == "end"
        )
    finally:
        runner.close()
        runtime.close()
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
