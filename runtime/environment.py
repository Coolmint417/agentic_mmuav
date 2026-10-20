"""Single-owner simulation thread; runners never call env.step()."""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import queue
import threading
import time

from agent.state_machine import AgentState, ThinkOutcome
from env.flight_state_machine import FlightState
from env.mmuav_env import MMUAVEnv, POSITION_AUTO_NONE
from observations import MuJoCoObservationProvider, ObservationConfig, collect_for_agent
from tools import FlyWaypointTool, LandTool, TakeoffTool
from tools.tasks import PlanningRequest
from .events import EventBus, describe_error
from .preview import PreviewRenderer, WorldCamera


@dataclass(frozen=True)
class RuntimeConfig:
    realtime: bool = True
    preview_enabled: bool = True
    preview_fps: float = 30.0
    preview_width: int = 1280
    preview_height: int = 960
    observation_width: int = 224
    observation_height: int = 224

    def __post_init__(self):
        if not 0 < self.preview_fps <= 60:
            raise ValueError("preview_fps must be in (0, 60]")
        ObservationConfig(self.preview_width, self.preview_height)
        ObservationConfig(self.observation_width, self.observation_height)


class EnvironmentRuntime:
    """Own environment, state machines, render contexts and cooperative tools.

    Public operations are marshalled to the owner thread. Trajectory solving
    uses a separate worker so RC polling and physical control remain active.
    Only this service advances the simulation in the interactive application.
    """

    def __init__(self, config=None, *, events=None, environment_factory=MMUAVEnv):
        self.config = config or RuntimeConfig()
        self.events = events or EventBus()
        self._factory = environment_factory
        self._commands = queue.Queue()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._status = {"ready": False, "authorized": False, "error": None}
        self._frame = None
        self._camera = "fpv_camera"
        self._cancel_callback = None
        self._active = None
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("EnvironmentRuntime can only be started once")
        self._thread = threading.Thread(
            target=self._run, name="environment-runtime", daemon=True
        )
        self._thread.start()

    def wait_ready(self, timeout=15):
        if not self._ready.wait(timeout):
            raise TimeoutError("Environment initialization timed out")
        if self.get_status().get("error"):
            raise RuntimeError(self.get_status()["error"])

    def get_status(self):
        with self._lock:
            return dict(self._status)

    def get_preview(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def set_cancel_callback(self, callback):
        self._cancel_callback = callback

    def _post(self, function):
        if self._stop.is_set() or self._thread is None or not self._thread.is_alive():
            raise RuntimeError("Environment runtime is not running")
        future = Future()
        self._commands.put((function, future))
        return future

    def _invoke(self, function, timeout=15):
        future = self._post(function)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()  # Do not execute a delayed request after its caller gives up.
            raise

    def _validate_turn(self, turn_id):
        self._env.require_agent_control()
        agent = self._env.agent_state_machine
        if (
            agent.turn_id != turn_id
            or agent.termination_requested
            or agent.state is AgentState.END
        ):
            raise RuntimeError("Agent turn was cancelled or replaced")

    def begin_turn(self, *, include_depth=False, depth_size=56):
        def begin():
            self._env.require_agent_control()  # Read Axis4 before any prompt is sent.
            config = replace(
                self._observer.config,
                include_depth=include_depth,
                depth_size=depth_size,
            )
            if config != self._observer.config:
                new_observer = MuJoCoObservationProvider(self._env, config)
                self._observer.close()
                self._observer = new_observer
            agent = self._env.agent_state_machine
            agent.start_turn()
            return agent.turn_id, self._observer.collect()

        return self._invoke(begin)

    def validate_turn(self, turn_id):
        return self._invoke(lambda: self._validate_turn(turn_id))

    def finish_turn(self, turn_id):
        def finish():
            self._validate_turn(turn_id)
            self._env.agent_state_machine.complete_think(ThinkOutcome.FINAL)
            self._publish_snapshot()

        return self._invoke(finish)

    def collect_observation(self, turn_id):
        def collect():
            self._validate_turn(turn_id)
            return collect_for_agent(self._observer, self._env)

        return self._invoke(collect)

    def execute_tool(self, command, turn_id):
        completion = Future()

        def start():
            self._validate_turn(turn_id)
            if self._active is not None:
                raise RuntimeError("Another tool is running")
            registry = {
                tool.name: tool for tool in (TakeoffTool, LandTool, FlyWaypointTool)
            }
            Tool = registry[command["tool"]]
            tool = Tool(self._env, self._env.agent_state_machine)
            self._env.agent_state_machine.complete_think(ThinkOutcome.TOOL)
            self._active = {
                "task": tool.iter_execute(command),
                "completion": completion,
                "planning": None,
            }

        self._invoke(start)
        return completion

    def cancel_turn(self, reason="用户终止任务"):
        # Nonblocking: a busy UI/model thread must never block RC handover.
        if (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop.is_set()
        ):
            future = Future()
            self._commands.put((lambda: self._cancel_now(reason), future))

    def set_camera(self, camera):
        if camera not in {"fpv_camera", "world_camera", "depth_camera"}:
            raise ValueError("Unsupported preview camera")
        return self._post(lambda: self._change_camera(camera))

    def _change_camera(self, camera):
        self._camera = camera
        with self._lock:
            self._frame = None
        self._publish_snapshot()

    def move_world_camera(self, action, dx, dy):
        def move():
            if self._camera == "world_camera":
                self._world_camera.move(action, dx, dy)
                self._publish_snapshot()

        return self._post(move)

    def set_world_camera_fov(self, value):
        def change():
            self._world_camera.set_fov(value)
            self._publish_snapshot()

        return self._post(change)

    def reset_world_camera(self):
        def reset():
            self._world_camera.reset()
            self._publish_snapshot()

        return self._post(reset)

    def _cancel_task(self, reason):
        active, self._active = self._active, None
        if active is not None:
            if active["planning"] is not None:
                active["planning"].cancel()
            # A require_agent_control() inside the generator may detect the
            # RC edge reentrantly. Do not close a currently executing generator.
            if not active["task"].gi_running:
                active["task"].close()
            if not active["completion"].done():
                active["completion"].set_result(
                    {"status": "terminated", "reason": reason}
                )

    def _cancel_now(self, reason):
        agent = self._env.agent_state_machine
        was_moving = (
            self._env.flight_state_machine.state is FlightState.FLY_AGENT
            or self._env.get_position_auto_mode() != POSITION_AUTO_NONE
        )
        agent.terminate(reason)
        if self._env.agent_control_enabled:
            self._env.hold_position()
            if was_moving:
                self._env.flight_state_machine.transition_to(FlightState.HOVERING)
        self._cancel_task(reason)
        self._publish_snapshot()

    def _authority_revoked(self):
        self._cancel_task("Axis4 已撤销 agent 授权")
        self._publish_snapshot()
        if self._cancel_callback is not None:
            self._cancel_callback("Axis4 已撤销 agent 授权，已恢复手动控制。")

    def _advance_task(self):
        active = self._active
        if active is None:
            return
        try:
            planning = active["planning"]
            if planning is not None:
                if not planning.done():
                    return
                active["planning"] = None
                try:
                    result = planning.result()
                except Exception as exc:
                    request = active["task"].throw(exc)
                else:
                    request = active["task"].send(result)
            else:
                request = next(active["task"])
            if self._active is not active:
                active["task"].close()
                return
            if isinstance(request, PlanningRequest):
                active["planning"] = self._planner.submit(request.compute)
        except StopIteration as done:
            if self._active is active:
                self._active = None
            if not active["completion"].done():
                active["completion"].set_result(done.value)
        except Exception as exc:
            self._active = None
            if not active["completion"].done():
                active["completion"].set_exception(exc)
            self._cancel_now(str(exc))

    def _publish_snapshot(self):
        state = self._env.observe()
        agent = self._env.agent_state_machine
        status = {
            "ready": True,
            "error": None,
            "authorized": self._env.agent_control_enabled,
            "authority_source": self._env.control_authority_source,
            "agent_state": agent.state.value,
            "flight_state": self._env.flight_state_machine.state.value,
            "control_mode": self._env.flight_state_machine.control_mode.value,
            "sim_time": state["time"],
            "position": state["position"].tolist(),
            "yaw": state["yaw"],
            "camera": self._camera,
            "world_camera": self._world_camera.state(),
        }
        with self._lock:
            self._status = status

    def _render_preview(self):
        if not self.config.preview_enabled:
            return
        try:
            if self._preview is None:
                self._preview = PreviewRenderer(
                    self._env,
                    self._world_camera,
                    self.config.preview_width,
                    self.config.preview_height,
                )
            image = self._preview.render(self._camera)
            with self._lock:
                self._frame = image
        except Exception as exc:
            self.events.publish(
                "chat", {"role": "error", "text": "预览失败：" + describe_error(exc)}
            )
            self.config = RuntimeConfig(
                **{**vars(self.config), "preview_enabled": False}
            )

    def _run(self):
        self._env = self._observer = self._preview = None
        self._planner = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="trajectory-planner"
        )
        try:
            self._env = self._factory()
            self._world_camera = WorldCamera(self._env.model, self._env.data)
            self._env.set_model_stop_callback(self._authority_revoked)
            self._observer = MuJoCoObservationProvider(
                self._env,
                ObservationConfig(
                    width=self.config.observation_width,
                    height=self.config.observation_height,
                ),
            )
            self._publish_snapshot()
            self._ready.set()
            deadline = time.monotonic()
            next_publish = next_preview = 0.0
            while not self._stop.is_set():
                for _ in range(32):
                    try:
                        function, future = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    if future.set_running_or_notify_cancel():
                        try:
                            future.set_result(function())
                        except Exception as exc:
                            future.set_exception(exc)
                self._env.step()
                self._advance_task()
                now = time.monotonic()
                if now >= next_publish:
                    self._publish_snapshot()
                    next_publish = now + 0.05
                if now >= next_preview:
                    self._render_preview()
                    next_preview = now + 1 / self.config.preview_fps
                if self.config.realtime:
                    deadline += float(self._env.model.opt.timestep)
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        self._stop.wait(delay)
                    elif delay < -0.1:
                        deadline = (
                            time.monotonic()
                        )  # Avoid catch-up bursts after rendering.
        except Exception as exc:
            error = describe_error(exc)
            with self._lock:
                self._status.update(ready=False, error=error)
            self.events.publish("chat", {"role": "error", "text": "环境运行错误：" + error})
            if self._cancel_callback is not None:
                self._cancel_callback("环境运行错误，本轮已终止。")
        finally:
            self._stop.set()
            self._ready.set()
            self._cancel_task("环境已关闭")
            while True:
                try:
                    _, future = self._commands.get_nowait()
                except queue.Empty:
                    break
                if not future.done():
                    future.set_exception(RuntimeError("Environment closed"))
            for resource in (self._preview, self._observer, self._env):
                if resource is not None:
                    resource.close()
            self._planner.shutdown(wait=False, cancel_futures=True)
            with self._lock:
                self._status["ready"] = False

    def close(self, *, wait=True):
        self._stop.set()
        if wait and self._thread is not None:
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                raise TimeoutError("Environment did not stop")
