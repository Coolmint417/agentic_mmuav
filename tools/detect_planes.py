"""Read-only agent tool; owner captures, existing executor fits/annotates."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading

from PIL import Image

from agent.state_machine import AgentState
from observations.planes import PlaneDetectionConfig, detect_planes
from ._auto_flight import parse_no_argument_call
from .tasks import ComputationRequest, execute_synchronously


@dataclass(frozen=True)
class DetectPlanesConfig:
    width: int = 640
    height: int = 480
    save_annotated_rgb: bool = True
    output_directory: str = "artifacts/planes"
    detector: PlaneDetectionConfig = field(default_factory=PlaneDetectionConfig)

    def __post_init__(self):
        if any(type(v) is not int or not 16 <= v <= 2048 for v in (self.width, self.height)):
            raise ValueError("plane capture width/height must be in [16,2048]")
        if type(self.save_annotated_rgb) is not bool:
            raise ValueError("save_annotated_rgb must be boolean")


def analyze_snapshot(snapshot, config, cancel_event):
    detection = detect_planes(snapshot, config.detector, cancel_event)
    result = detection.to_dict()
    if cancel_event.is_set():
        raise InterruptedError("plane detection cancelled")
    if config.save_annotated_rgb:
        # All file/PNG work stays on the same geometry worker, never the owner.
        directory = Path(config.output_directory)
        if not directory.is_absolute():
            directory = Path(__file__).resolve().parents[1] / directory
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / snapshot.observation_id
        image_path = stem.with_suffix(".png")
        json_path = stem.with_suffix(".json")
        Image.fromarray(detection.annotated_rgb).save(image_path)
        result["artifacts"] = {"annotated_rgb": str(image_path), "json": str(json_path)}
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return detection, result


class DetectPlanesTool:
    name = "detect_planes"

    def __init__(self, environment, agent_state_machine, *, capture, deliver,
                 config=None, cancel_event=None):
        self.environment = environment
        self.agent = agent_state_machine
        self.capture = capture
        self.deliver = deliver
        self.config = config or DetectPlanesConfig()
        self.cancel_event = cancel_event or threading.Event()

    def _executing(self, turn):
        return (not self.cancel_event.is_set() and self.environment.agent_control_enabled
                and self.agent.turn_id == turn and self.agent.state is AgentState.EXECUTE
                and not self.agent.termination_requested)

    def execute(self, command):
        return execute_synchronously(self.iter_execute(command), self.environment)

    def iter_execute(self, command):
        if self.agent.state is not AgentState.EXECUTE:
            raise ValueError("detect_planes requires agent EXECUTE state")
        turn = self.agent.turn_id
        try:
            parse_no_argument_call(command, self.name)
            self.environment.require_agent_control()
            snapshot, packet = self.capture(self.config.width, self.config.height)
            detection, result = yield ComputationRequest(analyze_snapshot, dict(
                snapshot=snapshot, config=self.config, cancel_event=self.cancel_event))
            self.environment.poll_control_authority()
            if not self._executing(turn):
                return {"tool": self.name, "status": "terminated"}
            image = Image.fromarray(detection.annotated_rgb)
            image = image.resize(packet.images["rgb"].size, Image.Resampling.LANCZOS)
            packet.images["annotated_rgb"] = image
            packet.data["annotated_rgb"] = {
                "observation_id": snapshot.observation_id,
                "image_id": result["visualization"]["image_id"],
                "source_width": self.config.width, "source_height": self.config.height,
            }
            packet.data["image_order"] = list(packet.images)
            packet.data["plane_detection"] = {"observation_id": snapshot.observation_id,
                                               "plane_ids": [p.plane_id for p in detection.planes]}
            self.deliver(turn, packet, detection)
            return result
        except Exception as exc:
            return {"tool": self.name, "status": "terminated" if not self._executing(turn) else "error",
                    "error": str(exc)}
        finally:
            if self._executing(turn):
                self.agent.complete_execute()
