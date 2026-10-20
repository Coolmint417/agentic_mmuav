"""Replaceable observation contract and JSON/multimodal transport."""
from abc import ABC, abstractmethod
import base64
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
import json
from typing import Any

from PIL import Image

from agent.state_machine import AgentState


@dataclass
class ObservationPacket:
    """Owned snapshot of JSON data and named RGB image attachments."""

    data: dict[str, Any]
    images: dict[str, Image.Image] = field(default_factory=dict)

    def __post_init__(self):
        self.data = deepcopy(self.data)
        self.images = {
            name: image.convert("RGB").copy() for name, image in self.images.items()
        }
        # Fail early for NaN, numpy arrays, or unsupported extension data.
        json.dumps(self.data, allow_nan=False)
        for name in self.images:
            if name in self.data and not isinstance(self.data[name], dict):
                raise ValueError(f"image metadata {name!r} must be a JSON object")

    def to_dict(self, *, embed_images: bool = True) -> dict[str, Any]:
        payload = deepcopy(self.data)
        for name, image in self.images.items():
            metadata = dict(payload.get(name, {}))
            metadata.update(
                width=image.width,
                height=image.height,
                color_space="RGB",
                mime_type="image/png",
            )
            if embed_images:
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                metadata.update(
                    encoding="base64",
                    data=base64.b64encode(buffer.getvalue()).decode("ascii"),
                )
            else:
                metadata.update(encoding="attachment", attachment=name)
            payload[name] = metadata
        return payload

    def to_json(self, *, embed_images: bool = True) -> str:
        """Self-contained JSON by default, with lossless PNG/base64 images."""
        return json.dumps(
            self.to_dict(embed_images=embed_images),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    def to_model_inputs(self) -> dict[str, Any]:
        """Primary RGB plus optional named image attachments, in insertion order."""
        inputs = {"prompt": self.to_json(embed_images=False)}
        if self.images:
            inputs["image"] = next(iter(self.images.values())).copy()
        if len(self.images) > 1:
            inputs["extra_images"] = {
                name: image.copy() for name, image in list(self.images.items())[1:]
            }
        return inputs


class ObservationProvider(ABC):
    """Implement collect() to replace the entire observation experiment."""

    @abstractmethod
    def collect(self) -> ObservationPacket:
        """Capture without stepping the simulator or advancing the agent."""

    def close(self) -> None:
        """Release resources owned by this provider."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ObservationCancelled(RuntimeError):
    """The agent lost authority or changed turns before observation delivery."""


def collect_for_agent(provider: ObservationProvider, environment) -> ObservationPacket:
    """Collect in OBSERVATION and advance to THINK only for the same active turn.

    Call on the simulation owner thread, while stepping is paused. Sensor
    failures leave OBSERVATION intact; cancelled or stale packets are dropped.
    """
    environment.require_agent_control()
    agent = environment.agent_state_machine
    if agent.state is not AgentState.OBSERVATION:
        raise ValueError("Agent must be in OBSERVATION state")
    turn_id = agent.turn_id
    packet = provider.collect()
    environment.poll_control_authority()
    if (
        not environment.agent_control_enabled
        or not agent.enabled
        or agent.termination_requested
        or agent.turn_id != turn_id
        or environment.agent_state_machine is not agent
        or agent.state is not AgentState.OBSERVATION
    ):
        raise ObservationCancelled(
            "Observation discarded: turn or control authority changed"
        )
    if not isinstance(packet, ObservationPacket):
        raise TypeError("ObservationProvider.collect() must return ObservationPacket")
    agent.complete_observation()
    return packet
