"""Replaceable UAV observation providers and model input serialization."""
from .base import (
    ObservationCancelled,
    ObservationPacket,
    ObservationProvider,
    collect_for_agent,
)

__all__ = [
    "ObservationCancelled",
    "ObservationPacket",
    "ObservationProvider",
    "collect_for_agent",
    "MuJoCoObservationProvider",
    "ObservationConfig",
]


def __getattr__(name):
    # Alternative providers can use the packet/contract without importing
    # MuJoCo or selecting an OpenGL backend.
    if name in {"MuJoCoObservationProvider", "ObservationConfig"}:
        from .mujoco_observation import MuJoCoObservationProvider, ObservationConfig

        return {
            "MuJoCoObservationProvider": MuJoCoObservationProvider,
            "ObservationConfig": ObservationConfig,
        }[name]
    raise AttributeError(name)
