"""Uncapped, per-tool RGB/motion sampling on the simulation owner thread."""
import math

from .base import ObservationPacket


class ProcessObservation:
    @staticmethod
    def validate_interval(interval):
        if (isinstance(interval, bool) or not isinstance(interval, (int, float))
                or not math.isfinite(interval) or interval <= 0):
            raise ValueError("process sample interval must be positive and finite")

    def __init__(self, tool, start_time, interval):
        self.validate_interval(interval)
        self.tool = tool
        self.start_time = start_time
        self.interval = interval
        self.next_time = start_time + interval
        self.frames = []
        self.images = {}
        self.end_time = None

    def sample(self, provider, sim_time):
        if sim_time + 1e-9 < self.next_time:
            return
        # Same private physics snapshot for RGB and all motion values. Never
        # render process depth, even if the final observation requests depth.
        packet = provider.collect(include_depth=False)
        data = packet.data
        time = data["sim_time_seconds"]
        name = f"process_rgb_{len(self.frames):06d}"
        self.frames.append({
            "time": time,
            "elapsed_seconds": time - self.start_time,
            "position": data["position_m"],
            "velocity": data["velocity_m_s"],
            "rpy": data["attitude"]["euler_rpy_rad"],
            "angular_velocity": data["angular_velocity_rad_s_body"],
            "state": data["flight_state"],
            "sucker_states": data["sucker_states"],
            "rgb_attachment": name,
        })
        self.images[name] = packet.images["rgb"]
        # A slow capture never creates several copies of the same instant.
        self.next_time = self.start_time + (
            math.floor((time - self.start_time + 1e-9) / self.interval) + 1
        ) * self.interval

    def attach(self, packet):
        end = self.end_time if self.end_time is not None else packet.data["sim_time_seconds"]
        frames = [f for f in self.frames if f["time"] < end - 1e-9]
        data = dict(packet.data)
        data["process"] = {
            "tool": self.tool, "sample_interval_seconds": self.interval,
            "start_time_seconds": self.start_time, "end_time_seconds": end,
            "coordinate_frame": "NWU", "position_unit": "m",
            "velocity_unit": "m/s", "rpy_unit": "rad",
            "angular_velocity_unit": "rad/s", "angular_velocity_frame": "body",
            "frames": frames,
        }
        images = dict(packet.images)
        images.update({f["rgb_attachment"]: self.images[f["rgb_attachment"]] for f in frames})
        data["image_order"] = list(images)
        return ObservationPacket(data, images)
