"""Parse one waypoint object, tolerating a single Markdown JSON wrapper."""
import json
import math
import re


def parse_waypoint(text: str) -> dict:
    payload = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", payload, flags=re.DOTALL)
    if fenced:
        payload = fenced.group(1)
    action = json.loads(payload)
    if not isinstance(action, dict) or set(action) != {"position", "yaw"}:
        raise ValueError("Waypoint must contain exactly position and yaw")
    position = action["position"]
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError("position must contain three coordinates")
    for value in [*position, action["yaw"]]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Waypoint values must be finite numbers")
    return action
