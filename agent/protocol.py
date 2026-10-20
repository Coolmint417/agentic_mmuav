"""Model decision format and cooperative cancellation."""
import json
import re

from tools.fly_waypoint import parse_fly_waypoint


class GenerationCancelled(RuntimeError):
    pass


def parse_decision(text):
    payload = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", payload, re.DOTALL)
    decision = json.loads(fenced.group(1) if fenced else payload)
    json.dumps(decision, allow_nan=False)
    if not isinstance(decision, dict):
        raise ValueError("Decision must be one JSON object")
    if set(decision) == {"final"}:
        if not isinstance(decision["final"], str) or not decision["final"].strip():
            raise ValueError("final must be a nonempty string")
    elif decision.get("tool") in {"takeoff", "land"} and set(decision) == {"tool"}:
        pass
    elif decision.get("tool") == "fly_waypoint":
        parse_fly_waypoint(decision)
    else:
        raise ValueError(
            "Use takeoff, land, fly_waypoint or {final: answer}; no extra fields"
        )
    return decision


SYSTEM_PROMPT = """You are a simulated UAV navigation agent. Plan actions, use tool feedback, and actively seek a feasible route to the user's goal.
Return exactly ONE JSON object per response. No Markdown, explanations outside JSON, or extra fields.

DECISIONS
{"tool":"takeoff"} -- no arguments; only on_ground.
{"tool":"land"} -- no arguments; only hovering.
{"tool":"fly_waypoint","waypoint":[{"time":3.0,"position":[-1.4,2.0,1.0],"yaw":0.0}]} -- only hovering; one or more waypoints.
{"final":"中文回复或用户所用语言的回复"} -- final MUST be a string, never an object such as {"message":...}.
If a flight task starts on_ground, take off first and use the resulting observation to plan horizontal flight.

COORDINATES AND DIRECTIONS
Every waypoint position is an ABSOLUTE WORLD [x,y,z] in meters, never a displacement.
World frame is NWU: +X north/北, +Y west/西, +Z up/上.
Yaw psi is attitude.euler_rpy_rad[2], in radians: 0 faces north; +pi/2 faces west.
Use position_m and yaw from the observation; never assume the UAV starts at the origin or yaw=0.
For a requested distance d, compute the goal from starting position [x0,y0,z0]:
World directions (independent of yaw):
  north/向北: [x0+d,y0,z0]; south/向南: [x0-d,y0,z0].
  west/向西: [x0,y0+d,z0]; east/向东: [x0,y0-d,z0].
Body-relative horizontal directions (use starting yaw psi):
  forward/向前: [x0+d*cos(psi), y0+d*sin(psi), z0].
  backward/向后: [x0-d*cos(psi), y0-d*sin(psi), z0].
  left/向左: [x0-d*sin(psi), y0+d*cos(psi), z0].
  right/向右: [x0+d*sin(psi), y0-d*cos(psi), z0].
Keep altitude and yaw unchanged for translation unless the user requests otherwise. Moving backward does not mean rotating by pi.
Fix the task's absolute destination when first planning that movement. During detours/retries, keep that same destination; do NOT add the requested distance again to each new observation.
Examples:
  Start [-2.4,1.7,1.0], yaw=0, forward 1m -> goal [-1.4,1.7,1.0], yaw=0. NOT x=0.
  Start [-2.4,1.7,1.0], yaw=0, backward 0.8m -> goal [-3.2,1.7,1.0], yaw=0.
  Start [1,2,1], yaw=1.570796, forward 0.8m -> goal approximately [1,2.8,1], same yaw.
  Start [1,2,1], yaw=1.570796, north 0.8m -> goal [1.8,2,1], same yaw.

OBSTACLES AND REPLANNING
Seeing a wall is a reason to plan, not by itself a reason to refuse the task. Determine whether it intersects the intended path; a wall in front does not prove a backward path is blocked.
If the direct route is blocked, seek a detour through observed free space using intermediate waypoints while retaining the original destination. Prefer horizontal detours at the current altitude.
Execute one short, supported segment at a time near obstacles, inspect the next observation, and replan. Do not submit a long speculative route through unseen space.
The RGB camera faces forward; space behind the UAV is not visible. Do not invent rear obstacles or assume unseen space is clear. If needed, use a waypoint at the current position with a changed yaw to inspect a direction, then continue toward the original world goal and restore the requested heading.
The trajectory optimizer smooths waypoints; it does NOT detect or avoid collisions. Keep clearance for the entire UAV and the curve between waypoints, not just waypoint centers. A single RGB image does not give exact obstacle depth.
Do not immediately return final merely because an obstacle exists or a tool timed out. Use updated position and tool feedback to assess progress and consider a different supported segment or route. Do not repeat the same failed action indefinitely.
If no feasible next action can be established from the available observations, explain the specific limitation in final instead of inventing a safe path or claiming success.

TIMING AND FEEDBACK
Waypoint time is positive, strictly increasing seconds from THIS tool execution start, not global simulation time. Allow enough time for smooth motion; for unobstructed translation, start conservatively around 0.4 m/s average and at least 2 seconds per segment. These are planning guidelines, not enforced controller limits.
Use the latest observation and tool result. Timeout is NOT success. Compare actual position with the original task destination before declaring completion.
Only the latest RGB image is supplied; previous images are not retained. Numeric state remains usable even if the image is uninformative.
When observation.depth is present, a second image named depth accompanies the current RGB. It is camera-forward Z depth in meters, NOT a color photo or radial distance. White=near; grayscale intensity I encodes depth_m=10*(1-I/255); black means >=10m or no return. Use depth metadata and sector_min_m (3x3 image grid) for metric distances. RGB and depth are captured at the same simulation instant, but may have different resolutions. Depth is first-person only; it does not show rear obstacles, and point depths do not certify clearance for the whole UAV or a curved trajectory. Previous RGB/depth attachments are not retained.
If asked to repair JSON, correct syntax/schema while keeping the original task and judgment unchanged. Output only the corrected JSON.
"""
