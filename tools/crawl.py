"""Cooperative, counted forward crawling using the environment's gait."""
import json
import math
import re

from agent.state_machine import AgentState
from env.flight_state_machine import FlightState
from env.mmuav_env import CRAWL_STAGE_DURATION
from .tasks import execute_synchronously


def parse_crawl(command):
    if isinstance(command, str):
        payload = command.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", payload, re.DOTALL)
        command = json.loads(fenced.group(1) if fenced else payload)
    if (not isinstance(command, dict) or set(command) != {"tool", "steps", "attached"}
            or command["tool"] != "crawl"):
        raise ValueError('Expected {"tool":"crawl","steps":positive_integer,"attached":boolean}')
    if type(command["steps"]) is not int or command["steps"] <= 0:
        raise ValueError("steps must be a positive integer")
    if type(command["attached"]) is not bool:
        raise ValueError("attached must be true or false")
    return command["steps"], command["attached"]


class CrawlTool:
    name = "crawl"

    def __init__(self, environment, agent_state_machine):
        self.environment = environment
        self.agent = agent_state_machine
        self.flight = environment.flight_state_machine
        environment.bind_agent_state_machine(self.agent)

    def _still_executing(self, turn_id):
        return (self.environment.agent_control_enabled
                and self.agent.turn_id == turn_id
                and self.agent.state is AgentState.EXECUTE
                and not self.agent.termination_requested)

    def execute(self, command):
        return execute_synchronously(self.iter_execute(command), self.environment)

    def iter_execute(self, command):
        if self.agent.state is not AgentState.EXECUTE:
            raise ValueError("crawl requires agent EXECUTE state")
        turn_id = self.agent.turn_id
        started = False
        completed = False
        try:
            steps, attached = parse_crawl(command)
            self.environment.require_agent_control()
            if self.flight.state not in {FlightState.ON_GROUND, FlightState.ADHERED}:
                raise ValueError("crawl requires UAV ON_GROUND or ADHERED state")
            start = previous = self.environment.observe()["time"]
            self.environment.start_crawl(steps, attached)
            started = True
            # Allow for timestep rounding and neutral/attachment cleanup.
            timeout = (4 * steps + 8) * CRAWL_STAGE_DURATION * 1.5
            while True:
                if not self._still_executing(turn_id):
                    return {"tool": self.name, "status": "terminated"}
                yield None
                if not self._still_executing(turn_id):
                    return {"tool": self.name, "status": "terminated"}
                observation = self.environment.observe()
                now = observation["time"]
                if not math.isfinite(now) or now <= previous:
                    raise RuntimeError("simulation time did not advance")
                previous = now
                progress = self.environment.get_crawl_progress()
                if not progress["running"]:
                    expected = FlightState.ADHERED if attached else FlightState.ON_GROUND
                    if progress["steps_completed"] != steps or self.flight.state is not expected:
                        raise RuntimeError("crawl stopped before reaching the requested result")
                    completed = True
                    return {"tool": self.name, "status": "success", "steps": steps,
                            "steps_completed": progress["steps_completed"], "attached": attached,
                            "flight_state": self.flight.state.value,
                            "elapsed_seconds": now - start,
                            "position": observation["position"].tolist(),
                            "yaw": float(observation["yaw"])}
                if now - start >= timeout:
                    return {"tool": self.name, "status": "timeout"}
        except Exception as exc:
            return {"tool": self.name, "status": "error", "error": str(exc)}
        finally:
            # Generator.close(), Axis4 revocation and user cancellation all
            # leave cleanup to the continuously stepping environment.
            if started and not completed:
                self.environment.request_crawl_stop()
            if self._still_executing(turn_id):
                self.agent.complete_execute()
