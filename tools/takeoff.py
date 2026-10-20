"""No-argument automatic takeoff tool."""
from env.flight_state_machine import FlightState
from ._auto_flight import AutoFlightTool


class TakeoffTool(AutoFlightTool):
    name = "takeoff"
    required_state = FlightState.ON_GROUND
    completed_state = FlightState.HOVERING

    def _start(self):
        return self.environment.start_auto_takeoff()
