"""No-argument automatic landing tool."""
from env.flight_state_machine import FlightState
from ._auto_flight import AutoFlightTool


class LandTool(AutoFlightTool):
    name = "land"
    required_state = FlightState.HOVERING
    completed_state = FlightState.ON_GROUND

    def _start(self):
        self.environment.start_auto_land()
        return None
