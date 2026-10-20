"""Explicit simulated remote input for headless control tests."""


class FakeJoystick:
    def __init__(self, agent_enabled=True):
        self.axes = {4: float(agent_enabled), 7: -1.0}
        self.reads = []
        self.on_update = lambda: None

    def update(self):
        self.on_update()

    def axis(self, number, default=0.0):
        self.reads.append(number)
        return self.axes.get(number, default)


class AuthorizedEnvironment:
    """Minimal authority protocol for isolated tool lifecycle tests."""
    agent_control_enabled = True

    def bind_agent_state_machine(self, machine):
        self.agent = machine

    def require_agent_control(self):
        if not self.agent_control_enabled:
            raise PermissionError("Agent control is not authorized")
