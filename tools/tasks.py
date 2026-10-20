"""Cooperative tool requests; the runtime is the sole simulation step owner."""
from dataclasses import dataclass
from typing import Any


@dataclass
class PlanningRequest:
    optimizer: Any
    arguments: dict

    def compute(self):
        return self.optimizer.optimize(**self.arguments)


def execute_synchronously(task, environment):
    """Headless compatibility driver. Interactive applications use EnvironmentRuntime."""
    response = None
    error = None
    while True:
        try:
            request = task.throw(error) if error is not None else task.send(response)
        except StopIteration as finished:
            return finished.value
        response = error = None
        try:
            if isinstance(request, PlanningRequest):
                response = request.compute()
            else:
                environment.step()
        except Exception as exc:
            error = exc
