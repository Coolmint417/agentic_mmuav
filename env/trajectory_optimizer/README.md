# Trajectory Optimizer

This module post-processes model waypoint outputs into a smooth piecewise
polynomial trajectory.

## Configuration

Edit `config_trajectory_optimizer.py`:

- `k_pos = 8`: polynomial degree for x/y/z
- `k_yaw = 5`: polynomial degree for yaw
- `continuity_order_pos = 4`: highest continuous derivative order for x/y/z
- `continuity_order_yaw = 2`: highest continuous derivative order for yaw
- `objective_derivative_order_pos = 4`: derivative order minimized for x/y/z
- `objective_derivative_order_yaw = 2`: derivative order minimized for yaw
- `waypoint_dt = 0.2`: default time spacing between model waypoints

The optimizer minimizes the squared integral of the configured derivative
orders. Segment connection continuity is controlled by `continuity_order_pos`
and `continuity_order_yaw`; defaults use C4/minimum snap for position and
C2/minimum angular acceleration for yaw.

## Nonuniform waypoint times

`optimize(..., waypoint_times=[0.0, 1.5, 4.0])` accepts one time in seconds
per knot, starting at zero and strictly increasing. This example gives
segment durations `[1.5, 2.5]`. Times and positions/yaws must be finite.
Alternatively, `segment_durations=[1.5, 2.5]` still works. Specify only one
of these arguments; omitting both preserves the `waypoint_dt` default.

The `fly_waypoint` tool prepends the current measured state at time zero,
so the model specifies only future waypoints, each with a positive time.

## Example

```python
import numpy as np

from trajectory_optimizer import (
    PolynomialTrajectoryOptimizer,
    TrajectoryOptimizerConfig,
)

positions = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.2, 0.1],
    [1.0, 0.0, 0.2],
])
yaws = np.array([0.0, 0.1, 0.2])
velocities = [
    np.array([0.0, 0.0, 0.0]),
    None,                          # no velocity constraint at this knot
    np.array([0.0, 0.0, 0.0]),
]                                  # optional
accelerations = np.zeros((3, 3))   # optional
yaw_rates = np.zeros(3)            # optional

optimizer = PolynomialTrajectoryOptimizer(TrajectoryOptimizerConfig())
trajectory = optimizer.optimize(
    positions=positions,
    yaws=yaws,
    waypoint_times=np.array([0.0, 1.5, 4.0]),
    velocities=velocities,
    accelerations=accelerations,
    yaw_rates=yaw_rates,
)

state = trajectory.evaluate(0.2)
print(state.position)
print(state.velocity)
print(state.acceleration)
print(state.yaw)
print(state.yaw_rate)
```

`trajectory.pos_coeffs` has shape `[num_segments, k_pos + 1, 3]`.
`trajectory.yaw_coeffs` has shape `[num_segments, k_yaw + 1]`.

For `velocities`, `accelerations`, and `yaw_rates`, each knot can be left
unconstrained by using `None` or `NaN`. At an unconstrained internal knot, the
optimizer enforces derivative continuity between neighboring polynomial
segments instead of pinning that derivative to a target value.
