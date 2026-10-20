import numpy as np
import os
import math
import errno
import fcntl
import struct
import mujoco 
import mujoco.viewer as viewer 

from dataclasses import dataclass

if __package__:
    from .controller.rate_controller import RatePIDController
    from .controller.gyro_filter import GyroLowPassFilter
    from .controller.se3_controller import SE3Controller, VehicleState
    from .controller.motor_mixer import Mixer
    from .flight_state_machine import FlightState, FlightStateMachine
    from . import geometry
else:
    from controller.rate_controller import RatePIDController
    from controller.gyro_filter import GyroLowPassFilter
    from controller.se3_controller import SE3Controller, VehicleState
    from controller.motor_mixer import Mixer
    from flight_state_machine import FlightState, FlightStateMachine
    import geometry

MAX_YAW_RATE = math.radians(90.0)
MAX_XY_SPEED = 1.0
MAX_Z_SPEED = 0.7
MAX_POSITION_TARGET_Z = 10.0
MIN_POSITION_TARGET_Z = 0.0

# 根据你的遥控器通道顺序改这里即可。Linux joystick 轴值范围会归一化到 [-1, 1]。
AXIS_ROLL = 0
AXIS_PITCH = 1
AXIS_THROTTLE = 2
AXIS_YAW = 3
AXIS_AGENT_ENABLE = 4
AXIS_TAKEOFF_LAND = 7

CTRL_MODE_POSITION = 2      # Always position control

POSITION_MODE_TAKEOFF_Z = 1.0
POSITION_AUTO_NONE = 0
POSITION_AUTO_TAKEOFF = 1
POSITION_AUTO_LAND = 2
TAKEOFF_RELATIVE_HEIGHT = 0.8
TAKEOFF_REACHED_RADIUS = 0.1
TAKEOFF_SETTLE_SPEED = 1.0
TAKEOFF_KV = 2.0
LAND_DESCENT_SPEED = 0.25
LAND_STILL_SPEED = 0.08
LAND_STILL_TIME = 1.0


class JoystickReader:
    JS_EVENT_FORMAT = "=IhBB"
    JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)
    JS_EVENT_AXIS = 0x02
    JS_EVENT_INIT = 0x80

    def __init__(self, path="/dev/input/js0", deadband=0.05):
        self.path = path
        self.deadband = deadband
        self.fd = None
        self.axes = {}
        self.open_failed = False
        self.open()

    def open(self):
        try:
            self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
            flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
            fcntl.fcntl(self.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            print(f"Joystick connected: {self.path}")
        except OSError as exc:
            self.fd = None
            self.open_failed = True
            print(f"Joystick unavailable ({self.path}): {exc}. Use neutral command.")

    def update(self):
        if self.fd is None:
            return

        while True:
            try:
                event = os.read(self.fd, self.JS_EVENT_SIZE)
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    print(f"Joystick read error: {exc}")
                    os.close(self.fd)
                    self.fd = None
                break

            if len(event) != self.JS_EVENT_SIZE:
                break

            _, value, event_type, number = struct.unpack(self.JS_EVENT_FORMAT, event)
            event_type &= ~self.JS_EVENT_INIT
            if event_type == self.JS_EVENT_AXIS:
                axis_value = value / 32767.0
                if abs(axis_value) < self.deadband:
                    axis_value = 0.0
                self.axes[number] = float(np.clip(axis_value, -1.0, 1.0))

    def axis(self, number, default=0.0):
        return self.axes.get(number, default)


class AgentOnlyInput:
    """Desktop fallback selected only when no joystick is available at startup."""

    def update(self):
        pass

    def axis(self, number, default=0.0):
        if number == AXIS_AGENT_ENABLE:
            return 1.0
        if number == AXIS_TAKEOFF_LAND:
            return -1.0
        return 0.0


def joystick_to_world_velocity(roll_axis, pitch_axis, throttle, yaw):
    def throttle_to_vertical_speed(throttle):
        throttle = float(np.clip(throttle, 0.0, 1.0))
        if throttle > 0.6:
            return (throttle - 0.6) / 0.4 * MAX_Z_SPEED
        if throttle < 0.4:
            return -(0.4 - throttle) / 0.4 * MAX_Z_SPEED
        return 0.0
    forward_speed = pitch_axis * MAX_XY_SPEED
    left_speed = -roll_axis * MAX_XY_SPEED
    vertical_speed = throttle_to_vertical_speed(throttle)

    forward_world = np.array([math.cos(yaw), math.sin(yaw), 0.0])  # X axis in the NWU coordination
    left_world = np.array([-math.sin(yaw), math.cos(yaw), 0.0])  # Y axis in the NWU coordination
    return forward_speed * forward_world + left_speed * left_world + np.array([0.0, 0.0, vertical_speed])

rate_controller = RatePIDController(kp=np.array([5e-1, 5e-1, 5e-1]), 
                                    ki=np.array([1e-1, 1e-1, 1e-1]), 
                                    kd=np.array([1e-2, 1e-2, 1e-2]), 
                                    integral_limit=np.array([3.0, 3.0, 3.0]), 
                                    torque_limit=np.array([1.0, 1.0, 0.4]))
gyro_filter = GyroLowPassFilter(cutoff_freq=30.0)
mixer = Mixer()
position_controller = SE3Controller(with_position_error=True)
position_controller.set_parameters(kx=8.0, kv=6.0, kR=8.0)

joystick = JoystickReader()
yaw_command = 0.0
position_target = np.zeros(3)
planner_target_provider = None
takeoff_land_axis_last_sign = 0
position_auto_mode = POSITION_AUTO_NONE
position_auto_target = np.zeros(3)
land_still_time = 0.0
flight_state_machine = FlightStateMachine()
agent_state_machine = None
model_stop_callback = None


xml_path = os.path.join(os.path.dirname(__file__), './scene.xml')


def set_planner_target_provider(provider):
    global planner_target_provider
    planner_target_provider = provider


def get_position_auto_mode():
    return position_auto_mode


def start_position_auto_land(d=None):
    if d is None:
        raise ValueError("start_position_auto_land requires current MjData")
    _start_position_auto_land(d)


def load_callback(m=None, d=None, *, flight_machine=None, agent_machine=None):
    global last_time, yaw_command, position_target
    global takeoff_land_axis_last_sign
    global position_auto_mode, position_auto_target, land_still_time
    global planner_target_provider, flight_state_machine, agent_state_machine
    mujoco.set_mjcb_control(None)
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    flight_state_machine = flight_machine if flight_machine is not None else FlightStateMachine()
    agent_state_machine = agent_machine
    if agent_state_machine is not None:
        agent_state_machine.set_enabled(flight_state_machine.agent_authorized)
    planner_target_provider = None
    last_time = 0.0
    yaw_command = 0.0
    takeoff_land_axis_last_sign = 0
    position_auto_mode = POSITION_AUTO_NONE
    position_auto_target = d.qpos[:3].copy()
    land_still_time = 0.0
    position_target = d.qpos[:3].copy()
    rate_controller.reset()
    gyro_filter.reset()
    position_controller.vehicle_mass = float(m.body_subtreemass[m.body("Body").id])
    position_controller.gravity = float(np.linalg.norm(m.opt.gravity))
    position_controller.set_parameters(12.0, 8.0, 8.0)
    if m is not None:
        mujoco.set_mjcb_control(lambda m, d: control_callback(m, d))  # 设置控制回调函数
    return m, d


def _axis_sign(value):
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _sync_takeoff_land_axis_state():
    global takeoff_land_axis_last_sign
    if flight_state_machine.agent_authorized:
        return
    sign = _axis_sign(joystick.axis(AXIS_TAKEOFF_LAND))
    if sign != 0:
        takeoff_land_axis_last_sign = sign


def _update_position_mode_takeoff_land_trigger(d):
    global takeoff_land_axis_last_sign

    axis_value = joystick.axis(AXIS_TAKEOFF_LAND)
    sign = _axis_sign(axis_value)
    if sign == 0:
        return

    if (takeoff_land_axis_last_sign < 0 and sign > 0
            and flight_state_machine.state is FlightState.ON_GROUND):
        _start_position_auto_takeoff(d)
    elif (takeoff_land_axis_last_sign > 0 and sign < 0
          and flight_state_machine.state is FlightState.HOVERING):
        _start_position_auto_land(d)

    takeoff_land_axis_last_sign = sign


def _start_position_auto_takeoff(d):
    global position_auto_mode, position_auto_target, position_target, land_still_time
    if flight_state_machine.state is not FlightState.ON_GROUND:
        raise ValueError("takeoff requires ON_GROUND")
    position_auto_mode = POSITION_AUTO_TAKEOFF
    position_auto_target = d.qpos[:3].copy()
    position_auto_target[2] = np.clip(
        max(POSITION_MODE_TAKEOFF_Z, d.qpos[2] + TAKEOFF_RELATIVE_HEIGHT),
        MIN_POSITION_TARGET_Z,
        MAX_POSITION_TARGET_Z,
    )
    position_target = position_auto_target.copy()
    land_still_time = 0.0
    print(f"Position auto takeoff: target = {position_auto_target}")


def _start_position_auto_land(d):
    global position_auto_mode, position_auto_target, position_target, land_still_time
    if flight_state_machine.state is not FlightState.HOVERING:
        raise ValueError("landing requires HOVERING")
    position_auto_mode = POSITION_AUTO_LAND
    position_auto_target = d.qpos[:3].copy()
    position_target = d.qpos[:3].copy()
    land_still_time = 0.0
    print("Position auto landing: joystick input blocked, descending slowly")


def _update_position_auto_control(d, dt):
    global position_auto_mode, position_target, land_still_time

    if position_auto_mode == POSITION_AUTO_TAKEOFF:
        position_target = position_auto_target.copy()
        target_velocity = TAKEOFF_KV * (position_target - d.qpos[:3].copy())
        z_error = abs(d.qpos[2] - position_auto_target[2])
        speed = np.linalg.norm(d.qvel[:3])
        if z_error < TAKEOFF_REACHED_RADIUS and speed < TAKEOFF_SETTLE_SPEED:
            position_auto_mode = POSITION_AUTO_NONE
            flight_state_machine.transition_to(FlightState.HOVERING)
            _sync_takeoff_land_axis_state()
            target_velocity = np.zeros(3)
            print("Position auto takeoff complete")
        return target_velocity

    if position_auto_mode == POSITION_AUTO_LAND:
        target_velocity = np.array([0.0, 0.0, -LAND_DESCENT_SPEED])

        position_target[2] = np.clip(
            position_target[2] - LAND_DESCENT_SPEED * dt,
            MIN_POSITION_TARGET_Z,
            MAX_POSITION_TARGET_Z,
        )
        if np.linalg.norm(d.qvel[:3]) < LAND_STILL_SPEED:
            land_still_time += dt
        else:
            land_still_time = 0.0

        if land_still_time >= LAND_STILL_TIME:
            position_auto_mode = POSITION_AUTO_NONE
            position_target = d.qpos[:3].copy()
            land_still_time = 0.0
            flight_state_machine.transition_to(FlightState.ON_GROUND)
            _sync_takeoff_land_axis_state()
            target_velocity = np.zeros(3)
            print("Position auto landing complete")
        return target_velocity

    return None


global last_time_crawl, crawl_stage
global original_upleft_sucker_geom_contype, original_upleft_sucker_geom_conaffinity
global original_upright_sucker_geom_contype, original_upright_sucker_geom_conaffinity
global original_bottomleft_sucker_geom_contype, original_bottomleft_sucker_geom_conaffinity
global original_bottomright_sucker_geom_contype, original_bottomright_sucker_geom_conaffinity
last_time_crawl = 0
crawl_stage = 0
def control_callback_crawl(m, d):
    global last_time_crawl, crawl_stage

    geom_sucker_upleft_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sucker_upleft")
    geom_sucker_upright_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sucker_upright")
    geom_sucker_bottomleft_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sucker_bottomleft")
    geom_sucker_bottomright_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sucker_bottomright")
    
    stage_duration = 1.0
    ctrl_val_max = 0.3
    dt_crawl = (d.time - last_time_crawl) / stage_duration
    if dt_crawl > 1.0:
        crawl_stage = crawl_stage + 1
        last_time_crawl = d.time
    
    if crawl_stage % 4 == 0:
        
        m.geom_contype[geom_sucker_upleft_id] = 1
        m.geom_contype[geom_sucker_bottomright_id] = 1
        m.geom_conaffinity[geom_sucker_upleft_id] = 1
        m.geom_conaffinity[geom_sucker_bottomright_id] = 1

        m.geom_contype[geom_sucker_upright_id] = 0
        m.geom_contype[geom_sucker_bottomleft_id] = 0
        m.geom_conaffinity[geom_sucker_upright_id] = 0
        m.geom_conaffinity[geom_sucker_bottomleft_id] = 0
        d.ctrl[1] = 1.0
        d.ctrl[2] = 0.0
        d.ctrl[3] = 1.0
        d.ctrl[4] = 0.0

        '''
        d.ctrl[9] = 1.0
        d.ctrl[10] = 0.0
        d.ctrl[11] = 1.0
        d.ctrl[12] = 0.0

        m.geom_friction[sucker_upleft_id][0] = 2.0
        m.geom_friction[sucker_bottomleft_id][0] = 0.0
        m.geom_friction[sucker_bottomright_id][0] = 2.0
        m.geom_friction[sucker_upright_id][0] = 0.0'''

        '''m.site_pos[site_sucker_upleft_id] = d.xpos[body_sucker_upleft_id]
        m.site_pos[site_sucker_bottomright_id] = d.xpos[body_sucker_bottomright_id]
        d.eq_active[0] = 1
        d.eq_active[1] = 0
        d.eq_active[2] = 1
        d.eq_active[3] = 0'''
       
    elif crawl_stage % 4 == 1:
        ctrl_val = dt_crawl * 0.5 - ctrl_val_max
        d.ctrl[0] = ctrl_val

        d.ctrl[1] = 1 - dt_crawl
        d.ctrl[2] = dt_crawl
        d.ctrl[3] = 1 - dt_crawl
        d.ctrl[4] = dt_crawl
    elif crawl_stage % 4 == 2:


        m.geom_contype[geom_sucker_upright_id] = 1
        m.geom_contype[geom_sucker_bottomleft_id] = 1
        m.geom_conaffinity[geom_sucker_upright_id] = 1
        m.geom_conaffinity[geom_sucker_bottomleft_id] = 1

        m.geom_contype[geom_sucker_upleft_id] = 0
        m.geom_contype[geom_sucker_bottomright_id] = 0
        m.geom_conaffinity[geom_sucker_upleft_id] = 0
        m.geom_conaffinity[geom_sucker_bottomright_id] = 0



        '''
        d.ctrl[9] = 0.0
        d.ctrl[10] = 1.0
        d.ctrl[11] = 0.0
        d.ctrl[12] = 1.0

        m.geom_friction[sucker_upleft_id][0] = 0.0
        m.geom_friction[sucker_bottomleft_id][0] = 2.0
        m.geom_friction[sucker_bottomright_id][0] = 0.0
        m.geom_friction[sucker_upright_id][0] = 2.0
        m.site_pos[site_sucker_bottomleft_id] = d.xpos[body_sucker_bottomleft_id]
        m.site_pos[site_sucker_upright_id] = d.xpos[body_sucker_upright_id]
        d.eq_active[0] = 0
        d.eq_active[1] = 1
        d.eq_active[2] = 0
        d.eq_active[3] = 1'''

    elif crawl_stage % 4 == 3:
        ctrl_val = -dt_crawl * 0.5 + 0.25
        d.ctrl[0] = ctrl_val
        
        d.ctrl[1] = dt_crawl
        d.ctrl[2] = 1 - dt_crawl
        d.ctrl[3] = dt_crawl
        d.ctrl[4] = 1 - dt_crawl

def _current_yaw(d):
    w, x, y, z = d.qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def poll_control_authority(d):
    """Axis4 is the sole authorization source; never read other axes in agent mode."""
    global position_auto_mode, position_target, yaw_command, land_still_time
    global planner_target_provider
    joystick.update()
    value = joystick.axis(AXIS_AGENT_ENABLE)
    enabled = value > 0.5
    was_auto = position_auto_mode != POSITION_AUTO_NONE
    changed = flight_state_machine.set_agent_authorized(enabled)
    if agent_state_machine is not None:
        agent_state_machine.set_enabled(enabled, "Axis4 revoked agent control")
    if changed:
        position_auto_mode = POSITION_AUTO_NONE
        land_still_time = 0.0
        planner_target_provider = None
        position_target = d.qpos[:3].copy()
        yaw_command = _current_yaw(d)
        rate_controller.reset()
        if enabled and was_auto:
            flight_state_machine.transition_to(FlightState.HOVERING)
        if not enabled:
            _sync_takeoff_land_axis_state()
            if model_stop_callback is not None:
                try:
                    model_stop_callback()
                except Exception as exc:
                    # Control must still return to the pilot if a future model
                    # cancellation callback fails. The agent stays disabled.
                    print(f"Model stop callback failed: {exc}")
    return changed


def control_callback(m, d):
    global last_time, yaw_command, position_target
    dt = d.time - last_time
    if dt <= 0:
        dt = m.opt.timestep
    authority_changed = poll_control_authority(d)
    agent_control = flight_state_machine.agent_authorized
    omega = gyro_filter.update(d.sensordata[:3].copy(), dt)
    qw, qx, qy, qz = d.sensordata[6:10]
    rotation = geometry.GeoQuaternion(qx, qy, qz, qw).getRotationMatrix()

    if not agent_control and not authority_changed and position_auto_mode == POSITION_AUTO_NONE:
        _update_position_mode_takeoff_land_trigger(d)
    auto_velocity = _update_position_auto_control(d, dt)
    velocity = np.zeros(3)
    acceleration = np.zeros(3)
    feedforward_omega = np.zeros(3)
    if auto_velocity is not None:
        velocity = auto_velocity
    elif agent_control and planner_target_provider is not None:
        target = planner_target_provider(m=m, d=d, current_rotation=rotation, omega=omega, dt=dt)
        if target is not None:
            position_target = np.asarray(target.position, dtype=float).copy()
            velocity = np.asarray(target.velocity, dtype=float)
            acceleration = np.asarray(target.acceleration, dtype=float)
            yaw_command = float(target.yaw)
            feedforward_omega = np.asarray(target.omega, dtype=float)
    elif (not agent_control and not authority_changed
          and flight_state_machine.state is FlightState.HOVERING):
        roll, pitch, raw_throttle, yaw = [joystick.axis(axis) for axis in
                                         (AXIS_ROLL, AXIS_PITCH, AXIS_THROTTLE, AXIS_YAW)]
        throttle = 0.5 * (raw_throttle + 1.0)
        yaw_rate = -yaw * MAX_YAW_RATE
        yaw_command += yaw_rate * dt
        velocity = joystick_to_world_velocity(roll, pitch, throttle, yaw_command)
        position_target += velocity * dt
        position_target[2] = np.clip(position_target[2], MIN_POSITION_TARGET_Z, MAX_POSITION_TARGET_Z)

    current = VehicleState(pos=d.qpos[:3].copy(), vel=d.qvel[:3].copy(), rotation=rotation, omega=omega)
    goal = VehicleState(pos=position_target.copy(), vel=velocity, rotation=np.eye(3), omega=feedforward_omega)
    position_controller.set_current_state(current)
    position_controller.set_goal_state(goal)
    target_rate = position_controller.control_update(goal_yaw=yaw_command, goal_acc=acceleration)
    thrust = float(np.clip(target_rate[0], 0.0, 4.0 * mixer.const_thrust * mixer.max_speed ** 2))
    target_omega = np.asarray(target_rate[1:], dtype=float)
    actuation = rate_controller.update(target_thrust=thrust, target_omega=target_omega,
                                      current_omega=omega, dt=dt)
    d.ctrl[5:9] = mixer.motor_distribute(thrust=actuation[0], mx=actuation[1],
                                      my=actuation[2], mz=actuation[3])
    last_time = d.time

@dataclass
class PlannerTarget:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: float
    yaw_rate: float
    omega: np.ndarray


class MMUAVEnv:
    """Headless stepping adapter for the existing module-level control loop.

    MuJoCo's control callback and these controllers are process-global, so
    only one active MMUAVEnv is supported. Tools and stepping run on one
    simulation owner thread; this adapter does not create a background loop.
    """

    _active = None

    def __init__(self, flight_state_machine=None, *, agent_state_machine=None,
                 joystick_reader=None, position_gains=(12.0, 8.0, 8.0),
                 auto_agent_without_joystick=False):
        global joystick
        if MMUAVEnv._active is not None:
            raise RuntimeError("Only one active MMUAVEnv is supported")
        gains = np.asarray(position_gains, dtype=float)
        if gains.shape != (3,) or not np.all(np.isfinite(gains)) or np.any(gains <= 0):
            raise ValueError("position_gains must contain three positive finite gains")
        self.flight_state_machine = (
            flight_state_machine if flight_state_machine is not None else FlightStateMachine()
        )
        self._previous_position_parameters = (
            position_controller.kx, position_controller.kv, position_controller.kR,
            position_controller.vehicle_mass, position_controller.gravity,
        )
        if agent_state_machine is None:
            from agent.state_machine import AgentStateMachine
            agent_state_machine = AgentStateMachine()
        self.agent_state_machine = agent_state_machine
        self._previous_joystick = joystick
        selected_reader = joystick_reader if joystick_reader is not None else joystick
        if (auto_agent_without_joystick and isinstance(selected_reader, JoystickReader)
                and selected_reader.fd is None):
            selected_reader = AgentOnlyInput()
        joystick = selected_reader
        self.control_authority_source = (
            "no_joystick" if isinstance(joystick, AgentOnlyInput) else "axis4"
        )
        self.model, self.data = load_callback(flight_machine=self.flight_state_machine,
                                              agent_machine=self.agent_state_machine)
        position_controller.set_parameters(*gains)
        self._closed = False
        mujoco.mj_forward(self.model, self.data)
        MMUAVEnv._active = self

    def _check_open(self):
        if self._closed:
            raise RuntimeError("MMUAVEnv is closed")

    def observe(self):
        self._check_open()
        w, x, y, z = self.data.qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return {
            "time": float(self.data.time),
            "position": self.data.qpos[:3].copy(),
            "velocity": self.data.qvel[:3].copy(),
            "yaw": yaw,
            "angular_velocity": self.data.qvel[3:6].copy(),
        }

    @property
    def agent_control_enabled(self):
        return self.flight_state_machine.agent_authorized

    def poll_control_authority(self):
        self._check_open()
        return poll_control_authority(self.data)

    def require_agent_control(self):
        self.poll_control_authority()
        if not self.agent_control_enabled:
            raise PermissionError("Axis4 must be 1 to authorize agent control")

    def bind_agent_state_machine(self, machine):
        global agent_state_machine
        self.poll_control_authority()
        if machine is not self.agent_state_machine and self.agent_state_machine.state.value != "end":
            raise RuntimeError("Environment already has an active agent")
        self.agent_state_machine = machine
        agent_state_machine = machine
        machine.set_enabled(self.agent_control_enabled, "Axis4 is not authorizing agent control")

    def set_model_stop_callback(self, callback):
        """Reserve an integration hook for cancelling generation on Axis4 revocation."""
        global model_stop_callback
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")
        model_stop_callback = callback

    def _install_provider(self, provider):
        global position_auto_mode
        self._check_open()
        position_auto_mode = POSITION_AUTO_NONE
        set_planner_target_provider(provider)

    def follow_trajectory(self, trajectory):
        self.require_agent_control()
        start_time = float(self.data.time)

        def provider(*, d, current_rotation, **kwargs):
            elapsed = max(0.0, float(d.time) - start_time)
            state = trajectory.evaluate(min(elapsed, trajectory.total_time))
            finished = elapsed >= trajectory.total_time
            yaw_rate = 0.0 if finished else state.yaw_rate
            return PlannerTarget(
                position=state.position.copy(),
                velocity=np.zeros(3) if finished else state.velocity.copy(),
                acceleration=np.zeros(3) if finished else state.acceleration.copy(),
                yaw=state.yaw,
                yaw_rate=yaw_rate,
                omega=current_rotation.T @ np.array([0.0, 0.0, yaw_rate]),
            )

        self._install_provider(provider)

    def hold_position(self, position=None, yaw=None):
        global position_target, yaw_command
        observed = self.observe()
        position = np.asarray(observed["position"] if position is None else position, dtype=float).copy()
        yaw = observed["yaw"] if yaw is None else float(yaw)
        if position.shape != (3,) or not np.all(np.isfinite(position)) or not math.isfinite(yaw):
            raise ValueError("hold target must contain finite position and yaw")

        def provider(**kwargs):
            return PlannerTarget(position.copy(), np.zeros(3), np.zeros(3), yaw, 0.0, np.zeros(3))

        position_target = position.copy()
        yaw_command = yaw
        self._install_provider(provider if self.agent_control_enabled else None)

    def start_auto_takeoff(self):
        self.require_agent_control()
        self.hold_position()
        _start_position_auto_takeoff(self.data)
        return position_auto_target.copy()

    def start_auto_land(self):
        self.require_agent_control()
        self.hold_position()
        _start_position_auto_land(self.data)

    def get_position_auto_mode(self):
        self._check_open()
        return get_position_auto_mode()

    def step(self):
        self._check_open()
        mujoco.mj_step(self.model, self.data)

    def close(self):
        global joystick, model_stop_callback, agent_state_machine
        if not self._closed:
            mujoco.set_mjcb_control(None)
            set_planner_target_provider(None)
            self.agent_state_machine.set_enabled(False, "Environment closed")
            joystick = self._previous_joystick
            model_stop_callback = None
            agent_state_machine = None
            kx, kv, kR, mass, gravity = self._previous_position_parameters
            position_controller.set_parameters(kx, kv, kR)
            position_controller.vehicle_mass = mass
            position_controller.gravity = gravity
            self._closed = True
            MMUAVEnv._active = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == '__main__':
    viewer.launch(loader=load_callback)
    
    
