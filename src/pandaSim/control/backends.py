"""Robot backends for the primitive executor: same calls in simulation and on the FER.

The executor only sends joint velocities. On the robot, panda-py's IntegratedVelocity controller (libfranka's
1 kHz thread, C++) integrates them into a position target q_d and tracks it with
    tau = Kp (q_d - q) - Kd dq            (integrated_velocity.cpp; gravity is compensated by the robot)
Three properties of that controller shape this file (read from the panda-py sources, see docs of PandaBackend):
  - it keeps integrating the LAST velocity command until a new one arrives: a stalled Python loop does not
    stop the arm (watchdog below);
  - q_d is not tied to the measured q: a blocked arm winds the torque up to the limits (tracking monitor);
  - the damping acts on dq, not on the velocity error, so the measured joints lag q_d by (Kd / Kp) dq
    (SimBackend(lag=...) reproduces this; the monitor subtracts it).
"""
from __future__ import annotations

import threading
import time

import numpy as np

OPEN_WIDTH = 0.08        # fully open Franka hand (m)

# panda-py 1.1.1 defaults (integrated_velocity.cpp, constants.h)
PANDA_KP = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])     # Nm/rad
PANDA_KD = np.array([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0])           # Nm s/rad
PANDA_LAG = PANDA_KD / PANDA_KP                                           # s: q lags q_d by lag * dq
# virtual joint walls that panda-py adds to every torque controller: a spring (1000 to 2000 Nm/rad on joints
# 1 to 4) pushes back within this distance of a joint limit, a damper within twice this distance
PANDA_WALL = np.array([0.12, 0.09, 0.09, 0.09, 0.0349, 0.0349, 0.0349])   # rad
# largest torque of the tracking term Kp (q_d - q) - Kd dq before the run is stopped. In free motion this
# torque only has to overcome friction, inertia and the payload (about 6 Nm on joint 2 for 1 kg at 0.6 m).
TRACK_TAU_MAX = np.array([20.0, 20.0, 20.0, 20.0, 6.0, 6.0, 4.0])         # Nm


class SimBackend:
    """Kinematic stand-in for the robot: q_d += theta_dot * dt per tick, time = ticks * dt.

    lag=None: the measured joints equal q_d (ideal tracking). lag = time constants [s] (e.g. PANDA_LAG):
    the measured joints follow q_d through a first-order lag, as the real controller does in steady motion.
    """

    def __init__(self, theta0, dt: float = 1e-3, lag=None):
        self._theta = np.array(theta0[:7], float)       # integrated command q_d
        self._q = self._theta.copy()                    # measured joints
        self.lag = None if lag is None else np.broadcast_to(np.asarray(lag, float), (7,)).copy()
        self.dt = dt
        self._t = 0.0
        self._qd = np.zeros(7)
        self.gripper_events = []

    def start(self):
        self._qd[:] = 0.0

    def stop(self):
        self._qd[:] = 0.0

    def tick(self) -> bool:
        self._theta += self._qd * self.dt
        if self.lag is None:
            self._q = self._theta.copy()
        else:
            self._q += (self._theta - self._q) * (1.0 - np.exp(-self.dt / self.lag))
        self._t += self.dt
        return True

    def time(self) -> float:
        return self._t

    def theta(self) -> np.ndarray:
        return self._q.copy()

    def target(self) -> np.ndarray:
        """Position target q_d (the integrated velocity command)."""
        return self._theta.copy()

    def fault(self):
        """None, or the reason the backend wants the run stopped (checked every tick by run_segment)."""
        return None

    def tracking(self) -> np.ndarray:
        """(q_d - q) - lag dq [rad]: zero in this model once the lag is in steady state."""
        if self.lag is None:
            return np.zeros(7)
        return self._theta - self._q - self.lag * self._qd

    def command(self, qd7):
        self._qd[:] = qd7

    def wrench(self):
        return np.zeros(6), np.zeros(7)

    def grasp(self, width: float, force: float, speed: float = 0.05, epsilon: float = 0.01) -> bool:
        self.gripper_events.append(("grasp", self._t, width, force))
        return True

    def open(self, width: float = OPEN_WIDTH, speed: float = 0.05) -> bool:
        self.gripper_events.append(("open", self._t, width))
        return True


class PandaBackend:
    """FER through panda-py: IntegratedVelocity torque controller, paced by PandaContext.

    `panda` is a connected panda_py.Panda, `gripper` a panda_py.libfranka.Gripper (or None).
    Gripper calls block; the controller holds the arm meanwhile (send zeros first).
    The robot state is read once per tick; theta() and wrench() use that same state.
    air=True: dry run without an object; "grasp" only closes the fingers to the object width (gripper.move), so
    a missing object is not an error.

    Safety (fault() is checked every tick by run_segment, which then stops the run):
      watchdog   a thread zeroes the velocity command if a non-zero command has not been renewed for
                 `stall_timeout` s. Without it the controller keeps integrating the last command for as long as
                 the Python loop is stuck (it cannot help if the interpreter itself is frozen inside a C call;
                 if the whole process dies the control thread dies with it and the robot stops on its own).
      tracking   |Kp (q_d - q) - Kd dq| above track_tau_max [Nm] on any joint, i.e. the torque with which the
                 controller pulls the arm to the target: the arm is blocked, saturated or pushed away. The
                 same number in rad, (q_d - q) - lag dq, is in tracking() and in the log.
    stop() ends the controller: the robot's own joint controller then holds the measured position, so a
    wound-up target is dropped. The gripper is never moved by stop().
    """

    def __init__(self, panda, gripper=None, frequency: float = 1000.0, stiffness=None, damping=None, air: bool = False,
                 stall_timeout: float = 0.05, track_tau_max=TRACK_TAU_MAX):
        self.panda, self.gripper, self.frequency, self.air = panda, gripper, frequency, air
        self._gains = {k: v for k, v in (("stiffness", stiffness), ("damping", damping)) if v is not None}
        self.kp = PANDA_KP if stiffness is None else np.asarray(stiffness, float)
        self.kd = PANDA_KD if damping is None else np.asarray(damping, float)
        self.lag = self.kd / self.kp
        self.stall_timeout = stall_timeout
        self.track_tau_max = None if track_tau_max is None else np.broadcast_to(np.asarray(track_tau_max, float), (7,))
        self.ctrl = self.ctx = self._state = None
        self._track = np.zeros(7)
        self._fault = None
        self._armed_since = None                   # monotonic time of the last non-zero command, or None
        self._lock = threading.Lock()
        self._watchdog = None
        self._running = threading.Event()

    # ---- life cycle ----
    def start(self):
        from panda_py import controllers

        self._fault, self._armed_since = None, None
        self.ctrl = controllers.IntegratedVelocity(**self._gains)
        self.panda.start_controller(self.ctrl)
        self.ctx = self.panda.create_context(frequency=self.frequency)
        if self.stall_timeout:
            self._running.set()
            self._watchdog = threading.Thread(target=self._watch, name="panda-watchdog", daemon=True)
            self._watchdog.start()

    def stop(self):
        """Zero velocity, then end the controller (the robot holds its position). Safe to call twice."""
        self._running.clear()
        try:
            if self.ctrl is not None:
                self.ctrl.set_control(np.zeros(7))
        finally:
            self.panda.stop_controller()
            if self._watchdog is not None and self._watchdog is not threading.current_thread():
                self._watchdog.join(timeout=1.0)
                self._watchdog = None

    def _watch(self):
        period = min(0.005, self.stall_timeout / 4)
        while self._running.is_set():
            time.sleep(period)
            with self._lock:
                since = self._armed_since
                if since is None or time.monotonic() - since <= self.stall_timeout:
                    continue
                self._armed_since = None
                self.ctrl.set_control(np.zeros(7))          # the arm stops here, whatever the main thread does
                if self._fault is None:
                    self._fault = (f"loop stall: no velocity command for more than {self.stall_timeout * 1e3:.0f} ms "
                                   "(the watchdog set the velocity to zero)")

    # ---- per tick ----
    def tick(self) -> bool:
        ok = self.ctx.ok()            # waits for the next period; raises on a robot error
        self._state = s = self.panda.get_state()
        if self.track_tau_max is not None:
            self._track = np.asarray(self.ctrl.get_qd(), float) - np.asarray(s.q, float) \
                - self.lag * np.asarray(s.dq, float)
            tau = self.kp * self._track
            j = int(np.argmax(np.abs(tau) / self.track_tau_max))
            if abs(tau[j]) > self.track_tau_max[j] and self._fault is None:
                self._fault = (f"tracking error: joint {j + 1} is {self._track[j] * 1e3:.0f} mrad from its target "
                               f"({abs(tau[j]):.0f} Nm of tracking torque, limit {self.track_tau_max[j]:.0f} Nm)")
        return ok

    def fault(self):
        return self._fault

    def tracking(self) -> np.ndarray:
        """(q_d - q) - lag dq [rad] at the last tick: zero for ideal tracking in free motion."""
        return self._track.copy()

    def time(self) -> float:
        return self.ctrl.get_time()

    def theta(self) -> np.ndarray:
        if self._state is None:       # before the first tick
            self._state = self.panda.get_state()
        return np.asarray(self._state.q, float)

    def target(self) -> np.ndarray:
        return np.asarray(self.ctrl.get_qd(), float)

    def command(self, qd7):
        qd7 = np.asarray(qd7, float)
        with self._lock:
            if self._fault is not None:                     # after a fault only zeros reach the robot
                qd7 = np.zeros(7)
            self._armed_since = time.monotonic() if np.any(qd7 != 0.0) else None
            self.ctrl.set_control(qd7)

    def wrench(self):
        s = self._state
        return np.asarray(s.O_F_ext_hat_K, float), np.asarray(s.tau_ext_hat_filtered, float)

    # ---- gripper ----
    def grasp(self, width: float, force: float, speed: float = 0.05, epsilon: float = 0.01) -> bool:
        if self.air:
            return self.gripper.move(width, speed)
        return self.gripper.grasp(width, speed, force, epsilon, epsilon)

    def open(self, width: float = OPEN_WIDTH, speed: float = 0.05) -> bool:
        return self.gripper.move(width, speed)
