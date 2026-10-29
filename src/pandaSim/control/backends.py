"""Robot backends for the primitive executor: same calls in simulation and on the FER.

The executor only sends joint velocities. On the robot, panda-py's IntegratedVelocity
controller (libfranka's 1 kHz thread, C++) integrates them into a position target and
tracks it; between our ticks it keeps the last command. SimBackend mimics that.
"""
from __future__ import annotations

import numpy as np


class SimBackend:
    """Kinematic stand-in for the robot: theta += theta_dot * dt per tick, time = ticks * dt."""

    def __init__(self, theta0, dt: float = 1e-3):
        self._theta = np.array(theta0[:7], float)
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
        self._t += self.dt
        return True

    def time(self) -> float:
        return self._t

    def theta(self) -> np.ndarray:
        return self._theta.copy()

    def command(self, qd7):
        self._qd[:] = qd7

    def wrench(self):
        return np.zeros(6), np.zeros(7)

    def grasp(self, width: float, force: float, speed: float = 0.05, epsilon: float = 0.01) -> bool:
        self.gripper_events.append(("grasp", self._t, width, force))
        return True

    def open(self, width: float = 0.08, speed: float = 0.05) -> bool:
        self.gripper_events.append(("open", self._t, width))
        return True


class PandaBackend:
    """FER through panda-py: IntegratedVelocity torque controller, paced by PandaContext.

    `panda` is a connected panda_py.Panda, `gripper` a panda_py.libfranka.Gripper (or None).
    Gripper calls block; the controller holds the arm meanwhile (send zeros first).
    The robot state is read once per tick; theta() and wrench() use that same state.
    """

    def __init__(self, panda, gripper=None, frequency: float = 1000.0, stiffness=None, damping=None):
        self.panda, self.gripper, self.frequency = panda, gripper, frequency
        self._gains = {k: v for k, v in (("stiffness", stiffness), ("damping", damping)) if v is not None}
        self.ctrl = self.ctx = self._state = None

    def start(self):
        from panda_py import controllers

        self.ctrl = controllers.IntegratedVelocity(**self._gains)
        self.panda.start_controller(self.ctrl)
        self.ctx = self.panda.create_context(frequency=self.frequency)

    def stop(self):
        try:
            if self.ctrl is not None:
                self.ctrl.set_control(np.zeros(7))
        finally:
            self.panda.stop_controller()

    def tick(self) -> bool:
        ok = self.ctx.ok()            # waits for the next period; raises on a robot error
        self._state = self.panda.get_state()
        return ok

    def time(self) -> float:
        return self.ctrl.get_time()

    def theta(self) -> np.ndarray:
        if self._state is None:       # before the first tick
            self._state = self.panda.get_state()
        return np.asarray(self._state.q, float)

    def command(self, qd7):
        self.ctrl.set_control(np.asarray(qd7, float))

    def wrench(self):
        s = self._state
        return np.asarray(s.O_F_ext_hat_K, float), np.asarray(s.tau_ext_hat_filtered, float)

    def grasp(self, width: float, force: float, speed: float = 0.05, epsilon: float = 0.01) -> bool:
        return self.gripper.grasp(width, speed, force, epsilon, epsilon)

    def open(self, width: float = 0.08, speed: float = 0.05) -> bool:
        return self.gripper.move(width, speed)
