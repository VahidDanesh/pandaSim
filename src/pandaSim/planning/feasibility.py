"""Kinematic check of planner steps: run each step through the executor in simulation.

A step is feasible if run_plan finishes it on SimBackend (coarse time step): the QP stays feasible
(joint limits and joint speed limits) and every segment reaches its target. This is the same code and the
same model (free rotation or rigid grasp) that later moves the robot, so the check and the execution agree.
Every step starts and ends at the ready configuration, so steps are checked alone and results are cached.
"""
from __future__ import annotations

import numpy as np


class Rollout:
    """`kin` of sequencer.Sequencer. check(step) -> (feasible, smallest distance to a joint limit [rad]).

    size: box dimensions; if given, a step is also infeasible if the palm of the hand, any link of the arm or the
    hand comes within `margin` of the box or the table at any time of the rollout
    (primitive_executor.check_clearance, planning.collision.ArmCollision)."""

    def __init__(self, robot, free_rotation=True, q_ready=None, dt=0.01, size=None, margin=0.002, **run_kwargs):
        from pandaSim.control.primitive_executor import contact_pose

        self.robot, self.free_rotation, self.dt, self.run_kwargs = robot, free_rotation, dt, run_kwargs
        self.size = None if size is None else np.asarray(size, float)
        self.arm = None
        if size is not None:
            from pandaSim.planning.collision import ArmCollision

            self.arm = ArmCollision(robot, size, margin)
        self.q_ready = np.asarray(robot.qr if q_ready is None else q_ready, float)[:7]
        self.T_ready = contact_pose(robot, self.q_ready)
        self._cache = {}

    def check(self, step):
        pl = step.plan
        key = (step.kind, pl.T_grasp.tobytes(), tuple((S.tobytes(), float(th)) for S, th in pl.screws))
        if key not in self._cache:
            self._cache[key] = self._run(step)
        return self._cache[key]

    def _run(self, step):
        from pandaSim.control.backends import SimBackend
        from pandaSim.control.primitive_executor import run_plan

        log, _ = run_plan(SimBackend(self.q_ready, dt=self.dt), self.robot, [step], step.T_before,
                          free_rotation=self.free_rotation, q_ready=self.q_ready, **self.run_kwargs)
        if not log.ok or not log.steps[-1]["ok"]:
            return False, 0.0
        if self.size is not None:
            from pandaSim.control.primitive_executor import check_clearance

            c = check_clearance(log, self.robot, step.T_before, self.size, period=0.02, arm=self.arm)
            if not (c["box_ok"] and c["table_ok"] and c["arm_ok"]):
                return False, 0.0
        theta = log.arrays()["theta"]
        lb, ub = self.robot.qlim[0, :7], self.robot.qlim[1, :7]
        return True, float(np.minimum(theta - lb, ub - theta).min())


def grasp_observability(step, min_arm=0.02, max_cos=0.9):
    """Is the bearing angle phi determined by the motion of the hand? (ok, margin, text)

    The fingertip bearing lets the object turn about the finger axis n (through the grasp point). phi is neither
    driven nor measured: it is what the object-environment contact makes it. So the grasp is only valid if the
    object cannot turn about n while it keeps that contact (manuscript Sec. 6, assumption behind the virtual joint):
      pivot  about a line (edge) without slipping: the object can only rotate about that line, so turning about
             n is possible only if n lies on it. margin = distance [m] of the grasp point from the pivot axis
             when n is parallel to it (required: >= min_arm); n not parallel to the axis is always valid.
      slide  on a face: the object can turn about any vertical axis, so n must not be vertical.
             margin = 1 - |n_z| (required: |n_z| <= max_cos).
    """
    pl = step.plan
    n, c = pl.T_grasp[:3, 1], pl.T_grasp[:3, 3]
    if step.kind == "pivot":
        S = np.asarray(pl.S, float)
        u = S[:3] / np.linalg.norm(S[:3])
        p_axis = np.cross(u, S[3:] / np.linalg.norm(S[:3]))             # point of the axis closest to the origin
        if np.linalg.norm(np.cross(n, u)) > 0.1:                        # more than 6 deg from parallel
            return True, float("inf"), "the finger axis is not parallel to the pivot axis"
        d = c - p_axis
        arm = float(np.linalg.norm(d - (d @ u) * u))
        return arm >= min_arm, arm, f"grasp point {arm * 1e3:.0f} mm from the pivot axis (at least {min_arm * 1e3:.0f} mm)"
    cos = abs(float(n[2]))
    return cos <= max_cos, 1.0 - cos, f"finger axis {np.degrees(np.arccos(min(cos, 1.0))):.0f} deg from the vertical"
