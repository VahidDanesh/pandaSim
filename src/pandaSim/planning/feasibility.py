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

    size: box dimensions; if given, a step is also infeasible if the palm of the hand touches the box or the
    table at any time of the rollout (primitive_executor.check_clearance)."""

    def __init__(self, robot, free_rotation=True, q_ready=None, dt=0.01, size=None, **run_kwargs):
        from pandaSim.control.primitive_executor import contact_pose

        self.robot, self.free_rotation, self.dt, self.run_kwargs = robot, free_rotation, dt, run_kwargs
        self.size = None if size is None else np.asarray(size, float)
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

            c = check_clearance(log, self.robot, step.T_before, self.size, period=0.02)
            if not (c["box_ok"] and c["table_ok"]):
                return False, 0.0
        theta = log.arrays()["theta"]
        lb, ub = self.robot.qlim[0, :7], self.robot.qlim[1, :7]
        return True, float(np.minimum(theta - lb, ub - theta).min())
