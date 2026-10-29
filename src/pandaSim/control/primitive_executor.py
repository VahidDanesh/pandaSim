"""One pivot experiment, identical in simulation and on the FER.

Sequence: open gripper -> approach the grasp -> grasp -> pivot -> release.
Every segment is the same loop: read theta, build q, reference twist V_c (omega, v)
in {c}, ContactQP -> joint velocities, send theta_dot, integrate phi_hat (free mode).
Both references are constant screw motions with time scaling, so they start and
end at zero velocity (no ramp on the QP output).
"""
from __future__ import annotations

import time

import numpy as np
from pytransform3d import transformations as pt

from pandaSim.control.contact_qp import EE, FER_QD_MAX, ContactQP
from pandaSim.planning.primitives import pivot_twist, pose_error_twist, time_scaling

PEAK_SDOT = {"q": 1.875, "c": 1.5, "l": 1.0}   # max of s_dot * Tf per time scaling

# libfranka 0.8.0 kMaxJointAcceleration (rad/s^2), reported by check_profile
FER_QDD_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])


def contact_pose(robot, q):
    """World pose of {c}: q = (theta, phi), or theta for the rigid grasp ({c} = {E}). Includes robot.base."""
    q8 = q if len(q) == 8 else np.r_[q, 0.0]
    return robot.base.A @ robot.ets(end=EE).eval(q8)


def approach_reference(fk, T_wc0, T_wd, K_e=1.0, v_max=0.1, w_max=0.3, method="quintic", Tf_min=1.0):
    """Screw motion from {c} at T_wc0 to the grasp pose T_wd, plus pose feedback.

    V_c(t) = s_dot V1 + K_e log(T_wc^-1 T_plan(s)),  V1 = log(T_wc0^-1 T_wd),  T_plan(s) = T_wc0 exp(V1 s).
    Tf makes the feed-forward peak equal to v_max (m/s) or w_max (rad/s). Returns (ref(t, q), Tf).
    """
    V1 = pose_error_twist(T_wc0, T_wd)
    peak = PEAK_SDOT[method[0].lower()]
    Tf = max(Tf_min, peak * max(np.linalg.norm(V1[:3]) / w_max, np.linalg.norm(V1[3:]) / v_max))

    def ref(t, q):
        s, sdot = time_scaling(t, Tf, method)
        T_plan = T_wc0 @ pt.transform_from_exponential_coordinates(V1 * s)
        return sdot * V1 + K_e * pose_error_twist(fk(q), T_plan)

    return ref, Tf


def pivot_reference(V1, Tf, method="quintic"):
    """Feed-forward pivot twist s_dot(t) V1 (manuscript Sec. 6). Returns ref(t, q)."""
    return lambda t, q: time_scaling(t, Tf, method)[1] * V1


class Log:
    """Per-tick records of one run, plus segment results and gripper events."""

    FIELDS = ("t", "segment", "theta", "phi", "qd", "V", "solve_time", "tick_time", "F_ext", "tau_ext")

    def __init__(self):
        self.rows = {k: [] for k in self.FIELDS}
        self.segments = []      # dicts: name, ok, cause, t0, t_end, Tf
        self.events = []        # (name, t, ok)

    def record(self, **row):
        for k in self.FIELDS:
            self.rows[k].append(row[k])

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.rows.items()}

    @property
    def ok(self):
        return bool(self.segments) and all(s["ok"] for s in self.segments)

    def save(self, path, **meta):
        np.savez(path, **self.arrays(), segments=np.array(self.segments, dtype=object),
                 events=np.array(self.events, dtype=object), **meta)


def run_segment(backend, ctrl, ref, Tf, log, name, phi=0.0, done=None, settle=2.0, f_max=30.0, tau_max=15.0,
                viewer=None):
    """Track ref(t, q) with ctrl until t >= Tf (and done(q), if given). Returns (ok, phi).

    Stops with zero velocity on an infeasible QP, an external force/torque above
    f_max (N) / tau_max (Nm), a robot error, or t > Tf + settle; the cause is logged.
    """
    free = ctrl.n == 8
    t0, t_last, phi_dot = backend.time(), 0.0, 0.0
    cause = None
    seg = len(log.segments)
    try:
        while backend.tick():
            tick_start = time.perf_counter()
            t = backend.time() - t0
            phi += phi_dot * (t - t_last)      # phi(t + dt) = phi(t) + phi_dot dt (phi is not measured)
            t_last = t
            theta = backend.theta()
            q = np.r_[theta, phi] if free else theta
            if t >= Tf and (done is None or done(q)):
                break
            if t > Tf + settle:
                cause = "timeout"
                break
            F_ext, tau_ext = backend.wrench()
            if np.linalg.norm(F_ext[:3]) > f_max or np.abs(tau_ext).max() > tau_max:
                cause = "external wrench"
                break
            V = ref(t, q)
            ts = time.perf_counter()
            qd = ctrl.solve(q, V)
            solve_time = time.perf_counter() - ts
            if qd is None:
                cause = "infeasible QP"
                break
            backend.command(qd[:7])
            phi_dot = qd[7] if free else 0.0
            phi_model = phi if free else 0.0   # rigid: {c} = {E}, the model has phi = 0
            if viewer is not None:
                viewer.update(np.r_[theta, phi_model])
            log.record(t=t0 + t, segment=seg, theta=theta, phi=phi_model, qd=np.r_[qd[:7], phi_dot], V=V,
                       solve_time=solve_time, tick_time=time.perf_counter() - tick_start, F_ext=F_ext, tau_ext=tau_ext)
    except RuntimeError as e:          # panda-py raises robot errors from PandaContext.ok()
        cause = f"robot error: {e}"
    finally:
        backend.command(np.zeros(7))
    log.segments.append(dict(name=name, ok=cause is None, cause=cause, t0=t0, t_end=backend.time(), Tf=Tf))
    return cause is None, phi


def run_pivot(backend, robot, T_wd, pivot, free_rotation=True, qp=None, approach=None, grasp=None, safety=None,
              viewer=None, T_w_obj=None):
    """Open gripper, approach {c} to the grasp pose T_wd, grasp, pivot, release.

    pivot: dict(q, s_axis, theta, Tf, method="quintic"), axis in the world frame.
    The approach always uses the free-rotation model without slack (shared by all modes);
    the pivot uses free_rotation and qp as given. qp: ContactQP kwargs; approach: approach_reference kwargs plus tol;
    grasp: dict(width, force); safety: run_segment kwargs (f_max, tau_max, settle).
    viewer: LiveViewer or None; T_w_obj: object pose (world), so the viewer can carry it.
    The object is released only after a successful pivot.
    """
    qp, approach, grasp, safety = dict(qp or {}), dict(approach or {}), dict(grasp or {}), dict(safety or {})
    tol = approach.pop("tol", 1e-3)
    fk = lambda q: contact_pose(robot, q)
    log = Log()

    def gripper(name, call, *args):
        backend.command(np.zeros(7))
        ok = call(*args)
        log.events.append((name, backend.time(), bool(ok)))
        return ok

    backend.start()
    try:
        if not gripper("open", backend.open):
            return log
        # approach, free model, phi_hat starts at 0
        approach_qp = ContactQP(robot, free_rotation=True, **{k: v for k, v in qp.items() if k != "slack"})
        q = np.r_[backend.theta(), 0.0]
        ref, Tf = approach_reference(fk, fk(q), T_wd, **approach)
        done = lambda q: np.linalg.norm(pose_error_twist(fk(q), T_wd)) < tol
        ok, phi = run_segment(backend, approach_qp, ref, Tf, log, "approach", phi=0.0, done=done, viewer=viewer, **safety)
        if not ok or not gripper("grasp", backend.grasp, grasp["width"], grasp.get("force", 20.0)):
            return log
        # pivot from the reached frame: {c} (free) or {E} (rigid)
        ctrl = ContactQP(robot, free_rotation=free_rotation, **qp)
        q = np.r_[backend.theta(), phi] if free_rotation else backend.theta()
        if viewer is not None and T_w_obj is not None:
            viewer.attach(np.linalg.inv(fk(q)) @ T_w_obj)
        V1 = pivot_twist(pivot["q"], pivot["s_axis"], pivot["theta"], fk(q))
        method = pivot.get("method", "quintic")
        ok, phi = run_segment(backend, ctrl, pivot_reference(V1, pivot["Tf"], method), pivot["Tf"], log, "pivot",
                              phi=phi, viewer=viewer, **safety)
        if ok:
            gripper("release", backend.open)
            if viewer is not None:
                viewer.detach()
        return log
    finally:
        backend.stop()


def check_profile(log, qd_scale=0.8):
    """Simulation gate before the robot: joint speeds within libfranka limits x qd_scale (hard),
    joint accelerations vs libfranka kMaxJointAcceleration (reported; under torque control a
    step in theta_dot changes the slope of the position target, not the target)."""
    a = log.arrays()
    qd, t, seg = a["qd"][:, :7], a["t"], a["segment"]
    same = seg[1:] == seg[:-1]
    qdd = np.abs(np.diff(qd, axis=0) / np.diff(t)[:, None])[same]
    vel_ratio = (np.abs(qd) / (qd_scale * FER_QD_MAX)).max(axis=0)
    acc_ratio = (qdd.max(axis=0) if len(qdd) else np.zeros(7)) / FER_QDD_MAX
    return dict(vel_ok=bool(np.all(vel_ratio <= 1 + 1e-9)), vel_ratio=vel_ratio,
                acc_ok=bool(np.all(acc_ratio <= 1)), acc_ratio=acc_ratio)
