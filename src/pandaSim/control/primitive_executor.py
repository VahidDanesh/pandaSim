"""Pivot and slide experiments, identical in simulation and on the FER.

run_pivot: open gripper -> approach the grasp -> grasp -> pivot -> release.
run_plan:  a sequence of planner steps (pivots and slides), each from the ready configuration.
Every segment is the same loop: read theta, build q, reference twist V_c (omega, v)
in {c}, ContactQP -> joint velocities, send theta_dot, integrate phi_hat (free mode).
Both references are constant screw motions with time scaling, so they start and
end at zero velocity (no ramp on the QP output).
"""
from __future__ import annotations

import time

import numpy as np
from pytransform3d import transformations as pt

from pandaSim.control.backends import OPEN_WIDTH
from pandaSim.control.contact_qp import EE, FER_QD_MAX, ContactQP
from pandaSim.planning.clearance import palm_hits_box, palm_lowest
from pandaSim.planning.primitives import pivot_twist, pose_error_twist, screw_twist, time_scaling

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
        self.segments = []      # dicts: name, ok, cause, t0, t_end, Tf (run_plan adds step)
        self.events = []        # (name, t, ok, width): gripper calls and the finger opening they command
        self.steps = []         # run_plan: dicts kind, ok, T_obj (object pose after the step)

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
                 events=np.array(self.events, dtype=object), steps=np.array(self.steps, dtype=object), **meta)


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
        width = args[0] if args else OPEN_WIDTH        # finger opening after the call
        log.events.append((name, backend.time(), bool(ok), float(width)))
        if ok and viewer is not None:
            viewer.fingers(width)
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


class JointMove:
    """Joint-space move theta0 -> theta_goal, with the controller interface of run_segment (rigid model):
    theta_dot = s_dot (theta_goal - theta0) + K (theta_plan(s) - theta). Tf makes the peak joint speed qd_max."""

    n = 7

    def __init__(self, theta0, theta_goal, qd_max=0.5, K=1.0, method="quintic", Tf_min=1.0, tol=1e-3, qd_scale=0.8):
        self.qd_lim = qd_scale * FER_QD_MAX
        self.theta0, self.goal = np.array(theta0, float), np.array(theta_goal, float)
        self.d = self.goal - self.theta0
        self.K, self.method, self.tol = K, method, tol
        self.Tf = max(Tf_min, PEAK_SDOT[method[0].lower()] * np.abs(self.d).max() / qd_max)
        self.s, self.sdot = 0.0, 0.0

    def ref(self, t, q):
        self.s, self.sdot = time_scaling(t, self.Tf, self.method)
        return np.zeros(6)             # no task-space twist; solve() uses the time scaling stored here

    def solve(self, q, V_c):
        qd = self.sdot * self.d + self.K * (self.theta0 + self.s * self.d - q)
        return np.clip(qd, -self.qd_lim, self.qd_lim)      # the feedback term must not exceed the robot's limits

    def done(self, q):
        return np.abs(q - self.goal).max() < self.tol


def run_plan(backend, robot, steps, T_w_obj, free_rotation=True, q_ready=None, qp=None, approach=None, pivot=None,
             ready=None, grasp=None, safety=None, lift=0.10, viewer=None, approach_free=True):
    """Execute planner steps (sequencer.Step: kind, plan.T_grasp, plan.screws, plan.width) in order.

    Each step starts and ends at the ready configuration q_ready (default robot.qr):
        pre (above the grasp) -> approach -> close -> pivot | slide -> open -> retreat (straight up) -> ready
    Pivot: approach with the free-rotation model (phi_hat starts at 0 at every grasp), then the pivot screw
    with free_rotation as given, feed-forward only (as run_pivot). phi stays within plan.phi_lim, the interval
    in which the palm is clear of the object (qp["phi_lim"] replaces it; None: no limits).
    approach_free=False: the hand reaches plan.T_grasp as planned, without turning about the finger axis
    (a rigid grasp with a prescribed hand orientation; used by experiments/ablation.py). Slide: rigid model throughout, the hand
    stays as planned; each planar screw is tracked with pose feedback. Retreat and ready use the rigid model.

    qp: ContactQP kwargs (slack only applies to pivots); approach: approach_reference kwargs plus tol, used for
    pre, approach, slide and retreat; pivot: dict(w_max, method, Tf_min); ready: JointMove kwargs;
    grasp: dict(force); safety: run_segment kwargs. T_w_obj: object pose at the start (world).
    The run stops at the first failed segment (the gripper is left as it is).
    Returns (log, T_w_obj): the object pose after the last finished step, from the motion of the hand (open loop).
    """
    qp, approach, pivot, ready = dict(qp or {}), dict(approach or {}), dict(pivot or {}), dict(ready or {})
    grasp, safety = dict(grasp or {}), dict(safety or {})
    tol = approach.pop("tol", 1e-3)
    qp_hard = {k: v for k, v in qp.items() if k != "slack"}
    q_ready = np.asarray(robot.qr if q_ready is None else q_ready, float)[:7]
    T_w_obj = np.array(T_w_obj, float)
    fk = lambda q: contact_pose(robot, q)
    log = Log()
    step = -1

    def segment(ctrl, ref, Tf, name, phi=0.0, done=None):
        ok, phi = run_segment(backend, ctrl, ref, Tf, log, name, phi=phi, done=done, viewer=viewer, **safety)
        log.segments[-1]["step"] = step
        return ok, phi

    def move(ctrl, T_d, name, phi=0.0):
        """Screw motion of the controlled frame to T_d with pose feedback."""
        q = np.r_[backend.theta(), phi] if ctrl.n == 8 else backend.theta()
        ref, Tf = approach_reference(fk, fk(q), T_d, **approach)
        return segment(ctrl, ref, Tf, name, phi, lambda q: np.linalg.norm(pose_error_twist(fk(q), T_d)) < tol)

    def to_ready():
        jm = JointMove(backend.theta(), q_ready, **ready)
        return segment(jm, jm.ref, jm.Tf, "ready", done=jm.done)[0]

    def gripper(name, call, *args):
        backend.command(np.zeros(7))
        ok = call(*args)
        width = args[0] if args else OPEN_WIDTH        # finger opening after the call
        log.events.append((name, backend.time(), bool(ok), float(width)))
        if ok and viewer is not None:
            viewer.fingers(width)
        return ok

    def up(T):
        T = T.copy()
        T[2, 3] += lift
        return T

    rigid = ContactQP(robot, free_rotation=False, **qp_hard)
    backend.start()
    try:
        if not gripper("open", backend.open):
            return log, T_w_obj
        if np.abs(backend.theta() - q_ready).max() > 1e-4 and not to_ready():
            return log, T_w_obj
        for step, st in enumerate(steps):
            pl, is_pivot = st.plan, st.kind == "pivot"
            log.steps.append(dict(kind=st.kind, ok=False, T_obj=T_w_obj))
            lim = {"phi_lim": getattr(pl, "phi_lim", None)}        # palm clear of the object; qp can override it
            ctrl = ContactQP(robot, free_rotation=True, **{**lim, **qp_hard}) if is_pivot and approach_free else rigid
            phi = 0.0
            for name, T_d in (("pre", up(pl.T_grasp)), ("approach", pl.T_grasp)):
                ok, phi = move(ctrl, T_d, name, phi)
                if not ok:
                    return log, T_w_obj
            if not gripper("grasp", backend.grasp, pl.width, grasp.get("force", 20.0), grasp.get("speed", 0.05),
                           grasp.get("epsilon", 0.01)):
                return log, T_w_obj
            # the object moves with the controlled frame: {c} (free) or {E} (rigid)
            if is_pivot:
                ctrl = ContactQP(robot, free_rotation=free_rotation, **{**lim, **qp})
            q_now = lambda: np.r_[backend.theta(), phi] if ctrl.n == 8 else backend.theta()
            T_c_obj = np.linalg.inv(fk(q_now())) @ T_w_obj
            if viewer is not None:
                viewer.attach(T_c_obj)
            for S, theta in pl.screws:
                T_c = fk(q_now())
                if is_pivot:
                    method = pivot.get("method", "quintic")
                    Tf = max(pivot.get("Tf_min", 1.0), PEAK_SDOT[method[0].lower()] * abs(theta) / pivot.get("w_max", 0.3))
                    ok, phi = segment(ctrl, pivot_reference(screw_twist(S, theta, T_c), Tf, method), Tf, "pivot", phi)
                else:
                    ok, phi = move(ctrl, pt.transform_from_exponential_coordinates(np.asarray(S) * theta) @ T_c, "slide")
                if not ok:
                    return log, fk(q_now()) @ T_c_obj
            T_w_obj = fk(q_now()) @ T_c_obj
            if not gripper("release", backend.open):
                return log, T_w_obj
            if viewer is not None:
                viewer.detach()
            log.steps[-1].update(T_obj=T_w_obj)
            if not move(rigid, up(fk(backend.theta())), "retreat")[0] or not to_ready():
                return log, T_w_obj
            log.steps[-1].update(ok=True)
        return log, T_w_obj
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


def check_clearance(log, robot, T_w_obj, size, period=0.01, clearance=0.002, arm=None, arm_every=2):
    """Palm of the hand against the box and the table over a run_plan log (the planner and the QP do not
    check collisions of the arm; this is the check after the fact). T_w_obj: box pose at the start.
    The box pose at each sample follows from the log: at rest between the steps, moving with {c} while held.
    The palm must stay `clearance` [m] away from the box and above the table (the samples are `period` s apart,
    at most 3 mm of hand motion at the speeds used). Returns dict(box_ok, table_ok, hits): hits lists (t, segment name) of the samples with contact.
    arm: planning.collision.ArmCollision; if given, every `arm_every`-th sample also tests all arm links and the
    hand against the box and the table (distance >= arm.margin) and the result has arm_ok, arm_hits
    (t, segment, link) and arm_min (smallest distance)."""
    a = log.arrays()
    seg, theta, phi, t = a["segment"], a["theta"], a["phi"], a["t"]
    T_rest = [np.asarray(T_w_obj, float)] + [np.asarray(st["T_obj"], float) for st in log.steps]   # before step i, after it
    hits, table_ok = [], True
    arm_hits, arm_min, count = [], np.inf, 0
    T_c_obj = {}
    for j, sg in enumerate(log.segments):
        idx = np.flatnonzero(seg == j)
        if not len(idx):
            continue
        i, held = sg.get("step", -1), sg["name"] in ("pivot", "slide")
        if held and i not in T_c_obj:
            T_c_obj[i] = np.linalg.inv(contact_pose(robot, np.r_[theta[idx[0]], phi[idx[0]]])) @ T_rest[i]
        stride = max(1, int(round(period / max(np.median(np.diff(t[idx])) if len(idx) > 1 else period, 1e-9))))
        for k in np.r_[idx[::stride], idx[-1]]:
            if held:
                T_obj = contact_pose(robot, np.r_[theta[k], phi[k]]) @ T_c_obj[i]
            else:
                T_obj = T_rest[max(i, 0) + (sg["name"] in ("retreat", "ready") and i >= 0)]
            T_E = contact_pose(robot, theta[k])
            table_ok &= palm_lowest(T_E) > clearance
            if palm_hits_box(T_E, T_obj, size, clearance):
                hits.append((float(t[k]), sg["name"]))
            if arm is not None and count % arm_every == 0:
                ok, who, d = arm.ok(theta[k], T_obj, phi[k])
                arm_min = min(arm_min, d)
                if not ok:
                    arm_hits.append((float(t[k]), sg["name"], who))
            count += 1
    return dict(box_ok=not hits, table_ok=bool(table_ok), hits=hits, arm_ok=not arm_hits, arm_hits=arm_hits,
                arm_min=float(arm_min))
