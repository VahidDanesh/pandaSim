"""Pivot and slide experiments, identical in simulation and on the FER.

run_plan: a sequence of planner steps (pivots, slides, gait steps), each from the ready configuration.
Every segment is the same loop: read theta, build q, reference twist V_c (omega, v)
in {c}, ContactQP -> joint velocities, send theta_dot, integrate phi_hat (free mode).
Both references are constant screw motions with time scaling, so they start and
end at zero velocity (no ramp on the QP output).
"""
from __future__ import annotations

import time

import numpy as np

from pandaSim.control.backends import OPEN_WIDTH
from pandaSim.control.contact_qp import EE, FER_QD_MAX, ContactQP
from pandaSim.planning import se3
from pandaSim.planning.clearance import hand_hits_box, palm_lowest
from pandaSim.planning.primitives import pose_error_twist, screw_twist, time_scaling

PEAK_SDOT = {"q": 1.875, "c": 1.5, "l": 1.0}   # max of s_dot * Tf per time scaling

# libfranka 0.8.0 kMaxJointAcceleration (rad/s^2), reported by check_profile
FER_QDD_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])


def contact_pose(robot, q):
    """World pose of {c}: q = (theta, phi), or theta for the rigid grasp ({c} = {E}). Includes robot.base."""
    q8 = q if len(q) == 8 else np.append(q, 0.0)      # np.append: the same array as np.r_, 2x faster per tick
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
        T_plan = T_wc0 @ se3.exp6(V1 * s)
        return sdot * V1 + K_e * pose_error_twist(fk(q), T_plan)

    return ref, Tf


def pivot_reference(V1, Tf, method="quintic"):
    """Feed-forward pivot twist s_dot(t) V1 (manuscript Sec. 6). Returns ref(t, q)."""
    return lambda t, q: time_scaling(t, Tf, method)[1] * V1


class Log:
    """Per-tick records of one run, plus segment results and gripper events."""

    FIELDS = ("t", "segment", "theta", "phi", "qd", "V", "solve_time", "tick_time", "F_ext", "tau_ext", "track", "target")

    def __init__(self):
        self.rows = {k: [] for k in self.FIELDS}
        self.segments = []      # dicts: name, ok, cause, t0, t_end, Tf (run_plan adds step)
        self.events = []        # (name, t, ok, width): gripper calls and the finger opening they command
        self.steps = []         # run_plan: dicts kind, ok, T_obj (object pose after the step)
        self.abort = None       # run_plan: dict(cause, segment, step, holding) if the run stopped early

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
                 events=np.array(self.events, dtype=object), steps=np.array(self.steps, dtype=object),
                 abort=np.array(self.abort, dtype=object), **meta)


def wrench_excess(F_ext, tau_ext, f_max, tau_max, load=None):
    """External wrench estimate against its thresholds: max(|F| / f_max, max_j |tau_j| / tau_max); above 1 the
    run stops. F_ext: force part of O_F_ext_hat_K (base frame), tau_ext: tau_ext_hat_filtered.

    load = (f, tau, band_f, band_tau, lo): the force [N, base frame] and joint torques [Nm] that the held object is
    predicted to put on the arm (predicted_load; no load is configured on the robot), the half-widths of the band
    around them (band_f [N], band_tau [Nm per joint]), and the smallest share lo of that load to expect (1 on an edge
    or a vertex; 0 while the box is partly on a face or in a slide, where the table may carry any part of it).
    The best-fitting share alpha in [lo, 1] of the load is removed, then up to the band of what is left; the rest
    is compared with the thresholds (design review D14: no longer any fraction of the full weight).
    """
    w = np.r_[np.asarray(F_ext, float)[:3] / f_max, np.asarray(tau_ext, float) / tau_max]
    if load is None:
        return float(max(np.linalg.norm(w[:3]), np.abs(w[3:]).max()))
    f, tau, band_f, band_tau, lo = load
    a = np.r_[np.asarray(f, float) / f_max, np.asarray(tau, float) / tau_max]
    r = w - np.clip(a @ w / max(a @ a, 1e-12), lo, 1.0) * a
    rf = max(float(np.linalg.norm(r[:3])) - band_f / f_max, 0.0)
    rt = np.maximum(np.abs(r[3:]) - np.asarray(band_tau, float) / tau_max, 0.0)
    return float(max(rf, rt.max()))


def predicted_load(robot, step, k, T_obj, box, kp=None, every=10):
    """The load of the held box on the arm during screw k of a planner step, for the wrench stop: a function
    q -> (f, tau, band_f, band_tau, lo) of wrench_excess. T_obj: the box pose at the start of the screw; box: size,
    mass, mu (sequencer.Box); kp: joint stiffness of the robot's controller (default PANDA_KP).
    The load is the closed chain of planning/load.py (bound b, as design review D14 asks), from the measured hand
    alone: at an edge or a vertex the box angle follows from the grasp point's turn about the screw axis (phi is
    not measured), in a slide the box moves with the hand. External wrench = -(F, M) of the hand on the box, joint
    torques -J^T (F, M). Band: load.band(F) on the force, 20 % of each joint torque plus the torque of load.BAND_N.
    Recomputed every `every` calls (10 ms at 1 kHz; the load changes slowly)."""
    from pandaSim.control.backends import PANDA_KP
    from pandaSim.planning import load as L

    ets = robot.ets(end=EE)
    Kinv = np.diag(1.0 / np.asarray(PANDA_KP if kp is None else kp, float))
    base = robot.base.A
    T_obj0 = np.asarray(T_obj, float)
    ct0 = L.contact(step, k, T_obj0, box.size)
    memo = dict(n=0, T_c0=None, out=None)

    def load(q):
        memo["n"] += 1
        if memo["out"] is not None and memo["n"] % every:
            return memo["out"]
        q8 = np.r_[np.asarray(q, float)[:7], 0.0]                 # c and n do not depend on phi
        T_c = base @ ets.eval(q8)
        if memo["T_c0"] is None:
            memo["T_c0"] = T_c
        if ct0["kind"] == "slide":
            T_o = T_c @ np.linalg.inv(memo["T_c0"]) @ T_obj0
        else:
            th = L.screw_angle(ct0["S"], memo["T_c0"][:3, 3], T_c[:3, 3])
            T_o = se3.exp6(ct0["S"] * th) @ T_obj0
        J = ets.jacob0(q8)[:, :7]
        ct = L.contact(step, k, T_o, box.size)
        w = L.wrench(ct, T_o, T_c, box.size, box.mass, box.mu, J @ Kinv @ J.T)
        W = -np.r_[w["F_b"], w["M_b"]]
        tau = J.T @ W
        lo = 0.0 if ct["kind"] == "slide" else 1.0
        if ct["kind"] == "edge":                                   # partly on a face within the deflection angle
            r = float(np.linalg.norm(L._perp(T_c[:3, 3] - ct["p"], ct["u"])))
            d = float(np.linalg.norm(w["x_free"][:3])) / r
            if L.face_ramp(th, ct["theta"], d, ct["starts_on_face"], ct["ends_on_face"]) < 1.0:
                lo = 0.0
        memo["out"] = (W[:3], tau, L.band(W[:3]), L.BAND_FRAC * np.abs(tau) + L.BAND_N * np.linalg.norm(J[:3], axis=0), lo)
        return memo["out"]

    return load


def run_segment(backend, ctrl, ref, Tf, log, name, phi=0.0, done=None, settle=2.0, f_max=30.0, tau_max=15.0,
                dt_max=0.05, load=None, lead=None, lead_T=0.02, viewer=None):
    """Track ref(t, q) with ctrl until t >= Tf (and done(q), if given). Returns (ok, phi).

    lead: None, or the time constants [s] by which the measured joints lag the position target of the robot's
    controller (backend.lag). The target is then led by lead * theta_dot, so that the measured joints follow the
    commanded motion without that lag: the velocity sent is theta_dot + u, where u moves the injected lead L
    towards lead * theta_dot with the time constant lead_T. L is bounded by lead * |theta_dot| and returns to zero
    when the motion stops, so nothing accumulates.

    Stops with zero velocity and logs the cause on: an infeasible QP; an external force/torque above
    f_max (N) / tau_max (Nm) (wrench_excess; load(q) -> its load argument for a held object (predicted_load), or None);
    a fault reported by the backend (watchdog, tracking error); more than dt_max s between two ticks
    (the arm kept its last velocity meanwhile); a robot error; t > Tf + settle; a backend that stops ticking.
    Any other exception (Ctrl-C included) also zeroes the velocity and is recorded, then it propagates.
    """
    free = ctrl.n == 8
    t0, t_last, phi_dot = backend.time(), 0.0, 0.0
    cause = "backend stopped"          # a tick() that returns False must not look like a finished segment
    seg = len(log.segments)
    L, u = np.zeros(7), np.zeros(7)    # lead of the position target [rad] and its rate
    qd_lim = getattr(ctrl, "qd_lim", getattr(ctrl, "qd_max", None))    # joint speed limits of the controller
    try:
        while backend.tick():
            tick_start = time.perf_counter()
            t = backend.time() - t0
            dt = t - t_last
            phi += phi_dot * dt                # phi(t + dt) = phi(t) + phi_dot dt (phi is not measured)
            L += u * dt
            t_last = t
            theta = backend.theta()
            q = np.append(theta, phi) if free else theta
            fault = backend.fault()
            if fault is not None:
                cause = fault
                break
            if dt > dt_max:
                cause = f"loop stall: {dt * 1e3:.0f} ms between two ticks"
                break
            if t >= Tf and (done is None or done(q)):
                cause = None
                break
            if t > Tf + settle:
                cause = "timeout"
                break
            F_ext, tau_ext = backend.wrench()
            if wrench_excess(F_ext, tau_ext, f_max, tau_max, None if load is None else load(q)) > 1.0:
                cause = "external wrench"
                break
            V = ref(t, q)
            ts = time.perf_counter()
            qd = ctrl.solve(q, V)
            solve_time = time.perf_counter() - ts
            if qd is None:
                cause = "infeasible QP"
                break
            if lead is None:
                backend.command(qd[:7])
            else:
                u = (lead * qd[:7] - L) / lead_T
                sent = qd[:7] + u
                if qd_lim is not None:
                    sent = np.clip(sent, -qd_lim, qd_lim)
                    u = sent - qd[:7]
                backend.command(sent)
            phi_dot = qd[7] if free else 0.0
            phi_model = phi if free else 0.0   # rigid: {c} = {E}, the model has phi = 0
            if viewer is not None:
                viewer.update(np.r_[theta, phi_model])
            log.record(t=t0 + t, segment=seg, theta=theta, phi=phi_model, qd=np.append(qd[:7], phi_dot), V=V,
                       solve_time=solve_time, tick_time=time.perf_counter() - tick_start, F_ext=F_ext, tau_ext=tau_ext,
                       track=backend.tracking(), target=backend.target())
    except RuntimeError as e:          # panda-py raises robot errors from PandaContext.ok()
        cause = f"robot error: {e}"
    except BaseException as e:         # Ctrl-C, or a bug: the arm is stopped below, then the exception goes on
        cause = f"{type(e).__name__}: {e}".rstrip(": ")
        raise
    finally:
        try:
            backend.command(np.zeros(7))
        finally:                       # recorded even if the zero command raises (its exception then goes on)
            log.segments.append(dict(name=name, ok=cause is None, cause=cause, t0=t0, t_end=backend.time(), Tf=Tf))
    return cause is None, phi


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
             ready=None, grasp=None, safety=None, lift=0.10, viewer=None, approach_free=True, box=None, lead=None,
             log=None, kp=None):
    """Execute planner steps (sequencer.Step: kind, plan.T_grasp, plan.screws, plan.width) in order.

    Each step starts and ends at the ready configuration q_ready (default robot.qr):
        pre (above the grasp) -> approach -> close -> pivot | slide -> open -> retreat (straight up) -> ready
    Pivot: approach with the free-rotation model (phi_hat starts at 0 at every grasp), then the pivot screw
    with free_rotation as given, feed-forward only. phi stays within plan.phi_lim, the interval
    in which the palm is clear of the object (qp["phi_lim"] replaces it; None: no limits).
    approach_free=False: the hand reaches plan.T_grasp as planned, without turning about the finger axis
    (a rigid grasp with a prescribed hand orientation; used by experiments/ablation.py). Slide: rigid model throughout, the hand
    stays as planned; each planar screw is tracked with pose feedback. Retreat and ready use the rigid model.

    qp: ContactQP kwargs; approach: approach_reference kwargs plus tol, used for
    pre, approach, slide and retreat; pivot: dict(w_max, method, Tf_min); ready: JointMove kwargs;
    grasp: dict(force, speed, epsilon); safety: run_segment kwargs (f_max, tau_max, settle, dt_max).
    T_w_obj: object pose at the start (world). box: the object (sequencer.Box: size, mass, mu) or None (level air,
    simulation); while it is held, its predicted load on the arm (predicted_load, with kp the joint stiffness of
    the robot's controller) and a band around it do not count towards f_max / tau_max (wrench_excess).
    lead: lag compensation of the robot's joint controller (run_segment; None: off).
    log: a Log to fill (so the caller keeps the records if an exception ends the run).

    Abort procedure. The run stops at the first failed segment or gripper call, and on any exception:
    the velocity command is set to zero, the controller is stopped (backend.stop(): on the robot its own joint
    controller then holds the position), and the gripper is left exactly as it is. If the object is in the
    gripper it stays there (opening would drop it); log.abort = dict(cause, segment, step, holding) says so.
    No motion is commanded after a failure.
    Returns (log, T_w_obj): the object pose after the last finished step, from the motion of the hand (open loop).
    """
    qp, approach, pivot, ready = dict(qp or {}), dict(approach or {}), dict(pivot or {}), dict(ready or {})
    grasp, safety = dict(grasp or {}), dict(safety or {})
    tol = approach.pop("tol", 1e-3)
    q_ready = np.asarray(robot.qr if q_ready is None else q_ready, float)[:7]
    T_w_obj = np.array(T_w_obj, float)
    fk = lambda q: contact_pose(robot, q)
    log = Log() if log is None else log
    step = -1
    state = dict(holding=False, cause=None, finished=False, load=None)

    def segment(ctrl, ref, Tf, name, phi=0.0, done=None):
        load = state["load"] if state["holding"] else None
        ok, phi = run_segment(backend, ctrl, ref, Tf, log, name, phi=phi, done=done, viewer=viewer, load=load,
                              lead=lead, **safety)
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
        try:
            ok = bool(call(*args))
            error = "" if ok else "the gripper reported failure"
        except RuntimeError as e:                      # libfranka command or network error of the hand
            ok, error = False, str(e)
        width = args[0] if args else OPEN_WIDTH        # finger opening after the call
        log.events.append((name, backend.time(), ok, float(width)))
        if not ok:
            state["cause"] = f"gripper {name} failed: {error}"
        elif viewer is not None:
            viewer.fingers(width)
        return ok

    def up(T):
        T = T.copy()
        T[2, 3] += lift
        return T

    rigid = ContactQP(robot, free_rotation=False, **qp)
    backend.start()
    try:
        if not gripper("open", backend.open):
            return log, T_w_obj
        if np.abs(backend.theta() - q_ready).max() > 1e-4 and not to_ready():
            return log, T_w_obj
        for step, st in enumerate(steps):
            pl, is_pivot = st.plan, st.kind in ("pivot", "gait")   # feed-forward screws (gait: five in one grasp)
            log.steps.append(dict(kind=st.kind, ok=False, T_obj=T_w_obj))
            lim = {"phi_lim": getattr(pl, "phi_lim", None)}        # palm clear of the object; qp can override it
            ctrl = ContactQP(robot, free_rotation=True, **{**lim, **qp}) if is_pivot and approach_free else rigid
            phi = 0.0
            for name, T_d in (("pre", up(pl.T_grasp)), ("approach", pl.T_grasp)):
                ok, phi = move(ctrl, T_d, name, phi)
                if not ok:
                    return log, T_w_obj
            state["holding"] = True                    # from here the fingers may be on the object
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
            T_o = pl.T_start                                  # planned box pose at the start of each screw
            for k, (S, theta) in enumerate(pl.screws):
                T_c = fk(q_now())
                if box is not None:
                    state["load"] = predicted_load(robot, st, k, T_o, box, kp)
                T_o = se3.exp6(np.asarray(S) * theta) @ T_o
                if is_pivot:
                    method = pivot.get("method", "quintic")
                    Tf = max(pivot.get("Tf_min", 1.0), PEAK_SDOT[method[0].lower()] * abs(theta) / pivot.get("w_max", 0.3))
                    ok, phi = segment(ctrl, pivot_reference(screw_twist(S, theta, T_c), Tf, method), Tf, "pivot", phi)
                else:
                    ok, phi = move(ctrl, se3.exp6(np.asarray(S) * theta) @ T_c, "slide")
                if not ok:
                    return log, fk(q_now()) @ T_c_obj
            T_w_obj = fk(q_now()) @ T_c_obj
            if not gripper("release", backend.open):
                return log, T_w_obj
            state["holding"], state["load"] = False, None
            if viewer is not None:
                viewer.detach()
            log.steps[-1].update(T_obj=T_w_obj)
            if not move(rigid, up(fk(backend.theta())), "retreat")[0] or not to_ready():
                return log, T_w_obj
            log.steps[-1].update(ok=True)
        state["finished"] = True
        return log, T_w_obj
    except BaseException as e:
        state["cause"] = state["cause"] or f"{type(e).__name__}: {e}".rstrip(": ")
        raise
    finally:
        try:
            backend.stop()
        finally:
            if not state["finished"]:
                failed = next((sg for sg in reversed(log.segments) if not sg["ok"]), None)
                last = log.segments[-1] if log.segments else None
                log.abort = dict(cause=state["cause"] or (failed["cause"] if failed else "stopped"),
                                 segment=last["name"] if last else None, step=step, holding=state["holding"])


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
    """Hand and arm against the box and the table over a run_plan log (the planner and the QP do not
    check collisions of the arm; this is the check after the fact). T_w_obj: box pose at the start.
    box_ok uses the same model of the hand as the phi limits (clearance.hand_hits_box: the hand's collision mesh,
    or its bounding box if coal is missing or clearance.HAND_MODEL == "box"); table_ok uses the bounding box.
    The box pose at each sample follows from the log: at rest between the steps, moving with {c} while held.
    The palm must stay `clearance` [m] away from the box and above the table (the samples are `period` s apart,
    at most 3 mm of hand motion at the speeds used). Returns dict(box_ok, table_ok, hits): hits lists (t, segment name) of the samples with contact.
    arm: planning.collision.ArmCollision; if given, every `arm_every`-th sample also tests all arm links, the
    hand and the fingers against the box and the table (distance >= arm.margin) and the result has arm_ok, arm_hits
    (t, segment, link) and arm_min (smallest distance). The fingers are at the opening the gripper was last
    commanded to (log.events); they are tested against the box whenever it is not held."""
    a = log.arrays()
    seg, theta, phi, t = a["segment"], a["theta"], a["phi"], a["t"]
    T_rest = [np.asarray(T_w_obj, float)] + [np.asarray(st["T_obj"], float) for st in log.steps]   # before step i, after it
    hits, table_ok = [], True
    arm_hits, arm_min, count = [], np.inf, 0
    from pandaSim.control.viewer import finger_widths

    width = finger_widths(t, log.events)                   # finger opening at each sample
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
            if hand_hits_box(T_E, T_obj, size, clearance):
                hits.append((float(t[k]), sg["name"]))
            if arm is not None and count % arm_every == 0:
                ok, who, d = arm.ok(theta[k], T_obj, phi[k], width=width[k], fingers_vs_box=not held)
                arm_min = min(arm_min, d)
                if not ok:
                    arm_hits.append((float(t[k]), sg["name"], who, arm.obstacle))
            count += 1
    return dict(box_ok=not hits, table_ok=bool(table_ok), hits=hits, arm_ok=not arm_hits, arm_hits=arm_hits,
                arm_min=float(arm_min))
