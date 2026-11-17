"""Start pose -> goal pose of the box: plan, check, replay in Swift, run on the Panda.

    python experiments/run_box.py --start 0.50,0.00,0,lying --goal 0.45,-0.15,60,standing
        plans, simulates the plan with the executor (same code as on the robot), prints the pre-flight report
        and saves the run folder experiments/runs/box/<name>/ (plan.json, expected.npz, preflight.json). No robot.
    ... --replay                     show the simulated run in Swift (a normal browser, WebGL)
    ... --execute --host 172.16.0.2 --level air --speed 0.3
        runs it on the Panda after the pre-flight checks and a typed confirmation; saves real.npz and compare.json.

Pose: x,y,yaw_deg,face. x, y: box centre on the table (the robot base is at the origin of the table frame), yaw
about z, face: the face on the table: lying (0.06 x 0.25 face), lying_flip (the opposite one), standing
(0.15 x 0.06 face), standing_flip. The largest face cannot be grasped. The box frame is fixed by a mark on one corner
of the real box: the marked corner is the (+x, +y, +z) corner of the box frame (MARK); the replay draws it on the
start pose, the moving box and the goal. Levels of the real ladder:
    air    no box, gripper open, the fingers close to the box width in the air (speed <= 0.5)
    low    box on the table, slow (speed <= 0.5)
    full   box on the table, speed up to 1.0
The robot only moves after you type the confirmation phrase; there is no flag that skips it.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_plan as rp  # noqa: E402

from pandaSim.control.backends import OPEN_WIDTH, PANDA_KP, PANDA_LAG, PANDA_WALL, SimBackend  # noqa: E402
from pandaSim.control.primitive_executor import Log, check_profile, contact_pose, run_plan  # noqa: E402
from pandaSim.planning.box_primitives import BASE_KEEPOUT  # noqa: E402
from pandaSim.planning.feasibility import grasp_observability  # noqa: E402
from pandaSim.planning.sequencer import pose_error  # noqa: E402

REAL_CONFIG = HERE / "configs" / "box_real.yaml"
FACES = {"lying": -90.0, "lying_flip": 90.0, "standing": 0.0, "standing_flip": 180.0}     # pitch about the box y axis
MARK = rp.MARK                      # the marked corner of the real box, in units of the half size (box frame)
LEVEL_SPEED = {"air": (0.3, 0.5), "low": (0.3, 0.5), "full": (1.0, 1.0)}      # default and maximum speed factor
G = 9.81
PAYLOAD = 3.0                       # kg, Franka Emika Panda (hand included)
REGION = dict(x=(0.30, 0.75), y=(-0.35, 0.35))
JOINT_MARGIN = (0.05, 0.15)         # rad: below the first the run is refused, below the second it is a warning
PIVOT_PATH_WARN = 2.0               # mm: warning if the hand leaves the pivot arc by more with the controller's lag
STOP_COAST_MAX = 0.005              # m: an expected stop is refused if the hand would coast further (controller lag)


def parse_pose(text):
    """'x,y,yaw_deg,face' -> {xy, rpy} of the config."""
    try:
        x, y, yaw, face = text.split(",")
        return dict(xy=[float(x), float(y)], rpy=[0.0, FACES[face.strip()], float(yaw)])
    except (ValueError, KeyError):
        raise SystemExit(f"cannot read the pose '{text}': use x,y,yaw_deg,face with face in {sorted(FACES)}")


def pose_text(T):
    """Resting pose -> 'x,y,yaw_deg,face' (inverse of parse_pose with run_plan.box_pose)."""
    from pandaSim.planning import se3

    R = np.asarray(T, float)[:3, :3]
    for face, pitch in FACES.items():
        M = R @ se3.exp3(np.array([0.0, math.radians(pitch), 0.0])).T       # = Rz(yaw) for the right face
        if M[2, 2] > 0.99:
            return f"{T[0, 3]:.3f},{T[1, 3]:.3f},{math.degrees(math.atan2(M[1, 0], M[0, 0])):.1f},{face}"
    raise ValueError("the box rests on its largest face: no pose of run_box.py")


def symmetric(cfg):
    """Is the box treated as symmetric (False: a marked box, only the goal itself counts)."""
    return cfg["planner"].get("symmetric", True)


def check(name, ok, detail="", hard=True):
    return dict(name=name, ok=bool(ok), detail=detail, hard=hard)


# checks of the simulation that fail by design when the run is expected to stop (rigid grasp, --expect-stop)
UNTIL_THE_STOP = ("all segments completed", "QP always feasible", "object at the goal", "arm back at ready")


def predicted_stop(log, steps):
    """Where a simulated run stops if it stops by an infeasible QP in a pivot: dict(step, angle_deg, of_deg, time_frac);
    None if it does not stop, or stops for another reason."""
    failed = next((sg for sg in log.segments if not sg["ok"]), None)
    if failed is None or failed["name"] != "pivot" or failed["cause"] != "infeasible QP":
        return None
    from pandaSim.planning.primitives import time_scaling

    k = sum(1 for sg in log.segments[:log.segments.index(failed)] if sg["name"] == "pivot" and sg["step"] == failed["step"])
    theta = steps[failed["step"]].plan.screws[k][1]
    tau = (failed["t_end"] - failed["t0"]) / failed["Tf"]
    s = time_scaling(failed["t_end"] - failed["t0"], failed["Tf"])[0]
    return dict(step=failed["step"], angle_deg=math.degrees(s * theta), of_deg=math.degrees(theta), time_frac=tau)


def preflight(res, cfg, expect_stop=False):
    """Pre-flight list for a planned and simulated task (res of run_plan.run_task): dicts name, ok, detail, hard.
    A failed hard check blocks the execution on the robot; a failed soft check is a warning.
    expect_stop: the run is meant to stop (a rigid grasp where the simulation predicts an infeasible QP): it must stop
    exactly so, in a pivot, and every other check holds until the stop."""
    plan, box = res["plan"], res["box"]
    out = []
    for key in ("start", "goal"):
        spec = cfg["tasks"]["custom"][key]
        x, y = spec["xy"]
        out.append(check(f"{key} pose inside the table area", REGION["x"][0] <= x <= REGION["x"][1]
                         and REGION["y"][0] <= y <= REGION["y"][1], f"x {x:.2f}, y {y:.2f}; area {REGION}"))
        # the planner keeps every corner of the box out of a cylinder around the base: a pose inside it has no plan
        T = res["T0"] if key == "start" else res["Tg"]
        corners = T[:3, 3] + (np.array(list(itertools.product((-0.5, 0.5), repeat=3))) * np.asarray(box.size)) @ T[:3, :3].T
        d = float(np.linalg.norm(corners[:, :2] - np.asarray(cfg["robot"]["base"][:2]), axis=1).min())
        out.append(check(f"{key} pose clear of the robot base", d >= BASE_KEEPOUT,
                         f"nearest box corner {d:.2f} m from the base axis, keep-out {BASE_KEEPOUT:.2f} m"))
    out.append(check("plan found", plan.found, "" if plan.found else "no feasible plan for this start and goal"))
    if not plan.found:
        return out
    ep, er = pose_error(res["T0"], res["Tg"], symmetric(cfg))
    out.append(check("start and goal differ", ep > 1e-3 or er > 1e-2))
    stop = predicted_stop(res["log"], plan.steps) if expect_stop else None
    if expect_stop:
        out.append(check("simulation stops by an infeasible QP in a pivot (expected: the robot then holds the tilted box)",
                         stop is not None, f"step {stop['step']} at {stop['angle_deg']:.0f} of {stop['of_deg']:.0f} deg"
                         if stop else "it does not: run it without --expect-stop"))
    if stop:
        # the joints are fast where the QP gives up; after the zero command they move on by the controller's lag
        a = res["log"].arrays()
        qd = a["qd"][-1, :7]
        J = res["robot"].jacob0(np.r_[a["theta"][-1], a["phi"][-1]])[:3, :7]
        coast = float(np.linalg.norm(J @ (PANDA_LAG * qd)))
        out.append(check("hand coasts little after the expected stop", coast <= STOP_COAST_MAX,
                         f"{coast * 1e3:.1f} mm (joints at up to {np.abs(qd).max():.2f} rad/s), at most "
                         f"{STOP_COAST_MAX * 1e3:.0f} mm: otherwise lower --speed"))
    for k, v in res["checks"].items():
        if k != "plan found" and not (stop and k in UNTIL_THE_STOP):
            out.append(check(f"simulation: {k}", v))
    for i, st in enumerate(plan.steps):                # phi is not measured: the contact must determine it
        ok, _, text = grasp_observability(st)
        out.append(check(f"step {i} ({st.kind}): the box cannot turn about the finger axis on its own", ok, text))
    a = res["log"].arrays()
    lb, ub = res["robot_limits"]
    margin = float(np.minimum(a["theta"] - lb, ub - a["theta"]).min())
    out.append(check("joint-limit margin", margin >= JOINT_MARGIN[0], f"{margin:.2f} rad", hard=True))
    if margin < JOINT_MARGIN[1]:
        out.append(check("joint-limit margin comfortable", False, f"{margin:.2f} rad < {JOINT_MARGIN[1]} rad", hard=False))
    # panda-py's virtual joint walls push back inside these zones, which the simulation does not model
    per_joint = np.minimum(a["theta"] - lb, ub - a["theta"]).min(axis=0)
    j = int(np.argmin(per_joint - PANDA_WALL))
    out.append(check("joints stay out of the controller's joint walls", bool(np.all(per_joint >= PANDA_WALL)),
                     f"joint {j + 1}: {per_joint[j]:.3f} rad from its limit, wall at {PANDA_WALL[j]:.3f} rad"))
    j = int(np.argmin(per_joint - 2 * PANDA_WALL))
    if per_joint[j] < 2 * PANDA_WALL[j]:
        out.append(check("joints stay out of the walls' damping zone", False,
                         f"joint {j + 1}: {per_joint[j]:.3f} rad < {2 * PANDA_WALL[j]:.3f} rad (more tracking lag there)", hard=False))
    tr = res.get("tracking")
    if tr is not None:
        mode = "with lag compensation" if tr["lead"] else "without lag compensation"
        if stop:
            same = bool(tr["fails"]) and all(f.startswith(f"pivot (step {stop['step']}): infeasible QP") for f in tr["fails"])
            out.append(check(f"simulation with the joint controller's lag ({mode}): stops in the same pivot", same,
                             "; ".join(tr["fails"]) or "it does not stop"))
        else:
            out.append(check(f"simulation with the joint controller's lag ({mode}): all segments completed", tr["ok"],
                             "" if tr["ok"] else "; ".join(tr["fails"]) + "  (slower speed, larger safety.settle, or --lead)"))
        if tr["ok"] and not stop:
            tol = cfg["tolerance"]
            out.append(check("  object at the goal with the lag", tr["err_mm"] < tol["position"] * 1e3
                             and tr["err_deg"] < tol["rotation_deg"], f"{tr['err_mm']:.1f} mm, {tr['err_deg']:.2f} deg", hard=False))
            out.append(check("  hand stays on the pivot arc with the lag", tr["pivot_dev_mm"] < PIVOT_PATH_WARN,
                             f"leaves it by {tr['pivot_dev_mm']:.1f} mm (the box is squeezed against the table by that much)", hard=False))
    out.append(check("arm and hand clearance", res["clearance"]["arm_min"] >= 0.005, f"smallest distance {res['clearance']['arm_min'] * 1e3:.1f} mm"))
    m, f_max = box.mass, (cfg["execution"].get("safety") or {}).get("f_max", 30.0)
    out.append(check("box mass within the payload", m <= PAYLOAD, f"{m:.2f} kg, payload {PAYLOAD} kg"))
    out.append(check("box weight is not close to the force stop", m * G + 5.0 < f_max, f"{m * G:.1f} N weight, stop at {f_max:.0f} N", hard=False))
    acc = check_profile(res["log"])["acc_ratio"].max()
    out.append(check("joint accelerations within the limits", acc <= 1.0, f"{acc:.2f} of the limit", hard=False))
    return out


def go(items):
    return all(c["ok"] or not c["hard"] for c in items)


def print_preflight(items):
    for c in items:
        mark = "ok  " if c["ok"] else ("FAIL" if c["hard"] else "warn")
        print(f"  [{mark}] {c['name']}" + (f"  ({c['detail']})" if c["detail"] else ""))
    print("  => " + ("GO" if go(items) else "NO-GO: a hard check failed"))


def robot_settings(cfg):
    """execution.robot of the config with its defaults: settings that only matter on the real robot."""
    r = dict(stall_timeout=0.05, track_tau_max=None, lead=False, reflex=None)
    r.update((cfg["execution"].get("robot") or {}))
    return r


def plan_json(res, cfg, speed):
    steps = [asdict(s) for s in res["plan"].steps] if res["plan"].found else []
    meta = dict(start=cfg["tasks"]["custom"]["start"], goal=cfg["tasks"]["custom"]["goal"], speed=speed,
                free_rotation=cfg["free_rotation"], box=cfg["box"], cost=res["plan"].cost if res["plan"].found else None,
                stats={k: v for k, v in res["plan"].stats.items() if isinstance(v, (int, float, str))}, steps=steps)
    return json.dumps(meta, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def plan_and_simulate(cfg, start, goal, speed=1.0, tracking=False, gait=False):
    """Plan with the feasibility model of the config and simulate the plan at `speed`; returns res (+ robot limits).
    tracking=True adds res["tracking"]: the same plan simulated with the joint controller's lag (simulate_tracking).
    gait=True: object gaiting (planning/gait.py) instead of the sequence of slides and pivots."""
    cfg["tasks"]["custom"] = dict(start=start, goal=goal)
    rp.scale_speed(cfg, speed)
    robot = rp.make_robot(cfg)
    cfg_plan = json.loads(json.dumps(cfg))
    rp.scale_speed(cfg_plan, 1.0)                     # the rollouts of the planner run at the nominal speed
    res = rp.run_task(cfg, robot, "custom", rp.make_planner(cfg_plan, robot, gait))
    res["robot_limits"] = (robot.qlim[0, :7], robot.qlim[1, :7])
    res["robot"] = robot
    if tracking and res["plan"].found:
        res["tracking"] = simulate_tracking(res, cfg, lead=robot_settings(cfg)["lead"])
    return res


def pivot_path_error(log, steps, robot):
    """Largest distance [m] of the grasp point from the planned arc over the pivot segments of a log (every screw of
    a pivot or gait step): change of the radius about the screw axis or of the position along it."""
    a, dev, count = log.arrays(), 0.0, {}
    for j, sg in enumerate(log.segments):
        idx = np.flatnonzero(a["segment"] == j)
        if sg["name"] != "pivot":
            continue
        k = count[sg["step"]] = count.get(sg["step"], -1) + 1             # k-th screw of the step (a gait has five)
        if len(idx) < 2:
            continue
        S = np.asarray(steps[sg["step"]].plan.screws[k][0], float)
        u, p_axis = S[:3], np.cross(S[:3], S[3:])
        P = np.array([contact_pose(robot, np.r_[a["theta"][k], a["phi"][k]])[:3, 3] for k in idx[::20]])
        d = P - p_axis
        ax = d @ u
        rad = np.linalg.norm(d - np.outer(ax, u), axis=1)
        dev = max(dev, float(np.abs(rad - rad[0]).max()), float(np.abs(ax - ax[0]).max()))
    return dev


def simulate_tracking(res, cfg, lead=False, dt=0.002):
    """The plan once more, with measured joints that lag the position target as on the robot (SimBackend(lag=...):
    panda-py's controller damps the joint velocity, so q lags q_d by Kd / Kp * velocity). The ideal simulation
    cannot show the two effects of that lag: segments that do not settle in time, and a pivot that leaves its arc."""
    kw = rp.exec_kwargs(cfg)
    backend = SimBackend(kw["q_ready"], dt=dt, lag=PANDA_LAG)
    log, T_end = run_plan(backend, res["robot"], res["plan"].steps, res["T0"], free_rotation=cfg["free_rotation"],
                          lead=PANDA_LAG if lead else None, **kw)
    ep, er = pose_error(T_end, res["Tg"], symmetric(cfg))
    return dict(lead=bool(lead), ok=log.ok, fails=[f"{s['name']} (step {s.get('step')}): {s['cause']}" for s in log.segments if not s["ok"]],
                err_mm=float(ep * 1e3), err_deg=float(math.degrees(er)),
                pivot_dev_mm=pivot_path_error(log, res["plan"].steps, res["robot"]) * 1e3)


def save_run(res, cfg, run_dir, speed):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(plan_json(res, cfg, speed))
    if res["log"] is not None:
        res["log"].save(run_dir / "expected.npz", T0=res["T0"], Tg=res["Tg"], size=np.array(res["box"].size),
                        base=np.array(cfg["robot"]["base"]))


# ---- comparison of a real run with the simulation ------------------------------------------------------
def compare(expected, real, T_end=None, T_goal=None, symmetric=True):
    """Per segment (same order in both logs): joint tracking against the simulated trajectory over the common
    time, in rad; plus the largest wrench, the loop timing and the final object pose error."""
    ea, ra = expected.arrays(), real.arrays()
    segs = []
    for j, sr in enumerate(real.segments):
        ie, ir = np.flatnonzero(ea["segment"] == j), np.flatnonzero(ra["segment"] == j)
        if len(ie) < 2 or len(ir) < 2:
            continue
        te, tr = ea["t"][ie] - ea["t"][ie][0], ra["t"][ir] - ra["t"][ir][0]
        n = tr <= te[-1]
        e = np.stack([np.interp(tr[n], te, ea["theta"][ie][:, i]) for i in range(7)], axis=1) - ra["theta"][ir][n]
        segs.append(dict(name=sr["name"], step=sr.get("step"), ok=sr["ok"], cause=sr["cause"],
                         duration=float(tr[-1]), expected_duration=float(te[-1]),
                         rms_joint_error=float(np.sqrt((e ** 2).mean())), max_joint_error=float(np.abs(e).max()),
                         max_force=float(np.linalg.norm(ra["F_ext"][ir][:, :3], axis=1).max())))
    out = dict(segments=segs, n_segments=len(real.segments), expected_segments=len(expected.segments),
               ok=real.ok and real.abort is None, abort=real.abort)
    if len(ra["t"]) > 2:
        dt = np.diff(ra["t"])[ra["segment"][1:] == ra["segment"][:-1]]
        dq = np.gradient(ra["theta"], ra["t"], axis=0)
        e = ra["target"] - ra["theta"]                                  # position target minus measured joints
        moving = np.abs(dq) > 0.02
        lag = [float((e[moving[:, i], i] @ dq[moving[:, i], i]) / (dq[moving[:, i], i] @ dq[moving[:, i], i]))
               if moving[:, i].sum() > 100 else None for i in range(7)]
        out.update(tick_ms_p99=float(np.percentile(ra["tick_time"], 99) * 1e3),
                   solve_ms_p99=float(np.percentile(ra["solve_time"], 99) * 1e3),
                   period_ms_max=float(dt.max() * 1e3) if len(dt) else None,
                   # tracking: (q_d - q) - lag dq, as a torque of the joint controller, and the lag itself
                   track_mrad_max=(np.abs(ra["track"]).max(axis=0) * 1e3).round(2).tolist(),
                   track_torque_max=(np.abs(ra["track"]).max(axis=0) * PANDA_KP).round(2).tolist(),
                   lag_measured_s=lag, lag_model_s=PANDA_LAG.round(3).tolist())
    if T_end is not None and T_goal is not None:
        ep, er = pose_error(T_end, T_goal, symmetric)
        out.update(object_error_mm=float(ep * 1e3), object_error_deg=float(math.degrees(er)))
    return out


def print_compare(c):
    print(f"\nreal run: {'all segments finished' if c['ok'] else 'STOPPED'}  ({c['n_segments']} of {c['expected_segments']} segments)")
    if "tick_ms_p99" in c:
        print(f"  loop: tick p99 {c['tick_ms_p99']:.2f} ms, QP p99 {c['solve_ms_p99']:.2f} ms, longest period {c['period_ms_max']:.1f} ms")
        print(f"  tracking torque, largest per joint [Nm]: {c['track_torque_max']}")
        print(f"  lag of the joints behind the target [s]: measured {[None if v is None else round(v, 3) for v in c['lag_measured_s']]}, "
              f"model {c['lag_model_s']}")
    for s in c["segments"]:
        print(f"  step {s['step']}  {s['name']:9s} {s['duration']:6.2f} s (sim {s['expected_duration']:6.2f})  "
              f"joint error rms {s['rms_joint_error'] * 1e3:6.1f} mrad, max {s['max_joint_error'] * 1e3:6.1f} mrad   "
              f"max force {s['max_force']:5.1f} N   {'ok' if s['ok'] else s['cause']}")
    if "object_error_mm" in c:
        print(f"  object pose by forward kinematics at the release: {c['object_error_mm']:.2f} mm, {c['object_error_deg']:.2f} deg "
              f"from the goal (not measured)")
    ab = c.get("abort")
    if ab:
        print(f"\nABORTED in '{ab['segment']}' (step {ab['step']}): {ab['cause']}")
        print("  The velocity was set to zero and the controller stopped; the robot holds its position. No motion follows.")
        if ab["holding"]:
            print("  THE FINGERS ARE STILL CLOSED ON THE BOX. Support the box by hand, then open the gripper from Desk.")
        print("  Before the next run: move the arm near the ready configuration by hand (guiding mode) and put the box back.")


# ---- the robot -------------------------------------------------------------------------------------------
READY_PATH_CLEARANCE = 0.03         # m: required distance to the table and the box on the joint-space move to ready


def robot_checks(state, q_ready, lim, gripper_state=None, arm=None, T_box=None):
    """Checks on the robot before the first motion (state: panda-py RobotState or a fake with the same fields).
    arm: planning.collision.ArmCollision, T_box: pose of the box on the table (None: no box, level air). With arm,
    the straight joint-space path from the measured configuration to q_ready (the path of panda-py's
    move_to_joint_position) is swept against the table, the box and the arm itself, fingers at their real opening."""
    q = np.asarray(state.q, float)
    margin = float(np.minimum(lim[1] - q, q - lim[0]).min())
    dist = float(np.abs(q - q_ready).max())
    force = float(np.linalg.norm(np.asarray(state.O_F_ext_hat_K, float)[:3]))
    items = [
        check("joints inside the limits", margin > JOINT_MARGIN[0], f"margin {margin:.2f} rad"),
        check("no robot error", not str(state.current_errors).strip("[]{} "), str(state.current_errors)[:80]),
        check("robot is at rest and unloaded", force < 5.0, f"|F_ext| {force:.1f} N"),
    ]
    if arm is None:
        items.append(check("robot is within 1.0 rad of the ready configuration (the move there is not collision-checked)",
                           dist < 1.0, f"largest joint distance {dist:.2f} rad"))
    else:
        far = np.eye(4)
        far[0, 3] = 10.0                               # no box on the table: park it out of reach
        width = OPEN_WIDTH if gripper_state is None else float(gripper_state.width)
        sw = arm.sweep(q, q_ready, far if T_box is None else T_box, width=width)
        items.append(check("path to the ready configuration is clear of the table and the box",
                           sw["d_min"] >= READY_PATH_CLEARANCE,
                           f"closest {sw['d_min'] * 1e3:.0f} mm" + (f" ({sw['link']} at {sw['s'] * 100:.0f} % of the path)" if sw["link"] else " or more")
                           + f", required {READY_PATH_CLEARANCE * 1e3:.0f} mm"))
        items.append(check("the arm does not touch itself on that path", not sw["self_collision"]))
        items.append(check("the move to the ready configuration is short", dist < 1.0, f"largest joint distance {dist:.2f} rad", hard=False))
    if gripper_state is not None:
        items.append(check("gripper answers", gripper_state.max_width > 0.07, f"max width {gripper_state.max_width:.3f} m"))
    return items


def confirm(phrase, ask=input):
    """The user must type the phrase; anything else, or no terminal, refuses."""
    if ask is input and not sys.stdin.isatty():
        print("no terminal: the robot is not moved")
        return False
    return ask(f"\nType  {phrase}  to move the robot (anything else cancels): ").strip() == phrase


FER_TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])     # Nm, maximum joint torques


def reflex_thresholds(reflex):
    """execution.robot.reflex -> absolute thresholds dict(torque=dict(lower, upper) [Nm], force=dict(lower, upper)):
    torque = alpha x torque_fraction x the maximum joint torque (at most the maximum), force = alpha x force."""
    a = float(reflex.get("alpha", 1.0))
    tq = {k: np.minimum(a * np.asarray(v, float) * FER_TAU_MAX, FER_TAU_MAX).tolist()
          for k, v in reflex["torque_fraction"].items()}
    return dict(torque=tq, force={k: (a * np.asarray(v, float)).tolist() for k, v in reflex["force"].items()})


def set_reflex(panda, reflex):
    """Collision thresholds of the robot itself (libfranka setCollisionBehavior): lower = contact, upper = stop.
    panda-py does not set them, so without this call they are whatever the robot was left with."""
    th = reflex_thresholds(reflex)
    t, f = th["torque"], th["force"]
    panda.get_robot().set_collision_behavior(t["lower"], t["upper"], f["lower"], f["upper"])


def execute(res, cfg, run_dir, host, level, speed, ask=input, panda=None, gripper=None, lead=None):
    """Run the planned and simulated task on the Panda. panda/gripper: connected objects (tests pass fakes).

    Order: connect (read only) -> checks on the robot -> typed confirmation -> brakes, reflex thresholds, move to
    ready -> run_plan. Whatever ends the run (failed segment, robot error, Ctrl-C, exception), the arm has been
    commanded to zero velocity and the controller stopped by run_plan, and real.npz and compare.json are written.
    Returns the comparison dict, or None if the robot was not moved."""
    from pandaSim.control.backends import PandaBackend

    robot, kw, rs = res["robot"], rp.exec_kwargs(cfg), robot_settings(cfg)
    lead = rs["lead"] if lead is None else lead
    creds = None
    if panda is None:
        import check_connection as cc
        import os
        import resource
        import panda_py
        from panda_py import libfranka

        rt_ok = os.geteuid() == 0 or resource.getrlimit(resource.RLIMIT_RTPRIO)[0] > 0
        print(f"realtime {'available' if rt_ok else 'NOT available (rtprio limit is 0): timing will be poor'}")
        panda = panda_py.Panda(host, realtime_config=libfranka.RealtimeConfig.kEnforce if rt_ok else libfranka.RealtimeConfig.kIgnore)
        gripper = libfranka.Gripper(host)
        creds = cc.desk_credentials()
    from pandaSim.planning.collision import ArmCollision

    items = robot_checks(panda.get_state(), kw["q_ready"], res["robot_limits"],
                         gripper.read_once() if gripper is not None else None,
                         arm=ArmCollision(robot, res["box"].size), T_box=None if level == "air" else res["T0"])
    print("\nrobot:")
    print_preflight(items)
    if not go(items):
        return None
    print(f"\nrun {run_dir.name}: level {level}, speed {speed}, {len(res['plan'].steps)} steps "
          f"({' -> '.join(s.kind for s in res['plan'].steps)}), about {res['log'].arrays()['t'][-1]:.0f} s, "
          f"lag compensation {'on' if lead else 'off'}")
    if rs["reflex"]:
        th = reflex_thresholds(rs["reflex"])
        print(f"the robot's collision reflex (alpha {rs['reflex'].get('alpha', 1.0)}) will stop it above: force "
              f"{[round(x, 1) for x in th['force']['upper'][:3]]} N, torque {[round(x, 1) for x in th['torque']['upper']]} Nm")
    if not confirm(f"RUN {level} {run_dir.name}", ask):
        print("cancelled; the robot was not moved")
        return None
    if creds is not None:                              # release the brakes (needed if the robot was locked)
        import panda_py

        panda_py.Desk(host, *creds, platform="panda").unlock()
    if rs["reflex"]:
        set_reflex(panda, rs["reflex"])
    if np.abs(np.asarray(panda.get_state().q) - kw["q_ready"]).max() > 1e-3:
        print("moving to the ready configuration (panda-py joint motion, slow)")
        reached = panda.move_to_joint_position(kw["q_ready"], speed_factor=0.1)
        off = float(np.abs(np.asarray(panda.get_state().q) - kw["q_ready"]).max())
        if reached is False or off > 0.02:
            print(f"the move to the ready configuration did not arrive ({off:.3f} rad off): the run is not started")
            return None
    backend = PandaBackend(panda, gripper, frequency=1.0 / cfg["execution"]["dt"], air=(level == "air"),
                           stall_timeout=rs["stall_timeout"],
                           **({} if rs["track_tau_max"] is None else {"track_tau_max": rs["track_tau_max"]}))
    log, T_end, error = Log(), None, None
    t0 = time.perf_counter()
    try:
        _, T_end = run_plan(backend, robot, res["plan"].steps, res["T0"], free_rotation=cfg["free_rotation"],
                            payload=0.0 if level == "air" else res["box"].mass, lead=backend.lag if lead else None,
                            log=log, **kw)
    except KeyboardInterrupt:
        error = "interrupted with Ctrl-C"
    except Exception as e:                             # the arm is already stopped (run_plan); keep the records
        error = e
    finally:
        print(f"finished in {time.perf_counter() - t0:.1f} s")
        log.save(run_dir / "real.npz", T0=res["T0"], Tg=res["Tg"], size=np.array(res["box"].size),
                 base=np.array(cfg["robot"]["base"]))
        c = compare(res["log"], log, T_end, res["Tg"], symmetric(cfg))
        (run_dir / "compare.json").write_text(json.dumps(c, indent=1, default=str))
        print_compare(c)
    if isinstance(error, Exception):
        raise error
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--start", required=True, help="x,y,yaw_deg,face")
    ap.add_argument("--goal", required=True, help="x,y,yaw_deg,face")
    ap.add_argument("--config", default=REAL_CONFIG)
    ap.add_argument("--rigid", action="store_true",
                    help="execute with the rigid grasp (original fingers); the plan is the one of free rotation, so both "
                         "grasps run the same motion")
    ap.add_argument("--plan-rigid", action="store_true", help="plan with the rigid grasp as well")
    ap.add_argument("--expect-stop", action="store_true",
                    help="the simulation predicts an infeasible QP in a pivot (rigid grasp): run up to that stop")
    ap.add_argument("--gait", action="store_true", help="object gaiting (rotations about vertices) instead of slides and pivots")
    ap.add_argument("--mass", type=float, help="box mass in kg (overrides the config)")
    ap.add_argument("--name", help="run folder name (default: timestamp)")
    ap.add_argument("--speed", type=float, help="speed factor of the execution, 0 < s <= 1 (default by level)")
    ap.add_argument("--level", choices=list(LEVEL_SPEED), default="air")
    ap.add_argument("--replay", nargs="?", const="expected", choices=["expected", "real"], help="show a run in Swift")
    ap.add_argument("--replay-speed", type=float, default=1.0)
    ap.add_argument("--execute", action="store_true", help="run on the robot (after the checks and a typed confirmation)")
    ap.add_argument("--host", default="172.16.0.2")
    ap.add_argument("--lead", action=argparse.BooleanOptionalAction, default=None,
                    help="lag compensation of the joint controller (default: execution.robot.lead of the config)")
    args = ap.parse_args(argv)

    cfg = rp.load_config(args.config, not args.rigid, not args.plan_rigid)
    if args.mass is not None:
        cfg["box"]["mass"] = args.mass
    default, cap = LEVEL_SPEED[args.level]
    speed = args.speed if args.speed is not None else default
    if not 0 < speed <= cap:
        raise SystemExit(f"speed {speed} is outside (0, {cap}] for level {args.level}")
    run_dir = HERE / "runs" / "box" / (args.name or time.strftime("%Y%m%d-%H%M%S"))

    if args.lead is not None:
        cfg["execution"]["robot"] = {**(cfg["execution"].get("robot") or {}), "lead": args.lead}
    res = plan_and_simulate(cfg, parse_pose(args.start), parse_pose(args.goal), speed, tracking=True, gait=args.gait)
    grasp = lambda free: "free rotation" if free else "rigid grasp"
    print(f"\nplan {args.start}  ->  {args.goal}   (planned with {grasp(cfg['plan_free_rotation'])}, executed with "
          f"{grasp(cfg['free_rotation'])}, speed {speed})")
    if res["plan"].found:
        rp.report(res)
    items = preflight(res, cfg, args.expect_stop)
    print("\npre-flight:")
    print_preflight(items)
    save_run(res, cfg, run_dir, speed)
    (run_dir / "preflight.json").write_text(json.dumps(items, indent=1))
    print(f"\nrun folder: {run_dir}")
    if args.replay == "expected" and res["log"] is not None:
        env = rp.show(run_dir / "expected.npz", args.replay_speed)
        time.sleep(2.0)
        env.close()
    ok = go(items)
    if args.execute:
        if not ok:
            print("NO-GO: the robot is not moved")
            return 2
        if args.expect_stop:
            st = predicted_stop(res["log"], res["plan"].steps)
            print(f"\nEXPECTED STOP: the simulation stops in step {st['step']} (pivot) at {st['angle_deg']:.0f} of "
                  f"{st['of_deg']:.0f} deg. The robot will then hold the tilted box: support it, open the gripper from Desk.")
        execute(res, cfg, run_dir, args.host, args.level, speed)
        if args.replay == "real" and (run_dir / "real.npz").exists():
            env = rp.show(run_dir / "real.npz", args.replay_speed)
            time.sleep(2.0)
            env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
