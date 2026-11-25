"""P2 step 1: can the box be rolled by pi (two same-sense pivots by pi/2 about parallel edges) in ONE grasp?

Purely geometric study (seconds, no rollouts); the rollouts at the end only confirm the best start poses.

    PYTHONPATH=src .venv/bin/python experiments/double_pivot_study.py [--no-rollouts] [--workers 8]

Writes experiments/results/double_pivot_study.{json,pdf,png}.

The pivots are about edges parallel to the finger axis n, so the problem is planar in the cross-section normal to n.
The hand is described by the direction of its palm p = -z_E in that plane:
  alpha  angle of p in the world plane spanned by the roll direction u and the vertical (90 deg: palm up, the hand
         points straight down as in every planner grasp; 0 deg: palm toward the roll direction)
  psi    angle of p in the box frame, atan2(p.z_b, p.x_b); the bearing (phi) changes psi and nothing else.
Clearance to the box depends on psi only (clearance.palm_hits_box: the bounding box of the hand, 5 mm, the model of
box_real.yaml), clearance to the table on alpha and the height of the grasp point (clearance.palm_lowest, 2 mm as in
primitive_executor.check_clearance).

For a grasp point c on the face normal to n, in the square of a corner (insets 10-40 mm from both edges):
  arc          longest arc of psi clear of the box; usable = arc - 2 rho: the phi damper (ContactQP, eq:qp_c) stops
               phi rho before each limit (rho = rho_s of box_real.yaml, 0.13 rad; also 0.05 rad and 0)
  tilt         the least rotation of the hand about n over the chain: 180 deg - usable arc
  windows      hand orientations alpha clear of box (usable arc) and table at the start and at the end; "connected":
               only those joined by a continuous path over the chain (sweep forward and backward over both arcs)
  lazy         what the executor's QP does: the weight nu = 1e-3 on phi_dot is small, so the hand keeps its
               orientation and the bearing turns until the damper stops phi; from there the hand turns with the
               box. The path from the best start orientation, its smallest palm-table distance, its end orientation
  rigid        phi fixed (the hand turns with the box by pi): start orientations whose palm clears the table over
               the whole chain
  observability  feasibility.grasp_observability of both pivots (the grasp point >= 20 mm from each pivot axis)
Start poses (lying, standing; x, y over the protocol's table area; yaw over 360 deg): the geometry above does not
depend on them; reach of the grasp point over both arcs (sequencer.REACH), base keep-out (keepout_ok) and the table
area (run_box.REGION) do.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pandaSim.planning import box_primitives as P
from pandaSim.planning import clearance as C
from pandaSim.planning import se3
from pandaSim.planning.feasibility import grasp_observability
from pandaSim.planning.sequencer import REACH, Sequencer, down_axis

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "double_pivot_study"
SIZE = (0.15, 0.06, 0.25)                      # box_real.yaml
FACES = {"lying": -90.0, "standing": 0.0}      # pitch about the box y axis (run_box.FACES)
REGION = dict(x=(0.30, 0.75), y=(-0.35, 0.35))  # run_box.REGION: where the box may be placed
Y_REF = np.array([0.0, -1.0, 0.0])             # y axis of {c} at the ready pose of box_real.yaml (finger sign rule)
STEP = 0.5                                     # deg, grid of alpha and psi
N = int(round(360 / STEP))
DTH = 1.5                                      # deg, samples of each pivot (a multiple of STEP: the grids map 1:1)
TABLE_MARGIN = 0.002                           # m, palm above the table (check_clearance)
RHOS = (0.13, 0.05, 0.0)                       # rad, phi damper margin: box_real rho_s, smaller, none
INSETS = np.arange(10, 40.1, 2.5) / 1e3        # m
CORNERS = ("back_top", "back_bottom", "front_top", "front_bottom")    # relative to the roll; front: the side it tips to
MIN_MARGIN = 10.0                              # deg, the margin asked for (free arc and end window)


# ---- the chains ------------------------------------------------------------------------------------------------
def rest_pose(face, x=0.5, y=0.0, yaw=0.0):
    """Resting pose as run_box.parse_pose + run_plan.box_pose."""
    R = se3.exp3(np.array([0, 0, math.radians(yaw)])) @ se3.exp3(np.array([0, math.radians(FACES[face]), 0]))
    return se3.rp(R, np.array([x, y, np.abs(R[2]) @ np.asarray(SIZE) / 2]))


def two_pivot_chains(T0, y_ref=Y_REF):
    """Every chain of two pivots that the planner allows (Sequencer._box_pivots twice):
    list of (kind, pl1, pl2), kind 'undo' (back on the start face) or 'roll' (net rotation pi)."""
    seq, box = Sequencer(y_ref=y_ref), SimpleNamespace(size=SIZE)
    out = []
    for _, p1 in seq._box_pivots(box, T0):
        for _, p2 in seq._box_pivots(box, p1.T_goal):
            ang = se3.angle_between(T0[:3, :3], p2.T_goal[:3, :3])
            out.append(("undo" if ang < 1e-6 else "roll", p1, p2))
    return out


def roll_chain(T0, d1_o=None, y_ref=Y_REF):
    """The same-sense roll: pivot toward d1_o (an object axis, horizontal, normal to n = object y), then pivot again
    toward the same world direction u. Returns (pl1, pl2, u)."""
    if d1_o is None:
        d1_o = np.array([0.0, 0, 1]) if abs(T0[2, 2]) < 0.5 else np.array([1.0, 0, 0])   # lying: z_b, standing: x_b
    pl1 = P.plan_box_pivot(T0, SIZE, d1_o, math.pi / 2, y_ref)
    u = T0[:3, :3] @ d1_o
    pl2 = P.plan_box_pivot(pl1.T_goal, SIZE, np.round(pl1.T_goal[:3, :3].T @ u), math.pi / 2, y_ref)
    return pl1, pl2, u


def chain_poses(pl1, pl2, dth=DTH):
    """Box poses over both pivots, every dth degrees (the shared pose at pi/2 once)."""
    th = np.radians(np.arange(0, 90 + 1e-9, dth))
    return ([se3.exp6(pl1.S * t) @ pl1.T_start for t in th]
            + [se3.exp6(pl2.S * t) @ pl2.T_start for t in th[1:]])


def corner_point(T0, u, corner, a, b):
    """Grasp point in the box frame: in the square of `corner`, a from the front/back face, b from the top/bottom."""
    half = np.asarray(SIZE) / 2
    R = T0[:3, :3]
    f_o, up_o = np.round(R.T @ u), down_axis(T0) * -1
    sf = 1.0 if corner.startswith("front") else -1.0
    st = 1.0 if corner.endswith("top") else -1.0
    return sf * (np.abs(f_o) @ half - a) * f_o + st * (np.abs(up_o) @ half - b) * up_o


# ---- the planar model ------------------------------------------------------------------------------------------
def palm_dirs_box():
    """Palm directions p(psi) in the box frame (rows), psi on the grid."""
    psi = np.radians(np.arange(N) * STEP)
    return np.c_[np.cos(psi), np.zeros(N), np.sin(psi)]


def hand_in_box(c_o, p_b):
    """{E} in the box frame: at c_o, palm toward p_b, y_E = n = y_b."""
    z, y = -p_b, np.array([0.0, 1, 0])
    return se3.rp(np.column_stack([np.cross(y, z), y, z]), c_o)


def free_mask(c_o, model="box", clearance=C.CLEARANCE):
    """psi grid -> hand clear of the box (model box: clearance.palm_hits_box; mesh: hand.stl by coal)."""
    if model == "box":
        return np.array([not C.palm_hits_box(hand_in_box(c_o, p), np.eye(4), SIZE, clearance) for p in palm_dirs_box()])
    return np.array([C.hand_distance(hand_in_box(c_o, p), np.eye(4), SIZE) >= clearance for p in palm_dirs_box()])


def longest_arc(mask):
    """(start index, length in cells) of the longest circular run of True."""
    if mask.all():
        return 0, len(mask)
    if not mask.any():
        return 0, 0
    k = int(np.argmin(mask))                       # a False cell: no run wraps past it
    r = np.roll(mask, -k)
    best, run, start = (0, 0), 0, 0
    for i, v in enumerate(r):
        run = run + 1 if v else 0
        if run == 1:
            start = i
        if run > best[1]:
            best = (start, run)
    return (best[0] + k) % len(mask), best[1]


def usable(mask, rho):
    """The longest arc, shrunk by rho [rad] at both ends (the phi damper). Returns (mask, lo, length)."""
    lo, L = longest_arc(mask)
    e = int(round(math.degrees(rho) / STEP))
    out = np.zeros(len(mask), bool)
    if L - 2 * e > 0:
        out[(lo + e + np.arange(L - 2 * e)) % len(mask)] = True
    return out, (lo + e) % len(mask), max(L - 2 * e, 0)


def components(mask):
    """Labels of the circular runs of True (-1 for False)."""
    if mask.all():
        return np.zeros(len(mask), int)
    k = int(np.argmin(mask))
    r = np.roll(mask, -k)
    lab = np.where(r, np.cumsum(r & ~np.roll(r, 1)) - 1, -1)
    return np.roll(lab, k)


def connected(masks):
    """Forward sweep: at each sample the parts of masks[k] reachable from masks[0] by a continuous path."""
    R = [masks[0]]
    for m in masks[1:]:
        lab = components(m)
        keep = np.unique(lab[R[-1] & m])
        R.append(np.isin(lab, keep[keep >= 0]))
    return R


def width(mask):
    return longest_arc(mask)[1] * STEP


def palm_low(cz, alpha):
    """clearance.palm_lowest for hands with y_E horizontal (the case here), vectorised over alpha [rad]."""
    s, c = np.sin(alpha), np.cos(alpha)
    return cz - C.PALM_CENTRE[2] * s - np.abs(c) * C.PALM_HALF[0] - np.abs(s) * C.PALM_HALF[2]


def planar(T0, u, poses):
    """psi index of every alpha at each pose (rows), and its inverse."""
    alpha = np.radians(np.arange(N) * STEP)
    p_w = np.outer(np.cos(alpha), u) + np.outer(np.sin(alpha), [0, 0, 1.0])
    psi_idx, inv = np.empty((len(poses), N), int), np.empty((len(poses), N), int)
    for k, T in enumerate(poses):
        pb = p_w @ T[:3, :3]                       # rows: R^T p
        psi = np.degrees(np.arctan2(pb[:, 2], pb[:, 0]))
        psi_idx[k] = np.round(psi / STEP).astype(int) % N
        inv[k, psi_idx[k]] = np.arange(N)
        assert len(np.unique(psi_idx[k])) == N, "alpha and psi grids must map one to one"
    return psi_idx, inv


def study_grasp(poses, psi_idx, inv, c_o, free, rho):
    """Everything about one grasp point and one damper margin (see the module docstring); angles in deg, m.
    poses, psi_idx, inv: chain_poses and planar of the chain."""
    F, lo, L = usable(free, rho)
    alpha = np.radians(np.arange(N) * STEP)
    cz = np.array([(T @ np.r_[c_o, 1])[2] for T in poses])
    table = np.array([palm_low(z, alpha) for z in cz])                       # (samples, alpha)
    ok = F[psi_idx] & (table > TABLE_MARGIN)                                 # hand clear of box and table
    fwd = connected(list(ok))
    bwd = connected(list(ok[::-1]))[::-1]

    # lazy: hand still until phi reaches the damper stop, then it turns with the box; from every connected start
    a = np.flatnonzero(bwd[0])
    res = dict(arc=float(width(free)), usable=L * STEP, tilt=180.0 - L * STEP,
               start_window=width(ok[0]), end_window=width(ok[-1]),
               start_window_connected=width(bwd[0]), end_window_connected=width(fwd[-1]))
    if len(a):
        pos = (psi_idx[0, a] - lo) % N                                       # position of psi in the usable arc
        low = table[0, a].copy()
        for k in range(1, len(poses)):
            d = (psi_idx[k, a] - psi_idx[k - 1, a] + N // 2) % N - N // 2   # psi change with the hand kept still
            pos = np.clip(pos + d, 0, L - 1)
            a = inv[k, (lo + pos) % N]
            low = np.minimum(low, table[k, a])
        best = int(np.argmax(low))
        a0 = int(np.flatnonzero(bwd[0])[best])
        res.update(lazy_alpha0=a0 * STEP, lazy_alpha_end=float(a[best] * STEP), lazy_table_min=float(low[best]),
                   lazy_ok=bool(low[best] > TABLE_MARGIN))
    else:
        res.update(lazy_alpha0=None, lazy_alpha_end=None, lazy_table_min=None, lazy_ok=False)

    # rigid: psi fixed at its start value (any psi clear of the box, no damper)
    a_r = inv[:, np.flatnonzero(free)]                                       # alpha of each psi at each sample
    low_r = table[np.arange(len(poses))[:, None], a_r].min(axis=0)
    ok_r = np.zeros(N, bool)
    ok_r[np.flatnonzero(free)] = low_r > TABLE_MARGIN
    j = int(np.argmax(low_r)) if len(low_r) else None
    res.update(rigid_window=width(ok_r), rigid_table_best=None if j is None else float(low_r[j]),
               rigid_alpha0=None if j is None else float(inv[0, np.flatnonzero(free)[j]] * STEP))
    return res


def observability(pl1, pl2, T_grasp):
    """grasp_observability margins [m] of both pivots for the grasp frame T_grasp at the start."""
    T2 = se3.exp6(pl1.S * pl1.theta) @ T_grasp
    out = []
    for pl, T in ((pl1, T_grasp), (pl2, T2)):
        step = SimpleNamespace(kind="pivot", plan=SimpleNamespace(T_grasp=T, screws=pl.screws))
        out.append(grasp_observability(step)[1])
    return out


def grasp_frame(T0, u, c_o, alpha0, y_ref=Y_REF):
    """{c} at the start: at the grasp point, palm toward alpha0 [deg], y = n with the planner's sign rule."""
    p = math.cos(math.radians(alpha0)) * u + math.sin(math.radians(alpha0)) * np.array([0, 0, 1.0])
    y = T0[:3, 1] * (1.0 if T0[:3, 1] @ y_ref >= 0 else -1.0)
    z = -p
    return se3.rp(np.column_stack([np.cross(y, z), y, z]), (T0 @ np.r_[c_o, 1])[:3])


# ---- start poses -----------------------------------------------------------------------------------------------
def pose_checks(face, x, y, yaw, grasps):
    """Reach margin [m] of each grasp point over both arcs, base keep-out, table area of start and end."""
    T0 = rest_pose(face, x, y, yaw)
    pl1, pl2, u = roll_chain(T0)
    poses = chain_poses(pl1, pl2, dth=7.5)
    keep = P.keepout_ok(pl1.S, pl1.theta, pl1.T_start, SIZE) and P.keepout_ok(pl2.S, pl2.theta, pl2.T_start, SIZE)
    end = pl2.T_goal[:3, 3]
    reach = {}
    for name, c_o in grasps.items():
        c_o = corner_point(T0, u, *c_o)
        r = np.array([np.linalg.norm((T @ np.r_[c_o, 1])[:2]) for T in poses])
        reach[name] = float(min((r - REACH[0]).min(), (REACH[1] - r).min()))
    inside = lambda p: REGION["x"][0] <= p[0] <= REGION["x"][1] and REGION["y"][0] <= p[1] <= REGION["y"][1]
    return dict(face=face, x=x, y=y, yaw=yaw, keepout=bool(keep), end_xy=[float(end[0]), float(end[1])],
                end_in_region=bool(inside(end)), reach=reach)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-rollouts", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plot", action="store_true", help="only redraw the figure from the saved json")
    ap.add_argument("--sweep", action="store_true", help="also roll out every start pose of a coarser grid (free, rigid)")
    args = ap.parse_args()
    if args.plot:
        return figure(json.load(open(OUT.with_suffix(".json"))))
    C.HAND_MODEL = "box"                           # box_real.yaml planner.hand_model
    out = dict(size=SIZE, step_deg=STEP, dth_deg=DTH, table_margin=TABLE_MARGIN, clearance=C.CLEARANCE,
               rhos=RHOS, insets=INSETS.tolist())

    # 1. the chains of two pivots
    out["chains"] = {}
    for face in FACES:
        T0 = rest_pose(face)
        out["chains"][face] = [dict(kind=k, down_end=down_axis(p2.T_goal).tolist(),
                                    net_rotation_deg=math.degrees(se3.angle_between(T0[:3, :3], p2.T_goal[:3, :3])),
                                    shift_m=float(np.linalg.norm(p2.T_goal[:3, 3] - T0[:3, 3])))
                               for k, p1, p2 in two_pivot_chains(T0)]
        print(face, [(c["kind"], round(c["net_rotation_deg"]), round(c["shift_m"], 3)) for c in out["chains"][face]])

    # 2. the free arc of psi in the box frame (the cross-section is symmetric: one table for every corner)
    out["arc"] = {}
    for a in INSETS:
        for b in INSETS:
            c_o = np.array([0.075 - a, 0.0, 0.125 - b])
            out["arc"][f"{a*1e3:.0f},{b*1e3:.0f}"] = width(free_mask(c_o))
    print("free arc (box model) [deg], x inset (rows) vs z inset:", out["arc"])

    # 3. per face, corner, inset, rho: windows, tilt, lazy and rigid paths, observability
    out["grasps"] = []
    for face in FACES:
        T0 = rest_pose(face)
        pl1, pl2, u = roll_chain(T0)
        poses = chain_poses(pl1, pl2)
        psi_idx, inv = planar(T0, u, poses)
        for corner in CORNERS:
            for a in INSETS:
                for b in INSETS:
                    c_o = corner_point(T0, u, corner, a, b)
                    free = free_mask(c_o)
                    row = dict(face=face, corner=corner, a=float(a), b=float(b), c_o=c_o.tolist())
                    for rho in RHOS:
                        row[f"rho{rho:.2f}"] = study_grasp(poses, psi_idx, inv, c_o, free, rho)
                    a0 = row["rho0.13"]["lazy_alpha0"]
                    row["obs_margin"] = observability(pl1, pl2, grasp_frame(T0, u, c_o, 90.0 if a0 is None else a0))
                    out["grasps"].append(row)

    # per corner two grasps: the largest smaller window (connected, rho_s of box_real) with a hand tilt of at most
    # 90 deg ("tilt90"), and without that bound ("any"); both pivots observable
    def score(r):
        g = r["rho0.13"]
        return min(g["start_window_connected"], g["end_window_connected"]) if min(r["obs_margin"]) >= 0.02 else -1
    out["best"] = {}
    for face in FACES:
        for corner in CORNERS:
            rows = [r for r in out["grasps"] if r["face"] == face and r["corner"] == corner]
            for variant, rs in (("tilt90", [r for r in rows if r["rho0.13"]["tilt"] <= 90]), ("any", rows)):
                r = max(rs, key=lambda r: (score(r), -r["rho0.13"]["tilt"]))
                key = f"{face}/{corner}/{variant}"
                out["best"][key] = dict(a=r["a"], b=r["b"], score=score(r), **r["rho0.13"], obs_margin=r["obs_margin"])
                print(key, {k: v for k, v in out["best"][key].items() if k in
                            ("a", "b", "score", "usable", "tilt", "lazy_alpha0", "lazy_alpha_end", "lazy_table_min")})
    # the hand mesh (5 mm) for comparison of the arc, at the best corner insets
    out["arc_mesh"] = {f"{a*1e3:.1f},{b*1e3:.1f}": width(free_mask(np.array([0.075 - a, 0, 0.125 - b]), "mesh"))
                       for a, b in ((0.01, 0.01), (0.02, 0.02), (0.01, 0.03), (0.01, 0.035), (0.03, 0.01))}

    # 4. start poses: reach, keep-out, table area, for the best grasp of the corners with a window
    usable_corners = [k for k, v in out["best"].items() if v["score"] >= MIN_MARGIN]
    jobs = [(face, x, y, yaw, {k.split("/", 1)[1]: (k.split("/")[1], out["best"][k]["a"], out["best"][k]["b"])
                               for k in usable_corners if k.startswith(face)})
            for face in FACES for x in np.arange(0.30, 0.751, 0.05) for y in np.arange(-0.35, 0.351, 0.05)
            for yaw in range(0, 360, 15)]
    with ProcessPoolExecutor(args.workers) as ex:
        out["poses"] = list(ex.map(_pose_job, jobs, chunksize=64))
    for face in FACES:
        rows = [p for p in out["poses"] if p["face"] == face]
        for corner in sorted({k.split("/", 1)[1] for k in usable_corners if k.startswith(face)}):
            ok = [p for p in rows if p["keepout"] and p["reach"][corner] >= 0]
            print(f"{face}/{corner}: {len(ok)} of {len(rows)} start poses with reach and keep-out "
                  f"({sum(p['end_in_region'] for p in ok)} of them end inside the table area)")
    save(out)

    # 5. confirmation rollouts of the best few start poses
    if not args.no_rollouts:
        out["rollouts"] = rollouts(out, usable_corners, args.workers)
        save(out)
    if args.sweep:
        out["sweep"] = sweep(out, [k for k in usable_corners if k.endswith("tilt90")], args.workers)
        save(out)
    figure(out)
    return out


def save(out):
    """The results as json, floats to 5 significant digits (size)."""
    r = lambda v: (float(f"{v:.5g}") if isinstance(v, float) else {k: r(x) for k, x in v.items()} if isinstance(v, dict)
                   else [r(x) for x in v] if isinstance(v, (list, tuple)) else v)
    json.dump(r(out), open(OUT.with_suffix(".json"), "w"))


def _pose_job(job):
    return pose_checks(*job)


# ---- confirmation rollouts -------------------------------------------------------------------------------------
_CACHE = {}


def _robot():
    """(robot, config) of box_real.yaml, once per process."""
    if not _CACHE:
        sys.path.insert(0, str(HERE))
        import run_plan as rp

        cfg = rp.load_config(HERE / "configs" / "box_real.yaml")
        _CACHE.update(robot=rp.make_robot(cfg), cfg=cfg, rp=rp)
        C.HAND_MODEL = "box"
    return _CACHE["robot"], _CACHE["cfg"]


def ik(robot, T, q0, iters=500):
    """theta with {E} (phi = 0) at T, inside the joint limits: damped least squares from q0, and from q0 with joint 1
    turned toward the grasp point. Returns (theta, pose error norm) of the seed that converges with the largest
    distance to the joint limits."""
    from pandaSim.control.contact_qp import EE, OMEGA_V
    from pandaSim.control.primitive_executor import contact_pose

    ets, lb, ub = robot.ets(end=EE), robot.qlim[0, :7] + 0.01, robot.qlim[1, :7] - 0.01
    out = []
    for j1 in (None, math.atan2(T[1, 3], T[0, 3])):
        q = np.array(q0, float)
        if j1 is not None:
            q[0] = j1
        for _ in range(iters):
            e = se3.log6(se3.inv(contact_pose(robot, q)) @ T)
            if np.linalg.norm(e) < 1e-10:
                break
            J = ets.jacobe(np.r_[q, 0.0])[OMEGA_V, :7]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.clip(e, -0.2, 0.2)), lb, ub)
        err = float(np.linalg.norm(se3.log6(se3.inv(contact_pose(robot, q)) @ T)))
        out.append((err > 1e-6, -float(np.minimum(q - lb, ub - q).min()), q, err))
    best = min(out, key=lambda o: o[:2])
    return best[2], best[3]


def setup(face, x, y, yaw, corner, a, b, alpha0):
    T0 = rest_pose(face, x, y, yaw)
    pl1, pl2, u = roll_chain(T0)
    c_o = corner_point(T0, u, corner, a, b)
    T_grasp = grasp_frame(T0, u, c_o, alpha0)
    return T0, pl1, pl2, u, T_grasp, C.phi_limits(se3.inv(T_grasp) @ T0, SIZE, model="box")


def direct_rollout(job):
    """ContactQP alone, the two screws in sequence from the grasp (IK), phi carried over (tests/unit/test_contact_qp.py
    pivot_run does the same for one screw). Every 20 ticks: palm above the table, palm clear of the box (box model,
    2 mm), arm and hand clear of box and table (ArmCollision), hand orientation alpha."""
    from pandaSim.control.contact_qp import ContactQP
    from pandaSim.control.primitive_executor import PEAK_SDOT, contact_pose
    from pandaSim.planning.collision import ArmCollision
    from pandaSim.planning.primitives import screw_twist, time_scaling

    pose, free, dt = job["pose"], job["free"], 1e-3
    robot, cfg = _robot()
    T0, pl1, pl2, u, T_grasp, phi_lim = setup(**pose)
    lb, ub = robot.qlim[0, :7], robot.qlim[1, :7]
    theta, err = ik(robot, T_grasp, cfg["robot"]["q_ready"])
    res = dict(job, phi_lim_deg=np.degrees(phi_lim).tolist(), ik_err=err)
    if err > 1e-6 or np.any(theta < lb) or np.any(theta > ub):
        return dict(res, ok=False, cause="no IK solution within the joint limits at the grasp", angle_deg=0.0)
    qp = dict(cfg["execution"]["qp"], **job["qp"])
    ctrl = ContactQP(robot, free_rotation=free, **{"phi_lim": phi_lim, **qp})    # qp phi_lim None: phi unconstrained
    q = np.r_[theta, 0.0] if free else theta.copy()
    fk = lambda q: contact_pose(robot, q)
    T_c_obj = se3.inv(fk(q)) @ T0
    arm = ArmCollision(robot, SIZE)
    w_max = cfg["execution"]["pivot"]["w_max"]
    trace = dict(angle=[], alpha=[], phi=[], table=[], arm=[], margin=[])
    box_hit, hit_deg, cause, done = False, None, None, 0.0
    for S, th in ((pl1.S, pl1.theta), (pl2.S, pl2.theta)):
        V1, Tf = screw_twist(S, th, fk(q)), PEAK_SDOT["q"] * th / w_max
        n = int(round(Tf / dt))
        for k in range(n + 1):
            s, sdot = time_scaling(k * dt, Tf)
            qd = ctrl.solve(q, sdot * V1)
            if qd is None:
                th7 = q[:7]
                rho = np.minimum(ub - th7, th7 - lb)
                J7 = ctrl.jacobian(q)[:, :7]                       # the arm alone (phi at its stop, or rigid)
                ls = np.linalg.pinv(J7) @ (sdot * V1)
                near = [f"j{j + 1} {rho[j]:.2f} rad" for j in np.flatnonzero(rho < ctrl.rho_e)]
                stop = ""
                if free and ctrl.phi_lim is not None:
                    r_phi = min(q[7] - phi_lim[0], phi_lim[1] - q[7])
                    stop = f"phi {math.degrees(q[7]):.0f} deg, {math.degrees(r_phi):.1f} deg from its limit; "
                cause = (f"infeasible QP at {math.degrees(done + s * th):.0f} of 180 deg; {stop}within rho_e of a "
                         f"joint limit: {', '.join(near) or 'none'}; arm alone: min-norm rate/bound "
                         f"{np.max(np.abs(ls) / ctrl.qd_max):.2f}, smallest singular value of J "
                         f"{np.linalg.svd(J7, compute_uv=False)[-1]:.3f}")
                break
            q = q + qd * dt
            if k % 20 == 0 or k == n:
                T_E, T_c = fk(q[:7]), fk(q)
                T_obj = T_c @ T_c_obj
                p = -T_E[:3, 2]
                if C.palm_hits_box(T_E, T_obj, SIZE, 0.002) and not box_hit:
                    box_hit, hit_deg = True, math.degrees(done + s * th)
                trace["angle"].append(math.degrees(done + s * th))
                trace["alpha"].append(math.degrees(math.atan2(p[2], p @ u)))
                trace["phi"].append(math.degrees(q[7]) if free else 0.0)
                trace["table"].append(C.palm_lowest(T_E))
                trace["arm"].append(arm.ok(q[:7], T_obj, q[7] if free else 0.0, width=pl1.width)[2])
                trace["margin"].append(float(np.minimum(ub - q[:7], q[:7] - lb).min()))
        if cause:
            break
        done += th
    if not trace["angle"]:                         # infeasible at the first tick: the start itself is out
        return dict(res, ok=False, cause=cause + " (at the start)", angle_deg=0.0)
    T_end = fk(q) @ T_c_obj
    E = se3.inv(pl2.T_goal) @ T_end
    trace["alpha"] = np.degrees(np.unwrap(np.radians(trace["alpha"])))
    a = trace["alpha"]
    res.update(ok=bool(cause is None and not box_hit and min(trace["table"]) > TABLE_MARGIN and min(trace["arm"]) >= 0.002),
               cause=cause, angle_deg=trace["angle"][-1] if trace["angle"] else 0.0, box_hit=bool(box_hit),
               box_hit_deg=hit_deg,
               table_min=min(trace["table"]), arm_min=min(trace["arm"]), margin_min=min(trace["margin"]),
               alpha_range=[float(a.min()), float(a.max())],
               phi_range=[min(trace["phi"]), max(trace["phi"])],
               pos_err=float(np.linalg.norm(E[:3, 3])) if cause is None else None,
               rot_err_deg=math.degrees(np.linalg.norm(se3.log3(E[:3, :3]))) if cause is None else None,
               trace={k: np.round(v, 4).tolist() for k, v in trace.items()})
    return res


def executor_rollout(job):
    """The chain as one step of the executor (run_plan: approach, grasp, both screws with phi carried over, release,
    retreat, ready) with the planner's check (feasibility.Rollout, dt 0.01, palm, arm and hand vs box and table)."""
    from pandaSim.planning.feasibility import Rollout
    from pandaSim.planning.sequencer import Step

    robot, cfg = _robot()
    rp = _CACHE["rp"]
    T0, pl1, pl2, u, T_grasp, phi_lim = setup(**job["pose"])
    kw = dict(rp.exec_kwargs(cfg), **job["qp"])                    # job["qp"]: run_plan options here (lift)
    kin = Rollout(robot, free_rotation=job["free"], dt=cfg["planner"]["dt"], size=SIZE, **kw)
    plan = SimpleNamespace(T_grasp=T_grasp, screws=[(pl1.S, pl1.theta), (pl2.S, pl2.theta)], width=pl1.width,
                           phi_lim=phi_lim)
    log, fail = kin._rollout(Step("pivot", T0, pl2.T_goal, plan))
    a = log.arrays()
    lb, ub = robot.qlim[0, :7], robot.qlim[1, :7]
    piv = [s for s in log.segments if s["name"] == "pivot"]
    reached = sum(90.0 * min(1.0, (s["t_end"] - s["t0"]) / s["Tf"]) for s in piv) if piv else 0.0
    return dict(job, ok=fail is None, cause=None if fail is None else f"{fail[0]}: {fail[1]}", pivot_deg=reached,
                margin_min=float(np.minimum(ub - a["theta"], a["theta"] - lb).min()) if len(a["theta"]) else None,
                phi_range=[float(np.degrees(a["phi"].min())), float(np.degrees(a["phi"].max()))] if len(a["phi"]) else None)


def _run(job):
    return direct_rollout(job) if job["kind"] == "direct" else executor_rollout(job)


def roll_class(T0, u):
    """Direction of the roll relative to the base: out, in (radial), left, right (tangential)."""
    r = T0[:2, 3] / np.linalg.norm(T0[:2, 3])
    c, s = float(u[:2] @ r), float(r[0] * u[1] - r[1] * u[0])
    return "out" if c > 0.71 else "in" if c < -0.71 else ("left" if s > 0 else "right")


def rollouts(out, usable_corners, workers):
    """The best start pose (largest reach margin) of each roll direction, for each usable corner; and the start of the
    protocol's seq/flip chain (0.50, 0.00, lying, rolled along -y: the slide target of its plan)."""
    cands = []
    for key in usable_corners:
        face, corner, variant = key.split("/")
        g, best = f"{corner}/{variant}", {}
        for p in out["poses"]:
            if p["face"] != face or not p["keepout"] or p["reach"][g] < 0:
                continue
            T0 = rest_pose(face, p["x"], p["y"], p["yaw"])
            cls = roll_class(T0, roll_chain(T0)[2])
            if cls not in best or p["reach"][g] > best[cls]["reach"][g]:
                best[cls] = p
        for cls, p in sorted(best.items()):
            cands.append(dict(label=f"{key}/{cls}", key=key, face=face, x=p["x"], y=p["y"], yaw=p["yaw"], corner=corner))
        if face == "lying":
            cands.append(dict(label=f"seq_flip/{corner}/{variant}", key=key, face=face, x=0.5, y=0.0, yaw=90.0,
                              corner=corner))
    jobs = []
    for c in cands:
        g = out["best"][c["key"]]
        pose = dict(face=c["face"], x=float(c["x"]), y=float(c["y"]), yaw=float(c["yaw"]), corner=c["corner"],
                    a=g["a"], b=g["b"], alpha0=g["lazy_alpha0"])
        # box_real's QP; with rho_s 0.05 (box_plan.yaml: the damper margin of phi smaller, and of the joints too);
        # without the centering and manipulability terms (K = lam = 0: the hand turns only when phi is at its stop)
        # the executor with its lift of 0.10 m and with 0.25 m (pre pose and retreat straight up above the grasp)
        for kind, free, qp in (("direct", True, {}), ("direct", False, {}), ("direct", True, dict(rho_s=0.05)),
                               ("direct", True, dict(K=0.0, lam=0.0)), ("direct", True, dict(phi_lim=None)),
                               ("executor", True, {}), ("executor", False, {}),
                               ("executor", True, dict(lift=0.25)), ("executor", False, dict(lift=0.25))):
            jobs.append(dict(label=c["label"], kind=kind, free=free, qp=qp, pose=pose))
        # the rigid grasp also from its own best start orientation (largest palm-table distance with phi fixed)
        jobs.append(dict(label=c["label"], kind="direct", free=False, qp={}, pose=dict(pose, alpha0=g["rigid_alpha0"])))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_run, jobs))
    for r in res:
        if not (r["label"].startswith("seq_flip") and not r["qp"]):
            r.pop("trace", None)                   # keep the traces of the protocol's flip only (size of the json)
    for r in res:
        print(f"{r['label']:40s} {r['kind']:8s} {'free' if r['free'] else 'rigid':5s} {str(r['qp']):24s}: "
              f"{'OK  ' if r['ok'] else 'FAIL'} {r.get('cause') or ''} "
              + (f"table {r['table_min']*1e3:.0f} mm, arm {r['arm_min']*1e3:.0f} mm, margin {r['margin_min']:.2f} rad, "
                 f"alpha {r['alpha_range'][0]:.0f}..{r['alpha_range'][1]:.0f}, phi {r['phi_range'][0]:.0f}..{r['phi_range'][1]:.0f}"
                 if r["kind"] == "direct" and "table_min" in r else ""))
    return res


def sweep(out, keys, workers):
    """Direct rollouts (free; rigid from the same and from its own start orientation) of every start pose with reach
    and keep-out on the grid x, y every 0.1 m, yaw every 30 deg. Returns the rows without traces."""
    jobs = []
    for key in keys:
        face, corner, variant = key.split("/")
        g = out["best"][key]
        for p in out["poses"]:
            on_grid = abs(p["x"] * 10 - round(p["x"] * 10)) < 1e-6 and abs(p["y"] * 10 - round(p["y"] * 10)) < 1e-6
            if p["face"] != face or not on_grid or p["yaw"] % 30 or not p["keepout"] or p["reach"][f"{corner}/{variant}"] < 0:
                continue
            pose = dict(face=face, x=float(p["x"]), y=float(p["y"]), yaw=float(p["yaw"]), corner=corner, a=g["a"],
                        b=g["b"], alpha0=g["lazy_alpha0"])
            for free, alpha0, qp in ((True, g["lazy_alpha0"], {}), (False, g["lazy_alpha0"], {}),
                                     (False, g["rigid_alpha0"], {}), (True, g["lazy_alpha0"], dict(phi_lim=None))):
                jobs.append(dict(label=key, kind="direct", free=free, qp=qp, pose=dict(pose, alpha0=alpha0)))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_run, jobs, chunksize=4))
    for r in res:
        r.pop("trace", None)
    for key in keys:
        rows = [r for r in res if r["label"] == key]
        n = len(rows) // 4
        free = [r for r in rows if r["free"] and not r["qp"]]
        unc = [r for r in rows if r["free"] and r["qp"]]
        rig = [r for r in rows if not r["free"]]
        poses_rigid = {(r["pose"]["x"], r["pose"]["y"], r["pose"]["yaw"]) for r in rig if r["ok"]}
        print(f"sweep {key}: {n} start poses; free {sum(r['ok'] for r in free)}, rigid {len(poses_rigid)} "
              f"(either start orientation); no IK {sum('IK' in (r['cause'] or '') for r in free)}; phi unconstrained: "
              f"{sum(r.get('cause') is None for r in unc)} complete, {sum(r['ok'] for r in unc)} ok, palm into the box in {sum(r.get('box_hit', False) for r in unc)}")
    return res


# ---- figure ----------------------------------------------------------------------------------------------------
def figure(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ins = np.array(out["insets"]) * 1e3
    grid = lambda f: np.array([[f(next(r for r in out["grasps"] if r["face"] == "lying" and r["corner"] == "back_top"
                                        and abs(r["a"] * 1e3 - a) < 1e-6 and abs(r["b"] * 1e3 - b) < 1e-6))
                                for a in ins] for b in ins])
    usable_arc = grid(lambda r: r["rho0.13"]["usable"])
    window = grid(lambda r: min(r["rho0.13"]["start_window_connected"], r["rho0.13"]["end_window_connected"]))
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.6), constrained_layout=True)
    ext = [ins[0] - 1.25, ins[-1] + 1.25, ins[0] - 1.25, ins[-1] + 1.25]
    for k, (A, title, vmax) in enumerate(((usable_arc, "(a) usable arc of $\\phi$ [deg], back-top", 140),
                                          (window, "(b) min(start, end window) [deg]", 30))):
        im = ax[0, k].imshow(A, origin="lower", extent=ext, cmap="cividis", vmin=0, vmax=vmax, aspect="equal")
        ax[0, k].contour(ins, ins, A, levels=[90.0 if k == 0 else MIN_MARGIN], colors="w", linewidths=1.0)
        ax[0, k].set(xlabel="inset from the back face $a$ [mm]", ylabel="inset from the top face $b$ [mm]", title=title)
        fig.colorbar(im, ax=ax[0, k], shrink=0.85)
    ax[0, 0].text(0.97, 0.95, "white: 90 deg (tilt 90 deg)", transform=ax[0, 0].transAxes, ha="right", va="top",
                  color="w", fontsize=8)
    ax[0, 1].text(0.97, 0.95, "white: 10 deg", transform=ax[0, 1].transAxes, ha="right", va="top", color="w", fontsize=8)
    # (c, d): the sweep: start poses by roll direction, stacked by the outcome of the chain (direct rollouts;
    # free: either back corner; rigid: either back corner and either start orientation)
    colours = {"free and rigid": "#E69F00", "free only": "#0072B2", "rigid only": "#D55E00", "neither": "#BBBBBB"}
    classes = ("left", "right", "in", "out")
    for k, face in enumerate(FACES):
        res = {}
        for r in out.get("sweep", []):
            p = r["pose"]
            if p["face"] != face:
                continue
            key = (p["x"], p["y"], p["yaw"])
            f, g = res.get(key, (False, False))
            res[key] = (f or (r["free"] and not r["qp"] and r["ok"]), g or (not r["free"] and r["ok"]))
        cnt = {c: dict.fromkeys(colours, 0) for c in classes}
        for (x, y, yaw), (f, g) in res.items():
            T0 = rest_pose(face, x, y, yaw)
            cnt[roll_class(T0, roll_chain(T0)[2])]["free and rigid" if f and g else "free only" if f else
                                                    "rigid only" if g else "neither"] += 1
        bottom = np.zeros(len(classes))
        for lab, c in colours.items():
            v = np.array([cnt[cl][lab] for cl in classes])
            ax[1, k].bar(classes, v, bottom=bottom, color=c, label=f"{lab} ({v.sum()})")
            bottom += v
        ax[1, k].set(xlabel="roll direction seen from the base", ylabel="start poses of the sweep",
                     title=f"({'cd'[k]}) {face}: chain in one grasp")
        ax[1, k].legend(fontsize=7, frameon=False)
    for ext_ in ("pdf", "png"):
        fig.savefig(OUT.with_suffix("." + ext_), dpi=200)


if __name__ == "__main__":
    main()
