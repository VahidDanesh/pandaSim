"""Free rotation against the rigid grasp on random trials, in simulation (the numbers of the manuscript).

    python experiments/ablation.py pivots          # random single 90 deg pivots, each run in three modes
    python experiments/ablation.py tasks           # random start and goal poses, planned and executed
    python experiments/ablation.py pivots --n 50 --workers 4 --seed 1

pivots: box poses are drawn on the table and kept if the pivot passes the planner's tests that do not depend
on the mode (base keep-out, reach of the grasp point). Every pivot is then executed as
    rigid          the hand is fixed to the box after the grasp; its tilt about the finger axis is the one the
                   approach ends with (the approach uses the free-rotation model, as in run_plan --rigid)
    rigid_tilt     the hand is fixed to the box with a prescribed tilt; every tilt of TILTS is run, and the
                   pivot counts as solved if one of them succeeds (a rigid grasp with the best of these tilts)
    free           the box may turn about the finger axis, phi within the limits that keep the palm clear
    free_nolimit   the same without the limits on phi
tasks: start and goal poses are drawn in a smaller region; each task is planned and executed with the rigid
model and with free rotation (planner and execution use the same model).

A run counts as a success if every segment finishes (the QP stays feasible), the box ends at its goal, the
palm stays 2 mm clear of the box and the table, and no link of the arm or the hand comes closer than 2 mm
to the box or the table (planning/collision.py, coal). Results: experiments/results/ablation_<name>.json; the same seed gives
the same trials. Figures and tables: experiments/ablation_figures.py.
"""
import argparse
import dataclasses
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_plan as rp  # noqa: E402

from pandaSim.control.backends import SimBackend  # noqa: E402
from pandaSim.planning.collision import ArmCollision
from pandaSim.control.contact_qp import EE  # noqa: E402
from pandaSim.control.primitive_executor import check_clearance, check_profile, run_plan  # noqa: E402
from pandaSim.planning.sequencer import pose_error  # noqa: E402

RESULTS = rp.HERE / "results"
PIVOT_MODES = ("rigid", "rigid_tilt", "free", "free_nolimit")
TILTS = (0.0, 22.5, 45.0, 67.5, 90.0)                     # [deg] hand tilt from the vertical, away from the box
TASK_MODES = ("rigid", "free")
PIVOT_REGION = dict(x=(0.20, 0.85), y=(-0.50, 0.50))     # box centre [m]; the planner's tests cut it down
TASK_REGION = dict(x=(0.35, 0.70), y=(-0.30, 0.30))
LYING, STANDING = -90.0, 0.0                              # pitch [deg]: on the 0.06 x 0.25 face / on the 0.15 x 0.06 face

_ctx = {}


def ctx(config):
    """Per-process robot, box and planners (built once)."""
    if not _ctx:
        cfg = rp.load_config(config)
        robot = rp.make_robot(cfg)
        planners = {m: rp.make_planner({**cfg, "plan_free_rotation": m == "free"}, robot) for m in TASK_MODES}
        _ctx.update(cfg=cfg, robot=robot, box=rp.make_box(cfg), planners=planners)
    return _ctx


def random_pose(rng, box, region):
    spec = dict(xy=[rng.uniform(*region["x"]), rng.uniform(*region["y"])],
                rpy=[0.0, float(rng.choice([LYING, STANDING])), rng.uniform(-180.0, 180.0)])
    return spec, rp.box_pose(box, spec)


_arm = {}


def arm_collision(robot, box):
    if "a" not in _arm:
        _arm["a"] = ArmCollision(robot, box.size)
    return _arm["a"]


def summarize(log, robot, cfg, box, T0, T_end, T_goal):
    """What one execution did: success, where it failed, and how close the arm came to its limits."""
    a = log.arrays()
    ep, er = pose_error(T_end, T_goal)
    tol = cfg["tolerance"]
    clear = check_clearance(log, robot, T0, box.size, arm=arm_collision(robot, box))
    names = [s["name"] for s in log.segments]
    held = np.isin(a["segment"], [j for j, n in enumerate(names) if n in ("pivot", "slide")])
    grasp = np.isin(a["segment"], [j for j, n in enumerate(names) if n == "approach"])
    lb, ub = robot.qlim[0, :7], robot.qlim[1, :7]
    ets, w = robot.ets(end=EE), []
    for th in a["theta"][held][::50]:
        J = ets.jacobe(np.r_[th, 0.0])[:, :7]
        w.append(math.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
    failed = next((s for s in log.segments if not s["ok"]), None)
    reached = ep < tol["position"] and er < math.radians(tol["rotation_deg"])
    return dict(
        success=bool(log.ok and reached and clear["box_ok"] and clear["table_ok"] and clear["arm_ok"]),
        completed=bool(log.ok), at_goal=bool(reached), palm_clear=bool(clear["box_ok"]), table_clear=clear["table_ok"], arm_clear=bool(clear["arm_ok"]),
        arm_min=clear["arm_min"], arm_link=(clear["arm_hits"][0][2] if clear["arm_hits"] else None),
        failed_segment=failed and failed["name"], cause=failed and failed["cause"],
        failed_at=failed and float((failed["t_end"] - failed["t0"]) / failed["Tf"]),     # fraction of the segment
        err_pos=ep, err_rot=er, duration=float(a["t"][-1]),
        joint_margin=float(np.minimum(a["theta"] - lb, ub - a["theta"]).min()),
        joint_margin_held=float(np.minimum(a["theta"][held] - lb, ub - a["theta"][held]).min()) if held.any() else None,
        manipulability_min=min(w) if w else None,
        phi_grasp=float(a["phi"][grasp][-1]) if grasp.any() else 0.0,                    # hand tilt chosen in the approach
        phi_min=float(a["phi"][held].min()) if held.any() else 0.0,
        phi_max=float(a["phi"][held].max()) if held.any() else 0.0,
        vel_ratio=float(check_profile(log)["vel_ratio"].max()),
    )


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


def execute(c, steps, T0, T_goal, mode, tilt=None):
    """tilt [rad]: the hand is turned by this angle about the finger axis at the grasp and the approach is rigid
    (T_c = T_E Rot_y(phi), so the hand frame is T_grasp Rot_y(-tilt))."""
    cfg, robot = c["cfg"], c["robot"]
    kw = rp.exec_kwargs(cfg)
    if mode == "free_nolimit":
        kw["qp"] = {**(kw["qp"] or {}), "phi_lim": None}
    if tilt is not None:
        steps = [dataclasses.replace(s, plan=dataclasses.replace(s.plan, T_grasp=s.plan.T_grasp @ rot_y(-tilt)))
                 for s in steps]
        kw["approach_free"] = False
    backend = SimBackend(kw["q_ready"], dt=cfg["execution"]["dt"])
    log, T_end = run_plan(backend, robot, steps, T0, free_rotation=mode != "rigid", **kw)
    return summarize(log, robot, cfg, c["box"], T0, T_end, T_goal)


# ---- random pivots ------------------------------------------------------------------------------
def draw_pivots(config, n, seed):
    """n pivots that pass the mode-independent tests: (pose spec, index of the pivot direction)."""
    c = ctx(config)
    box, seq = c["box"], c["planners"]["free"]
    rng, out, drawn = np.random.default_rng(seed), [], 0
    while len(out) < n:
        spec, T = random_pose(rng, box, PIVOT_REGION)
        j = int(rng.integers(2))
        drawn += 1
        if seq._pivot_ok(box, seq._box_pivots(box, T)[j][1]):
            out.append((spec, j))
    return out, drawn


def pivot_trial(args):
    config, k, spec, j = args
    c = ctx(config)
    box, seq = c["box"], c["planners"]["free"]
    T = rp.box_pose(box, spec)
    d_o, pl = seq._box_pivots(box, T)[j]
    step, _ = seq._pivot_step(box, T, d_o, pl)
    rec = dict(trial=k, start=spec, kind="tip up" if spec["rpy"][1] == LYING else "tip down",
               grasp_xy=pl.T_grasp[:2, 3].tolist(), goal_xy=pl.T_goal[:2, 3].tolist(), phi_lim=list(pl.phi_lim))
    for mode in ("rigid", "free", "free_nolimit"):
        rec[mode] = execute(c, [step], T, pl.T_goal, mode)
    # prescribed tilts lie on the side on which the palm has room (the wide side of phi_lim)
    side = 1.0 if pl.phi_lim[1] > -pl.phi_lim[0] else -1.0
    runs = [dict(execute(c, [step], T, pl.T_goal, "rigid", tilt=side * math.radians(a)), tilt_deg=a) for a in TILTS]
    best = max(runs, key=lambda r: (r["success"], r["completed"], r["joint_margin_held"] or -1.0))
    rec["rigid_tilt"] = dict(best, per_tilt=[dict(tilt_deg=r["tilt_deg"], success=r["success"], cause=r["cause"],
                                                  palm_clear=r["palm_clear"] and r["table_clear"],
                                                  failed_segment=r["failed_segment"]) for r in runs])
    return rec


# ---- random tasks -------------------------------------------------------------------------------
def draw_tasks(config, n, seed):
    c = ctx(config)
    rng = np.random.default_rng(seed)
    return [(random_pose(rng, c["box"], TASK_REGION)[0], random_pose(rng, c["box"], TASK_REGION)[0]) for _ in range(n)]


def task_trial(args):
    config, k, start, goal = args
    c = ctx(config)
    box = c["box"]
    T0, Tg = rp.box_pose(box, start), rp.box_pose(box, goal)
    rec = dict(trial=k, start=start, goal=goal)
    for mode in TASK_MODES:
        plan = c["planners"][mode].plan(box, T0, Tg)
        r = dict(found=plan.found, planning_s=plan.stats.get("time_s"), cost=plan.cost if plan.found else None,
                 steps=[s.kind for s in plan.steps], success=False)
        if plan.found:
            r.update(execute(c, plan.steps, T0, Tg, mode))
        rec[mode] = r
    return rec


# ---- summaries ----------------------------------------------------------------------------------
def pivot_summary(trials):
    out = {}
    for kind in ("all", "tip up", "tip down"):
        rows = [t for t in trials if kind in ("all", t["kind"])]
        out[kind] = dict(n=len(rows))
        for m in PIVOT_MODES:
            r = [t[m] for t in rows]
            out[kind][m] = dict(
                success=sum(x["success"] for x in r),
                qp_infeasible=sum(x["cause"] == "infeasible QP" for x in r),
                other_failure=sum(not x["completed"] and x["cause"] != "infeasible QP" for x in r),
                palm_contact=sum(x["completed"] and not (x["palm_clear"] and x["table_clear"]) for x in r),
                arm_contact=sum(x["completed"] and x["palm_clear"] and x["table_clear"] and not x["arm_clear"] for x in r),
                joint_margin_median=float(np.median([x["joint_margin_held"] for x in r if x["completed"]] or [math.nan])),
            )
        out[kind]["rigid_per_tilt"] = {f"{a:g}": sum(t["rigid_tilt"]["per_tilt"][i]["success"] for t in rows)
                                       for i, a in enumerate(TILTS)}
    return out


def task_summary(trials):
    out = dict(n=len(trials))
    for m in TASK_MODES:
        r = [t[m] for t in trials]
        ok = [x for x in r if x["success"]]
        out[m] = dict(
            planned=sum(x["found"] for x in r), success=len(ok),
            planning_s_median=float(np.median([x["planning_s"] for x in r])),
            steps_mean=float(np.mean([len(x["steps"]) for x in ok] or [math.nan])),
            pivots_mean=float(np.mean([x["steps"].count("pivot") for x in ok] or [math.nan])),
            duration_mean=float(np.mean([x["duration"] for x in ok] or [math.nan])),
            cost_mean=float(np.mean([x["cost"] for x in ok] or [math.nan])),
        )
    both = [t for t in trials if t["rigid"]["success"] and t["free"]["success"]]
    out["both"] = dict(n=len(both), **{f"{k}_{m}": float(np.mean([v(t[m]) for t in both] or [math.nan]))
                                        for m in TASK_MODES
                                        for k, v in (("duration", lambda x: x["duration"]), ("cost", lambda x: x["cost"]),
                                                     ("steps", lambda x: len(x["steps"])))})
    return out


def print_summary(name, s):
    if name == "pivots":
        for kind, row in s.items():
            print(f"{kind} (n = {row['n']})")
            for m in PIVOT_MODES:
                r = row[m]
                print(f"  {m:13s} success {r['success']:4d} ({100 * r['success'] / max(row['n'], 1):5.1f} %)   "
                      f"QP infeasible {r['qp_infeasible']:4d}   other {r['other_failure']:3d}   "
                      f"palm contact {r['palm_contact']:4d}   other link {r['arm_contact']:3d}   median joint margin {r['joint_margin_median']:.2f} rad")
            print(f"  rigid, successes per prescribed tilt [deg]: {row['rigid_per_tilt']}")
    else:
        for m in TASK_MODES:
            r = s[m]
            print(f"  {m:6s} planned {r['planned']:4d}   executed {r['success']:4d} of {s['n']}   "
                  f"planning {r['planning_s_median']:.1f} s (median)   steps {r['steps_mean']:.2f}   "
                  f"pivots {r['pivots_mean']:.2f}   duration {r['duration_mean']:.1f} s   cost {r['cost_mean']:.2f}")
        print(f"  solved in both modes: {s['both']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("experiment", choices=["pivots", "tasks"])
    ap.add_argument("--n", type=int, default=None, help="number of trials (default: 300 pivots, 100 tasks)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--config", default=str(rp.DEFAULT_CONFIG))
    ap.add_argument("--out", default=None, help="result file (default: experiments/results/ablation_<experiment>.json)")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    meta = dict(experiment=args.experiment, seed=args.seed, config=rp.load_config(args.config))
    if args.experiment == "pivots":
        n = args.n or 300
        samples, drawn = draw_pivots(args.config, n, args.seed)
        meta.update(region=PIVOT_REGION, drawn=drawn)
        jobs, fn, summary = [(args.config, k, s, j) for k, (s, j) in enumerate(samples)], pivot_trial, pivot_summary
    else:
        n = args.n or 100
        jobs = [(args.config, k, s, g) for k, (s, g) in enumerate(draw_tasks(args.config, n, args.seed))]
        meta.update(region=TASK_REGION)
        fn, summary = task_trial, task_summary
    with Pool(min(args.workers, len(jobs))) as pool:
        trials = pool.map(fn, jobs, chunksize=1)
    s = summary(trials)
    print_summary(args.experiment, s)
    out = Path(args.out) if args.out else RESULTS / f"ablation_{args.experiment}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta, n=n, summary=s, trials=trials), indent=1))
    print(f"{n} trials in {time.perf_counter() - t0:.0f} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
