"""Plan a box task and execute the plan in simulation.

    python experiments/run_plan.py                                # the config's default task, headless
    python experiments/run_plan.py --task all                     # every task of the config
    python experiments/run_plan.py --task slide_pivot --viewer    # then replay it in Swift (browser tab)
    python experiments/run_plan.py --task all --rigid             # ablation: no free rotation

The log of each run is saved to experiments/runs/<task>_<free|rigid>.npz.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from pandaSim.control.backends import SimBackend
from pandaSim.control.primitive_executor import check_clearance, check_profile, run_plan
from pandaSim.planning import se3
from pandaSim.planning.collision import ArmCollision
from pandaSim.planning.feasibility import Rollout
from pandaSim.planning.sequencer import Box, Sequencer, pose_error, snap_stable

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs" / "box_plan.yaml"


def load_config(path=DEFAULT_CONFIG, free_rotation=None, plan_free_rotation=None):
    """Read the YAML file; the two arguments override its ablation switches."""
    cfg = yaml.safe_load(Path(path).read_text())
    if free_rotation is not None:
        cfg["free_rotation"] = bool(free_rotation)
    if plan_free_rotation is not None:
        cfg["plan_free_rotation"] = bool(plan_free_rotation)
    if cfg.get("plan_free_rotation") is None:
        cfg["plan_free_rotation"] = cfg["free_rotation"]
    return cfg


def make_robot(cfg):
    import spatialmath as sm

    from pandaSim.geometry.utils import create_virtual_panda

    robot = create_virtual_panda()
    robot.base = sm.SE3(*cfg["robot"]["base"])
    return robot


def make_box(cfg):
    b = cfg["box"]
    return Box(tuple(b["size"]), b["mass"], b["mu"])


def box_pose(box, spec):
    """Resting pose from {xy: [x, y], rpy: [roll, pitch, yaw] in degrees}."""
    r, p, y = np.radians(spec["rpy"])
    R = se3.exp3(np.array([0.0, 0.0, y])) @ se3.exp3(np.array([0.0, p, 0.0])) @ se3.exp3(np.array([r, 0.0, 0.0]))
    return snap_stable(box, se3.rp(R, np.array([*spec["xy"], 0.0])))


def exec_kwargs(cfg):
    e = cfg["execution"]
    return dict(q_ready=np.array(cfg["robot"]["q_ready"], float), lift=e["lift"], pivot=e.get("pivot"),
                grasp=e.get("grasp"), qp=e.get("qp") or None)


def make_planner(cfg, robot):
    """Sequencer whose feasibility check rolls each step out with the executor (model: plan_free_rotation)."""
    p = cfg["planner"]
    kin = Rollout(robot, free_rotation=cfg["plan_free_rotation"], dt=p["dt"], size=make_box(cfg).size, **exec_kwargs(cfg))
    return Sequencer(kin, max_pivots=p["max_pivots"], max_kin_checks=p["max_kin_checks"],
                     base_xy=tuple(cfg["robot"]["base"][:2]))


def run_task(cfg, robot, name, planner=None):
    """Plan task `name` and execute the plan in simulation. Returns a dict with the plan, the log and the checks."""
    box = make_box(cfg)
    T0, Tg = (box_pose(box, cfg["tasks"][name][k]) for k in ("start", "goal"))
    planner = planner or make_planner(cfg, robot)
    plan = planner.plan(box, T0, Tg)
    res = dict(task=name, box=box, T0=T0, Tg=Tg, plan=plan, free_rotation=cfg["free_rotation"],
               plan_free_rotation=cfg["plan_free_rotation"], log=None, checks={"plan found": plan.found})
    if not plan.found:
        return res

    kw = exec_kwargs(cfg)
    backend = SimBackend(kw["q_ready"], dt=cfg["execution"]["dt"])
    t0 = time.perf_counter()
    log, T_end = run_plan(backend, robot, plan.steps, T0, free_rotation=cfg["free_rotation"], **kw)
    ep, er = pose_error(T_end, Tg)
    prof, tol = check_profile(log), cfg["tolerance"]
    clear = check_clearance(log, robot, T0, box.size, arm=ArmCollision(robot, box.size))
    res.update(log=log, T_end=T_end, err_pos=ep, err_rot=er, profile=prof, clearance=clear,
               wall_s=time.perf_counter() - t0)
    res["checks"].update({
        "all segments completed": log.ok,
        "QP always feasible": all(s["cause"] != "infeasible QP" for s in log.segments),
        "joint speeds within limits": prof["vel_ok"],
        "object at the goal": ep < tol["position"] and er < math.radians(tol["rotation_deg"]),
        "arm back at ready": bool(np.abs(backend.theta() - kw["q_ready"]).max() < 1e-3),
        "palm clear of the box and the table": clear["box_ok"] and clear["table_ok"],
        "arm and hand clear of the box and the table": clear["arm_ok"],
    })
    return res


def passed(res):
    return all(res["checks"].values())


def report(res):
    mode = lambda f: "free rotation" if f else "rigid"
    print(f"\n=== {res['task']}: executed with {mode(res['free_rotation'])}, "
          f"planned with {mode(res['plan_free_rotation'])} ===")
    plan = res["plan"]
    if not plan.found:
        print(f"no feasible plan ({plan.stats.get('time_s', 0):.1f} s)")
        return
    print(f"plan: {' -> '.join(s.kind for s in plan.steps)}   cost {plan.cost:.3f}   "
          f"planning {plan.stats['time_s']:.1f} s   joint-limit margin {plan.stats['joint_margin']:.2f} rad")
    log = res["log"]
    for s in log.segments:
        print(f"  step {s['step']}  {s['name']:9s} {s['t_end'] - s['t0']:6.2f} s  {'ok' if s['ok'] else s['cause']}")
    a = log.arrays()
    print(f"duration {a['t'][-1]:.1f} s (computed in {res['wall_s']:.1f} s)   "
          f"object error {res['err_pos'] * 1e3:.2f} mm, {math.degrees(res['err_rot']):.3f} deg   "
          f"max joint speed {res['profile']['vel_ratio'].max():.2f} of the limit   "
          f"max |phi| {math.degrees(np.abs(a['phi']).max()):.0f} deg")
    for k, v in res["checks"].items():
        print(f"  [{'ok' if v else 'FAIL'}] {k}")


def save(res, cfg, out_dir=HERE / "runs"):
    """Save the log with what replay needs. Returns the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{res['task']}_{'free' if res['free_rotation'] else 'rigid'}.npz"
    res["log"].save(path, T0=res["T0"], Tg=res["Tg"], size=np.array(res["box"].size), base=np.array(cfg["robot"]["base"]))
    return path


def show(path, speed=1.0, **launch):
    """Replay a saved run in Swift: the box in blue, its goal pose as a green ghost."""
    from pandaSim.control.viewer import replay

    d = np.load(path, allow_pickle=True)
    size, base = d["size"].tolist(), se3.rp(np.eye(3), d["base"])
    objects = [{"type": "cuboid", "scale": size, "pose": d["T0"], "color": "blue"},
               {"type": "cuboid", "scale": size, "pose": d["Tg"], "color": [0.1, 0.8, 0.1, 0.35]}]
    return replay(path, base, objects, speed=speed, **launch)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--task", help="task name from the config, or 'all' (default: the config's `task`)")
    ap.add_argument("--free-rotation", dest="free", action="store_true", default=None,
                    help="pivot with free rotation about the finger axis (overrides the config)")
    ap.add_argument("--rigid", dest="free", action="store_false", help="pivot with a rigid grasp")
    ap.add_argument("--plan-free-rotation", dest="plan_free", action="store_true", default=None,
                    help="feasibility model of the planner (default: same as the execution)")
    ap.add_argument("--plan-rigid", dest="plan_free", action="store_false")
    ap.add_argument("--viewer", action="store_true", help="replay each run in Swift afterwards")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.free, args.plan_free)
    task = args.task or cfg["task"]
    names = list(cfg["tasks"]) if task == "all" else [task]
    robot = make_robot(cfg)
    planner = make_planner(cfg, robot)          # shared: rollouts of repeated steps are cached
    results = []
    for name in names:
        res = run_task(cfg, robot, name, planner)
        report(res)
        results.append(res)
        if res["log"] is not None:
            path = save(res, cfg)
            print(f"log: {path}")
            if args.viewer:
                env = show(path, args.speed)
                time.sleep(2.0)
                env.close()
    if len(results) > 1:
        print("\nsummary")
        for r in results:
            print(f"  {r['task']:16s} {'PASS' if passed(r) else 'FAIL' if r['plan'].found else 'no plan'}")
    return 0 if all(passed(r) for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
