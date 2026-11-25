"""Plan a box task and execute the plan in simulation.

    python experiments/run_plan.py                                # the config's default task, headless
    python experiments/run_plan.py --task all                     # every task of the config
    python experiments/run_plan.py --task slide_pivot --viewer    # then replay it in Swift (browser tab)
    python experiments/run_plan.py --task all --rigid             # ablation: no free rotation

The log of each run is saved to experiments/runs/<task>_<free|rigid>.npz.
"""
import argparse
import inspect
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from pandaSim.control.backends import SimBackend
from pandaSim.control.contact_qp import ContactQP
from pandaSim.control.primitive_executor import check_clearance, check_profile, run_plan
from pandaSim.planning import clearance, se3
from pandaSim.planning.collision import ArmCollision
from pandaSim.planning.feasibility import Rollout
from pandaSim.planning.chain import ChainPlanner
from pandaSim.planning.gait import GaitPlanner
from pandaSim.planning.sequencer import Box, Sequencer, pose_error, snap_stable

HERE = Path(__file__).resolve().parent
QD_SCALE = inspect.signature(ContactQP).parameters["qd_scale"].default     # nominal joint velocity bound, x limit
QP_RATES = {k: inspect.signature(ContactQP).parameters[k].default for k in ("K", "lam", "eta")}    # rates of eq:qp
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
                grasp=e.get("grasp"), qp=e.get("qp") or None, approach=e.get("approach"), ready=e.get("ready"),
                safety=e.get("safety"))


def scale_speed(cfg, factor):
    """Slow the execution down by `factor` (0 < factor <= 1): approach, joint moves, pivots and the joint velocity
    bound of the QP (eq:qp_d), so that a run that is slowed down also cannot speed a joint up beyond the same fraction
    (near a singularity the QP drives the joints to its bound). The rates of the QP are scaled by the same factor:
    the centering gain K, the manipulability weight lam (eq:qp_a) and the damper rate eta (eq:qp_c). The QP is then
    homogeneous in the rates, so its null-space motion, and with it the joint path, is the same at every speed, only
    slower: the planner's rollouts (nominal speed) see the path that the robot runs (design review D4). The pose
    feedback gain K_e of the approach is not scaled: it acts only on tracking errors, not on the path (1e-5 rad), and
    a slower K_e lets the lag of the robot's controller outlast safety.settle after a slide (timeout).
    The nominal values are the config's (defaults: those of ContactQP); a config is scaled once
    (run_box.plan_and_simulate copies it before). The geometry of the plan does not change."""
    e = cfg["execution"]
    for key, nominal in (("pivot", dict(w_max=0.3)), ("approach", dict(v_max=0.1, w_max=0.3)),
                         ("ready", dict(qd_max=0.5)), ("qp", dict(qd_scale=QD_SCALE, **QP_RATES))):
        sub = dict(e.get(key) or {})
        sub.update({k: _times(sub.get(k, v), factor) for k, v in nominal.items()})
        e[key] = sub
    return cfg


def _times(v, factor):
    """v x factor for a number or a list (K may be given per joint), so the config stays plain YAML/JSON."""
    return [x * factor for x in v] if isinstance(v, (list, tuple)) else v * factor


def make_planner(cfg, robot, gait=False, chain=False):
    """Sequencer (gait=True: GaitPlanner, object gaiting; chain=True: ChainPlanner, a roll by pi in one grasp) whose
    feasibility check rolls each step out with the executor (model: plan_free_rotation)."""
    p = cfg["planner"]
    # model of the hand for the phi limits and the clearance test; always set, so that it does not carry over
    # from a planner made earlier in the same process with another config
    clearance.HAND_MODEL = p.get("hand_model") or clearance.DEFAULT_HAND_MODEL
    kin = Rollout(robot, free_rotation=cfg["plan_free_rotation"], dt=p["dt"], size=make_box(cfg).size, **exec_kwargs(cfg))
    if chain:
        return ChainPlanner(kin, base_xy=tuple(cfg["robot"]["base"][:2]), symmetric=p.get("symmetric", True))
    if gait:
        g = p["gait"]
        return GaitPlanner(kin, gamma=math.radians(g["gamma_deg"]), beta=math.radians(g["beta_deg"]),
                           alpha_max=math.radians(g["alpha_max_deg"]), k_max=g["k_max"],
                           max_kin_checks=g["max_kin_checks"], base_xy=tuple(cfg["robot"]["base"][:2]))
    return Sequencer(kin, max_pivots=p["max_pivots"], max_kin_checks=p["max_kin_checks"],
                     base_xy=tuple(cfg["robot"]["base"][:2]), symmetric=p.get("symmetric", True))


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
    ep, er = pose_error(T_end, Tg, cfg["planner"].get("symmetric", True))
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
          f"object tracking (model, FK) {res['err_pos'] * 1e3:.2f} mm, {math.degrees(res['err_rot']):.3f} deg   "
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


MARK = (1.0, 1.0, 1.0)      # marked corner of the real box (box frame, in half sizes); see run_box.py


def show(path, speed=1.0, **launch):
    """Replay a saved run in Swift: the box in blue, its goal pose as a green ghost, and the marked corner of the
    box as a small red sphere on both. launch: window=(t0, t1) of viewer.replay and the options of swift.Swift.launch
    (notebook=True: inside the cell of a Jupyter notebook)."""
    from pandaSim.control.viewer import replay

    d = np.load(path, allow_pickle=True)
    return replay(path, se3.rp(np.eye(3), d["base"]), scene_objects(d["T0"], d["Tg"], d["size"]), speed=speed, **launch)


def scene_objects(T0, Tg, size):
    """Objects of the replay (viewer.replay): the box at T0 (blue), its goal Tg (green ghost), the marked corner on both."""
    size = np.asarray(size, float).tolist()
    mark = lambda T: T @ se3.rp(np.eye(3), np.asarray(MARK) * np.asarray(size) / 2)
    return [{"type": "cuboid", "scale": size, "pose": T0, "color": "blue"},
            {"type": "cuboid", "scale": size, "pose": Tg, "color": [0.1, 0.8, 0.1, 0.35]},
            {"type": "sphere", "radius": 0.008, "pose": mark(T0), "color": "red", "follow": True},
            {"type": "sphere", "radius": 0.008, "pose": mark(Tg), "color": [0.9, 0.1, 0.1, 0.5]}]


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
