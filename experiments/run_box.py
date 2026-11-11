"""Start pose -> goal pose of the box: plan, check, replay in Swift, run on the Panda.

    python experiments/run_box.py --start 0.50,0.00,0,lying --goal 0.45,-0.15,60,standing
        plans, simulates the plan with the executor (same code as on the robot), prints the pre-flight report
        and saves the run folder experiments/runs/box/<name>/ (plan.json, expected.npz, preflight.json). No robot.
    ... --replay                     show the simulated run in Swift (a normal browser, WebGL)
    ... --execute --host 172.16.0.2 --level air --speed 0.3
        runs it on the Panda after the pre-flight checks and a typed confirmation; saves real.npz and compare.json.

Pose: x,y,yaw_deg,face. x, y: box centre on the table (the robot base is at the origin of the table frame), yaw
about z, face: the face on the table: lying (0.06 x 0.25 face), lying_flip (the opposite one), standing
(0.15 x 0.06 face), standing_flip. The largest face cannot be grasped. Levels of the real ladder:
    air    no box, gripper open, the fingers close to the box width in the air (speed <= 0.5)
    low    box on the table, slow (speed <= 0.5)
    full   box on the table, speed up to 1.0
The robot only moves after you type the confirmation phrase; there is no flag that skips it.
"""
from __future__ import annotations

import argparse
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

from pandaSim.control.primitive_executor import check_profile  # noqa: E402
from pandaSim.planning.sequencer import pose_error  # noqa: E402

REAL_CONFIG = HERE / "configs" / "box_real.yaml"
FACES = {"lying": -90.0, "lying_flip": 90.0, "standing": 0.0, "standing_flip": 180.0}
LEVEL_SPEED = {"air": (0.3, 0.5), "low": (0.3, 0.5), "full": (1.0, 1.0)}      # default and maximum speed factor
G = 9.81
PAYLOAD = 3.0                       # kg, Franka Emika Panda (hand included)
REGION = dict(x=(0.30, 0.75), y=(-0.35, 0.35))
JOINT_MARGIN = (0.05, 0.15)         # rad: below the first the run is refused, below the second it is a warning


def parse_pose(text):
    """'x,y,yaw_deg,face' -> {xy, rpy} of the config."""
    try:
        x, y, yaw, face = text.split(",")
        return dict(xy=[float(x), float(y)], rpy=[0.0, FACES[face.strip()], float(yaw)])
    except (ValueError, KeyError):
        raise SystemExit(f"cannot read the pose '{text}': use x,y,yaw_deg,face with face in {sorted(FACES)}")


def check(name, ok, detail="", hard=True):
    return dict(name=name, ok=bool(ok), detail=detail, hard=hard)


def preflight(res, cfg):
    """Pre-flight list for a planned and simulated task (res of run_plan.run_task): dicts name, ok, detail, hard.
    A failed hard check blocks the execution on the robot; a failed soft check is a warning."""
    plan, box = res["plan"], res["box"]
    out = []
    for key in ("start", "goal"):
        spec = cfg["tasks"]["custom"][key]
        x, y = spec["xy"]
        out.append(check(f"{key} pose inside the table area", REGION["x"][0] <= x <= REGION["x"][1]
                         and REGION["y"][0] <= y <= REGION["y"][1], f"x {x:.2f}, y {y:.2f}; area {REGION}"))
    out.append(check("plan found", plan.found, "" if plan.found else "no feasible plan for this start and goal"))
    if not plan.found:
        return out
    out.append(check("start and goal differ", pose_error(res["T0"], res["Tg"])[0] > 1e-3 or pose_error(res["T0"], res["Tg"])[1] > 1e-2))
    for k, v in res["checks"].items():
        if k != "plan found":
            out.append(check(f"simulation: {k}", v))
    a = res["log"].arrays()
    lb, ub = res["robot_limits"]
    margin = float(np.minimum(a["theta"] - lb, ub - a["theta"]).min())
    out.append(check("joint-limit margin", margin >= JOINT_MARGIN[0], f"{margin:.2f} rad", hard=True))
    if margin < JOINT_MARGIN[1]:
        out.append(check("joint-limit margin comfortable", False, f"{margin:.2f} rad < {JOINT_MARGIN[1]} rad", hard=False))
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


def plan_json(res, cfg, speed):
    steps = [asdict(s) for s in res["plan"].steps] if res["plan"].found else []
    meta = dict(start=cfg["tasks"]["custom"]["start"], goal=cfg["tasks"]["custom"]["goal"], speed=speed,
                free_rotation=cfg["free_rotation"], box=cfg["box"], cost=res["plan"].cost if res["plan"].found else None,
                stats={k: v for k, v in res["plan"].stats.items() if isinstance(v, (int, float, str))}, steps=steps)
    return json.dumps(meta, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def plan_and_simulate(cfg, start, goal, speed=1.0):
    """Plan with the feasibility model of the config and simulate the plan at `speed`; returns res (+ robot limits)."""
    cfg["tasks"]["custom"] = dict(start=start, goal=goal)
    rp.scale_speed(cfg, speed)
    robot = rp.make_robot(cfg)
    cfg_plan = json.loads(json.dumps(cfg))
    rp.scale_speed(cfg_plan, 1.0)                     # the rollouts of the planner run at the nominal speed
    res = rp.run_task(cfg, robot, "custom", rp.make_planner(cfg_plan, robot))
    res["robot_limits"] = (robot.qlim[0, :7], robot.qlim[1, :7])
    res["robot"] = robot
    return res


def save_run(res, cfg, run_dir, speed):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(plan_json(res, cfg, speed))
    if res["log"] is not None:
        res["log"].save(run_dir / "expected.npz", T0=res["T0"], Tg=res["Tg"], size=np.array(res["box"].size),
                        base=np.array(cfg["robot"]["base"]))


# ---- comparison of a real run with the simulation ------------------------------------------------------
def compare(expected, real, T_end=None, T_goal=None):
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
    out = dict(segments=segs, tick_ms_p99=float(np.percentile(ra["tick_time"], 99) * 1e3),
               solve_ms_p99=float(np.percentile(ra["solve_time"], 99) * 1e3), n_segments=len(real.segments),
               expected_segments=len(expected.segments), ok=real.ok)
    if T_end is not None and T_goal is not None:
        ep, er = pose_error(T_end, T_goal)
        out.update(object_error_mm=float(ep * 1e3), object_error_deg=float(math.degrees(er)))
    return out


def print_compare(c):
    print(f"\nreal run: {'all segments finished' if c['ok'] else 'STOPPED'}  ({c['n_segments']} of {c['expected_segments']} segments), "
          f"tick p99 {c['tick_ms_p99']:.2f} ms, QP p99 {c['solve_ms_p99']:.2f} ms")
    for s in c["segments"]:
        print(f"  step {s['step']}  {s['name']:9s} {s['duration']:6.2f} s (sim {s['expected_duration']:6.2f})  "
              f"joint error rms {s['rms_joint_error'] * 1e3:6.1f} mrad, max {s['max_joint_error'] * 1e3:6.1f} mrad   "
              f"max force {s['max_force']:5.1f} N   {'ok' if s['ok'] else s['cause']}")
    if "object_error_mm" in c:
        print(f"  object pose by forward kinematics at the release: {c['object_error_mm']:.2f} mm, {c['object_error_deg']:.2f} deg "
              f"from the goal (not measured)")


# ---- the robot -------------------------------------------------------------------------------------------
def robot_checks(state, q_ready, lim, gripper_state=None):
    """Checks on the robot before the first motion (state: panda-py RobotState or a fake with the same fields)."""
    q = np.asarray(state.q, float)
    margin = float(np.minimum(lim[1] - q, q - lim[0]).min())
    dist = float(np.abs(q - q_ready).max())
    force = float(np.linalg.norm(np.asarray(state.O_F_ext_hat_K, float)[:3]))
    items = [
        check("joints inside the limits", margin > JOINT_MARGIN[0], f"margin {margin:.2f} rad"),
        check("no robot error", not str(state.current_errors).strip("[]{} "), str(state.current_errors)[:80]),
        check("robot is at rest and unloaded", force < 5.0, f"|F_ext| {force:.1f} N"),
        check("robot is within 1.0 rad of the ready configuration (the move there is not collision-checked)", dist < 1.0,
              f"largest joint distance {dist:.2f} rad"),
    ]
    if gripper_state is not None:
        items.append(check("gripper answers", gripper_state.max_width > 0.07, f"max width {gripper_state.max_width:.3f} m"))
    return items


def confirm(phrase, ask=input):
    """The user must type the phrase; anything else, or no terminal, refuses."""
    if ask is input and not sys.stdin.isatty():
        print("no terminal: the robot is not moved")
        return False
    return ask(f"\nType  {phrase}  to move the robot (anything else cancels): ").strip() == phrase


def execute(res, cfg, run_dir, host, level, speed, ask=input, panda=None, gripper=None):
    """Run the planned and simulated task on the Panda. panda/gripper: connected objects (tests pass fakes)."""
    from pandaSim.control.backends import PandaBackend
    from pandaSim.control.primitive_executor import run_plan

    robot, kw = res["robot"], rp.exec_kwargs(cfg)
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
        if creds is not None:                          # release the brakes (needed if the robot was locked)
            panda_py.Desk(host, *creds, platform="panda").unlock()
    items = robot_checks(panda.get_state(), kw["q_ready"], res["robot_limits"],
                         gripper.read_once() if gripper is not None else None)
    print("\nrobot:")
    print_preflight(items)
    if not go(items):
        return None
    print(f"\nrun {run_dir.name}: level {level}, speed {speed}, {len(res['plan'].steps)} steps "
          f"({' -> '.join(s.kind for s in res['plan'].steps)}), about {res['log'].arrays()['t'][-1]:.0f} s")
    if not confirm(f"RUN {level} {run_dir.name}", ask):
        print("cancelled; the robot was not moved")
        return None
    if np.abs(np.asarray(panda.get_state().q) - kw["q_ready"]).max() > 1e-3:
        print("moving to the ready configuration (panda-py joint motion, slow)")
        panda.move_to_joint_position(kw["q_ready"], speed_factor=0.1)
    backend = PandaBackend(panda, gripper, frequency=1.0 / cfg["execution"]["dt"], air=(level == "air"))
    t0 = time.perf_counter()
    log, T_end = run_plan(backend, robot, res["plan"].steps, res["T0"], free_rotation=cfg["free_rotation"], **kw)
    print(f"finished in {time.perf_counter() - t0:.1f} s")
    log.save(run_dir / "real.npz", T0=res["T0"], Tg=res["Tg"], size=np.array(res["box"].size), base=np.array(cfg["robot"]["base"]))
    c = compare(res["log"], log, T_end, res["Tg"])
    (run_dir / "compare.json").write_text(json.dumps(c, indent=1))
    print_compare(c)
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--start", required=True, help="x,y,yaw_deg,face")
    ap.add_argument("--goal", required=True, help="x,y,yaw_deg,face")
    ap.add_argument("--config", default=REAL_CONFIG)
    ap.add_argument("--rigid", action="store_true", help="rigid grasp instead of free rotation (ablation)")
    ap.add_argument("--mass", type=float, help="box mass in kg (overrides the config)")
    ap.add_argument("--name", help="run folder name (default: timestamp)")
    ap.add_argument("--speed", type=float, help="speed factor of the execution, 0 < s <= 1 (default by level)")
    ap.add_argument("--level", choices=list(LEVEL_SPEED), default="air")
    ap.add_argument("--replay", nargs="?", const="expected", choices=["expected", "real"], help="show a run in Swift")
    ap.add_argument("--replay-speed", type=float, default=1.0)
    ap.add_argument("--execute", action="store_true", help="run on the robot (after the checks and a typed confirmation)")
    ap.add_argument("--host", default="172.16.0.2")
    args = ap.parse_args(argv)

    cfg = rp.load_config(args.config, not args.rigid, None)
    if args.mass is not None:
        cfg["box"]["mass"] = args.mass
    default, cap = LEVEL_SPEED[args.level]
    speed = args.speed if args.speed is not None else default
    if not 0 < speed <= cap:
        raise SystemExit(f"speed {speed} is outside (0, {cap}] for level {args.level}")
    run_dir = HERE / "runs" / "box" / (args.name or time.strftime("%Y%m%d-%H%M%S"))

    res = plan_and_simulate(cfg, parse_pose(args.start), parse_pose(args.goal), speed)
    print(f"\nplan {args.start}  ->  {args.goal}   ({'rigid' if args.rigid else 'free rotation'}, speed {speed})")
    if res["plan"].found:
        rp.report(res)
    items = preflight(res, cfg)
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
        execute(res, cfg, run_dir, args.host, args.level, speed)
        if args.replay == "real" and (run_dir / "real.npz").exists():
            env = rp.show(run_dir / "real.npz", args.replay_speed)
            time.sleep(2.0)
            env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
