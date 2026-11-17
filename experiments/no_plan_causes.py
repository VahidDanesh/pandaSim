"""Why do some random tasks have no plan? Classification of the unsolved tasks of an ablation result (analysis).

    python experiments/no_plan_causes.py [--data experiments/results/ablation_tasks.json] [--workers 8]

For every task without a plan (free rotation): the number of candidates that pass the planner's cheap tests
(base keep-out, reach, tipping margin), how many of the cheapest ones the simulation rejects and why, and whether
a wider search finds a plan (more candidates simulated; more pivots allowed). Result: a table and a JSON file.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_plan as rp  # noqa: E402

from pandaSim.control.backends import SimBackend  # noqa: E402
from pandaSim.control.primitive_executor import check_clearance, run_plan  # noqa: E402
from pandaSim.planning.collision import ArmCollision  # noqa: E402

WIDE = 60                        # candidates simulated in the wider search


def step_failure(cfg, robot, arm, step, size):
    """Why the rollout rejects a step: None, or (segment, cause)."""
    kw = rp.exec_kwargs(cfg)
    log, _ = run_plan(SimBackend(kw["q_ready"], dt=cfg["planner"]["dt"]), robot, [step], step.T_before,
                      free_rotation=cfg["plan_free_rotation"], **kw)
    bad = next((s for s in log.segments if not s["ok"]), None)
    if bad is not None:
        return bad["name"], bad["cause"]
    c = check_clearance(log, robot, step.T_before, size, period=0.02, arm=arm)
    if not c["box_ok"]:
        return c["hits"][0][1], "hand too close to the box"
    if not c["table_ok"]:
        return "?", "hand too close to the table"
    if not c["arm_ok"]:
        return c["arm_hits"][0][1], f"{c['arm_hits'][0][2]} too close to the {c['arm_hits'][0][3]}"
    return None


def analyse(args):
    config, trial = args
    cfg = rp.load_config(config)
    robot, box = rp.make_robot(cfg), rp.make_box(cfg)
    T0, Tg = rp.box_pose(box, trial["start"]), rp.box_pose(box, trial["goal"])
    out = dict(trial=trial["trial"], start=trial["start"], goal=trial["goal"])
    seq = rp.make_planner(cfg, robot)
    cands = seq.candidates(box, T0, Tg, None)                       # every candidate that passes the cheap tests
    out["candidates"] = len(cands)
    out["found_again"] = seq.plan(box, T0, Tg).found                # the planner as configured (12 simulated)
    if not cands:
        r0, rg = float(np.hypot(*trial["start"]["xy"])), float(np.hypot(*trial["goal"]["xy"]))
        out["cause"] = "no candidate passes the cheap tests (reach, base keep-out, tipping margin)"
        out["detail"] = f"start {r0:.2f} m and goal {rg:.2f} m from the base"
        return out
    arm = ArmCollision(robot, box.size)
    reasons, first_ok = collections.Counter(), None
    for rank, (cost, steps) in enumerate(cands[:WIDE]):
        fails = [(s.kind,) + f for s in steps if (f := step_failure(cfg, robot, arm, s, box.size)) is not None]
        if not fails:
            first_ok = rank
            break
        for kind, seg, cause in fails:
            reasons[f"{kind}/{seg}: {cause}"] += 1
    out.update(first_feasible_rank=first_ok, reasons=dict(reasons.most_common(6)))
    if first_ok is not None:
        out["cause"] = (f"search limit: candidate {first_ok + 1} by cost is feasible, the planner simulates "
                        f"{cfg['planner']['max_kin_checks']}")
    else:
        out["cause"] = f"none of the {min(len(cands), WIDE)} cheapest candidates can be executed"
        cfg4 = json.loads(json.dumps(cfg))
        cfg4["planner"].update(max_pivots=4, max_kin_checks=WIDE)
        out["found_with_4_pivots"] = rp.make_planner(cfg4, robot).plan(box, T0, Tg).found
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(HERE / "results" / "ablation_tasks.json"))
    ap.add_argument("--config", default=str(rp.DEFAULT_CONFIG))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "results" / "hardware-safety" / "no_plan_causes.json"))
    args = ap.parse_args(argv)
    trials = [t for t in json.load(open(args.data))["trials"] if not t["free"]["found"]]
    print(f"{len(trials)} tasks without a plan in {args.data}")
    with Pool(min(args.workers, max(len(trials), 1))) as pool:
        rows = pool.map(analyse, [(args.config, t) for t in trials], chunksize=1)
    for r in rows:
        s, g = r["start"], r["goal"]
        face = lambda p: "lying" if p["rpy"][1] else "standing"
        print(f"\ntrial {r['trial']}: {face(s)} ({s['xy'][0]:.2f}, {s['xy'][1]:+.2f}, {s['rpy'][2]:+.0f} deg) -> "
              f"{face(g)} ({g['xy'][0]:.2f}, {g['xy'][1]:+.2f}, {g['rpy'][2]:+.0f} deg)   candidates {r['candidates']}"
              f"   plan now: {r['found_again']}")
        print(f"   {r['cause']}" + (f"; {r['detail']}" if "detail" in r else ""))
        for k, v in (r.get("reasons") or {}).items():
            print(f"      {v:3d} x {k}")
        if "found_with_4_pivots" in r:
            print(f"   with 4 pivots and {WIDE} candidates simulated: {'plan found' if r['found_with_4_pivots'] else 'no plan'}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
