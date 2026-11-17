"""How exactly must the box be placed? Sensitivity of a planned run to an error of the start pose (simulation).

    python experiments/placement_tolerance.py                       # two reference tasks, table and JSON
    python experiments/placement_tolerance.py --start 0.5,0,0,lying --goal 0.5,0,0,standing

The executor has no perception: it assumes the box is exactly at the planned pose. Here the real box is displaced
from that pose (along the finger axis n, across it in the table plane, and in yaw) and the simulated hand motion is
kept. Kinematics only (no contact forces), with this model of what happens:
  approach  the open fingers (0.08 m) and the hand come down on the displaced box: smallest distance to it over
            `pre` and `approach` (ArmCollision with the fingers open). Below 0: a finger lands on the box.
  grasp     the closing fingers centre the box between them and turn its faces flat onto the pads (the box
            slides on the table): the error along n and the yaw error are removed, the error across n stays.
  pivot     the hand turns about the planned edge, the box about its real one, displaced by the error e across n.
            The two disagree by 2 sin(theta / 2) |e| at the end: the edge must slip on the table by that much
            (or the arm gives way). The clearance between the hand and the box over the phi range changes with e.
  release   the box ends displaced by e from the goal (pivot and slide alike).
Result: experiments/results/hardware-safety/placement_tolerance.json and a tolerance for the jig.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_box as rb  # noqa: E402
import run_plan as rp  # noqa: E402

from pandaSim.control.backends import OPEN_WIDTH  # noqa: E402
from pandaSim.planning import clearance as cl  # noqa: E402
from pandaSim.planning import se3  # noqa: E402
from pandaSim.planning.collision import ArmCollision  # noqa: E402

TASKS = [("0.50,0.00,0,lying", "0.50,0.00,0,standing"), ("0.55,0.20,90,standing", "0.55,0.20,90,lying")]
APPROACH_MIN = 0.002            # m: required distance of fingers and hand to the box while they come down
RHO_S = 0.05                    # rad: the QP stops phi this far inside its limits


def displaced(T_obj, T_grasp, e_n=0.0, e_x=0.0, yaw=0.0):
    """Box pose displaced by e_n along the finger axis, e_x across it (both horizontal) and yaw about the
    vertical through the box centre. Directions are those of the planned grasp frame (y = n, x across)."""
    T = np.array(T_obj, float)
    T[:3, :3] = se3.exp3(np.array([0.0, 0.0, yaw])) @ T[:3, :3]
    T[:3, 3] += e_n * T_grasp[:3, 1] + e_x * T_grasp[:3, 0]
    return T


def step_samples(res, i):
    """(theta, phi) samples of the approach of step i (pre + approach) and of its motion (pivot or slide)."""
    log, a = res["log"], res["log"].arrays()
    down, held = [], []
    for j, sg in enumerate(log.segments):
        if sg.get("step") != i:
            continue
        idx = np.flatnonzero(a["segment"] == j)[::25]
        if sg["name"] in ("pre", "approach"):
            down += list(idx)
        elif sg["name"] in ("pivot", "slide"):
            held += list(idx)
    return a["theta"][down], a["phi"][down], a["theta"][held], a["phi"][held]


def analyse_step(res, i, arm, e_n, e_x, yaw):
    st, size = res["plan"].steps[i], res["box"].size
    pl = st.plan
    th_d, ph_d, th_h, ph_h = step_samples(res, i)
    T_real = displaced(st.T_before, pl.T_grasp, e_n, e_x, yaw)
    d_app, who = min((arm.clearance(th, T_real, ph, width=OPEN_WIDTH, fingers_vs_box=True) for th, ph in zip(th_d, ph_d)),
                     key=lambda r: r[0])
    out = dict(step=i, kind=st.kind, e_n=e_n, e_x=e_x, yaw=yaw, approach_mm=min(d_app, 0.05) * 1e3, approach_link=who,
               final_error_mm=abs(e_x) * 1e3)
    if st.kind == "pivot":
        T_c_obj = se3.inv(pl.T_grasp) @ displaced(st.T_before, pl.T_grasp, 0.0, e_x, 0.0)      # after the grasp: e_x stays
        lo, hi = pl.phi_lim
        worst = [lo + RHO_S, hi - RHO_S] + list(np.linspace(ph_h.min(), ph_h.max(), 15))
        out.update(edge_slip_mm=2 * math.sin(abs(pl.theta) / 2) * abs(e_x) * 1e3,
                   hand_mm=min(cl.hand_box_distance(T_c_obj, size, p) for p in worst) * 1e3)
    return out


def tolerance(rows, e_grid, yaw_grid):
    """Largest |e_n| and |yaw| (together, on the grid) for which every row has the approach clearance."""
    ok = lambda e, y: all(r["approach_mm"] >= APPROACH_MIN * 1e3 for r in rows
                          if abs(r["e_n"]) <= e + 1e-12 and abs(r["yaw"]) <= y + 1e-12 and abs(r["e_x"]) <= e + 1e-12)
    return [(e, y) for e in e_grid for y in yaw_grid if ok(e, y)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start")
    ap.add_argument("--goal")
    ap.add_argument("--config", default=rb.REAL_CONFIG)
    ap.add_argument("--out", default=str(HERE / "results" / "hardware-safety" / "placement_tolerance.json"))
    args = ap.parse_args(argv)
    tasks = [(args.start, args.goal)] if args.start else TASKS
    e_grid = [0.0, 0.002, 0.004, 0.006, 0.008]
    yaw_grid = [0.0, math.radians(2), math.radians(4), math.radians(6), math.radians(10)]
    rows = []
    for start, goal in tasks:
        cfg = rp.load_config(args.config)
        res = rb.plan_and_simulate(cfg, rb.parse_pose(start), rb.parse_pose(goal), 1.0)
        if not res["plan"].found:
            print(f"no plan for {start} -> {goal}")
            continue
        arm = ArmCollision(res["robot"], res["box"].size)
        print(f"\n{start} -> {goal}: {' -> '.join(s.kind for s in res['plan'].steps)}")
        for i, st in enumerate(res["plan"].steps):
            for e_n, e_x, yaw in itertools.product(sorted({-e for e in e_grid} | set(e_grid)), sorted({-e for e in e_grid} | set(e_grid)),
                                                   sorted({-y for y in yaw_grid} | set(yaw_grid))):
                if sum(v != 0 for v in (e_n, e_x, yaw)) > 1 and not (abs(e_n) == abs(e_x) or e_x == 0):
                    continue                                   # axes alone, e_n with yaw, and the corners
                rows.append(dict(task=f"{start} -> {goal}", **analyse_step(res, i, arm, e_n, e_x, yaw)))
            mine = [r for r in rows if r["task"] == f"{start} -> {goal}" and r["step"] == i]
            one = lambda **k: [r for r in mine if all(abs(r[a] - v) < 1e-12 for a, v in k.items())]
            print(f"  step {i} ({st.kind}): approach clearance [mm] to the box (closest of fingers, hand, arm)")
            print("     e_n [mm]  " + "  ".join(f"{e * 1e3:+5.0f}" for e in sorted({r['e_n'] for r in mine})))
            print("               " + "  ".join(f"{one(e_n=e, e_x=0.0, yaw=0.0)[0]['approach_mm']:5.1f}" for e in sorted({r['e_n'] for r in mine})))
            print("     yaw [deg] " + "  ".join(f"{math.degrees(y):+5.0f}" for y in sorted({r['yaw'] for r in mine})))
            print("               " + "  ".join(f"{one(e_n=0.0, e_x=0.0, yaw=y)[0]['approach_mm']:5.1f}" for y in sorted({r['yaw'] for r in mine})))
            print("     e_x [mm]  " + "  ".join(f"{e * 1e3:+5.0f}" for e in sorted({r['e_x'] for r in mine})))
            print("               " + "  ".join(f"{one(e_n=0.0, e_x=e, yaw=0.0)[0]['approach_mm']:5.1f}" for e in sorted({r['e_x'] for r in mine})))
            if st.kind == "pivot":
                print("     hand-box clearance over the phi range [mm], by e_x: "
                      + "  ".join(f"{one(e_n=0.0, e_x=e, yaw=0.0)[0]['hand_mm']:5.1f}" for e in sorted({r['e_x'] for r in mine})))
                print("     edge slip at the end of the pivot [mm], by e_x:     "
                      + "  ".join(f"{one(e_n=0.0, e_x=e, yaw=0.0)[0]['edge_slip_mm']:5.1f}" for e in sorted({r['e_x'] for r in mine})))
    feasible = tolerance(rows, e_grid, yaw_grid)
    best = max(feasible, key=lambda p: (min(p[0] / 0.008, p[1] / math.radians(10)), p[0])) if feasible else None
    summary = dict(approach_min_mm=APPROACH_MIN * 1e3,
                   feasible=[(e * 1e3, round(math.degrees(y), 1)) for e, y in feasible],
                   tolerance_mm_deg=None if best is None else (best[0] * 1e3, round(math.degrees(best[1]), 1)),
                   hand_mm_min=min((r["hand_mm"] for r in rows if "hand_mm" in r and abs(r["e_x"]) <= (best[0] if best else 0) + 1e-12), default=None))
    print(f"\nposition error and yaw error that keep {APPROACH_MIN * 1e3:.0f} mm between the open fingers and the box in every case "
          f"(position along and across the fingers, together with yaw): {summary['feasible']}")
    print(f"=> balanced tolerance: {summary['tolerance_mm_deg']} (mm, deg); hand-box clearance in the pivots then >= {summary['hand_mm_min']:.1f} mm")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
