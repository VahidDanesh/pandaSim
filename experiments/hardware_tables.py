"""Tables of Sec. 7 from the robot runs of the protocol (experiments/configs/protocol.yaml).

    python experiments/hardware_tables.py                 summary per entry and LaTeX rows -> results/hardware_tables.tex

A run folder experiments/runs/box/<experiment>_<task>_<grasp>_r<k> counts once it has compare.json (run_box.py
--execute). Its final error is the measured one (measured.json of calibrate_box.py --measure) when it exists, else the
forward kinematics at the release, marked with a dagger in the LaTeX rows: a tracking number of the model (FK, not
measured), blind to slip and compliance, not accuracy. A run succeeds if it finished
every segment and its error is within the tolerance of box_real.yaml (and the box is on the right face, if measured).
A stopped run is "infeasible" if the QP became infeasible, "stopped" for any other cause (force or torque threshold,
robot error, tracking, Ctrl-C).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "box"
OUT = HERE / "results" / "hardware_tables.tex"
SHORT = {"slide": "S", "pivot": "P", "gait": "G"}


def run_record(run_dir, tol, qlim):
    """Outcome of one robot run folder, or None if it was not run."""
    run_dir = Path(run_dir)
    if not (run_dir / "compare.json").exists():
        return None
    comp = json.loads((run_dir / "compare.json").read_text())
    plan = json.loads((run_dir / "plan.json").read_text())
    meas = json.loads((run_dir / "measured.json").read_text()) if (run_dir / "measured.json").exists() else None
    cause = (comp.get("abort") or {}).get("cause")
    rec = dict(name=run_dir.name, completed=bool(comp["ok"]), cause=cause, measured=meas is not None,
               steps=[s["kind"] for s in plan["steps"]], plan_time_s=(plan.get("stats") or {}).get("time_s"))
    if meas is not None:
        rec.update(error_mm=meas.get("error_mm"), error_deg=meas.get("error_deg"), face_ok=meas.get("face_ok", False))
    else:
        rec.update(error_mm=comp.get("object_error_mm"), error_deg=comp.get("object_error_deg"), face_ok=True)
    within = rec["error_mm"] is not None and rec["error_mm"] <= tol["position"] * 1e3 and rec["error_deg"] <= tol["rotation_deg"]
    rec["success"] = rec["completed"] and within and rec["face_ok"]
    rec["outcome"] = "success" if rec["success"] else ("infeasible" if cause == "infeasible QP" else
                                                       "stopped" if cause else "off target")
    d = np.load(run_dir / "real.npz", allow_pickle=True)
    names = [s["name"] for s in d["segments"]]
    piv = np.isin(d["segment"], [j for j, n in enumerate(names) if n == "pivot"])
    th = d["theta"][piv] if piv.any() else d["theta"]
    rec["margin_rad"] = float(np.minimum(th - qlim[0], qlim[1] - th).min()) if len(th) else None
    rec["phi_deg"] = [float(np.degrees(d["phi"].min())), float(np.degrees(d["phi"].max()))] if len(d["phi"]) else None
    rec["duration_s"] = float(d["t"][-1] - d["t"][0]) if len(d["t"]) else None
    return rec


def collect(protocol, runs=RUNS, tol=None, qlim=None):
    """{(experiment, task, grasp): [record, ...]} over the repetitions of the protocol that were run."""
    if tol is None:
        tol = yaml.safe_load((HERE / "configs" / "box_real.yaml").read_text())["tolerance"]
    if qlim is None:
        from pandaSim.geometry.utils import create_virtual_panda

        r = create_virtual_panda()
        qlim = (r.qlim[0, :7], r.qlim[1, :7])
    out = {}
    for exp, e in protocol["experiments"].items():
        for task in e["tasks"]:
            for grasp in e.get("grasps", ["free"]):
                recs = [run_record(Path(runs) / f"{exp}_{task}_{grasp}_r{k}", tol, qlim) for k in range(1, e["repetitions"] + 1)]
                out[(exp, task, grasp)] = [r for r in recs if r is not None]
    return out


def mean_sd(values, fmt="{:.1f}"):
    v = [x for x in values if x is not None]
    if not v:
        return "--"
    return (fmt + r" $\pm$ " + fmt).format(np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0)


def error_cell(recs):
    """'mm, deg' over the successful runs; a dagger if any of them was not measured."""
    ok = [r for r in recs if r["success"]]
    dagger = r"$^\dagger$" if any(not r["measured"] for r in ok) else ""
    return f"{mean_sd([r['error_mm'] for r in ok])}, {mean_sd([r['error_deg'] for r in ok], '{:.2f}')}{dagger}"


def rows(protocol, data):
    """LaTeX rows per table: dict table -> list of rows."""
    out = {"prim": [], "gait": [], "seq": [], "fr": []}
    for (exp, task, grasp), recs in data.items():
        n, s = len(recs), sum(r["success"] for r in recs)
        task_tex = task.replace("_", r"\_")
        if exp == "prim":
            out["prim"].append(f"{task_tex} & {s}/{n} & {error_cell(recs)} & {mean_sd([r['margin_rad'] for r in recs], '{:.2f}')} \\\\")
        elif exp == "gait":
            k = len(recs[0]["steps"]) if recs else "--"
            out["gait"].append(f"{task_tex} & {k} & {s}/{n} & {error_cell(recs)} & {mean_sd([r['duration_s'] for r in recs], '{:.0f}')} \\\\")
        elif exp == "seq":
            plan = "-".join(SHORT.get(k, k) for k in recs[0]["steps"]) if recs else "--"
            t = mean_sd([r["plan_time_s"] for r in recs])
            out["seq"].append(f"{task_tex} & {plan} & {t} & {s}/{n} & {error_cell(recs)} \\\\")
    for grasp in ("rigid", "free"):                     # tab:exp_freerotation: one row per grasp over all configurations
        recs = [r for (exp, _, g), rs in data.items() if exp == "fr" and g == grasp for r in rs]
        if recs:
            count = lambda o: sum(r["outcome"] == o for r in recs)
            label = "rigid" if grasp == "rigid" else "free rotation"
            out["fr"].append(f"{label} & {count('success')}/{len(recs)} & {count('infeasible')} & {count('stopped')} & "
                             f"{mean_sd([r['margin_rad'] for r in recs], '{:.2f}')} \\\\")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--protocol", default=str(HERE / "configs" / "protocol.yaml"))
    ap.add_argument("--runs", default=str(RUNS))
    args = ap.parse_args(argv)
    protocol = yaml.safe_load(Path(args.protocol).read_text())
    data = collect(protocol, args.runs)
    for (exp, task, grasp), recs in data.items():
        outcomes = ", ".join(f"{r['name'].rsplit('_', 1)[1]} {r['outcome']}" + (f" ({r['cause']})" if r["cause"] else "")
                             + ("" if r["measured"] or not r["completed"] else " [FK, not measured]") for r in recs)
        print(f"{exp}_{task}_{grasp:6s} {len(recs)}/{protocol['experiments'][exp]['repetitions']} runs   {outcomes or '-'}")
    tex = rows(protocol, data)
    text = "\n".join(f"% {k}\n" + "\n".join(v) for k, v in tex.items()) + "\n% $^\\dagger$ tracking (model): forward kinematics, not measured\n"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text)
    print("\n" + text + f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
