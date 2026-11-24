"""The hardware experiments of the paper (experiments/configs/protocol.yaml): check every run in simulation, print
the commands.

    python experiments/protocol.py                    plan and simulate every entry, print the verdicts and commands
    python experiments/protocol.py --only fr          one experiment
    python experiments/protocol.py --host 172.16.0.2  host in the printed commands

Each entry is planned and simulated exactly as run_box.py does it (pre-flight with the joint controller's lag). A rigid
entry whose simulation stops by an infeasible QP in a pivot is a predicted failure: its command carries --expect-stop
(the robot runs up to the stop and then holds the tilted box). The verdicts go to experiments/results/protocol.json.
No robot is used here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_box as rb  # noqa: E402
import run_plan as rp  # noqa: E402

PROTOCOL = HERE / "configs" / "protocol.yaml"
OUT = HERE / "results" / "protocol.json"


def entries(protocol, only=None):
    """(experiment, task, grasp, spec) of every entry, in the order of the file."""
    for exp, e in protocol["experiments"].items():
        if only and exp != only:
            continue
        for task, spec in e["tasks"].items():
            for grasp in e.get("grasps", ["free"]):
                yield exp, task, grasp, dict(spec, gait=bool(e.get("gait")), repetitions=e["repetitions"],
                                             speed=e.get("speed", protocol["speed"]))


def check_entry(protocol, spec, grasp, config=rb.REAL_CONFIG):
    """Plan and simulate one entry as run_box.py does: dict with the verdict and the run_box.py flags."""
    cfg = rp.load_config(config, grasp == "free", True)          # the plan of free rotation in both grasps
    cfg["execution"]["robot"] = {**(cfg["execution"].get("robot") or {}), "lead": bool(protocol["lead"])}
    res = rb.plan_and_simulate(cfg, rb.parse_pose(spec["start"]), rb.parse_pose(spec["goal"]), spec["speed"],
                               tracking=True, gait=spec["gait"])
    found = res["plan"].found
    stop = rb.predicted_stop(res["log"], res["plan"].steps, rb.pivot_method(cfg)) if found and grasp == "rigid" else None
    items = rb.preflight(res, cfg, expect_stop=stop is not None)
    flags = (["--gait"] if spec["gait"] else []) + (["--rigid"] if grasp == "rigid" else []) \
        + (["--expect-stop"] if stop else []) + (["--lead"] if protocol["lead"] else [])
    return dict(go=rb.go(items), steps=[s.kind for s in res["plan"].steps] if found else [],
                duration_s=float(res["log"].arrays()["t"][-1]) if found else None, stop=stop, flags=flags,
                failed=[c["name"] + (f" ({c['detail']})" if c["detail"] else "") for c in items if not c["ok"] and c["hard"]],
                warnings=[c["name"] + (f" ({c['detail']})" if c["detail"] else "") for c in items if not c["ok"] and not c["hard"]])


def command(protocol, spec, flags, name, host):
    return (f".venv/bin/python experiments/run_box.py --start {spec['start']} --goal {spec['goal']} "
            + " ".join(flags + [f"--level {protocol['level']}", f"--speed {spec['speed']}", f"--name {name}",
                                f"--host {host}", "--execute"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--protocol", default=str(PROTOCOL))
    ap.add_argument("--only", help="one experiment of the protocol")
    ap.add_argument("--host", default="172.16.0.2")
    args = ap.parse_args(argv)
    protocol = yaml.safe_load(Path(args.protocol).read_text())

    results, runs = [], {"free": [], "rigid": []}
    for exp, task, grasp, spec in entries(protocol, args.only):
        name = f"{exp}_{task}_{grasp}"
        r = check_entry(protocol, spec, grasp)
        verdict = "GO" if r["go"] else "NO-GO"
        stop = f", expected stop in step {r['stop']['step']} at {r['stop']['angle_deg']:.0f} of {r['stop']['of_deg']:.0f} deg" if r["stop"] else ""
        print(f"{name:22s} {verdict:5s} {' -> '.join(r['steps']) or 'no plan':40s} "
              f"{(r['duration_s'] or 0):5.0f} s{stop}", flush=True)
        for f in r["failed"]:
            print(f"    FAIL {f}")
        for w in r["warnings"]:
            print(f"    warn {w}")
        results.append(dict(name=name, experiment=exp, task=task, grasp=grasp, **spec, **r))
        if r["go"]:
            runs[grasp] += [(command(protocol, spec, r["flags"], f"{name}_r{k}", args.host),
                             f".venv/bin/python experiments/calibrate_box.py {args.host} --measure experiments/runs/box/{name}_r{k}")
                            for k in range(1, spec["repetitions"] + 1)]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1, default=str))
    for grasp, title in (("free", "fingertips with the bearing"), ("rigid", "original fingers (swap them first)")):
        if runs[grasp]:
            print(f"\n# --- {title}: {len(runs[grasp])} runs; after each, measure the final pose ---")
            for run, measure in runs[grasp]:
                print(run)
                print(measure)
    print(f"\n{sum(r['go'] for r in results)} of {len(results)} entries GO; verdicts in {OUT}")
    return 0 if all(r["go"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
