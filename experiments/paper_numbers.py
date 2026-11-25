"""Every number of Sec. 7 of the manuscript, taken from the code, the configs and experiments/results.

    python experiments/paper_numbers.py                 # print the numbers and check them in main.tex
    python experiments/paper_numbers.py --tex PATH      # another copy of main.tex
    python experiments/paper_numbers.py --no-check      # only print

Parameters come from the code that runs the robot (ContactQP defaults with the overrides of configs/box_real.yaml,
planner constants, the rest of the config),
so the table of the paper cannot drift from the implementation. Simulation results come from
experiments/results/ablation_{pivots,tasks}.json (rates with 95 % Wilson intervals). The check looks for each
parameter in main.tex in the form the paper writes it and lists what is missing or different.
Exit status 1 if a parameter in main.tex does not match.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TEX = HERE.parents[1] / "Manuscript" / "Manipulation_with_the_help_of_environmental_contact" / "main.tex"
CONFIG = HERE / "configs" / "box_real.yaml"


def fmt(x):
    return f"{x:.4f}".rstrip("0").rstrip(".")


def wilson(k, n, z=1.96):
    if n == 0:
        return math.nan, math.nan
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def parameters(config=CONFIG):
    """(symbol, description, value, unit, the strings that main.tex must contain)."""
    from pandaSim.control.contact_qp import ContactQP
    from pandaSim.control.primitive_executor import check_clearance
    from pandaSim.planning import box_primitives, clearance, collision, sequencer

    qp = {k: v.default for k, v in inspect.signature(ContactQP).parameters.items() if v.default is not inspect._empty}
    cfg = yaml.safe_load(Path(config).read_text())
    prot = yaml.safe_load((Path(config).parent / "protocol.yaml").read_text())
    e, p = cfg["execution"], cfg["planner"]
    qp.update(e.get("qp") or {})                       # the values the robot runs with: config overrides the defaults
    d_col = inspect.signature(collision.ArmCollision).parameters["margin"].default
    d_palm = inspect.signature(check_clearance).parameters["clearance"].default
    # each check: a template of main.tex with {} for the numbers, and the values the numbers must have
    rows = [
        ("K", "gain toward the joint centers", qp["K"], "", [("& ${}\\,\\boldsymbol{I}_7$", [qp["K"]])]),
        ("lambda", "manipulability weight", qp["lam"], "", [("manipulability weight & ${}$", [qp["lam"]])]),
        ("rho_e, rho_s", "damper distance and clearance", (qp["rho_e"], qp["rho_s"]), "rad",
         [("damper distance and clearance & ${}$, ${}$~rad", [qp["rho_e"], qp["rho_s"]])]),
        ("eta", "damper gain", qp["eta"], "", [("damper gain & ${}$", [qp["eta"]])]),
        ("qd_max", "joint velocity bound", qp["qd_scale"], "x limit", [("${}\\times$ limit", [qp["qd_scale"]])]),
        ("d_phi", "end-effector clearance (phi limits)", clearance.CLEARANCE, "m",
         [("end-effector clearance & ${}$~mm", [clearance.CLEARANCE * 1e3])]),
        ("omega_max", "peak pivoting velocity", e["pivot"]["w_max"], "rad/s", [("peak pivoting velocity & ${}$~rad/s", [e["pivot"]["w_max"]])]),
        ("m", "box mass (box_real.yaml; placeholder until weighed)", cfg["box"]["mass"], "kg",
         [("The object is a box of ${}\\times{}\\times{}$~m and ${}$~kg", [*cfg["box"]["size"], cfg["box"]["mass"]])]),
        ("f_g", "grasp force", e["grasp"]["force"], "N", [("grasp force & ${}$~N", [e["grasp"]["force"]])]),
        ("r_b", "base keep-out radius", box_primitives.BASE_KEEPOUT, "m", [("base keep-out radius & ${}$~m", [box_primitives.BASE_KEEPOUT])]),
        ("[r_min, r_max]", "reach of the grasp points", sequencer.REACH, "m",
         [("reach of the grasp points & $[{}, {}]$~m", list(sequencer.REACH))]),
        ("d_min", "minimum tipping margin", box_primitives.SLIDE_MARGIN_MIN, "m",
         [("minimum tipping margin & ${}$~m", [box_primitives.SLIDE_MARGIN_MIN])]),
        ("kappa", "regrasp cost", sequencer.REGRASP_COST, "J", [("regrasp cost & ${}$~J", [sequencer.REGRASP_COST])]),
        ("k_max", "maximum pivoting motions", p["max_pivots"], "", [("maximum pivoting motions & ${}$", [p["max_pivots"]])]),
        ("N_max", "simulated candidates", p["max_kin_checks"], "", [("simulated candidates & ${}$", [p["max_kin_checks"]])]),
        ("d_col", "collision clearance (arm, hand, palm)", d_col, "m", [("collision clearance & ${}$~mm", [d_col * 1e3])]),
        ("f_max", "force stop", e["safety"]["f_max"], "N", [("external force exceeds ${}$~N", [e["safety"]["f_max"]])]),
        ("tau_max", "joint torque stop", e["safety"]["tau_max"], "Nm", [("external joint torque exceeds ${}$~Nm", [e["safety"]["tau_max"]])]),
        ("lift", "approach and retreat height", e["lift"], "m", [("approach from ${}$~m above the grasp", [e["lift"]])]),
        ("rate", "control rate", 1.0 / e["dt"], "Hz", [("quadprog at ${}$~kHz", [1e-3 / e["dt"]])]),
        ("gamma, beta", "tilting angles in object gaiting", (p["gait"]["gamma_deg"], p["gait"]["beta_deg"]), "deg",
         [("tilting angles in object gaiting & ${}^\\circ$, ${}^\\circ$", [p["gait"]["gamma_deg"], p["gait"]["beta_deg"]])]),
        ("alpha_max", "maximum rotation in object gaiting", p["gait"]["alpha_max_deg"], "deg",
         [("maximum rotation in object gaiting & ${}^\\circ$", [p["gait"]["alpha_max_deg"]])]),
        ("speed", "speed factors of the robot runs (protocol.yaml)", (prot["speed"], prot["experiments"]["fr"]["speed"]), "",
         [("bound $\\dot{\\boldsymbol{\\theta}}_{\\max}$ are scaled by ${}$, and by ${}$ in", [prot["speed"], prot["experiments"]["fr"]["speed"]])]),
        ("tolerance", "final pose tolerance", (cfg["tolerance"]["position"], cfg["tolerance"]["rotation_deg"]), "m, deg",
         [("is within ${}$~mm and ${}^\\circ$ of", [cfg["tolerance"]["position"] * 1e3, cfg["tolerance"]["rotation_deg"]])]),
    ]
    if d_palm != d_col:
        rows.append(("d_palm", "palm clearance in the check after the run", d_palm, "m", []))
    return rows


def simulation(results=RESULTS):
    """Lines with the simulation results, ready for the text or a table."""
    out = []
    piv = json.loads((results / "ablation_pivots.json").read_text())
    s, n = piv["summary"]["all"], piv["n"]
    names = dict(rigid="rigid grasp", rigid_tilt="rigid, best of 5 tilts", free="free rotation", free_nolimit="free, no limits on phi")
    out.append(f"Random pivots: n = {n} (drawn {piv['drawn']}, seed {piv['seed']})")
    for m, label in names.items():
        k = s[m]["success"]
        lo, hi = wilson(k, n)
        out.append(f"  {label:24s} {k:3d}/{n} = {100 * k / n:.1f} % [{100 * lo:.1f}, {100 * hi:.1f}]  infeasible QP {s[m]['qp_infeasible']}, "
                   f"palm in object {s[m]['palm_contact']}, other link {s[m].get('arm_contact', 0)}, "
                   f"median joint margin {s[m]['joint_margin_median']:.2f} rad")
    for kind in ("tip up", "tip down"):
        r = piv["summary"][kind]
        out.append(f"  {kind:9s} n = {r['n']:3d}: " + ", ".join(f"{m} {r[m]['success']}" for m in names))
    arm = [t["free"]["arm_min"] for t in piv["trials"] if t["free"]["success"] and "arm_min" in t["free"]]
    if arm:
        out.append(f"  smallest arm/hand distance in successful free-rotation pivots: {min(arm) * 1e3:.1f} mm")
    tasks = json.loads((results / "ablation_tasks.json").read_text())
    t, n = tasks["summary"], tasks["n"]
    out.append(f"Random tasks: n = {n} (seed {tasks['seed']})")
    for m in ("rigid", "free"):
        r = t[m]
        out.append(f"  {m:6s} plan found {r['planned']:3d}, executed {r['success']:3d}, median planning time {r['planning_s_median']:.1f} s "
                   "(wall clock, machine dependent)")
    b = t["both"]
    out.append(f"  solved in both modes: {b['n']}; mean cost rigid {b['cost_rigid']:.1f} vs free {b['cost_free']:.1f}; "
               f"mean duration rigid {b['duration_rigid']:.0f} s vs free {b['duration_free']:.0f} s")
    return out


NUMBER = r"(\d+(?:\.\d+)?)"


def found(template, values, tex):
    """True if main.tex contains the template with numbers equal to values (0.30 and 0.3 are the same)."""
    pattern = NUMBER.join(re.escape(part) for part in template.split("{}"))
    return any(all(math.isclose(float(g), v, rel_tol=1e-9, abs_tol=1e-12) for g, v in zip(m.groups(), values))
               for m in re.finditer(pattern, tex))


def check(rows, tex):
    """Checks that fail: [(symbol, expected text)]."""
    return [(sym, template.replace("{}", "%s") % tuple(fmt(v) for v in values))
            for sym, _, _, _, checks in rows for template, values in checks if not found(template, values, tex)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tex", default=str(TEX))
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args(argv)
    rows = parameters(a.config)
    print("Parameters (Table tab:parameters and the text of Sec. 7):")
    for sym, desc, val, unit, _ in rows:
        print(f"  {sym:15s} {desc:42s} {val} {unit}")
    print()
    print("\n".join(simulation(Path(a.results))))
    if a.no_check:
        return 0
    tex_path = Path(a.tex)
    if not tex_path.exists():
        print(f"\nmain.tex not found at {tex_path}: no check")
        return 0
    missing = check(rows, tex_path.read_text())
    print(f"\nCheck of {tex_path.name}: " + ("all parameters match" if not missing else f"{len(missing)} not found as expected"))
    for sym, snip in missing:
        print(f"  {sym:15s} expected  {snip}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
