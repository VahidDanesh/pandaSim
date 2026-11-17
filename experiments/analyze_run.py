"""Plan against measurement for one run on the robot: figure and text summary.

    python experiments/analyze_run.py experiments/runs/box/<name>

Reads expected.npz (the simulated run of the plan), real.npz (the run on the robot) and, if present,
compare.json from the run folder; writes analysis.pdf, analysis.png and summary.txt there. A run that stopped early
is shown up to where it stopped, with the cause. Panels:
  A  joint angles normalised to their range, robot (solid) and simulation (dashed), per segment
  B  bearing angle phi-hat (integrated, not measured), robot and simulation, with the limits of phi of each pivot
  C  external force |F_ext| and the stop threshold
  D  loop timing: tick and QP solve time per control period
The simulation is aligned to the robot segment by segment (each segment starts at the same time in both), since
approach segments end on arrival and can differ in length.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]
HELD = ("pivot", "slide")


def load(path):
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["segments"] = list(out["segments"])
    out["events"] = list(out["events"])
    return out


def aligned_simulation(exp, real):
    """Simulated samples moved onto the time of the robot run: segment j of the simulation starts when segment j
    of the robot run starts. Returns dict t, theta, phi (only the segments the robot started)."""
    t, th, ph = [], [], []
    for j, sr in enumerate(real["segments"]):
        ie = np.flatnonzero(exp["segment"] == j)
        if j >= len(exp["segments"]) or not len(ie):
            continue
        t.append(exp["t"][ie] - exp["t"][ie][0] + sr["t0"])
        th.append(exp["theta"][ie])
        ph.append(exp["phi"][ie])
    if not t:
        return dict(t=np.zeros(0), theta=np.zeros((0, 7)), phi=np.zeros(0))
    return dict(t=np.concatenate(t), theta=np.concatenate(th), phi=np.concatenate(ph))


def summary(exp, real, comp=None, f_max=None):
    """Text summary of the run (list of lines)."""
    segs = real["segments"]
    stopped = next((s for s in segs if not s["ok"]), None)
    t0 = real["t"][0] if len(real["t"]) else 0.0
    lines = [f"segments: {len(segs)} of {len(exp['segments'])} planned"
             + (f"; STOPPED in '{stopped['name']}' (step {stopped.get('step')}) at {stopped['t_end'] - t0:.2f} s: {stopped['cause']}"
                if stopped else "; all finished"),
             f"duration: {real['t'][-1] - t0:.1f} s (simulation {exp['t'][-1] - exp['t'][0]:.1f} s)" if len(real["t"]) else "no samples"]
    if len(real["t"]):
        F = np.linalg.norm(real["F_ext"][:, :3], axis=1)
        lines.append(f"largest external force: {F.max():.1f} N" + (f" (stop at {f_max:.0f} N)" if f_max else ""))
        lines.append(f"largest external joint torque: {np.abs(real['tau_ext']).max():.1f} Nm")
        lines.append(f"tick time p50/p99/max: {np.percentile(real['tick_time'], 50) * 1e3:.2f} / "
                     f"{np.percentile(real['tick_time'], 99) * 1e3:.2f} / {real['tick_time'].max() * 1e3:.2f} ms; "
                     f"QP p99 {np.percentile(real['solve_time'], 99) * 1e3:.2f} ms")
        lines.append(f"phi-hat range: {math.degrees(real['phi'].min()):.0f} to {math.degrees(real['phi'].max()):.0f} deg")
    if comp:
        lines.append("per segment (joint error against the simulation over the common time):")
        for s in comp["segments"]:
            lines.append(f"  step {s['step']}  {s['name']:9s} {s['duration']:6.2f} s (sim {s['expected_duration']:6.2f})  "
                         f"rms {s['rms_joint_error'] * 1e3:6.1f} mrad  max {s['max_joint_error'] * 1e3:6.1f} mrad  "
                         f"|F| max {s['max_force']:5.1f} N  {'ok' if s['ok'] else s['cause']}")
        if "object_error_mm" in comp:
            lines.append(f"object pose from forward kinematics at the release: {comp['object_error_mm']:.2f} mm, "
                         f"{comp['object_error_deg']:.2f} deg from the goal (not measured)")
    return lines


def phi_limits(run_dir):
    """{step: (lo, hi)} rad, the limits of phi of each pivot or gait step, from plan.json ({} if it has none)."""
    path = Path(run_dir) / "plan.json"
    if not path.exists():
        return {}
    steps = json.loads(path.read_text()).get("steps", [])
    return {i: tuple(s["plan"]["phi_lim"]) for i, s in enumerate(steps) if (s.get("plan") or {}).get("phi_lim")}


def figure(exp, real, qlim, f_max=None, width=7.0, phi_lim=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42, "mathtext.default": "regular"})
    sim = aligned_simulation(exp, real)
    t0 = real["t"][0] if len(real["t"]) else 0.0
    t, ts = real["t"] - t0, sim["t"] - t0
    lb, ub = qlim
    fig, axs = plt.subplots(4, 1, figsize=(width, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1, 0.8, 0.8], hspace=0.12))
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        for s in real["segments"]:                      # grey band while the object is held
            if s["name"] in HELD:
                ax.axvspan(s["t0"] - t0, s["t_end"] - t0, color="#e6e6e6", lw=0, zorder=0)
        stop = next((s for s in real["segments"] if not s["ok"]), None)
        if stop is not None:
            ax.axvline(stop["t_end"] - t0, color="#D55E00", lw=0.9, ls="--", zorder=1)
    a = axs[0]
    for j in range(7):
        a.plot(t, (real["theta"][:, j] - lb[j]) / (ub[j] - lb[j]), color=OKABE_ITO[j], lw=1.0, label=f"joint {j + 1}")
        a.plot(ts, (sim["theta"][:, j] - lb[j]) / (ub[j] - lb[j]), color=OKABE_ITO[j], lw=0.8, ls=(0, (3, 2)), alpha=0.8)
    a.set_ylim(-0.02, 1.02)
    a.set_ylabel("Joint angle\n(fraction of range)")
    a.legend(frameon=False, ncol=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.0)
    axs[1].plot(t, np.degrees(real["phi"]), color="#0072B2", lw=1.1, label="robot")
    axs[1].plot(ts, np.degrees(sim["phi"]), color="#000000", lw=0.8, ls=(0, (3, 2)), label="simulation")
    for s in real["segments"]:                          # limits of phi while a pivot holds the object
        lim = (phi_lim or {}).get(s.get("step"))
        if s["name"] == "pivot" and lim:
            axs[1].hlines(np.degrees(lim), s["t0"] - t0, s["t_end"] - t0, color="#D55E00", lw=0.8, ls=(0, (1, 1)))
    axs[1].set_ylabel("$\\hat{\\phi}$ (deg)")
    from matplotlib.patches import Patch
    handles, _ = axs[1].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    axs[1].legend(handles + [Patch(color="#e6e6e6"), Line2D([], [], color="#D55E00", lw=0.8, ls=(0, (1, 1)))],
                  ["robot", "simulation", "object held", "limits of $\\phi$"], frameon=False, loc="upper left", ncol=4)
    F = np.linalg.norm(real["F_ext"][:, :3], axis=1) if len(t) else np.zeros(0)
    axs[2].plot(t, F, color="#009E73", lw=0.9)
    if f_max:
        axs[2].axhline(f_max, color="#D55E00", lw=0.8)
        axs[2].text(0.0, f_max, " stop", va="bottom", fontsize=6, color="#1a1a1a")
    axs[2].set_ylabel("$|F_{ext}|$ (N)")
    axs[3].plot(t, real["tick_time"] * 1e3, color="#56B4E9", lw=0.6, label="tick")
    axs[3].plot(t, real["solve_time"] * 1e3, color="#CC79A7", lw=0.6, label="QP")
    axs[3].axhline(1.0, color="#1a1a1a", lw=0.6, ls=":")
    axs[3].set_ylabel("Time per\nperiod (ms)")
    axs[3].set_xlabel("Time (s)")
    axs[3].legend(frameon=False, loc="upper right", ncol=2)
    for ax, lab in zip(axs, "ABCD"):
        ax.text(-0.13, 1.0, lab, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    fig.align_ylabels(axs)
    return fig


def analyze(run_dir, f_max=None, qlim=None):
    run_dir = Path(run_dir)
    exp, real = load(run_dir / "expected.npz"), load(run_dir / "real.npz")
    comp = json.loads((run_dir / "compare.json").read_text()) if (run_dir / "compare.json").exists() else None
    if qlim is None:
        from pandaSim.geometry.utils import create_virtual_panda
        r = create_virtual_panda()
        qlim = (r.qlim[0, :7], r.qlim[1, :7])
    lines = summary(exp, real, comp, f_max)
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    fig = figure(exp, real, qlim, f_max, phi_lim=phi_limits(run_dir))
    fig.savefig(run_dir / "analysis.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(run_dir / "analysis.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--config", default=str(HERE / "configs" / "box_real.yaml"), help="for the force threshold")
    a = ap.parse_args(argv)
    import yaml
    f_max = ((yaml.safe_load(Path(a.config).read_text()).get("execution") or {}).get("safety") or {}).get("f_max")
    print("\n".join(analyze(a.run_dir, f_max)))
    print(f"\nwritten: {Path(a.run_dir) / 'analysis.pdf'}, analysis.png, summary.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
