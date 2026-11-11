"""Figure of the ablation: python experiments/ablation_figures.py [--out DIR] [--width INCHES]

Reads experiments/results/ablation_pivots.json (written by ablation.py pivots) and reruns one pivot to
show the bearing angle phi with and without its limits.
  A  success rate of a pivot in each mode (rigid; rigid with the best of 5 prescribed hand tilts; free rotation; free without limits), with the 95 % Wilson interval (n = number of pivots)
  B  box centre before the pivot, by outcome (rigid and free rotation)
  C  bearing angle phi(t) of one pivot with the limits and without; shaded: the palm is inside the box
Okabe-Ito colours, redundant with marker shape and hatching; sans-serif text of at least 6 pt at print size;
vector PDF and a 600 dpi PNG. Print width: --width (default 6.5 in; a column of 3.4 in needs a redesign).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
INK = "#1a1a1a"
BLUE, VERM, GREY, ORANGE = "#0072B2", "#D55E00", "#8c8c8c", "#E69F00"
MODES = [("rigid", "Rigid"), ("rigid_tilt", "Rigid,\n5 tilts"), ("free", "Free,\nlimits"), ("free_nolimit", "Free,\nno limits")]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def phi_history(trial, config):
    """phi(t) of the pivot of `trial` with and without the limits (rerun of the simulation)."""
    import ablation as ab
    import run_plan as rp
    from pandaSim.control.backends import SimBackend
    from pandaSim.control.primitive_executor import run_plan

    c = ab.ctx(config)
    box, seq = c["box"], c["planners"]["free"]
    T = rp.box_pose(box, trial["start"])
    # the trial records the grasp point, not the index of the pivot direction: pick the matching one
    d_o, pl = min(seq._box_pivots(box, T), key=lambda dp: np.linalg.norm(dp[1].T_grasp[:2, 3] - trial["grasp_xy"]))
    step, _ = seq._pivot_step(box, T, d_o, pl)
    out = {}
    for mode in ("free", "free_nolimit"):
        kw = rp.exec_kwargs(c["cfg"])
        if mode == "free_nolimit":
            kw["qp"] = {**(kw["qp"] or {}), "phi_lim": None}
        log, _ = run_plan(SimBackend(kw["q_ready"], dt=c["cfg"]["execution"]["dt"]), c["robot"], [step], T, free_rotation=True, **kw)
        a = log.arrays()
        names = [s["name"] for s in log.segments]
        first = names.index("approach")
        m = a["segment"] >= first
        m &= np.isin(a["segment"], [j for j, n in enumerate(names) if n in ("approach", "pivot")])
        t0 = a["t"][m][0]
        piv = a["segment"][m] == names.index("pivot")
        out[mode] = (a["t"][m] - t0, np.degrees(a["phi"][m]), a["t"][m][piv][0] - t0)
    return out, np.degrees(pl.phi_lim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "results" / "ablation_pivots.json"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--width", type=float, default=6.5)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    d = json.load(open(a.data))
    T, n = d["trials"], len(d["trials"])

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
                         "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular"})
    fig = plt.figure(figsize=(a.width, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.55, left=0.075, right=0.99, top=0.9, bottom=0.36)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    for ax in (axA, axB, axC):
        ax.spines[["top", "right"]].set_visible(False)

    # A: success rate with Wilson 95 % intervals
    k = [sum(t[m]["success"] for t in T) for m, _ in MODES]
    rate = [100 * x / n for x in k]
    ci = np.array([wilson(x, n) for x in k]) * 100
    cols, hatch = [GREY, GREY, BLUE, ORANGE], ["", "//", "", "xx"]
    for i in range(4):
        axA.bar(i, rate[i], color=cols[i], width=0.62, hatch=hatch[i], edgecolor="white", linewidth=0)
    axA.errorbar(range(4), rate, yerr=[np.array(rate) - ci[:, 0], ci[:, 1] - np.array(rate)], fmt="none", ecolor=INK,
                 elinewidth=0.7, capsize=2.5, capthick=0.7)
    for i in range(4):
        axA.text(i, 106, f"{rate[i]:.0f}", ha="center", va="bottom", fontsize=7, color=INK)
    axA.set_xticks(range(4), [l for _, l in MODES])
    axA.tick_params(axis="x", labelsize=6.5)
    axA.set_ylim(0, 118)
    axA.set_yticks(range(0, 101, 20))
    axA.set_ylabel("Successful pivots (%)", labelpad=2)

    # B: where the pivots fail
    cat = lambda t: "both" if t["rigid"]["success"] and t["free"]["success"] else ("free" if t["free"]["success"] else "none")
    style = {"both": (GREY, "o", "Rigid and free succeed"), "free": (BLUE, "s", "Only free rotation succeeds"),
             "none": (VERM, "X", "Neither succeeds")}
    for key, (c, mk, lab) in style.items():
        pts = np.array([t["start"]["xy"] for t in T if cat(t) == key]).reshape(-1, 2)
        axB.scatter(pts[:, 1], pts[:, 0], s=9 if key != "none" else 22, c=c, marker=mk, edgecolors="white", linewidths=0.3,
                    label=f"{lab} ({len(pts)})", zorder=3)
    axB.plot(0, 0, marker="s", color=INK, ms=4)
    axB.annotate("Base", (0, 0), (-0.09, 0.0), fontsize=6.5, color=INK, va="center")
    axB.set_xlim(0.55, -0.55)
    axB.set_ylim(-0.06, 0.9)
    axB.set_aspect("equal")
    axB.set_xlabel("y (m)", labelpad=1)
    axB.set_ylabel("x (m)", labelpad=1)
    axB.set_xticks([0.4, 0, -0.4])
    axB.set_yticks([0, 0.4, 0.8])
    axB.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), handletextpad=0.2, borderaxespad=0.0, fontsize=6.5)

    # C: phi(t) with and without the limits
    cfg = a.config or str(HERE / "configs" / "box_plan.yaml")
    trial = next(t for t in T if cat(t) == "free" and t["kind"] == "tip down")
    try:
        h, lim = phi_history(trial, cfg)
        hi = lim[1] if lim[1] > -lim[0] else lim[0]            # the wide side, in which the pivot turns the hand
        sg = 1.0 if hi > 0 else -1.0
        for mode, c, ls, lab in (("free_nolimit", ORANGE, "--", "No limits"), ("free", BLUE, "-", "With limits")):
            t, phi, _ = h[mode]
            axC.plot(t, sg * phi, color=c, ls=ls, lw=1.3, label=lab)
        axC.axhline(abs(hi), color=VERM, lw=0.8)
        axC.axhspan(abs(hi), abs(hi) + 60, color=VERM, alpha=0.18, lw=0)
        axC.text(0.03, 0.97, "Palm in the object", transform=axC.transAxes, fontsize=6.5, va="top", color=INK)
        axC.set_ylim(-10, abs(hi) + 55)
        axC.set_xlabel("Time (s)", labelpad=1)
        axC.set_ylabel("Hand angle $\\phi$ (deg)", labelpad=2)
        axC.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.7), handlelength=1.8, fontsize=6.5)
    except Exception as e:                                      # the figure must not silently lose a panel
        raise SystemExit(f"panel C failed: {e!r}")

    for ax, lab, x, y in ((axA, "A", -0.2, 1.0), (axB, "B", -0.3, 1.0), (axC, "C", -0.25, 1.0)):
        ax.text(x, y, lab, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "ablation_pivots.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "ablation_pivots.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    print("written to", out)


if __name__ == "__main__":
    main()
