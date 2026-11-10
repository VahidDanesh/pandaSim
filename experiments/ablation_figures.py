"""Figures of the ablation (experiments/results/ablation_pivots.json): python experiments/ablation_figures.py [--out DIR]

(a) success rate of the pivot in each mode, (b) top view of the box centres, by outcome.
Colours (Okabe-Ito) are also told apart by marker shape; the robot base is at the origin.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MODES = [("rigid", "rigid\\n(approach tilt)"), ("rigid_tilt", "rigid\\n(best of 5 tilts)"), ("free", "free rotation\\n(limits on $\\\\phi$)"),
         ("free_nolimit", "free rotation\\n(no limits)")]
INK, MUTED = "#222222", "#6b6b6b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "results" / "ablation_pivots.json"))
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    d = json.load(open(a.data))
    T = d["trials"]
    n = len(T)
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1, 1.25]))

    rates = [100 * sum(t[m]["success"] for t in T) / n for m, _ in MODES]
    cols = ["#999999", "#999999", "#0072B2", "#999999"]
    bars = ax1.bar(range(4), rates, color=cols, width=0.6)
    for b, r in zip(bars, rates):
        ax1.text(b.get_x() + b.get_width() / 2, r + 1.5, f"{r:.0f} %", ha="center", va="bottom")
    ax1.set_xticks(range(4), [l.replace("\\n", "\n").replace("\\\\", "\\") for _, l in MODES], fontsize=6.5)
    ax1.set_ylim(0, 112)
    ax1.set_ylabel(f"successful pivots [%] (n = {n})")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("(a) success rate", loc="left", fontsize=8)

    cat = lambda t: "both" if t["rigid"]["success"] and t["free"]["success"] else ("free only" if t["free"]["success"] else "neither")
    style = {"both": ("#999999", "o", "rigid and free succeed"), "free only": ("#0072B2", "s", "only free rotation succeeds"),
             "neither": ("#D55E00", "X", "neither succeeds")}
    for k, (c, mk, lab) in style.items():
        pts = np.array([t["start"]["xy"] for t in T if cat(t) == k]).reshape(-1, 2)
        # table frame: x forward, y left -> plot x vertical so the robot sits at the bottom
        ax2.scatter(pts[:, 1], pts[:, 0], s=14, c=c, marker=mk, edgecolors="white", linewidths=0.4,
                    label=f"{lab} ({len(pts)})", zorder=3)
    ax2.plot(0, 0, marker="s", color=INK, ms=6)
    ax2.annotate("base", (0, 0), (0.02, 0.03), fontsize=7, color=MUTED)
    ax2.set_xlim(0.55, -0.55)          # y to the left
    ax2.set_ylim(-0.05, 0.9)
    ax2.set_aspect("equal")
    ax2.set_xlabel("$y$ [m]")
    ax2.set_ylabel("$x$ [m]")
    ax2.grid(color="#e4e4e4", lw=0.5, zorder=0)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    ax2.set_title("(b) box centre before the pivot", loc="left", fontsize=8)
    fig.tight_layout()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"ablation_pivots.{ext}", dpi=200, bbox_inches="tight")
    print("written to", out)


if __name__ == "__main__":
    main()
