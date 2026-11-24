"""Panels (b) and (c) of the primitive-motion figure of Sec. 7 (fig:exp_primitives) from one robot run folder.

    python experiments/paper_figure.py experiments/runs/box/prim_pivot_free_r1 --out <Manuscript>/Figures/exp_pivot

(b) joint angles normalized by their ranges, (c) the integrated angle phi-hat of the free rotational motion with its
limits phi- and phi+, over the time of the held motion (grasp to release; the approach and the joint moves are left
out). Single IEEE column, vector PDF (and a PNG for checking). Panel (a), the snapshots, are photographs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_run as ar  # noqa: E402

OKABE_ITO = ar.OKABE_ITO
STYLES = ["-", "--", "-", "--", "-", "--", "-"]       # second encoding of the joints besides the colour


def held_window(real):
    """Time [t0, t1] from the start of the first held segment to the end of the last one."""
    held = [s for s in real["segments"] if s["name"] in ar.HELD]
    return (held[0]["t0"], held[-1]["t_end"]) if held else (real["t"][0], real["t"][-1])


def figure(real, qlim, phi_lim, width=3.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 6.5, "axes.linewidth": 0.6, "pdf.fonttype": 42, "mathtext.default": "regular"})
    t0, t1 = held_window(real)
    m = (real["t"] >= t0) & (real["t"] <= t1)
    t = real["t"][m] - t0
    lb, ub = qlim
    fig, axs = plt.subplots(2, 1, figsize=(width, 2.9), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1], hspace=0.15))
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(7):
        axs[0].plot(t, (real["theta"][m, j] - lb[j]) / (ub[j] - lb[j]), color=OKABE_ITO[j], ls=STYLES[j], lw=1.0,
                    label=f"{j + 1}")
    axs[0].axhline(0.0, color="#1a1a1a", lw=0.6)
    axs[0].axhline(1.0, color="#1a1a1a", lw=0.6)
    axs[0].set_ylim(-0.03, 1.03)
    axs[0].set_ylabel("Normalized\njoint angle")
    axs[0].legend(title="Joint", frameon=False, ncol=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.6,
                  columnspacing=0.8, title_fontsize=6.5)
    axs[1].plot(t, np.degrees(real["phi"][m]), color="#0072B2", lw=1.1, label=r"$\hat{\phi}$")
    first = True
    for s in real["segments"]:
        lim = phi_lim.get(s.get("step"))
        if s["name"] == "pivot" and lim and s["t_end"] >= t0 and s["t0"] <= t1:
            axs[1].hlines(np.degrees(lim), s["t0"] - t0, s["t_end"] - t0, color="#D55E00", lw=0.9, ls=(0, (2, 1.5)),
                          label=r"$\phi^{-}$, $\phi^{+}$" if first else None)
            first = False
    axs[1].set_ylabel(r"$\hat{\phi}$ (deg)")
    axs[1].set_xlabel("Time (s)")
    axs[1].legend(frameon=False, loc="best", ncol=2)
    for ax, lab in zip(axs, ("(b)", "(c)")):
        ax.text(-0.2, 1.0, lab, transform=ax.transAxes, fontsize=8, va="top")
    fig.align_ylabels(axs)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--out", help="output path without extension (default: <run_dir>/paper_figure)")
    a = ap.parse_args(argv)
    from pandaSim.geometry.utils import create_virtual_panda

    r = create_virtual_panda()
    real = ar.load(Path(a.run_dir) / "real.npz")
    fig = figure(real, (r.qlim[0, :7], r.qlim[1, :7]), ar.phi_limits(a.run_dir))
    out = Path(a.out) if a.out else Path(a.run_dir) / "paper_figure"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"written: {out.with_suffix('.pdf')}, .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
