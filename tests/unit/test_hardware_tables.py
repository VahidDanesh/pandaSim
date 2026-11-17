"""experiments/hardware_tables.py and paper_figure.py on run folders of the fake robot of test_run_box."""
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import hardware_tables as ht  # noqa: E402
import paper_figure as pf  # noqa: E402
from test_analyze_run import FailingPanda, planned, run  # noqa: E402,F401
from test_run_box import Panda  # noqa: E402

PROTOCOL = dict(experiments=dict(prim=dict(repetitions=3, tasks=dict(slide={})),
                                 fr=dict(repetitions=1, grasps=["free", "rigid"], tasks=dict(a={}))))


@pytest.fixture(scope="module")
def runs(planned, tmp_path_factory):                       # noqa: F811
    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("runs")
    done = run(planned, base, mp, Panda)                    # a complete run
    stopped = run(planned, base, mp, FailingPanda, 4000)    # stopped by a robot error
    mp.undo()
    out = base / "box"
    for name, src in (("prim_slide_free_r1", done), ("prim_slide_free_r2", done), ("fr_a_free_r1", stopped),
                      ("fr_a_rigid_r1", stopped)):
        shutil.copytree(src, out / name)
    (out / "prim_slide_free_r2" / "measured.json").write_text(json.dumps(dict(error_mm=10.0, error_deg=1.0, face_ok=True)))
    c = json.loads((out / "fr_a_rigid_r1" / "compare.json").read_text())
    c["abort"]["cause"] = "infeasible QP"
    (out / "fr_a_rigid_r1" / "compare.json").write_text(json.dumps(c))
    return out


def test_outcomes_and_rows(runs):
    data = ht.collect(PROTOCOL, runs)
    prim = data[("prim", "slide", "free")]
    assert [r["outcome"] for r in prim] == ["success", "off target"]          # r3 was not run
    assert not prim[0]["measured"] and prim[1]["measured"] and prim[1]["error_mm"] == 10.0
    assert data[("fr", "a", "rigid")][0]["outcome"] == "infeasible" and data[("fr", "a", "free")][0]["outcome"] == "stopped"
    tex = ht.rows(PROTOCOL, data)
    assert tex["prim"][0].startswith(r"slide & 1/2 & ") and r"$^\dagger$" in tex["prim"][0]     # FK error, not measured
    assert tex["fr"][0].startswith("rigid & 0/1 & 1 & 0 & ") and tex["fr"][1].startswith("free rotation & 0/1 & 0 & 1 & ")


def test_paper_figure(runs, tmp_path):
    import analyze_run as ar
    from pandaSim.geometry.utils import create_virtual_panda

    r = create_virtual_panda()
    run_dir = runs / "prim_slide_free_r1"
    fig = pf.figure(ar.load(run_dir / "real.npz"), (r.qlim[0, :7], r.qlim[1, :7]), {0: (-1.0, 1.0)})
    fig.savefig(tmp_path / "f.pdf")
    assert (tmp_path / "f.pdf").stat().st_size > 0 and len(fig.axes) == 2
