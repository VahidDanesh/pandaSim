"""experiments/analyze_run.py on runs of the fake robot of test_run_box (a complete run and one that stops)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import analyze_run as ar  # noqa: E402
import run_box as rb  # noqa: E402
import run_plan as rp  # noqa: E402
from test_run_box import GOAL, START, Ctrl, Ctx, Gripper, Panda  # noqa: E402


@pytest.fixture(scope="module")
def planned():
    cfg = rp.load_config(rb.REAL_CONFIG)
    return cfg, rb.plan_and_simulate(cfg, rb.parse_pose(START), rb.parse_pose(GOAL), speed=1.0)


class FailingCtx(Ctx):
    def __init__(self, panda, after):
        super().__init__(panda)
        self.after, self.n = after, 0

    def ok(self):
        self.n += 1
        if self.n > self.after:
            raise RuntimeError("cartesian_reflex")           # what panda-py raises on a robot error
        return super().ok()


class FailingPanda(Panda):
    def __init__(self, q, after):
        super().__init__(q)
        self.after = after

    def create_context(self, frequency):
        return FailingCtx(self, self.after)


def run(planned, tmp_path, monkeypatch, panda_cls, *args):
    from panda_py import controllers

    monkeypatch.setattr(controllers, "IntegratedVelocity", Ctrl)
    cfg, res = planned
    run_dir = tmp_path / panda_cls.__name__
    run_dir.mkdir()
    rb.save_run(res, cfg, run_dir, 1.0)
    q_ready = np.array(cfg["robot"]["q_ready"])
    rb.execute(res, cfg, run_dir, None, "low", 1.0, ask=lambda p: f"RUN low {run_dir.name}",
               panda=panda_cls(q_ready, *args), gripper=Gripper())
    return run_dir


def test_complete_run(planned, tmp_path, monkeypatch):
    run_dir = run(planned, tmp_path, monkeypatch, Panda)
    lines = ar.analyze(run_dir, f_max=30.0)
    assert "all finished" in lines[0]
    assert any("object pose" in l for l in lines)
    for name in ("analysis.pdf", "analysis.png", "summary.txt"):
        assert (run_dir / name).stat().st_size > 0


def test_run_that_stopped(planned, tmp_path, monkeypatch):
    run_dir = run(planned, tmp_path, monkeypatch, FailingPanda, 4000)  # stops 4 s into the run
    lines = ar.analyze(run_dir, f_max=30.0)
    assert "STOPPED" in lines[0] and "robot error" in lines[0]
    exp, real = ar.load(run_dir / "expected.npz"), ar.load(run_dir / "real.npz")
    sim = ar.aligned_simulation(exp, real)
    assert sim["t"].max() <= real["segments"][-1]["t0"] + exp["t"][-1]  # only the segments the robot started
    assert (run_dir / "analysis.pdf").exists()


def test_phi_limits_come_from_the_plan_of_the_run(tmp_path):
    import json

    (tmp_path / "plan.json").write_text(json.dumps(dict(steps=[dict(kind="pivot", plan=dict(phi_lim=[-1.0, 0.5])),
                                                               dict(kind="slide", plan=dict(width=0.06))])))
    assert ar.phi_limits(tmp_path) == {0: (-1.0, 0.5)} and ar.phi_limits(tmp_path / "missing") == {}
