"""experiments/run_plan.py: config loading, pose convention, one task end to end."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import run_plan as rp  # noqa: E402


def test_config_switches():
    cfg = rp.load_config()
    assert cfg["free_rotation"] is True and cfg["plan_free_rotation"] is True
    cfg = rp.load_config(free_rotation=False)
    assert cfg["free_rotation"] is False and cfg["plan_free_rotation"] is False
    cfg = rp.load_config(free_rotation=False, plan_free_rotation=True)
    assert cfg["free_rotation"] is False and cfg["plan_free_rotation"] is True


def test_box_pose_rests_on_the_table():
    cfg = rp.load_config()
    box = rp.make_box(cfg)
    lx, ly, lz = cfg["box"]["size"]
    lying = rp.box_pose(box, {"xy": [0.5, 0.1], "rpy": [0, -90, 30]})
    standing = rp.box_pose(box, {"xy": [0.5, 0.1], "rpy": [0, 0, 30]})
    assert np.allclose(lying[:2, 3], [0.5, 0.1]) and np.allclose(standing[:2, 3], [0.5, 0.1])
    assert lying[2, 3] == pytest.approx(lx / 2)       # box x axis vertical
    assert standing[2, 3] == pytest.approx(lz / 2)
    assert abs(lying[2, 0]) == pytest.approx(1.0)
    yaw = np.arctan2(standing[1, 0], standing[0, 0])
    assert yaw == pytest.approx(np.radians(30))


def test_slide_task_passes(tmp_path):
    cfg = rp.load_config()
    cfg["execution"]["dt"] = 0.005
    robot = rp.make_robot(cfg)
    res = rp.run_task(cfg, robot, "slide")
    assert [s.kind for s in res["plan"].steps] == ["slide"]
    assert rp.passed(res), res["checks"]
    path = rp.save(res, cfg, tmp_path)
    data = np.load(path, allow_pickle=True)
    assert path.name == "slide_free.npz"
    assert np.allclose(data["Tg"], res["Tg"]) and np.allclose(data["size"], cfg["box"]["size"])
