"""experiments/run_box.py: poses, pre-flight, confirmation and the execution on a fake panda-py (no robot)."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import run_box as rb  # noqa: E402
import run_plan as rp  # noqa: E402

START, GOAL = "0.50,0.00,0,lying", "0.60,0.10,20,lying_flip"      # a slide: about 13 s of motion


class Ctrl:
    def __init__(self, **kw):
        self.qd, self.t = np.zeros(7), 0.0

    def set_control(self, qd):
        self.qd = np.array(qd)

    def get_time(self):
        return self.t


class Ctx:
    def __init__(self, panda):
        self.panda = panda

    def ok(self):
        c = self.panda.ctrl
        c.t += 1e-3
        self.panda.q = self.panda.q + c.qd * 1e-3          # the IntegratedVelocity controller of libfranka
        return True


class State:
    def __init__(self, panda):
        self.q, self.O_F_ext_hat_K, self.tau_ext_hat_filtered = panda.q.copy(), np.zeros(6), np.zeros(7)
        self.current_errors = "[]"


class Panda:
    def __init__(self, q):
        self.q, self.ctrl, self.moves = np.array(q, float), None, []

    def start_controller(self, ctrl):
        self.ctrl = ctrl

    def stop_controller(self):
        self.ctrl = None

    def create_context(self, frequency):
        return Ctx(self)

    def get_state(self):
        return State(self)

    def move_to_joint_position(self, q, speed_factor):
        self.moves.append(speed_factor)
        self.q = np.array(q, float)


class Gripper:
    def __init__(self):
        self.calls = []

    def move(self, width, speed):
        self.calls.append(("move", width))
        return True

    def grasp(self, width, speed, force, e1, e2):
        self.calls.append(("grasp", width))
        return True

    def read_once(self):
        class G:
            width, max_width, is_grasped = 0.08, 0.08, False
        return G()


@pytest.fixture(scope="module")
def planned():
    cfg = rp.load_config(rb.REAL_CONFIG)
    return cfg, rb.plan_and_simulate(cfg, rb.parse_pose(START), rb.parse_pose(GOAL), speed=1.0)


def test_parse_pose():
    assert rb.parse_pose("0.5,-0.1,30,standing") == dict(xy=[0.5, -0.1], rpy=[0.0, 0.0, 30.0])
    assert rb.parse_pose("0.5,0,0,lying")["rpy"][1] == -90.0
    with pytest.raises(SystemExit):
        rb.parse_pose("0.5,0,0,flat")
    with pytest.raises(SystemExit):
        rb.parse_pose("0.5,0,0")


def test_preflight_go_and_the_checks_that_block(planned):
    cfg, res = planned
    items = rb.preflight(res, cfg)
    assert rb.go(items), [c for c in items if not c["ok"]]
    heavy = copy.deepcopy(cfg)
    heavy["box"]["mass"] = 4.0
    res_heavy = dict(res, box=type(res["box"])(res["box"].size, 4.0, res["box"].mu))
    assert not rb.go(rb.preflight(res_heavy, heavy))                    # over the payload
    outside = copy.deepcopy(cfg)
    outside["tasks"]["custom"]["goal"]["xy"] = [0.95, 0.0]
    assert not rb.go(rb.preflight(res, outside))


def test_no_plan_is_no_go():
    cfg = rp.load_config(rb.REAL_CONFIG)
    res = rb.plan_and_simulate(cfg, rb.parse_pose("0.5,0.0,0,lying"), rb.parse_pose("0.5,0.0,0,lying"))
    assert not rb.go(rb.preflight(res, cfg))                              # identical start and goal: nothing to do


def test_confirm_needs_the_exact_phrase():
    assert rb.confirm("RUN air x", lambda prompt: "RUN air x")
    assert not rb.confirm("RUN air x", lambda prompt: "yes")
    assert not rb.confirm("RUN air x", lambda prompt: "")


def test_execute_on_a_fake_panda_matches_the_simulation(planned, tmp_path, monkeypatch):
    from panda_py import controllers

    monkeypatch.setattr(controllers, "IntegratedVelocity", Ctrl)
    cfg, res = planned
    q_ready = rp.exec_kwargs(cfg)["q_ready"]
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    panda, gripper = Panda(q_ready + 0.01), Gripper()
    assert rb.execute(res, cfg, run_dir, None, "low", 1.0, ask=lambda p: "wrong", panda=panda, gripper=gripper) is None
    assert panda.moves == [] and gripper.calls == []                    # not confirmed: nothing moved
    c = rb.execute(res, cfg, run_dir, None, "low", 1.0, ask=lambda p: f"RUN low {run_dir.name}", panda=panda, gripper=gripper)
    assert panda.moves == [0.1] and c["ok"] and c["n_segments"] == c["expected_segments"]
    assert max(s["max_joint_error"] for s in c["segments"]) < 5e-3     # the fake follows the commands
    assert c["object_error_mm"] < 3.0
    assert [g[0] for g in gripper.calls] == ["move", "grasp", "move"]
    assert json.loads((run_dir / "compare.json").read_text())["ok"]


def test_robot_checks_refuse_a_bad_start():
    q_ready = np.array(rp.load_config(rb.REAL_CONFIG)["robot"]["q_ready"])
    lim = (np.full(7, -2.8), np.full(7, 2.8))
    panda = Panda(q_ready)
    assert rb.go(rb.robot_checks(panda.get_state(), q_ready, lim))
    far = Panda(q_ready + 1.5)
    assert not rb.go(rb.robot_checks(far.get_state(), q_ready, lim))
    err = panda.get_state()
    err.current_errors = "[communication_constraints_violation]"
    assert not rb.go(rb.robot_checks(err, q_ready, lim))
