"""planning/chain.py: a roll by pi in one grasp (two pivots), and its protocol entries."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from pandaSim.planning import clearance as C
from pandaSim.planning import se3
from pandaSim.planning.chain import CHAIN_ALPHA0, CHAIN_INSET, CHAIN_LIFT, ChainPlanner, plan_box_roll
from pandaSim.planning.sequencer import Box, down_axis

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import protocol as proto  # noqa: E402
import run_box as rb  # noqa: E402
import run_plan as rp  # noqa: E402

SIZE = (0.15, 0.06, 0.25)
Y_REF = np.array([0.0, -1.0, 0.0])


@pytest.fixture(autouse=True)
def box_hand_model(monkeypatch):
    monkeypatch.setattr(C, "HAND_MODEL", "box")                # box_real.yaml; restored after each test


def lying(x, y, yaw):
    return rp.box_pose(Box(SIZE), rb.parse_pose(f"{x},{y},{yaw},lying"))


def test_roll_is_two_pivots_by_pi_2_in_one_grasp_near_the_bottom_back_corner():
    T0 = lying(0.5, -0.2, 90)
    d = np.array([0.0, 0, -1])                                 # z_b points to -y at yaw 90: this rolls toward +y
    pl = plan_box_roll(T0, SIZE, d, Y_REF)
    assert len(pl.screws) == 2 and all(th == pytest.approx(math.pi / 2) for _, th in pl.screws)
    T = T0
    for S, th in pl.screws:
        T = se3.exp6(S * th) @ T
    assert np.allclose(T, pl.T_goal)
    assert np.allclose(pl.T_goal[:3, 3], [0.5, 0.2, 0.075])    # rolled by W + H = 0.40 m
    assert np.allclose(down_axis(pl.T_goal), -down_axis(T0))  # onto the opposite face
    assert rb.pose_text(pl.T_goal) == "0.500,0.200,90.0,lying_flip"
    c = (se3.inv(T0) @ pl.T_grasp)[:3, 3]                      # grasp point in the box frame: back (-d), bottom
    assert c @ -d == pytest.approx(0.125 - CHAIN_INSET[0]) and c @ down_axis(T0) == pytest.approx(0.075 - CHAIN_INSET[1])
    u, p = T0[:3, :3] @ d, -pl.T_grasp[:3, 2]                   # palm direction: tilted, behind the box
    assert math.atan2(p[2], p @ u) == pytest.approx(CHAIN_ALPHA0)
    assert abs(pl.T_grasp[:3, 1] @ T0[:3, 1]) == pytest.approx(1.0)    # fingers along the thin axis
    lo, hi = pl.phi_lim
    assert (lo, hi) == C.phi_limits(se3.inv(pl.T_grasp) @ T0, SIZE)
    assert lo < 0 < hi and math.degrees(hi - lo) > 120         # the free arc of the band (box model)
    fl = plan_box_roll(T0, SIZE, d, Y_REF, flip=True)          # the hand turned by pi about its axis
    assert np.allclose(fl.T_grasp[:3, 1], -pl.T_grasp[:3, 1]) and np.allclose(fl.T_grasp[:3, 2], pl.T_grasp[:3, 2])
    assert fl.phi_lim == pytest.approx((-hi, -lo), abs=1e-9)


class FakeKin:
    T_ready = se3.rp(np.diag([1.0, -1.0, -1.0]), np.array([0.3, 0.0, 0.5]))   # y of {c} at ready: -y

    def __init__(self, ok=True):
        self.ok, self.steps = ok, []

    def check(self, step):
        self.steps.append(step)
        return self.ok, 0.5


def test_planner_returns_one_step_with_both_screws_to_the_goal():
    T0 = lying(0.5, -0.2, 90)
    kin = FakeKin()
    Tg = rp.box_pose(Box(SIZE), rb.parse_pose("0.5,0.2,90,lying_flip"))
    plan = ChainPlanner(kin, symmetric=False).plan(Box(SIZE), T0, Tg)
    assert plan.found and len(plan.steps) == 1
    st = plan.steps[0]
    assert st.kind == "pivot" and len(st.plan.screws) == 2 and np.allclose(st.T_after, st.plan.T_goal)
    assert np.allclose(st.plan.T_grasp[:3, 1], plan_box_roll(T0, SIZE, np.array([0.0, 0, -1]), kin.T_ready[:3, 1]).T_grasp[:3, 1])
    # a goal that no roll reaches, and a rollout that fails: no plan
    assert not ChainPlanner(FakeKin()).plan(Box(SIZE), T0, lying(0.5, 0.0, 90)).found
    assert not ChainPlanner(FakeKin(ok=False)).plan(Box(SIZE), T0, plan.steps[0].plan.T_goal).found


def test_protocol_chain_entries_are_free_and_rigid_with_the_chain_flag():
    protocol = yaml.safe_load(proto.PROTOCOL.read_text())
    es = list(proto.entries(protocol, "chain"))
    assert [(t, g) for _, t, g, _ in es] == [("a", "free"), ("a", "rigid"), ("b", "free"), ("b", "rigid")]
    for _, _, _, spec in es:
        assert spec["chain"] and not spec["gait"] and spec["speed"] == 0.15
        T0, Tg = (rp.box_pose(Box(SIZE), rb.parse_pose(spec[k])) for k in ("start", "goal"))
        assert np.allclose(down_axis(Tg), -down_axis(T0))
    assert all(not s["chain"] for e, _, _, s in proto.entries(protocol) if e != "chain")


def test_chain_task_a_plans_and_runs_in_simulation():
    """Protocol entry chain/a with free rotation, at full speed without the lag model: one step of two screws, every
    check of run_plan passes, the lift is raised for the tilted hand."""
    cfg = rp.load_config(rb.REAL_CONFIG)
    spec = yaml.safe_load(proto.PROTOCOL.read_text())["experiments"]["chain"]["tasks"]["a"]
    res = rb.plan_and_simulate(cfg, rb.parse_pose(spec["start"]), rb.parse_pose(spec["goal"]), 1.0, chain=True)
    assert cfg["execution"]["lift"] == CHAIN_LIFT
    assert [s.kind for s in res["plan"].steps] == ["pivot"] and len(res["plan"].steps[0].plan.screws) == 2
    assert all(res["checks"].values()), res["checks"]
    phi = np.degrees(res["log"].arrays()["phi"])
    assert phi.max() - phi.min() > 90                          # the bearing takes more than one pivot's worth
