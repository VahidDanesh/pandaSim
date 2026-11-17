"""experiments/placement_tolerance.py: a displaced box against the planned approach and pivot (simulation)."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import placement_tolerance as pt  # noqa: E402
import run_box as rb  # noqa: E402
import run_plan as rp  # noqa: E402

pytest.importorskip("coal")


@pytest.fixture(autouse=True)
def keep_hand_model(monkeypatch):
    """The real config selects the hand model (a module setting): put it back after every test."""
    from pandaSim.planning import clearance

    monkeypatch.setattr(clearance, "HAND_MODEL", clearance.HAND_MODEL)



def test_displaced_box_against_the_planned_grasp():
    from pandaSim.planning.collision import ArmCollision

    cfg = rp.load_config(rb.REAL_CONFIG)
    res = rb.plan_and_simulate(cfg, rb.parse_pose("0.55,0.20,90,standing"), rb.parse_pose("0.55,0.20,90,lying"), 1.0)
    assert [s.kind for s in res["plan"].steps][0] == "pivot"
    arm = ArmCollision(res["robot"], res["box"].size)
    run = lambda **e: pt.analyse_step(res, 0, arm, **{**dict(e_n=0.0, e_x=0.0, yaw=0.0), **e})
    exact = run()
    assert 8.0 < exact["approach_mm"] < 10.5                   # about 10 mm between each open finger and the box
    assert exact["hand_mm"] > 5.0 and exact["edge_slip_mm"] == 0.0
    assert run(e_n=0.004)["approach_mm"] == pytest.approx(exact["approach_mm"] - 4.0, abs=1.6)  # 1 mm per mm along the fingers
    assert run(e_n=0.012)["approach_mm"] < 0                                                    # a finger lands on the box
    assert run(yaw=math.radians(10))["approach_mm"] < 0 < run(yaw=math.radians(2))["approach_mm"]
    across = run(e_x=0.004)                                    # across the fingers: no collision, but the box ends 4 mm off
    assert across["approach_mm"] == pytest.approx(exact["approach_mm"], abs=0.5)
    assert across["final_error_mm"] == pytest.approx(4.0) and across["edge_slip_mm"] == pytest.approx(4.0 * math.sqrt(2), rel=0.01)
    assert min(run(e_x=0.006)["hand_mm"], run(e_x=-0.006)["hand_mm"]) < exact["hand_mm"] - 4.0   # the palm margin is used up
