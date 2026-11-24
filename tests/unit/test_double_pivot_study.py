"""experiments/double_pivot_study.py: the geometry of two pivots in one grasp (no rollouts)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import double_pivot_study as dp  # noqa: E402

from pandaSim.planning import clearance as C  # noqa: E402
from pandaSim.planning import se3  # noqa: E402
from pandaSim.planning.box_primitives import plan_box_pivot  # noqa: E402
from pandaSim.planning.sequencer import down_axis  # noqa: E402


@pytest.mark.parametrize("face", ["lying", "standing"])
def test_two_undo_chains_and_two_pi_rolls(face):
    T0 = dp.rest_pose(face, 0.5, 0.1, 30.0)
    chains = dp.two_pivot_chains(T0)
    assert sorted(k for k, _, _ in chains) == ["roll", "roll", "undo", "undo"]
    n = T0[:3, 1]                                                       # the finger axis: the thin box axis
    for kind, p1, p2 in chains:
        T2 = p2.T_goal
        if kind == "undo":
            assert np.allclose(T2, T0, atol=1e-9)
            continue
        w = se3.log3(T0[:3, :3].T @ T2[:3, :3])                         # net rotation, box frame
        assert np.linalg.norm(w) == pytest.approx(math.pi)
        assert abs(T0[:3, :3] @ w / math.pi @ n) == pytest.approx(1.0)   # about n: onto the opposite face
        assert np.allclose(down_axis(T2), -down_axis(T0))
        assert np.linalg.norm(T2[:3, 3] - T0[:3, 3]) == pytest.approx(0.15 + 0.25)   # rolled by W + H
    pl1, pl2, u = dp.roll_chain(T0)                                     # the study's chain is one of the rolls
    assert any(np.allclose(pl2.T_goal, p2.T_goal) for k, _, p2 in chains if k == "roll")
    assert np.allclose(pl1.T_goal[:3, :3] @ np.round(pl1.T_goal[:3, :3].T @ u), u)


def test_free_arc_is_zero_inside_the_face_and_matches_phi_limits_at_the_corner():
    for c_o in ([0.0, 0, 0.0], [0.075 - 0.05, 0, 0.125 - 0.05], [0.0, 0, 0.125 - 0.04]):
        assert dp.width(dp.free_mask(np.array(c_o))) == 0.0             # interior grasps: no hand rotation clear
    T0 = dp.rest_pose("standing")                                       # the planner's pivot grasp: inset 20, 20
    pl = plan_box_pivot(T0, dp.SIZE, np.array([1.0, 0, 0]), math.pi / 2, dp.Y_REF)
    lo, hi = C.phi_limits(se3.inv(pl.T_grasp) @ T0, dp.SIZE, model="box")
    c_o = (se3.inv(T0) @ pl.T_grasp)[:3, 3]
    assert dp.width(dp.free_mask(c_o)) == pytest.approx(math.degrees(hi - lo), abs=2 * dp.STEP)


def test_arcs_and_components_on_a_circle():
    m = np.zeros(12, bool)
    m[[0, 1, 10, 11]] = True                                            # one run across the wrap, of 4 cells
    m[5] = True
    assert dp.longest_arc(m) == (10, 4)
    lab = dp.components(m)
    assert lab[0] == lab[11] != lab[5] and lab[3] == -1
    F, lo, L = dp.usable(m, math.radians(dp.STEP))                      # shrink by one cell at each end
    assert (lo, L) == (11, 2) and F.sum() == 2
    nxt = np.zeros(12, bool)
    nxt[[1, 2, 6, 7]] = True                                            # run 1-2 overlaps m at 1, run 6-7 nothing
    R = dp.connected([m, nxt])
    assert R[1][1] and R[1][2] and not R[1][6]


def test_palm_low_is_clearance_palm_lowest():
    T0 = dp.rest_pose("lying", 0.5, -0.1, 45.0)
    pl1, pl2, u = dp.roll_chain(T0)
    c_o = dp.corner_point(T0, u, "back_top", 0.01, 0.03)
    for alpha in (-30.0, 0.0, 22.0, 94.0, 158.0, 200.0):
        T = dp.grasp_frame(T0, u, c_o, alpha)
        assert dp.palm_low(T[2, 3], math.radians(alpha)) == pytest.approx(C.palm_lowest(T), abs=1e-12)


def test_windows_do_not_depend_on_the_start_position_and_yaw():
    res = []
    for x, y, yaw in ((0.5, 0.0, 0.0), (0.4, 0.2, 75.0)):
        T0 = dp.rest_pose("lying", x, y, yaw)
        pl1, pl2, u = dp.roll_chain(T0)
        poses = dp.chain_poses(pl1, pl2)
        c_o = dp.corner_point(T0, u, "back_top", 0.01, 0.03)
        res.append(dp.study_grasp(poses, *dp.planar(T0, u, poses), c_o, dp.free_mask(c_o), 0.13))
    assert res[0].keys() == res[1].keys()
    for k in res[0]:
        assert res[0][k] == pytest.approx(res[1][k], abs=1e-9), k
    g = res[0]
    assert g["tilt"] == pytest.approx(180 - g["usable"])
    assert g["start_window_connected"] > 10 and g["end_window_connected"] > 10 and g["lazy_ok"]
