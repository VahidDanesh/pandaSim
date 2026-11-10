"""Planner: parity with reference plans exported from the original (torch) planner, and basic properties.

tests/data/planner_reference.json holds, for random box tasks, the 12 cheapest candidate plans
(costs, step kinds, object pose after each step, grasp pose of each step)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pandaSim.planning import box_primitives as P
from pandaSim.planning import se3
from pandaSim.planning.sequencer import Box, Sequencer, box_symmetric_goals, down_axis, pose_error, snap_stable

REF = json.loads((Path(__file__).parents[1] / "data" / "planner_reference.json").read_text())
BOX = Box(tuple(REF["size"]), REF["mass"], REF["mu"])
TOL_C, TOL_T = 1e-6, 1e-6


def _same(steps, ref):
    return ([s.kind for s in steps] == ref["kinds"]
            and all(np.abs(s.T_after - np.array(T)).max() < TOL_T for s, T in zip(steps, ref["T_after"]))
            and all(np.abs(s.plan.T_grasp - np.array(T)).max() < TOL_T for s, T in zip(steps, ref["T_grasp"])))


@pytest.mark.parametrize("i", range(len(REF["tasks"])))
def test_candidates_match_reference(i):
    """Same costs in the same order; every reference plan cheaper than the last cost (plans that tie
    with the last one may be cut differently) is found with the same steps, poses and grasps."""
    task = REF["tasks"][i]
    seq = Sequencer(y_ref=REF["y_ref"])
    cands = seq.candidates(BOX, np.array(task["T0"]), np.array(task["Tg"]), k=REF["k"])
    ref = task["candidates"]
    assert len(cands) == len(ref)
    assert np.allclose([c for c, _ in cands], [r["cost"] for r in ref], atol=TOL_C, rtol=0)
    full = len(ref) == REF["k"]
    for r in ref:
        if full and r["cost"] > ref[-1]["cost"] - TOL_C:
            continue
        assert any(abs(c - r["cost"]) < TOL_C and _same(steps, r) for c, steps in cands)


def _pose(x, y, yaw, tilt=0.0):
    R = se3.exp3(np.array([0.0, 0.0, yaw])) @ se3.exp3(np.array([0.0, tilt, 0.0]))
    return snap_stable(BOX, se3.rp(R, np.array([x, y, 0.0])))


def test_se3_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        V = rng.normal(size=6)
        T = se3.exp6(V)
        assert np.allclose(se3.exp6(se3.log6(T)), T, atol=1e-9)
        assert np.allclose(se3.inv(T) @ T, np.eye(4), atol=1e-12)
        S, th = se3.axis_angle(V)
        assert np.allclose(S * th, V)


def test_pivot_plan_rotates_about_table_edge():
    T = _pose(0.5, 0.0, 0.3, -math.pi / 2)          # lying on its 0.06 x 0.25 face, thin axis y horizontal
    u = down_axis(T)
    d = np.cross(u, np.array([0.0, 1.0, 0.0]))
    pl = P.plan_box_pivot(T, BOX.size, d, math.pi / 2, np.array([0.0, -1.0, 0.0]))
    assert np.allclose(pl.T_goal, se3.exp6(pl.S * pl.theta) @ T)
    assert abs(pl.S[2]) < 1e-12 and abs(np.linalg.norm(pl.S[:3]) - 1) < 1e-12      # horizontal axis
    q = np.cross(pl.S[:3], pl.S[3:])                                                # point on the axis
    assert abs(q[2] - (q @ pl.S[:3]) * pl.S[2]) < 1e-9                             # on the table
    assert np.allclose(pl.T_goal, snap_stable(BOX, pl.T_goal), atol=1e-9)          # ends in a resting pose
    assert abs(abs(pl.T_grasp[:3, 1] @ pl.S[:3]) - 1) < 1e-9                       # bearing axis parallel to the edge
    assert pl.width == pytest.approx(0.06)


def test_slide_plan_reaches_goal_and_keeps_grasp():
    Ta, Tb = _pose(0.5, 0.0, 0.0, -math.pi / 2), _pose(0.55, 0.12, 0.7, -math.pi / 2)
    pl = P.plan_box_slide(Ta, BOX.size, Tb, np.array([0.0, -1.0, 0.0]), BOX.mu, BOX.mass)
    T = Ta
    for S, th in pl.screws:
        assert abs(S[0]) < 1e-12 and abs(S[1]) < 1e-12 and abs(S[5]) < 1e-12       # planar motion
        T = se3.exp6(S * th) @ T
    assert np.allclose(T, Tb, atol=1e-9)
    assert pl.T_grasp[2, 2] == pytest.approx(-1.0)                                 # hand straight down
    assert P.slide_tip_margin(BOX.size, pl, BOX.mu, BOX.mass) > 0


def test_plans_end_at_a_symmetric_goal():
    seq = Sequencer(y_ref=(0.0, -1.0, 0.0))
    T0 = _pose(0.5, 0.0, 0.0, -math.pi / 2)
    for Tg in (_pose(0.55, 0.15, 0.6, -math.pi / 2), _pose(0.45, -0.1, 1.0, 0.0), _pose(0.6, 0.1, -0.4, math.pi / 2)):
        cands = seq.candidates(BOX, T0, Tg, k=4)
        assert cands
        for cost, steps in cands:
            T = T0
            for s in steps:
                assert np.allclose(s.T_before, T, atol=1e-9)
                T = s.T_after
            ep, er = pose_error(T, Tg)
            assert ep < 1e-9 and er < 1e-6
            assert any(np.allclose(T, Tv, atol=1e-6) for Tv in box_symmetric_goals(Tg))
        assert all(a[0] <= b[0] + 1e-12 for a, b in zip(cands, cands[1:]))
