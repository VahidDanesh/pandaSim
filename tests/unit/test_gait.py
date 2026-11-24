"""Object gaiting with one arm (planning/gait.py): geometry of a step, the angle solver, the planner, execution."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from pandaSim.planning import gait, se3
from pandaSim.planning.feasibility import grasp_observability
from pandaSim.planning.sequencer import Box, Step, snap_stable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import run_box as rb  # noqa: E402
import run_plan as rp  # noqa: E402

BOX = Box((0.15, 0.06, 0.25), 1.0, 0.5)
G, B = math.radians(15), math.radians(30)
CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * np.array(BOX.size) / 2


def pose(text):
    return rp.box_pose(BOX, rb.parse_pose(text))


def corners(T):
    return CORNERS @ T[:3, :3].T + T[:3, 3]


@pytest.mark.parametrize("text", ["0.55,0.10,0,standing", "0.50,-0.10,60,lying", "0.60,0.00,-120,standing_flip"])
@pytest.mark.parametrize("end,side", [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)])
def test_a_step_turns_the_box_about_its_vertex_without_lifting_the_vertex_or_entering_the_table(text, end, side):
    T0, alpha = pose(text), math.radians(35)
    pl = gait.gait_step(T0, BOX.size, end, side, alpha, G, B)
    a = pl.vertex
    assert abs(a[2]) < 1e-12 and len(pl.screws) == 5
    T, low = T0, []
    for i, (S, th) in enumerate(pl.screws):
        for s in np.linspace(0, 1, 11):
            Ts = se3.exp6(S * th * s) @ T
            c = corners(Ts)
            assert c[:, 2].min() > -1e-9                                     # no corner below the table
            assert np.linalg.norm(c - a, axis=1).min() < 1e-9               # the vertex stays where it is
        T = se3.exp6(S * th) @ T
        low.append(np.sort(corners(T)[:, 2])[1])                             # the second lowest corner
    assert low[0] < 1e-9 and low[1] > 0.005                                  # on the edge, then on the vertex alone
    R = se3.exp3(np.array([0.0, 0.0, alpha]))                                # net: rotation by alpha about the vertical
    assert np.allclose(pl.T_goal[:3, :3], R @ T0[:3, :3], atol=1e-9)
    assert np.allclose(pl.T_goal[:3, 3], a + R @ (T0[:3, 3] - a), atol=1e-9)
    ok, margin, _ = grasp_observability(Step("gait", T0, pl.T_goal, pl))
    assert ok and margin > 0.02


def test_the_solver_reaches_the_goal_exactly_with_alternating_ends():
    planner = gait.GaitPlanner(None, G, B)
    planner.y_ref = np.array([0.0, -1.0, 0.0])
    T0, Tg = pose("0.50,0.10,0,lying"), pose("0.55,-0.08,30,lying")
    sols = gait.solve_angles(T0, Tg, BOX.size, k_max=4)
    assert sols and len(sols[0][1]) <= 4
    assert [len(a) for _, a in sols] == sorted(len(a) for _, a in sols)
    for seq, alphas in sols[:5]:
        assert np.abs(alphas).max() <= math.radians(45) + 1e-9
        assert all(seq[j][0] == -seq[j + 1][0] for j in range(len(seq) - 1))    # opposite ends in turn
        steps = planner.steps(BOX, T0, seq, alphas)
        assert np.allclose(steps[-1].T_after, Tg, atol=1e-6)
    assert gait.solve_angles(T0, pose("0.55,-0.08,30,standing"), BOX.size) == []   # another face: no gait
    big = snap_stable(BOX, se3.rp(se3.exp3(np.array([math.pi / 2, 0.0, 0.0])), np.array([0.5, 0.0, 0.0])))   # largest face down
    assert gait.support_vertices(big, BOX.size) == []


class FakeKin:
    T_ready = np.eye(4)

    def __init__(self, refuse=0):
        self.refuse, self.checked = refuse, []

    def check(self, step):
        self.checked.append(step)
        return (len(self.checked) > self.refuse, 0.5)


def test_the_planner_takes_the_first_candidate_that_passes_the_rollouts():
    T0, Tg = pose("0.50,0.10,0,lying"), pose("0.55,-0.08,30,lying")
    p = gait.GaitPlanner(FakeKin(), G, B, k_max=4).plan(BOX, T0, Tg)
    assert p.found and all(s.kind == "gait" for s in p.steps) and np.allclose(p.steps[-1].T_after, Tg, atol=1e-6)
    p2 = gait.GaitPlanner(FakeKin(refuse=1), G, B, k_max=4).plan(BOX, T0, Tg)   # the first step of the best is refused
    assert p2.found and p2.stats["kin_checks"] == 2
    assert not gait.GaitPlanner(FakeKin(refuse=10**6), G, B, k_max=4, max_kin_checks=3).plan(BOX, T0, Tg).found


def test_one_gait_step_through_the_executor_and_run_box():
    """A goal one rotation about a vertex away: one step, executed by run_plan (ideal and with the controller's lag),
    the pre-flight of run_box.py is GO and the hand stays on every arc."""
    cfg = rp.load_config(rb.REAL_CONFIG)
    T0 = pose("0.55,0.10,0,standing")
    Tg = gait.gait_step(T0, BOX.size, -1.0, 1.0, math.radians(30), G, B).T_goal
    goal = dict(xy=Tg[:2, 3].tolist(), rpy=[0.0, 0.0, math.degrees(math.atan2(Tg[1, 0], Tg[0, 0]))])   # exact, not rounded
    res = rb.plan_and_simulate(cfg, rb.parse_pose("0.55,0.10,0,standing"), goal, 0.3, tracking=True, gait=True)
    assert [s.kind for s in res["plan"].steps] == ["gait"]
    assert rp.passed(res) and res["err_pos"] < 1e-3
    items = rb.preflight(res, cfg)
    assert rb.go(items), [c for c in items if not c["ok"]]
    assert rb.pivot_path_error(res["log"], res["plan"].steps, res["robot"]) < 1e-3
    assert res["tracking"]["ok"]


def test_constraint_jacobian_of_the_angles_matches_finite_differences():
    """gait._apply_jac (the Jacobian SLSQP gets for the position constraint) against central differences."""
    rng = np.random.default_rng(3)
    for k in (1, 2, 3, 5):
        verts = [rng.uniform(-0.15, 0.15, 2) for _ in range(4)]
        seq, a, p = list(rng.integers(0, 4, k)), rng.uniform(-0.8, 0.8, k), rng.uniform(-0.5, 0.5, 2)
        J = gait._apply_jac(p, verts, seq, a)
        h = 1e-6
        fd = np.stack([(gait._apply(p, 0.0, verts, seq, a + h * e)[0] - gait._apply(p, 0.0, verts, seq, a - h * e)[0]) / (2 * h)
                       for e in np.eye(k)], axis=1)
        assert J.shape == (2, k) and np.abs(J - fd).max() < 1e-8
