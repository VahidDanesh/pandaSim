"""planning/load.py against hand calculations (quasi-static load of the held box, design review D8, D14, D17)."""
import math

import numpy as np
import pytest

from pandaSim.planning import box_primitives as P
from pandaSim.planning import gait, se3
from pandaSim.planning import load as L
from pandaSim.planning.sequencer import Step

m, g = 5.0, 9.81
Y = np.array([0.0, 1.0, 0.0])


def test_edge_force_only_holds_the_box_with_the_smallest_force():
    # edge along y through the origin, grasp point h above it, centre of mass d beyond it: F = m g d / h toward -x
    d, h = 0.1, 0.25
    F = L.edge_force_only(np.array([0, 0, h]), np.array([d, 0, 0.07]), np.zeros(3), Y, m)
    assert F == pytest.approx([-m * g * d / h, 0, 0])
    # general grasp point: |F| |r| = m g x_com (the bearing: no moment), F perpendicular to r and the edge
    c, com = np.array([0.2, 0.0, 0.1]), np.array([0.1, 0.0, 0.05])
    F = L.edge_force_only(c, com, np.zeros(3), Y, m)
    assert np.linalg.norm(F) * np.linalg.norm(c) == pytest.approx(m * g * 0.1)
    assert F @ c == pytest.approx(0) and F @ Y == pytest.approx(0)
    assert L.edge_moment(F, np.zeros(3), c, com, np.zeros(3), m) @ Y == pytest.approx(0)      # the edge is a hinge


def test_vertex_needs_a_moment_that_no_force_at_the_grasp_can_give():
    v, n, com = np.zeros(3), Y, np.array([0.05, 0.02, 0.1])
    c = np.array([0.0, 0.0, 0.3])                        # grasp point above the vertex: the force alone suffices
    F, M = L.vertex_min_wrench(c, com, v, n, m)
    assert M == pytest.approx(np.zeros(3)) and F == pytest.approx([-m * g * 0.05 / 0.3, -m * g * 0.02 / 0.3, 0])
    rng = np.random.default_rng(0)
    for _ in range(20):                                  # any geometry: all three moments about v balance
        c, com, n = rng.normal(size=3) * 0.2, rng.normal(size=3) * 0.1, rng.normal(size=3)
        n /= np.linalg.norm(n)
        F, M = L.vertex_min_wrench(c, com, v, n, m)
        b = -np.cross(com, L.GRAVITY * m)
        e = c / np.linalg.norm(c)
        assert np.cross(c, F) + M == pytest.approx(b, abs=1e-9)
        assert M @ n == pytest.approx(0, abs=1e-12) and F @ c == pytest.approx(0, abs=1e-12)
        assert np.linalg.norm(M) == pytest.approx(abs(b @ e) / math.sqrt(1 - (e @ n) ** 2))   # gait_vertex_moment.py


def test_closed_chain_equals_the_minimum_norm_force_for_an_isotropic_hand_and_adds_an_internal_force_otherwise():
    d, h, k, kr = 0.1, 0.25, 1000.0, 50.0                 # N/m, Nm/rad
    c, com = np.array([0, 0, h]), np.array([d, 0, 0.07])
    K = np.diag([k, k, k, kr, kr, kr])
    F, M, x = L.closed_chain(K, c, com, np.zeros(3), [Y], Y, m)
    assert F == pytest.approx([-m * g * d / h, 0, 0]) and M == pytest.approx(np.zeros(3), abs=1e-12)
    assert x[:3] == pytest.approx([m * g * d / (k * h), 0, 0])                           # deflection |F| / k
    # stiffness coupling x-y: the hinge holds the hand along the edge, so a force along it appears
    K[0, 1] = K[1, 0] = 300.0
    F, M, x = L.closed_chain(K, c, com, np.zeros(3), [Y], Y, m)
    assert F == pytest.approx([-m * g * d / h, -0.3 * m * g * d / h, 0])
    assert np.cross(c, F) @ Y + np.cross(com, L.GRAVITY * m) @ Y == pytest.approx(0)      # moment about the edge
    assert M @ Y == pytest.approx(0, abs=1e-12)                                           # the bearing
    # vertex: all three moments balance, whatever the stiffness
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    K = A @ A.T + 6 * np.eye(6)
    F, M, x = L.closed_chain(K, np.array([0.1, 0.03, 0.2]), com, np.zeros(3), list(np.eye(3)), Y, m)
    assert np.cross([0.1, 0.03, 0.2], F) + M + np.cross(com, L.GRAVITY * m) == pytest.approx(np.zeros(3), abs=1e-9)
    assert M @ Y == pytest.approx(0, abs=1e-9)


def test_slide_friction_translation_and_turn_about_the_centre():
    size, mu = (0.1, 0.1, 0.1), 0.5
    T = np.eye(4)
    T[2, 3] = 0.05
    c = np.array([0, 0, 0.08])
    F, M = L.slide_wrench(T, size, np.r_[0, 0, 0, 1.0, 0, 0], c, m, mu)
    assert F == pytest.approx([mu * m * g, 0, 0]) and M == pytest.approx(np.zeros(3), abs=1e-9)
    F, M = L.slide_wrench(T, size, np.r_[0, 0, 1.0, 0, 0, 0], c, m, mu, n_grid=60)
    a = 0.1                                               # mean distance from the centre of a square: a/6 (sqrt2 + ln(1 + sqrt2))
    assert np.linalg.norm(F) < 1e-9 and M[2] == pytest.approx(mu * m * g * a / 6 * (math.sqrt(2) + math.log(1 + math.sqrt(2))), rel=2e-3)


def test_table_friction_pads_and_the_face():
    d, h = 0.1, 0.25
    F = np.array([-m * g * d / h, 0, 0])
    assert L.table_reaction(F, m) == pytest.approx([m * g * d / h, 0, m * g])
    assert L.friction_needed(F, m) == pytest.approx(d / h)
    # an internal force of 5 N along the radius (here vertical) can unload the edge: the worst case
    assert L.friction_needed(F, m, [np.array([0, 0, 1.0])], 5.0) == pytest.approx((m * g * d / h) / (m * g - 5.0))
    assert L.friction_needed(np.array([0, 0, 60.0]), m) == math.inf                       # lifted off the table
    n = Y
    assert L.pad_use([84.0, 0, 0], np.zeros(3), n, 0.06, 70.0, 1.2) == pytest.approx(0.5)
    assert L.pad_use([0, 84.0, 0], np.zeros(3), n, 0.06, 70.0, 1.2) == pytest.approx(0.0)   # along n: the fingers
    assert L.pad_use(np.zeros(3), [0, 0, 2.52], n, 0.06, 70.0, 1.2) == pytest.approx(0.5)   # pad couple 5.04 Nm
    assert L.pad_use(np.zeros(3), [0, 2.52, 0], n, 0.06, 70.0, 1.2) == pytest.approx(0.0)   # about n: the bearing
    assert L.band([3.0, 4.0, 0]) == pytest.approx(6.0)
    th = math.radians(90)
    assert L.face_ramp(0.0, th, 0.02, True, True) == 0.0 and L.face_ramp(0.01, th, 0.02, True, True) == 0.5
    assert L.face_ramp(th / 2, th, 0.02, True, True) == 1.0 and L.face_ramp(th, th, 0.02, True, True) == 0.0
    assert L.face_ramp(0.0, th, 0.02, False, True) == 1.0


def test_contact_along_a_pivot_and_a_gait_step():
    size = (0.15, 0.06, 0.25)
    T = se3.rp(se3.exp3(np.array([0, -math.pi / 2, 0])), np.array([0.5, 0.0, 0.075]))     # lying
    pl = P.plan_box_pivot(T, size, np.array([0, 0, 1.0]), math.pi / 2)
    st = Step("pivot", T, pl.T_goal, pl)
    ct = L.contact(st, 0, T, size)
    assert ct["kind"] == "edge" and ct["L"] == pytest.approx(0.06) and ct["starts_on_face"] and ct["ends_on_face"]
    assert ct["p"][2] == pytest.approx(0.0) and np.cross(ct["p"] - T[:3, 3], ct["u"]) @ ct["u"] == pytest.approx(0)
    # the angle about the edge from the hand position alone
    c0 = pl.T_grasp[:3, 3]
    c = (se3.exp6(pl.S * 0.3) @ pl.T_grasp)[:3, 3]
    assert L.screw_angle(pl.S, c0, c) == pytest.approx(0.3)
    # at the start the hand holds the box with m g x_com / |r| (the 5 kg box: about 23 N, review D17)
    w = L.wrench(ct, T, pl.T_grasp, size, m)
    r = np.linalg.norm(L._perp(c0 - ct["p"], ct["u"]))
    assert np.linalg.norm(w["F_a"]) == pytest.approx(m * g * 0.125 / r)
    gp = gait.gait_step(T, size, 1.0, 1.0, 0.3, math.radians(15), math.radians(30), Y)
    gs = Step("gait", T, gp.T_goal, gp)
    kinds = [L.contact(gs, k, T, size)["kind"] for k in range(5)]
    assert kinds == ["edge", "vertex", "vertex", "vertex", "edge"]
    first, last = L.contact(gs, 0, T, size), L.contact(gs, 4, T, size)
    assert first["starts_on_face"] and not first["ends_on_face"] and last["ends_on_face"] and not last["starts_on_face"]


def test_where_the_edge_cannot_hold_the_hinge_the_chain_is_taken_with_the_edge_as_a_point():
    d, h = 0.12, 0.25
    c, com = np.array([0, 0, h]), np.array([d, 0, 0.07])
    T_c, T_obj = se3.rp(np.eye(3), c), se3.rp(np.eye(3), com)            # {c}: y = n along the edge
    ct = dict(kind="edge", S=np.r_[Y, 0, 0, 0], p=np.zeros(3), u=Y, L=0.06, theta=1.0, starts_on_face=False,
              ends_on_face=False)
    K = np.diag([1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0])
    w = L.wrench(ct, T_obj, T_c, (0.15, 0.06, 0.25), m, 0.5, np.linalg.inv(K))
    assert not w["rolls"] and w["roll"] == pytest.approx(0, abs=1e-9) and w["F_b"] == pytest.approx(w["F_h"])
    K[0, 1] = K[1, 0] = 300.0                    # the hinge now pushes along the edge: roll 0.3 m g d > N L / 2
    w = L.wrench(ct, T_obj, T_c, (0.15, 0.06, 0.25), m, 0.5, np.linalg.inv(K))
    assert w["roll"] == pytest.approx(0.3 * m * g * d) and w["roll_cap"] == pytest.approx(m * g * 0.03) and w["rolls"]
    F, M, x = L.closed_chain(K, c, com, np.zeros(3), list(np.eye(3)), Y, m)
    assert w["F_b"] == pytest.approx(F) and w["M_b"] == pytest.approx(M)
    assert L.edge_moment(F, M, c, com, np.zeros(3), m) == pytest.approx(np.zeros(3), abs=1e-9)   # asks nothing of it


def test_at_5_kg_the_standing_box_can_be_turned_about_the_grasp_but_not_translated():
    """Design review D17 (8): tau_grip / (m g) + d_lead - mu h_g along the finger axis, 0.0514 + 0.03 - 0.115 m < 0 at
    5 kg (+0.172 m at 1 kg); across it 0.075 - 0.115 m at any mass; a turn about the vertical through the grasp
    does not move the grasp point (no tipping moment)."""
    size, mu = (0.15, 0.06, 0.25), 0.5
    T = se3.rp(np.eye(3), np.array([0.5, 0.0, 0.125]))                  # standing on its 0.15 x 0.06 face
    T_grasp, width = P.box_grasp(T, size, Y)
    assert width == pytest.approx(0.06) and abs(T_grasp[:3, 1] @ Y) == pytest.approx(1.0)
    h_g = 0.25 - P.GRASP_DEPTH
    tau = 2 * P.PAD_MU * P.GRIP_N * 0.03 * P.GRIP_SAFETY                   # 2.52 Nm
    along, across = np.r_[0, 0, 0, Y], np.r_[0, 0, 0, 1.0, 0, 0]
    for mass, expect in ((5.0, tau / (5.0 * g) + 0.03 - mu * h_g), (1.0, tau / (1.0 * g) + 0.03 - mu * h_g)):
        assert P.tip_margin(T, size, T_grasp, along, mu, mass) == pytest.approx(expect)
    assert P.tip_margin(T, size, T_grasp, along, mu, 5.0) == pytest.approx(-0.0336, abs=1e-4)
    assert P.tip_margin(T, size, T_grasp, across, mu, 5.0) == pytest.approx(0.075 - mu * h_g)
    turn = np.r_[0, 0, 1.0, -np.cross([0, 0, 1.0], T_grasp[:3, 3])]
    assert P.tip_margin(T, size, T_grasp, turn, mu, 5.0) == math.inf
    # so the planner's slide of the standing 5 kg box: a turn about the grasp passes, a translation does not
    goal_turn = se3.exp6(turn * 0.5) @ T
    goal_move = se3.rp(np.eye(3), np.array([0.5, 0.05, 0.125]))
    assert P.slide_tip_margin(size, P.plan_box_slide(T, size, goal_turn, Y, mu, 5.0), mu, 5.0) > P.SLIDE_MARGIN_MIN
    assert P.slide_tip_margin(size, P.plan_box_slide(T, size, goal_move, Y, mu, 5.0), mu, 5.0) <= P.SLIDE_MARGIN_MIN
    assert P.slide_tip_margin(size, P.plan_box_slide(T, size, goal_move, Y, mu, 1.0), mu, 1.0) > P.SLIDE_MARGIN_MIN
