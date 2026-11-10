"""Planner steps through the executor in simulation: run_plan, JointMove, the rollout check, replay bookkeeping."""
import math

import numpy as np
import pytest
import spatialmath as sm

from pandaSim.control.backends import SimBackend
from pandaSim.control.contact_qp import FER_QD_MAX
from pandaSim.control.primitive_executor import (JointMove, Log, check_clearance, check_profile, contact_pose, run_plan,
                                                  run_segment)
from pandaSim.control.viewer import finger_widths, grasp_intervals
from pandaSim.planning import clearance, se3
from pandaSim.planning.feasibility import Rollout
from pandaSim.planning.primitives import pivot_twist, screw_twist
from pandaSim.planning.sequencer import Box, Sequencer, Step, pose_error, snap_stable

BOX = Box((0.15, 0.06, 0.25), 5.0, 0.5)
DT = 0.005


def pose(x, y, yaw, tilt=0.0):
    R = se3.exp3(np.array([0.0, 0.0, yaw])) @ se3.exp3(np.array([0.0, tilt, 0.0]))
    return snap_stable(BOX, se3.rp(R, np.array([x, y, 0.0])))


T0 = pose(0.5, 0.0, 0.0, -math.pi / 2)            # lying on its 0.06 x 0.25 face
TG = pose(0.45, -0.15, 1.0)                        # standing: needs a slide and a pivot


@pytest.fixture(scope="module")
def robot():
    from pandaSim.geometry.utils import create_virtual_panda

    r = create_virtual_panda()
    r.base = sm.SE3(0, 0, 0.013)
    return r


@pytest.fixture(scope="module")
def plans(robot):
    return {free: Sequencer(Rollout(robot, free_rotation=free)).plan(BOX, T0, TG) for free in (True, False)}


def test_screw_twist_matches_pivot_twist_and_adjoint():
    T = se3.exp6(np.array([0.3, -0.2, 0.5, 0.4, 0.1, 0.3]))
    q, s = np.array([0.6, 0.1, 0.0]), np.array([0.0, 1.0, 0.0])
    S = np.r_[s, np.cross(q, s)]
    assert np.allclose(screw_twist(S, 1.2, T), pivot_twist(q, s, 1.2, T))
    S = np.array([0.0, 0.0, 1.0, 0.2, -0.1, 0.0])
    assert np.allclose(screw_twist(S, 0.7, T), se3.adjoint(se3.inv(T)) @ (S * 0.7))


def test_joint_move_reaches_goal_within_speed(robot):
    q0 = robot.qr[:7]
    goal = q0 + np.array([0.4, -0.2, 0.3, 0.1, -0.5, 0.2, 0.6])
    backend, log, jm = SimBackend(q0, dt=DT), Log(), JointMove(q0, goal, qd_max=0.5)
    ok, _ = run_segment(backend, jm, jm.ref, jm.Tf, log, "ready", done=jm.done)
    assert ok and np.abs(backend.theta() - goal).max() < 1e-3
    assert np.abs(log.arrays()["qd"][:, :7]).max() <= 0.5 * 1.02


@pytest.mark.parametrize("free", [True, False])
def test_planned_slide_and_pivot_run(robot, plans, free):
    plan = plans[free]
    assert plan.found and [s.kind for s in plan.steps] == ["slide", "pivot"]
    backend = SimBackend(robot.qr[:7], dt=DT)
    log, T_end = run_plan(backend, robot, plan.steps, T0, free_rotation=free)
    assert log.ok, log.segments
    assert [(s["step"], s["name"]) for s in log.segments] == [
        (i, n) for i, kind in enumerate(["slide", "pivot"]) for n in ("pre", "approach", kind, "retreat", "ready")]
    assert [e[0] for e in log.events] == ["open", "grasp", "release", "grasp", "release"]
    assert [e[3] for e in backend.gripper_events if e[0] == "grasp"] == [20.0, 20.0]
    assert [e[2] for e in backend.gripper_events if e[0] == "grasp"] == pytest.approx([s.plan.width for s in plan.steps])
    assert all(s["ok"] for s in log.steps)

    ep, er = pose_error(T_end, TG)
    print(f"free={free}: object {ep * 1e3:.2f} mm, {math.degrees(er):.3f} deg from the goal")
    assert ep < 3e-3 and er < math.radians(2.0)
    assert np.allclose(log.steps[0]["T_obj"], plan.steps[0].T_after, atol=3e-3)
    assert np.abs(backend.theta() - robot.qr[:7]).max() < 1e-3          # back at ready

    a = log.arrays()
    pivot = a["segment"] == 7
    assert (np.ptp(a["phi"][pivot]) > 0.5) == free                       # phi_hat moves only with the free model
    slide = np.isin(a["segment"], [0, 1, 2, 3])
    assert np.all(a["phi"][slide] == 0.0)                                # slides use the rigid model
    prof = check_profile(log)
    assert prof["vel_ok"] and np.all(np.abs(a["qd"][:, :7]) <= 0.8 * FER_QD_MAX + 1e-9)
    lo, hi = plan.steps[1].plan.phi_lim                                  # the palm stays clear of the box
    assert lo + 0.05 - 1e-6 <= a["phi"][a["segment"] >= 5].min() and a["phi"][a["segment"] >= 5].max() <= hi - 0.05 + 1e-6
    clear = check_clearance(log, robot, T0, BOX.size)
    assert clear["box_ok"] and clear["table_ok"], clear["hits"][:3]

    # replay bookkeeping gives the same final object pose
    held = grasp_intervals(a["t"], np.c_[a["theta"], a["phi"]], log.events, T0, lambda q: contact_pose(robot, q))
    assert len(held) == 2 and np.allclose(held[-1][3], T_end, atol=1e-4)
    # the viewer shows the fingers at the object width between grasp and release, open otherwise
    w = finger_widths(a["t"], log.events)
    for (k0, k1, _, _), st in zip(held, plan.steps):
        assert np.allclose(w[k0:k1], st.plan.width) and st.plan.width < 0.08
    assert w[0] == 0.08 and w[-1] == 0.08
    assert np.all(finger_widths(a["t"], [e[:3] for e in log.events]) == 0.08)   # logs without widths


def test_phi_limits_keep_the_palm_clear_of_the_box(robot):
    # pivot grasp: 0.02 m below the top face and 0.02 m inside the far side face
    seq = Sequencer(y_ref=np.array([0.0, 1.0, 0.0]))
    T = pose(0.5, 0.2, math.pi / 2, -math.pi / 2)
    (_, pa), (_, pb) = seq._box_pivots(BOX, T)
    T_c_obj = se3.inv(pa.T_grasp) @ T
    lo, hi = clearance.phi_limits(T_c_obj, BOX.size, clearance=0.0)
    # palm corner (0.0316, 0.0374) from the fingertips: over the top face at 25.7 deg, into the side face at 115.7 deg
    assert sorted(np.degrees([abs(lo), abs(hi)])) == pytest.approx([25.7, 115.7], abs=0.5)
    assert pa.phi_lim == clearance.phi_limits(T_c_obj, BOX.size) and pb.phi_lim == pytest.approx((-pa.phi_lim[1], -pa.phi_lim[0]))
    assert lo < pa.phi_lim[0] < 0 < pa.phi_lim[1] < hi                   # 5 mm clearance narrows the interval
    for phi, hit in ((0.0, False), (lo + 0.02, False), (lo - 0.02, True), (hi - 0.02, False), (hi + 0.02, True)):
        T_E = pa.T_grasp @ se3.rp(se3.exp3(np.array([0.0, -phi, 0.0])), np.zeros(3))
        assert clearance.palm_hits_box(T_E, T, BOX.size) == hit

    # without the limits the QP turns the palm into the box; with them the same pivot is clean
    step = Step("pivot", T, pa.T_goal, pa)
    for qp, ok in ((None, True), ({"phi_lim": None}, False)):
        log, _ = run_plan(SimBackend(robot.qr[:7], dt=DT), robot, [step], T, qp=qp)
        assert log.ok
        assert check_clearance(log, robot, T, BOX.size)["box_ok"] == ok


def test_unreachable_step_fails_safely_and_is_cached(robot):
    Ta, Tb = pose(0.95, 0.0, 0.0, -math.pi / 2), pose(0.95, 0.1, 0.0, -math.pi / 2)
    step = Step("slide", Ta, Tb, Sequencer(y_ref=(0.0, -1.0, 0.0))._slide(BOX, Ta, Tb))
    backend = SimBackend(robot.qr[:7], dt=DT)
    log, T_end = run_plan(backend, robot, [step], Ta)
    assert not log.ok and log.segments[-1]["cause"] in ("infeasible QP", "timeout")
    assert [e[0] for e in log.events] == ["open"] and np.all(backend._qd == 0)
    assert np.allclose(T_end, Ta) and not log.steps[0]["ok"]

    kin = Rollout(robot, dt=DT)
    assert kin.check(step) == (False, 0.0)
    kin._run = None                                                      # a second check must not run again
    assert kin.check(step) == (False, 0.0)


def test_no_feasible_candidate_gives_empty_plan():
    class Never:
        T_ready = np.diag([1.0, -1.0, -1.0, 1.0])

        def check(self, step):
            return False, 0.0

    plan = Sequencer(Never()).plan(BOX, T0, TG)
    assert not plan.found and plan.steps == []


def test_arm_collision_sees_every_link(robot):
    """ArmCollision (coal): a clear box, the box in the hand, and an arm link near the table."""
    pytest.importorskip("coal")
    from pandaSim.planning.collision import ArmCollision

    ac = ArmCollision(robot, BOX.size)
    q = robot.qr[:7]
    T_far = pose(0.5, 0.0, 0.0, -math.pi / 2)
    assert ac.ok(q, T_far)[0]
    T_E = contact_pose(robot, q)
    T_hit = T_far.copy()
    T_hit[:3, 3] = T_E[:3, 3] + np.array([0.0, 0.0, 0.12])        # the box where the hand is
    ok, who, d = ac.ok(q, T_hit)
    assert not ok and d < 0
    q_low = np.array([0.0, 1.7, 0.0, -0.5, 0.0, 2.0, 0.7854])      # the hand in the table
    ok, who, d = ac.ok(q_low, pose(-0.6, 0.0, 0.0, -math.pi / 2))
    assert not ok and d < 0


def test_planned_runs_are_clear_of_the_box_with_the_arm(robot, plans):
    pytest.importorskip("coal")
    from pandaSim.planning.collision import ArmCollision

    plan = plans[True]
    backend = SimBackend(robot.qr[:7], dt=DT)
    log, _ = run_plan(backend, robot, plan.steps, T0, free_rotation=True)
    c = check_clearance(log, robot, T0, BOX.size, arm=ArmCollision(robot, BOX.size))
    assert c["arm_ok"] and c["arm_min"] > 0.002
