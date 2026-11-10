"""Planner steps through the executor in simulation: run_plan, JointMove, the rollout check, replay bookkeeping."""
import math

import numpy as np
import pytest
import spatialmath as sm

from pandaSim.control.backends import SimBackend
from pandaSim.control.contact_qp import FER_QD_MAX
from pandaSim.control.primitive_executor import JointMove, Log, check_profile, contact_pose, run_plan, run_segment
from pandaSim.control.viewer import grasp_intervals
from pandaSim.planning import se3
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

    # replay bookkeeping gives the same final object pose
    held = grasp_intervals(a["t"], np.c_[a["theta"], a["phi"]], log.events, T0, lambda q: contact_pose(robot, q))
    assert len(held) == 2 and np.allclose(held[-1][3], T_end, atol=1e-4)


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
