"""ContactQP (manuscript Sec. 6). Twists are (omega, v)."""
import numpy as np
import pytest
import spatialmath as sm
import spatialgeometry as sg
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt

from pandaSim.control.contact_qp import EE, FER_QD_MAX, ContactQP, arm_manipulability_jacobian
from pandaSim.planning.primitives import lifting_angle, pivot_twist, planned_pose


@pytest.fixture(scope="module")
def robot():
    from pandaSim.geometry.utils import create_virtual_panda

    return create_virtual_panda()


def q_test(robot, phi=0.4):
    q = robot.qr + np.r_[0.1, -0.2, 0.15, 0.3, -0.1, 0.2, -0.3, 0.0]
    q[7] = phi
    return q


def test_jacobian_is_the_body_jacobian_of_c(robot):
    ctrl, q = ContactQP(robot), q_test(robot)
    J, ets = ctrl.jacobian(q), robot.ets(end=EE)
    qd, eps = np.linspace(-0.5, 0.6, 8), 1e-6
    V_fd = pt.exponential_coordinates_from_transform(np.linalg.inv(ets.eval(q)) @ ets.eval(q + eps * qd)) / eps
    assert np.allclose(J @ qd, V_fd, atol=1e-5)
    assert np.allclose(J[:, 7], [0, 1, 0, 0, 0, 0])                      # B_c = (n, 0), n = y

    # eq:J_aug: [Ad_{T_cE}(phi) J_E | B_c] with J_E the Jacobian at phi = 0 (M_Ec = I)
    J_E = ctrl.jacobian(np.r_[q[:7], 0.0])[:, :7]
    T_Ec = pt.transform_from(pr.active_matrix_from_angle(1, q[7]), np.zeros(3))
    assert np.allclose(pt.adjoint_from_transform(np.linalg.inv(T_Ec)) @ J_E, J[:, :7])


def test_manipulability_gradient_is_of_the_arm_only(robot):
    ets = robot.ets(end=EE)

    def w(theta, phi):
        J = ets.jacobe(np.r_[theta, phi])[:, :7]
        return np.sqrt(np.linalg.det(J @ J.T))

    q, eps = q_test(robot), 1e-6
    Jm = arm_manipulability_jacobian(ets, q, ets.jacobe(q))
    fd = np.array([(w(q[:7] + eps * e, q[7]) - w(q[:7] - eps * e, q[7])) / (2 * eps) for e in np.eye(7)])
    assert np.allclose(Jm, fd, atol=1e-6)
    assert w(q[:7], 0.0) == pytest.approx(w(q[:7], 2.0))                  # independent of phi


def test_tracks_twist_within_bounds(robot):
    for free in (True, False):
        ctrl = ContactQP(robot, free_rotation=free)
        q = q_test(robot) if free else q_test(robot)[:7]
        V = np.array([0.3, -0.2, 0.4, 0.15, -0.1, 0.2])
        qd = ctrl.solve(q, V)
        assert qd.shape == (8 if free else 7,)
        assert np.allclose(ctrl.jacobian(q) @ qd, V, atol=1e-8)
        assert np.all(np.abs(qd[:7]) <= ctrl.qd_max + 1e-9)


def test_velocity_limits_are_libfranka_times_scale(robot):
    assert np.allclose(FER_QD_MAX, [2.129, 2.1515, 2.144, 2.1365, 2.564, 2.549, 2.549], atol=1e-4)
    assert np.allclose(ContactQP(robot).qd_max, 0.8 * FER_QD_MAX)


def test_damper_limits_approach_to_joint_limit(robot):
    ctrl = ContactQP(robot, eta=1.0, rho_e=0.3, rho_s=0.05)
    q = q_test(robot)
    V = ctrl.jacobian(q)[:, 3] * 1.0                                      # joint 4 at 1 rad/s
    assert ctrl.solve(q, V)[3] > 0.5                                      # far from limits: free to move
    q[3] = robot.qlim[1, 3] - 0.1                                         # rho = 0.1 < rho_e, upper side
    V = ctrl.jacobian(q)[:, 3] * 1.0
    qd = ctrl.solve(q, V)
    assert qd[3] <= 1.0 * (0.1 - 0.05) / (0.3 - 0.05) + 1e-9
    assert np.allclose(ctrl.jacobian(q) @ qd, V, atol=1e-8)


def test_phi_is_unbounded_and_free_without_the_bound(robot):
    ctrl = ContactQP(robot)                                               # phi_lim=None by default
    q = q_test(robot, phi=3.1)                                            # beyond the URDF's +-pi soon
    V = 5.0 * ctrl.jacobian(q)[:, 7]                                      # spin about n faster than any joint
    qd = ctrl.solve(q, V)
    assert qd[7] > 4.5                                                    # theta moves a little (centering)
    assert np.allclose(ctrl.jacobian(q) @ qd, V, atol=1e-8)


def test_phi_limits_move_the_rotation_to_the_arm(robot):
    ctrl = ContactQP(robot, eta=1.0, rho_e=0.3, rho_s=0.05, phi_lim=(-0.5, 2.0))
    assert ContactQP(robot, free_rotation=False, phi_lim=(-0.5, 2.0)).phi_lim is None
    for sign, lim in ((1.0, 2.0), (-1.0, -0.5)):
        q = q_test(robot, phi=1.0)                                        # far from both limits: phi takes the spin
        V = sign * 0.5 * ctrl.jacobian(q)[:, 7]
        assert sign * ctrl.solve(q, V)[7] > 0.4
        q = q_test(robot, phi=lim - sign * 0.1)                           # 0.1 rad from the limit
        V = sign * 0.5 * ctrl.jacobian(q)[:, 7]
        qd = ctrl.solve(q, V)
        assert sign * qd[7] <= 1.0 * (0.1 - 0.05) / (0.3 - 0.05) + 1e-9   # same damper as the joints
        assert np.linalg.norm(qd[:7]) > 0.2                               # the arm turns the hand instead
        assert np.allclose(ctrl.jacobian(q) @ qd, V, atol=1e-8)           # the twist is still tracked exactly
        assert sign * ctrl.solve(q, -V)[7] < -0.4                         # moving away from the limit is free


def test_infeasible_and_slack(robot):
    q, V = q_test(robot)[:7], np.array([3.0, 0, 0, 2.0, 0, 0])            # far beyond joint speed limits
    assert ContactQP(robot, free_rotation=False).solve(q, V) is None
    ctrl = ContactQP(robot, free_rotation=False, slack=100.0)
    qd = ctrl.solve(q, V)
    delta = V - ctrl.jacobian(q) @ qd
    assert qd.shape == (7,) and np.all(np.abs(qd) <= ctrl.qd_max + 1e-9)
    assert 0 < np.linalg.norm(delta) < np.linalg.norm(V)


def test_rejects_bad_damper_distances(robot):
    with pytest.raises(ValueError):
        ContactQP(robot, rho_e=0.3, rho_s=0.5)


# --- closed loop: pivot a cuboid by 90 deg from a grasp, Euler integration at 1 kHz ---

def pivot_run(robot, free, slack=None, Tf=5.0, dt=1e-3):
    from pandaSim.geometry.rtb_adapter import RoboticsToolboxAdapter
    from pandaSim.planning.primitives import time_scaling
    from pandaSim.planning.screw_motion_planner import ScrewMotionPlanner

    adapter = RoboticsToolboxAdapter({"viewer": None})
    box_size = [0.27, 0.065, 0.207]
    box = sg.Cuboid(scale=box_size, pose=sm.SE3(0.5, 0.0, box_size[2] / 2))
    grasp_pose, qs, s_axes = ScrewMotionPlanner(adapter).compute_grasp(
        box, adapter, grasp_height="top", prefer_closer_grasp=True, gripper_offset=sm.SE3.Rx(np.pi).A)
    q_axis, s_axis = qs.reshape(-1, 3)[0], s_axes.reshape(-1, 3)[0]
    theta = lifting_angle(q_axis, s_axis, np.pi / 2, box.T[:3, 3])

    ets = robot.ets(end=EE)
    sol = robot.ik_LM(grasp_pose, end=EE, q0=robot.qr)
    assert sol.success, "IK to the grasp pose failed"
    q8 = np.asarray(sol.q)
    q = q8 if free else q8[:7]
    ctrl = ContactQP(robot, free_rotation=free, slack=slack)
    fk = lambda q: ets.eval(q if free else np.r_[q, 0.0])

    T_wc0 = fk(q)
    V1 = pivot_twist(q_axis, s_axis, theta, T_wc0)
    for k in range(int(round(Tf / dt))):
        qd = ctrl.solve(q, time_scaling(k * dt, Tf)[1] * V1)
        if qd is None:
            return dict(ok=False, s=time_scaling(k * dt, Tf)[0])
        q = q + qd * dt
    E = np.linalg.inv(planned_pose(q_axis, s_axis, theta, T_wc0, 1.0)) @ fk(q)
    margin = np.minimum(robot.qlim[1, :7] - q[:7], q[:7] - robot.qlim[0, :7]).min()
    return dict(ok=True, pos_err=np.linalg.norm(E[:3, 3]), rot_err=np.degrees(pr.axis_angle_from_matrix(E[:3, :3])[3]),
                margin=margin)


def test_free_rotation_pivot_reaches_planned_pose(robot):
    r = pivot_run(robot, free=True)
    print("free:", r)
    assert r["ok"] and r["pos_err"] < 1e-3 and r["rot_err"] < 0.1 and r["margin"] > 0.05


@pytest.mark.parametrize("slack", [None, 100.0])
def test_rigid_pivot_outcome_is_reported(robot, slack):
    print(f"rigid slack={slack}:", pivot_run(robot, free=False, slack=slack))
