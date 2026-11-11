"""The code against the manuscript's mathematics, with independent implementations written from the paper.

Sec. 2: dual quaternions (D_T = Q_R + eps 1/2 Q_p Q_R, eq:screw_dq, ScLERP, V = 2 D^-1 D_dot, eq:twist_dq/twist_vec).
Sec. 6: the QP (eq:qp_a - eq:qp_d, eq:rho, eq:damper_sign, eq:J_aug), built literally with the diagonal matrix S
and a row for every joint, and solved with a second call to quadprog.
"""
import numpy as np
import pytest
import quadprog
from scipy.spatial.transform import Rotation

from pandaSim.control.contact_qp import EE, FER_QD_MAX, OMEGA_V, ContactQP
from pandaSim.planning import se3
from pandaSim.planning.primitives import pivot_twist, planned_pose, screw_twist

rng = np.random.default_rng(7)


# ---- dual quaternions as in Sec. 2 (arrays (p0, p1, p2, p3, q0, q1, q2, q3)) ---------------------------------
def qmul(a, b):
    a0, av, b0, bv = a[0], a[1:], b[0], b[1:]
    return np.r_[a0 * b0 - av @ bv, a0 * bv + b0 * av + np.cross(av, bv)]


def qconj(a):
    return np.r_[a[0], -a[1:]]


def dmul(s1, s2):                      # sigma1 sigma2 = p1 p2 + eps (p1 q2 + q1 p2)
    return np.r_[qmul(s1[:4], s2[:4]), qmul(s1[:4], s2[4:]) + qmul(s1[4:], s2[:4])]


def dconj(s):
    return np.r_[qconj(s[:4]), qconj(s[4:])]


def dq_from_T(T):                      # D_T = Q_R + eps 1/2 Q_p Q_R
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    QR = np.array([w, x, y, z])
    return np.r_[QR, 0.5 * qmul(np.r_[0.0, T[:3, 3]], QR)]


def T_from_dq(D):
    p, q = D[:4], D[4:]
    R = Rotation.from_quat([*p[1:], p[0]]).as_matrix()
    return se3.rp(R, 2 * qmul(q, qconj(p))[1:])


def screw_dq(theta, d, u, m):          # eq:screw_dq
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.r_[c, s * u, -d / 2 * s, d / 2 * c * u + s * m]


def random_pose():
    return se3.rp(Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix(), rng.uniform(-0.5, 0.5, 3))


def screw_parameters(T12):
    """(theta, d, u, m) of the displacement T12, from the code's log: S = (u, m + h u), d = h theta."""
    S, theta = se3.axis_angle(se3.log6(T12))
    u, v = S[:3], S[3:]
    return theta, theta * (u @ v), u, v - (u @ v) * u


def same_dq(a, b, tol=1e-9):
    return min(np.abs(a - b).max(), np.abs(a + b).max()) < tol        # D and -D are the same pose


def test_unit_dual_quaternions_and_the_product_of_poses():
    for _ in range(5):
        T1, T2 = random_pose(), random_pose()
        D1, D2 = dq_from_T(T1), dq_from_T(T2)
        assert D1[:4] @ D1[:4] == pytest.approx(1) and D1[:4] @ D1[4:] == pytest.approx(0, abs=1e-12)   # unit
        assert same_dq(dmul(D1, dconj(D1)), np.r_[1.0, np.zeros(7)])                                     # D D* = 1
        assert np.allclose(T_from_dq(D1), T1)
        assert np.allclose(T_from_dq(dmul(D1, D2)), T1 @ T2)


def test_screw_displacement_equation_is_the_relative_pose():
    for _ in range(5):
        T1, T2 = random_pose(), random_pose()
        D12 = dmul(dconj(dq_from_T(T1)), dq_from_T(T2))                  # D12 = D1* D2
        assert same_dq(screw_dq(*screw_parameters(se3.inv(T1) @ T2)), D12)


def test_sclerp_is_the_screw_path_of_the_code():
    for _ in range(3):
        T1, T2 = random_pose(), random_pose()
        theta, d, u, m = screw_parameters(se3.inv(T1) @ T2)
        X = se3.log6(se3.inv(T1) @ T2)
        for s in (0.0, 0.25, 0.6, 1.0):
            D = dmul(dq_from_T(T1), screw_dq(s * theta, s * d, u, m))   # D(s) = D1 D12(s)
            assert np.allclose(T_from_dq(D), T1 @ se3.exp6(X * s), atol=1e-9)


def test_pivot_pose_and_the_special_cases_of_eq_D12s():
    T0 = random_pose()
    q, u = rng.uniform(-0.3, 0.3, 3), rng.normal(size=3)
    u /= np.linalg.norm(u)
    theta = 1.1
    for s in (0.3, 1.0):                                                  # h = 0: D12(s) = (cos, sin u) + eps sin m
        Rw = Rotation.from_rotvec(u * theta * s).as_matrix()
        p_screw = np.r_[1.0, np.zeros(7)]
        # the world-frame screw conjugated into the frame of T0 gives the body screw (u_b, m_b)
        u_b = T0[:3, :3].T @ u
        m_b = np.cross(T0[:3, :3].T @ (q - T0[:3, 3]), u_b)
        D = dmul(dq_from_T(T0), screw_dq(theta * s, 0.0, u_b, m_b))
        assert np.allclose(T_from_dq(D), planned_pose(q, u, theta, T0, s), atol=1e-9)
        assert np.allclose(T_from_dq(D)[:3, :3], Rw @ T0[:3, :3], atol=1e-9)
    # h = infinity: 1 + eps (s d / 2) u, a translation by s d along u
    D = dmul(dq_from_T(T0), np.r_[1.0, np.zeros(3), 0.0, 0.5 * 0.2 * u_b])
    assert np.allclose(T_from_dq(D)[:3, 3], T0[:3, 3] + T0[:3, :3] @ (0.2 * u_b), atol=1e-9)


def test_body_twist_is_2_Dinv_Ddot_and_eq_twist_vec():
    T1, T2 = random_pose(), random_pose()
    theta, d, u, m = screw_parameters(se3.inv(T1) @ T2)
    V_paper = np.r_[theta * u, d * u + theta * m]                         # eq:twist_vec with s_dot = 1
    D = lambda s: dmul(dq_from_T(T1), screw_dq(s * theta, s * d, u, m))
    for s in (0.2, 0.7):
        h = 1e-6
        Ddot = (D(s + h) - D(s - h)) / (2 * h)
        Vbar = 2 * dmul(dconj(D(s)), Ddot)                                # V = 2 D^-1 D_dot, a pure dual quaternion
        assert abs(Vbar[0]) < 1e-6 and abs(Vbar[4]) < 1e-6
        assert np.allclose(np.r_[Vbar[1:4], Vbar[5:8]], V_paper, atol=1e-6)
    # the code: the body twist of the frame moving with the world screw of the same displacement
    S_w, th_w = se3.axis_angle(se3.log6(T2 @ se3.inv(T1)))
    assert np.allclose(screw_twist(S_w, th_w, T1), V_paper, atol=1e-9)


def test_pivot_twist_is_eq_twist_vec_with_pitch_zero():
    T0 = random_pose()
    q, u = rng.uniform(-0.3, 0.3, 3), rng.normal(size=3)
    u /= np.linalg.norm(u)
    theta = -0.8
    u_b = T0[:3, :3].T @ u
    m_b = np.cross(T0[:3, :3].T @ (q - T0[:3, 3]), u_b)                   # m = r x u in {c}
    assert np.allclose(pivot_twist(q, u, theta, T0), np.r_[theta * u_b, theta * m_b])


# ---- the QP of Sec. 6, built literally ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def robot():
    from pandaSim.geometry.utils import create_virtual_panda

    return create_virtual_panda()


def paper_qp(robot, q, V, K, lam, nu, rho_e, rho_s, eta, qd_scale, free):
    """eq:qp_a - eq:qp_d with S = diag(s_j) and rows for all joints; J_m by finite differences of w."""
    ets, n = robot.ets(end=EE), 8 if free else 7
    lb, ub = robot.qlim[0, :7], robot.qlim[1, :7]
    theta = q[:7]
    bar = (ub + lb) / 2                                                  # central position
    w = lambda th: np.sqrt(np.linalg.det((lambda J: J @ J.T)(ets.jacobe(np.r_[th, 0.0])[:, :7])))
    Jm = np.array([(w(theta + 1e-6 * e) - w(theta - 1e-6 * e)) / 2e-6 for e in np.eye(7)])
    Je = ets.jacobe(np.asarray(q if free else np.r_[q, 0.0], float))
    J = Je[OMEGA_V, :n]
    # 1/2 ||qd - K (bar - theta)||^2 - lam Jm^T qd  ->  Q = diag(I, nu), C = [K (theta - bar) - lam Jm; 0]
    Q = np.eye(n)
    if free:
        Q[7, 7] = nu
    C = np.zeros(n)
    C[:7] = K * (theta - bar) - lam * Jm
    rho = np.minimum(ub - theta, theta - lb)                              # eq:rho
    s = np.where(rho < rho_e, np.sign(theta - bar), 0.0)                  # eq:damper_sign
    S = np.zeros((7, n))
    S[:, :7] = np.diag(s)
    rhs = eta * (rho - rho_s) / (rho_e - rho_s)                           # eq:qp_c
    qd_max = qd_scale * FER_QD_MAX
    G = np.vstack([S, np.eye(7, n), -np.eye(7, n)])
    h = np.r_[rhs, qd_max, qd_max]
    sol = quadprog.solve_qp(Q, -C, np.vstack([J, -G]).T, np.r_[V, -h], meq=6)[0]
    return sol, J


@pytest.mark.parametrize("free", [True, False])
@pytest.mark.parametrize("case", ["interior", "near_upper", "near_lower", "two_joints"])
def test_contact_qp_equals_the_paper_formulation(robot, free, case):
    lb, ub = robot.qlim[0, :7], robot.qlim[1, :7]
    theta = robot.qr[:7] + np.r_[0.1, -0.2, 0.15, 0.3, -0.1, 0.2, -0.3]
    if case == "near_upper":
        theta[3] = ub[3] - 0.12
    if case == "near_lower":
        theta[1] = lb[1] + 0.08
    if case == "two_joints":
        theta[[0, 5]] = [ub[0] - 0.2, lb[5] + 0.1]
    q = np.r_[theta, 0.4] if free else theta
    pars = dict(K=0.2, lam=0.2, nu=1e-3, rho_e=0.3, rho_s=0.05, eta=1.0, qd_scale=0.8)
    ctrl = ContactQP(robot, free_rotation=free, **pars)
    for _ in range(3):
        V = ctrl.jacobian(q) @ rng.uniform(-0.4, 0.4, ctrl.n)            # a twist the QP can realise
        want, J = paper_qp(robot, q, V, free=free, **pars)
        got = ctrl.solve(q, V)
        assert np.allclose(J @ got, V, atol=1e-8)
        assert np.allclose(got, want, atol=1e-6), np.abs(got - want).max()
