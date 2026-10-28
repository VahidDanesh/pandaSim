"""QP of manuscript Sec. 6 (eq:qp) for the fingertip bearing (free rotation phi about n).

    min  1/2 ||theta_dot - K (theta_ctr - theta)||^2 - lam J_m^T theta_dot      (eq:qp_a)
    s.t. J_c^aug(q) q_dot = V_c                                                (eq:qp_b)
         S theta_dot <= eta (rho - rho_s) / (rho_e - rho_s)                    (eq:qp_c)
         -theta_dot_max <= theta_dot <= theta_dot_max                          (eq:qp_d)

Twists are ordered (omega, v) as in the manuscript; RTB Jacobians are (v, omega)
and are converted here only. With M_Ec = I_4, J_c^aug is the body Jacobian of
the virtual 8-DOF model at the virtual link, whose last column is B_c = (n, 0).
"""
from __future__ import annotations

import numpy as np
import quadprog

from pandaSim.geometry.utils import manipulability_jacobian

EE = "panda_finger_virtual"
OMEGA_V = [3, 4, 5, 0, 1, 2]  # RTB (v, omega) rows -> (omega, v)

# libfranka 0.8.0 rate_limiting.h: kMaxJointVelocity = limit - kLimitEps - 3 lost packets * 1 ms * kMaxJointAcceleration
_ACC_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
FER_QD_MAX = np.r_[[2.175] * 4, [2.61] * 3] - 1e-3 - 3 * 1e-3 * (_ACC_MAX - 1e-3)


def arm_manipulability_jacobian(ets, q8, Je):
    """J_m of w = sqrt(det(J_E J_E^T)) over the 7 arm joints (manuscript, below eq:qp).

    Uses the first 7 columns of the body Jacobian at {c}; with M_Ec = I they differ
    from J_E by a rotation about n, which leaves w unchanged.
    """
    return manipulability_jacobian(Je[:, :7], ets.hessiane(q8, Je=Je)[:7, :, :7])


def _qp(Q, c, A_eq, b_eq, G, h):
    """min 1/2 x'Qx + c'x  s.t.  A_eq x = b_eq, G x <= h.  Returns None if infeasible.

    quadprog solves min 1/2 x'Qx - a'x s.t. C'x >= b with the first meq rows as equalities.
    """
    C = np.vstack([A_eq, -G]).T
    b = np.r_[b_eq, -h]
    try:
        return quadprog.solve_qp(Q, -c, C, b, meq=len(b_eq))[0]
    except ValueError as e:
        if "inconsistent" in str(e):
            return None
        raise


class ContactQP:
    """Joint velocities that track the body twist V_c of the contact frame {c}.

    free_rotation=True:  solve(q=(theta, phi)) -> (theta_dot, phi_dot), J = J_c^aug
    free_rotation=False: solve(q=theta)        -> theta_dot,            J = J_E ({c} = {E}, rigid grasp)

    slack=None keeps eq:qp_b as a hard equality (returns None when infeasible).
    slack=beta relaxes it to J q_dot + delta = V_c with cost beta/2 ||delta||^2;
    the slack used is then V_c - J q_dot.
    """

    def __init__(self, robot, free_rotation=True, K=0.2, lam=0.2, nu=1e-3,
                 rho_e=0.3, rho_s=0.05, eta=1.0, qd_scale=0.8, theta_ctr=None, slack=None):
        if not rho_e > rho_s > 0:
            raise ValueError(f"need rho_e > rho_s > 0, got rho_e={rho_e}, rho_s={rho_s}")
        self.ets = robot.ets(end=EE)
        self.n = 8 if free_rotation else 7
        self.lb, self.ub = robot.qlim[0, :7], robot.qlim[1, :7]
        self.mid = (self.lb + self.ub) / 2
        self.theta_ctr = self.mid if theta_ctr is None else np.asarray(theta_ctr)
        self.K = np.broadcast_to(np.asarray(K, float), (7,))
        self.lam, self.rho_e, self.rho_s, self.eta = lam, rho_e, rho_s, eta
        self.qd_max = qd_scale * FER_QD_MAX
        self.slack = slack

        nx = self.n + (6 if slack else 0)
        self.Q = np.eye(nx)
        if free_rotation:
            self.Q[7, 7] = nu
        if slack:
            self.Q[self.n:, self.n:] *= slack
        self.bounds = np.vstack([np.eye(7, nx), -np.eye(7, nx)])

    def jacobian(self, q):
        """J_c in (omega, v): 6 x 8 (free) or 6 x 7 (rigid)."""
        return self._jacobians(q)[0]

    def _jacobians(self, q):
        q8 = q if self.n == 8 else np.r_[q, 0.0]
        Je = self.ets.jacobe(q8)
        return Je[OMEGA_V, : self.n], Je, q8

    def solve(self, q, V_c):
        J, Je, q8 = self._jacobians(np.asarray(q, float))
        theta = q8[:7]

        # eq:qp_a on theta only
        J_m = arm_manipulability_jacobian(self.ets, q8, Je)
        c = np.zeros(len(self.Q))
        c[:7] = self.K * (theta - self.theta_ctr) - self.lam * J_m

        # eq:qp_b, optionally with slack delta
        A_eq = np.hstack([J, np.eye(6)]) if self.slack else J

        # eq:rho, eq:damper_sign, eq:qp_c: rows only for joints within rho_e of a limit
        rho = np.minimum(self.ub - theta, theta - self.lb)
        near = rho < self.rho_e
        G_damp = np.zeros((near.sum(), len(self.Q)))
        G_damp[np.arange(near.sum()), np.flatnonzero(near)] = np.sign(theta - self.mid)[near]
        h_damp = self.eta * (rho[near] - self.rho_s) / (self.rho_e - self.rho_s)

        # eq:qp_d
        G = np.vstack([G_damp, self.bounds])
        h = np.r_[h_damp, self.qd_max, self.qd_max]

        x = _qp(self.Q, c, A_eq, np.asarray(V_c, float), G, h)
        return None if x is None else x[: self.n]
