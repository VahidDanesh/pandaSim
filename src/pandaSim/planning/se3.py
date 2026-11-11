"""Small SE(3) helpers for the planner (numpy, Lynch & Park conventions).

Twists and screw axes are ordered (omega, v). A screw axis S in the world frame
moves any attached frame by T(theta) = exp([S] theta) T(0).
"""
from __future__ import annotations

import math

import numpy as np

_EPS = 1e-9


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def rp(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def inv(T):
    R, p = T[:3, :3], T[:3, 3]
    return rp(R.T, -R.T @ p)


def _coeffs(theta):
    """sin(t)/t, (1 - cos t)/t^2, (t - sin t)/t^3, with Taylor series near 0."""
    if theta < 1e-4:
        t2 = theta * theta
        return 1 - t2 / 6, 0.5 - t2 / 24, 1.0 / 6 - t2 / 120
    return math.sin(theta) / theta, (1 - math.cos(theta)) / theta**2, (theta - math.sin(theta)) / theta**3


def exp3(w):
    """Rotation vector omega*theta -> R (Rodrigues)."""
    W = skew(w)
    a, b, _ = _coeffs(float(np.linalg.norm(w)))
    return np.eye(3) + a * W + b * (W @ W)


def log3(R):
    """R -> rotation vector omega*theta, theta in [0, pi]."""
    cos_t = min(max((np.trace(R) - 1) / 2, -1.0), 1.0)
    theta = math.acos(cos_t)
    sk = 0.5 * (R - R.T)
    s = np.array([sk[2, 1], sk[0, 2], sk[1, 0]])              # sin(theta) * omega_hat
    if theta > math.pi - 1e-2:                                # near pi: omega omega^T = (sym(R) - cos I)/(1 - cos)
        M = (0.5 * (R + R.T) - cos_t * np.eye(3)) / max(1 - cos_t, _EPS)
        k = int(np.argmax(np.diag(M)))
        w_hat = M[:, k] / math.sqrt(max(M[k, k], _EPS))
        return (-theta if w_hat @ s < 0 else theta) * w_hat
    return (1 + theta**2 / 6 if theta < 1e-4 else theta / math.sin(theta)) * s


def exp6(V):
    """Exponential coordinates S*theta (omega, v) -> T."""
    W, u = skew(V[:3]), np.asarray(V[3:], float)
    a, b, c = _coeffs(float(np.linalg.norm(V[:3])))
    W2 = W @ W
    return rp(np.eye(3) + a * W + b * W2, (np.eye(3) + b * W + c * W2) @ u)


def log6(T):
    """T -> exponential coordinates S*theta (omega, v)."""
    w = log3(T[:3, :3])
    theta = float(np.linalg.norm(w))
    W = skew(w)
    coef = 1.0 / 12 + theta**2 / 720 if theta < 1e-4 else (1 - (theta / 2) / math.tan(theta / 2)) / theta**2
    return np.r_[w, (np.eye(3) - 0.5 * W + coef * (W @ W)) @ T[:3, 3]]


def adjoint(T):
    """[Ad_T] for twists ordered (omega, v)."""
    R, p = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3], A[3:, 3:], A[3:, :3] = R, R, skew(p) @ R
    return A


def axis_angle(V):
    """Exponential coordinates S*theta -> (S, theta); theta is the distance for a pure translation."""
    theta = float(np.linalg.norm(V[:3]))
    if theta < _EPS:
        theta = float(np.linalg.norm(V[3:]))
    return np.asarray(V, float) / max(theta, _EPS), theta


def angle_between(R1, R2):
    return float(np.linalg.norm(log3(R1.T @ R2)))
