"""Quasi-static load of the held box on the hand (design review D8, D14, D17).

What the hand must apply at the grasp point c so that the box moves as planned; what that asks of the table contact,
of the grasp and of the joints; how far the joint impedance lets the hand deflect under it. Plain functions of the
geometry; the pre-flight (experiments/run_box.py) and the wrench stop (control/primitive_executor.py) use them.

Conventions. World frame, z up, g = (0, 0, -9.81). (F, M) is the wrench the HAND applies to the BOX at c. The robot's
estimate of the external wrench at its TCP ({c} is the TCP of the Franka Hand) is -(F, M), and the joint torques it
explains are -J^T (F, M), J the geometric Jacobian of {c} with rows (v, w). The bearing passes no moment about the
finger axis n. The table carries the rest: R = -(F + m g), at the contact.

Contact of the box with the table:
  edge    a pivot, and the first and last screw of a gait step: the box turns about an edge (axis u, || n, length L).
          Moment about the edge:  u . ((c - q) x F) = -Q,   Q = u . ((com - q) x m g)   (M . u = 0: the bearing).
  vertex  screws 2 to 4 of a gait step: a point contact at v, so all three moments about v must balance:
          (c - v) x F + M = b,   b = -(com - v) x m g.
  face    the box lands at the end of a pivot (the table carries it): see face_ramp.
  slide   friction mu m g / A per unit area of the support face (uniform pressure), against the velocity of each
          point; the hand supplies the resultant force and its moment about the vertical through c. The moments
          about horizontal axes are left to the support (the planner's tipping margin, box_primitives.tip_margin).

The split between hand and table is statically indeterminate, so two bounds:
  (a) minimum norm. edge: the force alone, F = -Q t / |t|^2 with t = u x r, r = c - q (perpendicular to u), M = 0.
      vertex: M only for the part of b along e = r / |r| that no force at c can produce, M = (b . e) / (w . e) w
      with w = e made perpendicular to n; F = (b - M) x r / |r|^2.
  (b) closed chain. The hand hangs on the joint impedance, stiffness K = (J Kp^-1 J^T)^-1 at c (6x6, (v, w)); the box
      moves only as the contact lets it (edge: rotation about u; vertex: any rotation about v), and the hand turns
      freely about n. With A = [(a x (c - p), a) for each allowed axis a] + [(0, n)] and Q_a = a . ((com - p) x m g):
          A^T K A alpha = (Q_a, 0),   x = A alpha  (hand displacement from its target),   (F, M) = -K x.
      At an edge this is the stiffest case: the box holds the hand's rotations perpendicular to n, with a moment the
      edge line contact must react (at most N L / 2 before the box rolls onto a corner of the edge).
Deflection of the hand (reported, not fed forward: design review D8): closed chain |x_v|; free moments: the hand
loaded by the bound (a) wrench with every rotation free, |(C (F, M))_v|, C = J Kp^-1 J^T.
"""
from __future__ import annotations

import math

import numpy as np

from pandaSim.planning import se3
from pandaSim.planning.box_primitives import GRIP_SAFETY, PAD_MU

G = 9.81
GRAVITY = np.array([0.0, 0.0, -G])
Z = np.array([0.0, 0.0, 1.0])
BAND_N, BAND_FRAC = 5.0, 0.20       # band of the predicted force: +-(5 N + 20 %) (design review D14)


def band(F):
    """Half-width [N] of the band around a predicted force."""
    return BAND_N + BAND_FRAC * float(np.linalg.norm(F))


def _perp(r, u):
    return r - (r @ u) * u


def edge_force_only(c, com, q, u, m):
    """Bound (a) at an edge (point q, unit axis u): the smallest force at c, no moment."""
    r = _perp(c - q, u)
    t = np.cross(u, r)
    Q = u @ np.cross(com - q, m * GRAVITY)
    return -Q * t / (t @ t)


def vertex_min_wrench(c, com, v, n, m):
    """Bound (a) at a vertex v: (F, M), M perpendicular to n only where a force at c cannot do it."""
    r = c - v
    b = -np.cross(com - v, m * GRAVITY)
    e = r / np.linalg.norm(r)
    w = _perp(e, n)
    if np.linalg.norm(w) < 1e-9:                          # n along c - v: that moment is out of reach of the hand
        return np.full(3, np.nan), np.full(3, np.nan)
    w = w / np.linalg.norm(w)
    M = (b @ e) / (w @ e) * w
    return np.cross(b - M, r) / (r @ r), M


def closed_chain(K, c, com, p, axes, n, m):
    """Bound (b): (F, M, x) with the box turning about the unit axes through p that the contact allows."""
    A = np.column_stack([np.r_[np.cross(a, c - p), a] for a in axes] + [np.r_[np.zeros(3), n]])
    Q = np.r_[[a @ np.cross(com - p, m * GRAVITY) for a in axes], 0.0]
    x = A @ np.linalg.solve(A.T @ K @ A, Q)
    W = -K @ x
    return W[:3], W[3:], x


def slide_wrench(T_obj, size, S, c, m, mu, n_grid=24):
    """(F, M) of the hand in a slide along the screw S (omega, v): uniform pressure on the support face, friction
    against the velocity of each point. M is about the vertical through c."""
    half = np.asarray(size, float) / 2
    R, p = T_obj[:3, :3], T_obj[:3, 3]
    i = int(np.argmax(np.abs(R[2])))                      # object axis that is vertical
    j, k = [a for a in range(3) if a != i]
    s = (np.arange(n_grid) + 0.5) / n_grid * 2 - 1        # cell centres in (-1, 1)
    a, b = np.meshgrid(s * half[j], s * half[k])
    pts = np.zeros((a.size, 3))
    pts[:, i], pts[:, j], pts[:, k] = -np.sign(R[2, i]) * half[i], a.ravel(), b.ravel()
    P = pts @ R.T + p                                     # world points of the support face
    V = S[3:] + np.cross(S[:3], P)
    V[:, 2] = 0.0
    speed = np.linalg.norm(V, axis=1)
    f = np.zeros_like(V)
    ok = speed > 1e-9
    f[ok] = -mu * m * G / len(P) * V[ok] / speed[ok, None]   # friction of the table on each cell
    F_fric = f.sum(axis=0)
    M_fric = (np.cross(P - c, f).sum(axis=0) @ Z) * Z
    return -F_fric, -M_fric


def table_reaction(F, m):
    """Force of the table on the box."""
    return -(np.asarray(F, float) + m * GRAVITY)


def friction_needed(F, m, dirs=(), b=0.0):
    """|R_t| / R_n of the table reaction, worst over an internal force of size b along the unit directions dirs
    (the hand force plus d, d in the span of dirs, |d| <= b). inf if the box would lift off the table."""
    R0 = table_reaction(F, m)
    D = [np.zeros(3)]
    if b > 0 and len(dirs):
        E = np.column_stack(dirs)
        ang = np.linspace(0, 2 * math.pi, 73)[:-1]
        D += [b * E @ (np.array([math.cos(t), math.sin(t)])[:E.shape[1]] if E.shape[1] == 2 else np.array([math.cos(t)]))
              for t in ang]
    worst = 0.0
    for d in D:
        R = R0 - d
        if R[2] <= 1e-9:
            return math.inf
        worst = max(worst, float(np.hypot(R[0], R[1]) / R[2]))
    return worst


def edge_moment(F, M, c, com, p, m):
    """Moment the table must apply to the box about the point p on the contact edge."""
    return -(np.cross(c - p, F) + M + np.cross(com - p, m * GRAVITY))


def pad_use(F, M, n, width, f_g, mu_pad=PAD_MU):
    """Share of the friction of the two pads (width apart along n, squeeze f_g each) that the hand's force and
    moment perpendicular to n use: |F_t| / (2 mu f_g) + |M_t| / (mu f_g width). 1 = slip; the pre-flight allows
    GRIP_SAFETY (0.5)."""
    F_t, M_t = _perp(np.asarray(F, float), n), _perp(np.asarray(M, float), n)
    return float(np.linalg.norm(F_t) / (2 * mu_pad * f_g) + np.linalg.norm(M_t) / (mu_pad * f_g * width))


def face_ramp(theta, theta_end, d_theta, starts_on_face, ends_on_face):
    """Fraction of the edge load the hand carries (the rest is on the face of the box). The hand pulls the box off
    the face only after its target is d_theta ahead (the deflection), and the box lands d_theta before the target
    ends, after which the hand's load falls to zero at the end."""
    if d_theta <= 0:
        return 1.0
    f = 1.0
    if starts_on_face:
        f = min(f, abs(theta) / d_theta)
    if ends_on_face:
        f = min(f, abs(theta_end - theta) / d_theta)
    return float(min(f, 1.0))


# ---- the contact along a planned step ----------------------------------------------------------------------------
def contact(step, k, T_obj, size):
    """Contact of the box with the table during screw k of a planner step, at the box pose T_obj:
    dict(kind=edge, S, theta, p, u, L, starts_on_face, ends_on_face) | dict(kind=vertex, S, theta, p) |
    dict(kind=slide, S, theta)."""
    S, theta = step.plan.screws[k]
    S = np.asarray(S, float)
    if step.kind == "slide":
        return dict(kind="slide", S=S, theta=float(theta))
    if step.kind == "gait" and k in (1, 2, 3):
        return dict(kind="vertex", S=S, p=np.asarray(step.plan.vertex, float), theta=float(theta))
    u = S[:3] / np.linalg.norm(S[:3])
    q = np.cross(u, S[3:])                                # point of the axis nearest the origin
    p = q + ((T_obj[:3, 3] - q) @ u) * u                  # the middle of the edge (below the centre of mass)
    L = float(2 * np.abs(T_obj[:3, :3].T @ u) @ (np.asarray(size, float) / 2))
    last = len(step.plan.screws) - 1
    return dict(kind="edge", S=S, p=p, u=u, L=L, theta=float(theta), starts_on_face=(k == 0), ends_on_face=(k == last))


def screw_angle(S, c0, c):
    """Angle [rad] by which the grasp point has turned from c0 to c about the axis of the screw S (a pivot or a gait
    screw: the box turns with the hand about it), so that the box pose follows from the hand position (phi is not
    measured)."""
    u = S[:3] / np.linalg.norm(S[:3])
    p = np.cross(u, S[3:])
    r0, r = _perp(c0 - p, u), _perp(c - p, u)
    return math.atan2(u @ np.cross(r0, r), r0 @ r)


def wrench(ct, T_obj, T_c, size, m, mu=0.5, C=None):
    """Both bounds of the hand's wrench at c for the contact ct (contact()) at the box pose T_obj, {c} at T_c.
    C: compliance J Kp^-1 J^T of {c} (6x6, (v, w)); without it only bound (a). Returns dict(F_a, M_a; with C also
    F_b, M_b, x_b (closed-chain hand displacement), x_free (free-moment displacement under the bound (a) wrench))."""
    c, n, com = T_c[:3, 3], T_c[:3, 1], T_obj[:3, 3]
    out = {}
    if ct["kind"] == "slide":
        F, M = slide_wrench(T_obj, size, ct["S"], c, m, mu)
        out.update(F_a=F, M_a=M, F_b=F, M_b=M)
        if C is not None:
            out.update(x_b=-(C @ np.r_[F, M]), x_free=-(C @ np.r_[F, M]))
        return out
    if ct["kind"] == "edge":
        out.update(F_a=edge_force_only(c, com, ct["p"], ct["u"], m), M_a=np.zeros(3))
        axes = [ct["u"]]
    else:
        F, M = vertex_min_wrench(c, com, ct["p"], n, m)
        out.update(F_a=F, M_a=M)
        axes = list(np.eye(3))
    if C is not None:
        F, M, x = closed_chain(np.linalg.inv(C), c, com, ct["p"], axes, n, m)
        out.update(F_b=F, M_b=M, x_b=x, x_free=-(C @ np.r_[out["F_a"], out["M_a"]]))
    return out
