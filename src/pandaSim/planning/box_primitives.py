"""Pivot and slide of a box on a table, as constant screw motions in the world (table) frame.

Pivot: rotation by theta about a bottom edge, S = (e, -e x q). Grasp near the top on the side
opposite the edge, fingers closing along the edge direction e (the bearing axis).
Slide: planar displacement with the box flat on the table. Any planar displacement is one screw
(rotation about a vertical axis, or a translation). Grasp top-down at the box centre.

The grasp frame {c} has z down, y along the finger closing (bearing) axis, x = y x z.
During a primitive {c} moves with the object: T_wc(s) = exp([S] theta s) T_grasp.

Tipping check of a slide (quasi-static): table friction acts h_g below the grasp, so dragging the
box across the bearing axis is stable only if mu * h_g < d_lead (distance from the grasp to the
leading edge of the support). If the direct screw fails, the slide is a chain of tip-free screws
in one grasp: rotate about the vertical through the grasp, translate, rotate to the goal yaw.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pandaSim.planning import se3
from pandaSim.planning.clearance import phi_limits

GRASP_INSET = 0.02     # m, pivot grasp: distance from the top face and from the far side face
GRASP_DEPTH = 0.02     # m, slide grasp: distance below the top face
GRIPPER_MAX = 0.08     # m, widest object extent the gripper can hold
BASE_KEEPOUT = 0.30    # m, box corners stay this far (horizontally) from the robot base axis
SLIDE_MARGIN_MIN = 0.03                          # m, tipping margin required for a slide
PAD_MU, GRIP_N, GRIP_SAFETY = 1.2, 70.0, 0.5     # fingertip friction, squeeze force per finger, safety factor
Z = np.array([0.0, 0.0, 1.0])
G = 9.81


def quintic(k, n):
    x = min(max(k / n, 0.0), 1.0)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def hand_frame(p, y_axis, y_ref=None):
    """Grasp frame at p: z down, y along +-y_axis (horizontal part), sign nearest y_ref."""
    z = -Z
    y = np.asarray(y_axis, float)
    y = y - (y @ z) * z
    y = y / np.linalg.norm(y)
    if y_ref is not None and y @ np.asarray(y_ref, float) < 0:
        y = -y
    return se3.rp(np.column_stack([np.cross(y, z), y, z]), p)


@dataclass
class PivotPlan:
    S: np.ndarray          # screw axis (omega, v), world frame
    theta: float           # rad
    T_grasp: np.ndarray    # {c} at the start
    T_goal: np.ndarray     # object pose after the pivot
    T_start: np.ndarray    # object pose before the pivot
    width: float           # object extent between the fingers [m]
    phi_lim: tuple = None  # (lo, hi) rad: rotation of the hand about the finger axis that keeps the palm clear of the box

    @property
    def screws(self):
        return [(self.S, self.theta)]


@dataclass
class SlidePlan:
    screws: list           # [(S, theta), ...] world frame, applied in order in one grasp
    T_grasp: np.ndarray
    T_goal: np.ndarray
    T_start: np.ndarray
    width: float

    def seg_starts(self):
        """(object pose, {c} pose) at the start of each screw."""
        To, Th, out = self.T_start, self.T_grasp, []
        for S, th in self.screws:
            out.append((To, Th))
            E = se3.exp6(S * th)
            To, Th = E @ To, E @ Th
        return out


def grasp_end(plan):
    """{c} at the end of the primitive."""
    T = plan.T_grasp
    for S, th in plan.screws:
        T = se3.exp6(S * th) @ T
    return T


def plan_box_pivot(T_wo, size, edge_dir_o, theta, y_ref=None):
    """Pivot about the bottom edge on the side edge_dir_o (a signed object axis, now horizontal)."""
    half = np.asarray(size, float) / 2
    d = np.asarray(edge_dir_o, float)
    R, p = T_wo[:3, :3], T_wo[:3, 3]
    up = R.T @ Z
    i = int(np.argmax(np.abs(up)))
    up_o = np.zeros(3)
    up_o[i] = np.sign(up[i])                       # object axis pointing up
    e_o = np.cross(up_o, d)                        # edge direction: +theta tips toward d
    q = p + R @ (d * half - up_o * half)           # a point on the edge
    e = R @ e_o
    S = np.r_[e, -np.cross(e, q)]
    p_grasp = p + R @ (-d * (half - GRASP_INSET) + up_o * (half - GRASP_INSET))
    T_grasp = hand_frame(p_grasp, e, y_ref)
    return PivotPlan(S, theta, T_grasp, se3.exp6(S * theta) @ T_wo, T_wo, float(2 * np.abs(e_o) @ half),
                     phi_limits(se3.inv(T_grasp) @ T_wo, size))


def planar_screw(T_from, T_to):
    """(S, theta) of the world-frame displacement T_to T_from^-1."""
    return se3.axis_angle(se3.log6(T_to @ se3.inv(T_from)))


def box_grasp(T_wo, size, y_ref=None):
    """Top-down grasp at the box centre; returns ({c} pose, width), or None if no horizontal axis fits."""
    half = np.asarray(size, float) / 2
    R, p = T_wo[:3, :3], T_wo[:3, 3]
    horiz = [i for i in range(3) if abs(R[2, i]) < 0.5]
    i = min(horiz, key=lambda k: half[k])          # closing axis: smallest horizontal extent
    if 2 * half[i] >= GRIPPER_MAX:
        return None
    top = p[2] + np.abs(R[2]) @ half
    return hand_frame(np.array([p[0], p[1], top - GRASP_DEPTH]), R[:, i], y_ref), float(2 * half[i])


def tip_free_screws(T_wo, T_goal, T_grasp, across=False):
    """Rotate about the vertical through the grasp, translate along the bearing axis (or across it),
    rotate to the goal yaw. The rotations do not move the grasp point."""
    g = T_grasp[:3, 3]
    delta = T_goal[:3, 3] - T_wo[:3, 3]
    delta[2] = 0.0
    a = T_grasp[:3, 1].copy()
    a[2] = 0.0
    a = a / np.linalg.norm(a)
    if across:
        a = np.cross(Z, a)
    screws = []
    dist = float(np.linalg.norm(delta))
    if dist > 1e-6:
        beta = math.atan2(np.cross(a, delta)[2], a @ delta)
        if beta > math.pi / 2:
            beta -= math.pi
        elif beta < -math.pi / 2:
            beta += math.pi
        if abs(beta) > 1e-6:
            screws.append((np.r_[Z, -np.cross(Z, g)], beta))
        screws.append((np.r_[np.zeros(3), delta / dist], dist))
    To = T_wo
    for S, th in screws:
        To = se3.exp6(S * th) @ To
    S3, th3 = planar_screw(To, T_goal)
    if abs(th3) > 1e-6:
        screws.append((S3, th3))
    return screws


def tip_margin(T_wo, size, T_hand, S, mu, mass):
    """Quasi-static tipping margin [m] for dragging the box along the unit screw S (> 0: no tipping).

    Across the bearing axis a only the support resists:  d_lead(u_perp) - mu h_g |u_perp|.
    Along a the fingertip friction couple also resists:  tau_grip/(m g) + d_lead(+-a) - mu h_g |u_par|.
    """
    half = np.asarray(size, float) / 2
    R, c = T_wo[:3, :3], T_wo[:3, 3]
    p_g = T_hand[:3, 3]
    v_g = S[3:] + np.cross(S[:3], p_g)             # grasp-point velocity
    v_g = v_g - v_g[2] * Z
    speed = float(np.linalg.norm(v_g))
    if speed < 1e-4:                               # grasp point (almost) still: no tipping moment
        return math.inf
    u = v_g / speed
    a = T_hand[:3, 1] - T_hand[2, 1] * Z
    a = a / np.linalg.norm(a)
    h_g = float(p_g[2] - (c[2] - np.abs(R[2]) @ half))

    def d_lead(w):
        return float(np.abs(R.T @ w) @ half - (p_g - c) @ w)

    u_par = float(u @ a)
    u_perp = u - u_par * a
    n_perp = float(np.linalg.norm(u_perp))
    m = math.inf
    if n_perp > 1e-6:
        m = min(m, d_lead(u_perp / n_perp) - mu * h_g * n_perp)
    if abs(u_par) > 1e-6:
        w_a = 2 * float(np.abs(R.T @ a) @ half)
        tau = 2 * PAD_MU * GRIP_N * (w_a / 2) * GRIP_SAFETY
        m = min(m, tau / (mass * G) + d_lead(a if u_par > 0 else -a) - mu * h_g * abs(u_par))
    return m


def slide_tip_margin(size, plan, mu, mass):
    return min(tip_margin(To, size, Th, S, mu, mass) for (To, Th), (S, _) in zip(plan.seg_starts(), plan.screws))


def plan_box_slide(T_wo, size, T_goal, y_ref=None, mu=0.5, mass=5.0):
    """Direct screw if it passes the tipping check, else the first tip-free chain that does
    (translate along, then across, the bearing axis); else the direct screw (it then fails the check).
    Returns None if the box does not fit in the gripper in this pose."""
    grasp = box_grasp(T_wo, size, y_ref)
    if grasp is None:
        return None
    T_grasp, width = grasp
    direct = SlidePlan([planar_screw(T_wo, T_goal)], T_grasp, T_goal, T_wo, width)
    for plan in (direct,
                 SlidePlan(tip_free_screws(T_wo, T_goal, T_grasp, False), T_grasp, T_goal, T_wo, width),
                 SlidePlan(tip_free_screws(T_wo, T_goal, T_grasp, True), T_grasp, T_goal, T_wo, width)):
        if slide_tip_margin(size, plan, mu, mass) > SLIDE_MARGIN_MIN:
            return plan
    return direct


_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def keepout_ok(S, theta, T_wo, size, base_xy=(0.0, 0.0), n=12):
    """True if the box corners stay outside the base keep-out cylinder along the screw."""
    pts = _SIGNS * (np.asarray(size, float) / 2)
    for k in range(n + 1):
        T = se3.exp6(S * theta * k / n) @ T_wo
        c = pts @ T[:3, :3].T + T[:3, 3]
        if np.linalg.norm(c[:, :2] - base_xy, axis=1).min() < BASE_KEEPOUT:
            return False
    return True


def plan_keepout_ok(plan, size, base_xy=(0.0, 0.0)):
    To = plan.T_start
    for S, th in plan.screws:
        if not keepout_ok(S, th, To, size, base_xy):
            return False
        To = se3.exp6(S * th) @ To
    return True


def hand_path(plan, T_start, lift=0.10, n=(40, 20, 60, 15)):
    """Waypoints of {c} for one primitive, for the kinematic check:
    (approach: start -> above the grasp -> grasp;  motion with the object, then lift)."""
    up = np.array([0.0, 0.0, lift])
    T_pre = plan.T_grasp.copy()
    T_pre[:3, 3] += up
    approach = se3.sclerp(T_start, T_pre, n[0]) + se3.sclerp(T_pre, plan.T_grasp, n[1])
    m = max(1, n[2] // len(plan.screws))
    motion, T = [], plan.T_grasp
    for S, th in plan.screws:
        motion += [se3.exp6(S * th * quintic(k, m)) @ T for k in range(1, m + 1)]
        T = motion[-1]
    T_post = T.copy()
    T_post[:3, 3] += up
    return approach, motion, se3.sclerp(T, T_post, n[3])
