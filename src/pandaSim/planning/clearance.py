"""Clearance between the hand and a box held at the fingertips.

The fingers hold the box on two opposite faces; the hand body (palm) is the part that can touch it.
The palm is the bounding box of the hand collision mesh, written in the hand frame {E}
(origin between the fingertips, z out of the hand, y along the finger closing axis n).

With free rotation the hand turns about n relative to the box: T_c = T_E Rot_y(phi), and {c} is fixed
to the box. phi_limits gives the interval of phi in which the palm stays clear of the box; ContactQP
keeps phi inside it (phi_lim).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

PALM_CENTRE = np.array([0.0, 0.0, -0.0834])     # m, in {E}: hand.stl spans z = -0.026 .. 0.066 in panda_hand,
PALM_HALF = np.array([0.0316, 0.104, 0.046])    # and {E} is 0.1034 m further along z
CLEARANCE = 0.005                               # m, the palm is grown by this much for phi_limits


def obb_overlap(Ta, ha, Tb, hb):
    """True if two boxes (pose 4x4, half extents) overlap: separating axis test."""
    Ra, Rb = Ta[:3, :3], Tb[:3, :3]
    L = np.vstack([Ra.T, Rb.T, np.cross(Ra.T[:, None, :], Rb.T[None, :, :]).reshape(9, 3)])
    n = np.linalg.norm(L, axis=1)
    L = L[n > 1e-9] / n[n > 1e-9, None]
    gap = np.abs(L @ (Tb[:3, 3] - Ta[:3, 3])) - np.abs(L @ Ra) @ ha - np.abs(L @ Rb) @ hb
    return bool(np.all(gap <= 0))


def palm_pose(T_E):
    T = np.array(T_E, float)
    T[:3, 3] += T[:3, :3] @ PALM_CENTRE
    return T


def palm_hits_box(T_E, T_obj, size, clearance=0.0):
    """True if the palm of the hand at T_E overlaps the box at T_obj (same frame for both)."""
    return obb_overlap(palm_pose(T_E), PALM_HALF + clearance, T_obj, np.asarray(size, float) / 2)


def palm_lowest(T_E):
    """Height of the lowest palm corner (frame of T_E)."""
    T = palm_pose(T_E)
    return float(T[2, 3] - np.abs(T[2, :3]) @ PALM_HALF)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


def phi_limits(T_c_obj, size, clearance=CLEARANCE, step=math.radians(0.25)):
    """(phi_lo, phi_hi): the interval around phi = 0 in which the palm, grown by `clearance`, is clear
    of the box. T_c_obj: box pose in the grasp frame {c}. Returns (0, 0) if the palm touches at phi = 0."""
    return _phi_limits(tuple(np.round(np.asarray(T_c_obj, float), 6).ravel()),
                       tuple(float(s) for s in size), float(clearance), float(step))


@lru_cache(maxsize=256)
def _phi_limits(T_flat, size, clearance, step):
    T_c_obj = np.array(T_flat).reshape(4, 4)
    hit = lambda phi: palm_hits_box(_rot_y(-phi), T_c_obj, size, clearance)   # {E} in {c} is Rot_y(-phi)
    if hit(0.0):
        return 0.0, 0.0
    lim = []
    for sign in (-1.0, 1.0):
        k = 1
        while k * step <= math.pi and not hit(sign * k * step):
            k += 1
        lim.append(sign * min((k - 1) * step, math.pi))
    return lim[0], lim[1]
