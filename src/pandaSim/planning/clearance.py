"""Clearance between the hand and a box held at the fingertips.

The fingers hold the box on two opposite faces; the hand body (palm) is the part that can touch it.
The palm is the bounding box of the hand collision mesh, written in the hand frame {E}
(origin between the fingertips, z out of the hand, y along the finger closing axis n).

With free rotation the hand turns about n relative to the box: T_c = T_E Rot_y(phi), and {c} is fixed
to the box. phi_limits gives the interval of phi in which the hand stays clear of the box; ContactQP
keeps phi inside it (phi_lim).

Two models of the hand body:
  "mesh"  the collision mesh of the hand (hand.stl) against the box, signed distance by coal. This is the same
          geometry and the same library as planning/collision.ArmCollision, which checks every simulated run, so
          the limits (clearance 5 mm) lie inside what that check accepts (2 mm). Default.
  "box"   the bounding box of that mesh (PALM_*), separating-axis test in numpy. It encloses the mesh, so it is
          the more cautious of the two; used when coal is not installed, and by palm_hits_box / palm_lowest.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

HAND_MESH = Path(__file__).resolve().parents[3] / "model" / "franka_description" / "meshes" / "collision" / "hand.stl"
E_TO_HAND = -0.1034                             # m: origin of the hand link (and of hand.stl) along z of {E}
# model of the hand for the phi limits and for the palm test of check_clearance: "mesh" or "box" (see above).
# PANDASIM_HAND_MODEL=box gives the limits used before the mesh model existed. A config's planner.hand_model
# replaces it (run_plan.make_planner), and a config without one sets the default back.
DEFAULT_HAND_MODEL = os.environ.get("PANDASIM_HAND_MODEL", "mesh")
HAND_MODEL = DEFAULT_HAND_MODEL

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


@lru_cache(maxsize=8)
def _hand_and_box(size):
    """(hand mesh, box) as spatialgeometry collision shapes, or None if coal or the mesh is not available."""
    try:
        import coal  # noqa: F401
        import spatialgeometry as sg
    except ImportError:
        return None
    if not HAND_MESH.exists():
        return None
    return sg.Mesh(filename=str(HAND_MESH), collision=True), sg.Cuboid(scale=list(size), collision=True)


def hand_distance(T_E, T_obj, size):
    """Signed distance [m] between the hand mesh, with the fingertip frame {E} at T_E, and the box at T_obj (both
    in the same frame). None if coal is not available; distances above 1 m read 1 m."""
    from spatialmath import SE3

    shapes = _hand_and_box(tuple(float(x) for x in size))
    if shapes is None:
        return None
    hand, box = shapes
    T_hand = np.array(T_E, float)                # the hand link is E_TO_HAND along z of {E}
    T_hand[:3, 3] += T_hand[:3, 2] * E_TO_HAND
    hand.T = SE3(T_hand, check=False)
    box.T = SE3(np.asarray(T_obj, float), check=False)
    hand.update()
    box.update()
    d = hand.closest_point(box, 1.0)[0]
    return 1.0 if d is None else float(d)


def hand_box_distance(T_c_obj, size, phi):
    """hand_distance when the hand is turned by phi about n relative to the grasp frame {c} (T_c_obj: box pose
    in {c}): {E} in {c} is Rot_y(-phi)."""
    return hand_distance(_rot_y(-phi), T_c_obj, size)


def hand_hits_box(T_E, T_obj, size, clearance=0.0, model=None):
    """True if the hand at T_E is closer than `clearance` to the box at T_obj, with the hand model `model`
    (None: HAND_MODEL; "mesh" falls back to "box" without coal). The test that goes with phi_limits."""
    model = HAND_MODEL if model is None else model
    if model == "mesh":
        d = hand_distance(T_E, T_obj, size)
        if d is not None:
            return d < clearance
    return palm_hits_box(T_E, T_obj, size, clearance)


def phi_limits(T_c_obj, size, clearance=CLEARANCE, step=math.radians(0.25), model=None):
    """(phi_lo, phi_hi): the interval around phi = 0 in which the hand stays `clearance` away from the box.
    T_c_obj: box pose in the grasp frame {c}. model: "mesh" (hand.stl by coal; falls back to "box" without coal)
    or "box" (bounding box of the hand); None: HAND_MODEL. Returns (0, 0) if the hand is too close at phi = 0."""
    model = HAND_MODEL if model is None else model
    if model == "mesh" and _hand_and_box(tuple(float(x) for x in size)) is None:
        model = "box"
    return _phi_limits(tuple(np.round(np.asarray(T_c_obj, float), 6).ravel()),
                       tuple(float(s) for s in size), float(clearance), float(step), model)


@lru_cache(maxsize=256)
def _phi_limits(T_flat, size, clearance, step, model="box"):
    T_c_obj = np.array(T_flat).reshape(4, 4)
    if model == "mesh":
        hit = lambda phi: hand_box_distance(T_c_obj, size, phi) < clearance
    else:
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
