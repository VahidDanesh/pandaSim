"""Pivot primitive as a constant screw motion (manuscript Sec. 2.2, 4.2).

All twists are ordered (omega, v), as in the manuscript (Lynch & Park).
Robotics Toolbox Jacobians use (v, omega): convert only where they are used.

A pivot is a pure rotation (h = 0, d = 0) about an axis through the fixed
object-table contact: unit direction `s_axis` through point `q`, both in world.
"""
from __future__ import annotations

import numpy as np
from pytransform3d import transformations as pt


def time_scaling(t: float, Tf: float, method: str = "quintic") -> tuple[float, float]:
    """Path parameter s(t) and its rate s_dot(t); s_dot = 0 outside [0, Tf].

    method: "quintic" (default), "cubic" or "linear".
    """
    tau = min(max(t / Tf, 0.0), 1.0)
    m = method[0].lower()
    if m == "q":
        return tau**3 * (10 - 15 * tau + 6 * tau**2), 30 * tau**2 * (1 - tau) ** 2 / Tf
    if m == "c":
        return tau**2 * (3 - 2 * tau), 6 * tau * (1 - tau) / Tf
    if m == "l":
        return tau, (1.0 / Tf if 0.0 <= t < Tf else 0.0)
    raise ValueError(f"unknown time scaling {method!r}: use quintic, cubic or linear")


def lifting_angle(q: np.ndarray, s_axis: np.ndarray, theta: float, center: np.ndarray) -> float:
    """Signed pivot angle (|theta|) about `s_axis` that lifts the object's `center`."""
    vz = np.cross(s_axis, np.asarray(center) - np.asarray(q))[2]  # centre velocity, unit rate
    if abs(vz) < 1e-9:
        raise ValueError("centre is straight above the pivot axis: lifting direction is undefined")
    return float(np.copysign(abs(theta), vz))


def pivot_twist(q: np.ndarray, s_axis: np.ndarray, theta: float, T_wc0: np.ndarray) -> np.ndarray:
    """Body twist [theta*u_c; theta*m_c] of frame {c} for s_dot = 1 (eq:twist_vec, d = 0).

    u_c and m_c are the axis direction and moment expressed in {c} at the start
    of the primitive. The body twist is constant over the primitive, so the
    commanded twist at time t is s_dot(t) * pivot_twist(...).
    """
    R, p = T_wc0[:3, :3], T_wc0[:3, 3]
    u_c = R.T @ (s_axis / np.linalg.norm(s_axis))
    m_c = np.cross(R.T @ (np.asarray(q) - p), u_c)
    return theta * np.r_[u_c, m_c]


def planned_pose(q: np.ndarray, s_axis: np.ndarray, theta: float, T0: np.ndarray, s: float) -> np.ndarray:
    """Pose at path parameter s of any frame attached to the object (ScLERP of the pivot)."""
    S = pt.screw_axis_from_screw_parameters(q, s_axis / np.linalg.norm(s_axis), 0.0)
    return pt.transform_from_exponential_coordinates(S * theta * s) @ T0


def pose_error_twist(T_wc: np.ndarray, T_wd: np.ndarray) -> np.ndarray:
    """Body twist log(T_wc^-1 T_wd) that moves {c} to the target in unit time (approach phase)."""
    return pt.exponential_coordinates_from_transform(pt.invert_transform(T_wc, check=False) @ T_wd, check=False)
