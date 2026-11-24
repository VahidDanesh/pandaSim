"""Pivot primitive math (planning/primitives.py). Twists are (omega, v)."""
import numpy as np
import pytest
import spatialmath as sm
from pytransform3d import transformations as pt

from pandaSim.planning.primitives import (
    lifting_angle,
    pivot_twist,
    planned_pose,
    pose_error_twist,
    time_scaling,
)

THETA = np.pi / 2


def body_twist_fd(T_a, T_b, ds):
    return pt.exponential_coordinates_from_transform(np.linalg.inv(T_a) @ T_b) / ds


def some_pose(seed=0):
    rng = np.random.default_rng(seed)
    return pt.transform_from_exponential_coordinates(rng.normal(size=6) * 0.5)


AXIS_Q, AXIS_S = np.array([0.5, -0.2, 0.0]), np.array([0.0, 1.0, 0.0])


@pytest.mark.parametrize("s", [0.0, 0.3, 0.9])
def test_pivot_twist_is_the_body_twist_of_planned_pose(s):
    T_wc0, ds = some_pose(), 1e-6
    T_a = planned_pose(AXIS_Q, AXIS_S, THETA, T_wc0, s)
    T_b = planned_pose(AXIS_Q, AXIS_S, THETA, T_wc0, s + ds)
    assert np.allclose(pivot_twist(AXIS_Q, AXIS_S, THETA, T_wc0), body_twist_fd(T_a, T_b, ds), atol=1e-5)


def test_planned_pose_matches_dual_quaternion_sclerp():
    T0 = some_pose(1)
    D1 = pt.dual_quaternion_from_transform(T0)
    D2 = pt.dual_quaternion_from_transform(planned_pose(AXIS_Q, AXIS_S, THETA, T0, 1.0))
    for s in np.linspace(0, 1, 7):
        T_sclerp = pt.transform_from_dual_quaternion(pt.dual_quaternion_sclerp(D1, D2, s))
        assert np.allclose(planned_pose(AXIS_Q, AXIS_S, THETA, T0, s), T_sclerp, atol=1e-9)


def test_twist_order_is_omega_v():
    T = np.eye(4)
    T[0, 3] = 0.1
    assert np.allclose(pose_error_twist(np.eye(4), T), [0, 0, 0, 0.1, 0, 0])
    V = np.array([0.1, -0.2, 0.3, 0.05, 0.0, -0.04])
    T_wc = some_pose(2)
    assert np.allclose(pose_error_twist(T_wc, T_wc @ pt.transform_from_exponential_coordinates(V)), V)


@pytest.mark.parametrize("method", ["quintic", "cubic", "linear"])
def test_time_scaling(method):
    Tf = 2.0
    assert time_scaling(0.0, Tf, method)[0] == 0.0
    assert time_scaling(Tf, Tf, method) == (1.0, 0.0)
    assert time_scaling(-0.1, Tf, method) == (0.0, 0.0)
    assert time_scaling(Tf + 0.1, Tf, method) == (1.0, 0.0)
    if method != "linear":
        assert time_scaling(0.0, Tf, method)[1] == 0.0
    t = np.linspace(0, Tf, 20001)
    sdot = np.array([time_scaling(ti, Tf, method)[1] for ti in t])
    assert np.trapezoid(sdot, t) == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(ValueError):
        time_scaling(0.5, Tf, "spline")


# --- pivot geometry about the ground edges of a box ---

# (size, pose): a box, and the bounding box of an upright cylinder (r 0.03, l 0.15)
OBJECTS = {
    "cuboid": ([0.27, 0.065, 0.207], (sm.SE3(0.5, -0.2, 0.207 / 2) * sm.SE3.Rz(np.pi / 2)).A),
    "cylinder_bbox": ([0.06, 0.06, 0.15], (sm.SE3(0.45, 0.25, 0.075) * sm.SE3.Rz(0.4)).A),
}


def ground_edges(size, T):
    """(corners, [(point, unit direction)] of the four bottom edges) of a box resting on the table."""
    h = np.asarray(size) / 2
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    corners = (signs * h) @ T[:3, :3].T + T[:3, 3]
    low = signs[:, 2] < 0
    edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            if low[a] and low[b] and np.sum(signs[a] != signs[b]) == 1:
                d = corners[b] - corners[a]
                edges.append((corners[a], d / np.linalg.norm(d)))
    return corners, edges


@pytest.mark.parametrize("name", OBJECTS)
def test_pivot_about_every_ground_edge(name):
    size, T0 = OBJECTS[name]
    corners_w, edges = ground_edges(size, T0)
    corners = np.c_[corners_w, np.ones(8)]
    table = corners_w[:, 2].min()
    assert len(edges) == 4
    for q, s_axis in edges:
        theta = lifting_angle(q, s_axis, THETA, T0[:3, 3])
        for s in np.linspace(0, 1, 21):
            D = planned_pose(q, s_axis, theta, T0, s) @ np.linalg.inv(T0)
            assert np.allclose(D[:3, :3] @ q + D[:3, 3], q, atol=1e-12)              # axis stays fixed
            assert np.allclose(D[:3, :3] @ s_axis, s_axis, atol=1e-12)
            assert (corners @ D.T)[:, 2].min() >= table - 1e-9                      # never below table
        D = planned_pose(q, s_axis, theta, T0, 1.0) @ np.linalg.inv(T0)
        tilt = np.cross([0, 0, 1], s_axis)                                          # horizontal, normal to the axis
        width = np.ptp(corners_w @ tilt)
        assert np.ptp((corners @ D.T)[:, 2]) == pytest.approx(width, abs=1e-9)      # new height
