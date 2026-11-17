"""Measure the table and the box with the robot's forward kinematics while you guide the arm by hand.

    python experiments/calibrate_box.py 172.16.0.2

READ-ONLY: this script never starts a controller or sends a motion command; it only reads the joint angles.
Unlock the brakes (Desk) and move the arm by hand with the guiding buttons; close the gripper so the two fingertips
touch, and touch each point with the fingertip centre (the frame {E} of the model, the middle between the pads).
  1. three points on the table, spread apart          -> table height and tilt
  2. three corners of the box's top face: two along the longest top edge, then the corner next to the second
     -> box centre, yaw, height (which face is down), the top-face dimensions
It prints the pose in the form of run_box.py (--start x,y,yaw,face) and compares the size with the config.
The touch point is the fingertip centre; pad thickness and how firmly you touch add about 1-2 mm.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
# vertical box axis -> face of run_box.py (axis 1 vertical: the box lies on its largest face, not graspable)
FACE_BY_HEIGHT = {0: "lying", 2: "standing"}


def plane(points):
    """Mean height and tilt (deg) of the plane through the points (least squares)."""
    P = np.asarray(points, float)
    c = P.mean(axis=0)
    n = np.linalg.svd(P - c)[2][-1]
    n = n if n[2] > 0 else -n
    return float(c[2]), float(math.degrees(math.acos(min(1.0, n[2]))))


def box_from_top_corners(c1, c2, c3, table_z, size):
    """Box pose from three top corners: c1-c2 along the longest top edge, c3 next to c2.
    Returns dict x, y, yaw_deg (in (-90, 90], the box is symmetric), face, height, edges, and the size check."""
    c1, c2, c3 = (np.asarray(c, float) for c in (c1, c2, c3))
    top_z = float(np.mean([c1[2], c2[2], c3[2]]))
    height = top_z - table_z
    size = np.asarray(size, float)
    axis = int(np.argmin(np.abs(size - height)))          # which box axis is vertical
    e1, e2 = np.linalg.norm((c2 - c1)[:2]), np.linalg.norm((c3 - c2)[:2])
    d = (c2 - c1)[:2] / max(e1, 1e-9)
    # run_box.py poses: lying (pitch -90) puts the box z (longest) axis along Rz(yaw) (-1, 0, 0);
    # standing (pitch 0) puts the box x axis along Rz(yaw) (1, 0, 0); either way yaw mod 180 is the same pose
    yaw = math.degrees(math.atan2(d[1], d[0]))
    yaw = (yaw + 90.0) % 180.0 - 90.0
    yaw = 90.0 if yaw == -90.0 else yaw
    centre = (c1 + c3)[:2] / 2
    others = np.delete(size, axis)
    return dict(x=float(centre[0]), y=float(centre[1]), yaw_deg=yaw, height=height, face=FACE_BY_HEIGHT.get(axis),
                edges=(float(e1), float(e2)), expected_edges=tuple(sorted(others, reverse=True)),
                height_error=float(height - size[axis]))


def read_point(panda, robot, n=50):
    """Fingertip centre {E} in the table frame, averaged over n states (no command is sent)."""
    from pandaSim.control.primitive_executor import contact_pose

    q = np.mean([np.asarray(panda.get_state().q, float) for _ in range(n)], axis=0)
    return contact_pose(robot, q)[:3, 3]


def session(panda, robot, size, ask=input, out=print):
    """Guided measurement; ask(prompt) waits for the user, out(text) prints. Returns the result dict."""
    pts = []
    for k in range(3):
        ask(f"table point {k + 1} of 3: touch the table, then press Enter ")
        pts.append(read_point(panda, robot))
    table_z, tilt = plane(pts)
    out(f"table: height {table_z * 1e3:.1f} mm in the table frame of the model (expected 0), tilt {tilt:.2f} deg")
    corners = []
    for name in ("first corner of the longest top edge", "second corner of that edge", "the corner next to the second"):
        ask(f"box: touch the {name}, then press Enter ")
        corners.append(read_point(panda, robot))
    r = box_from_top_corners(*corners, table_z, size)
    out(f"box: centre ({r['x']:.3f}, {r['y']:.3f}) m, yaw {r['yaw_deg']:.1f} deg, height {r['height'] * 1e3:.1f} mm "
        f"({'face ' + r['face'] if r['face'] else 'on its largest face: cannot be grasped'})")
    out(f"     top edges {r['edges'][0] * 1e3:.1f} x {r['edges'][1] * 1e3:.1f} mm, config "
        f"{r['expected_edges'][0] * 1e3:.0f} x {r['expected_edges'][1] * 1e3:.0f} mm, height off by {r['height_error'] * 1e3:.1f} mm")
    if r["face"]:
        out(f"run_box.py:  --start {r['x']:.3f},{r['y']:.3f},{r['yaw_deg']:.1f},{r['face']}")
    return dict(r, table_z=table_z, tilt=tilt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("host")
    ap.add_argument("--config", default=str(HERE / "configs" / "box_real.yaml"))
    a = ap.parse_args(argv)
    import panda_py
    import spatialmath as sm
    from pandaSim.geometry.utils import create_virtual_panda

    cfg = yaml.safe_load(Path(a.config).read_text())
    robot = create_virtual_panda()
    robot.base = sm.SE3(*cfg["robot"]["base"])
    panda = panda_py.Panda(a.host)          # connection only: no controller is started, nothing is commanded
    session(panda, robot, cfg["box"]["size"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
