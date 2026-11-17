"""Measure the table and the box with the robot's forward kinematics while you guide the arm by hand.

    python experiments/calibrate_box.py 172.16.0.2                                  start pose before a run
    python experiments/calibrate_box.py 172.16.0.2 --measure experiments/runs/box/<name>   final pose after it

READ-ONLY: this script never starts a controller or sends a motion command; it only reads the joint angles.
Unlock the brakes (Desk) and move the arm by hand with the guiding buttons; close the gripper so the two fingertips
touch, and touch each point with the fingertip centre (the frame {E} of the model, the middle between the pads).
  1. three points on the table, spread apart          -> table height and tilt (skipped with --measure: 0 is used)
  2. three corners of the box's top face: first the top corner of the vertical edge that carries the mark (the
     mark is on the corner at the top or at the bottom of that edge), then the other end of the longest top edge
     from there, then the corner next to that one; and whether the mark is at the top
     -> box centre, yaw over 360 deg, which face is down, the top-face dimensions
It prints the pose in the form of run_box.py (--start x,y,yaw,face) and compares the size with the config. With
--measure it compares the pose with the goal of the run and writes measured.json into the run folder.
The touch point is the fingertip centre; pad thickness and how firmly you touch add about 1-2 mm.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_box import FACES, MARK, pose_text  # noqa: E402


def plane(points):
    """Mean height and tilt (deg) of the plane through the points (least squares)."""
    P = np.asarray(points, float)
    c = P.mean(axis=0)
    n = np.linalg.svd(P - c)[2][-1]
    n = n if n[2] > 0 else -n
    return float(c[2]), float(math.degrees(math.acos(min(1.0, n[2]))))


def _ry(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def box_from_top_corners(c1, c2, c3, table_z, size, mark_on_top=True):
    """Box pose from three top corners: c1 on the vertical edge that carries the mark, c2 the other end of the longest
    top edge from c1, c3 next to c2; mark_on_top: the mark is at the top end of that edge.
    Returns dict x, y, yaw_deg (in (-180, 180]), face (None: on its largest face), height, edges, the size check, and
    fit_mm: the largest distance between a touched corner and the corner of the fitted pose."""
    c1, c2, c3 = (np.asarray(c, float) for c in (c1, c2, c3))
    top_z = float(np.mean([c1[2], c2[2], c3[2]]))
    height = top_z - table_z
    size, half = np.asarray(size, float), np.asarray(size, float) / 2
    axis = int(np.argmin(np.abs(size - height)))          # which box axis is vertical
    e1, e2 = np.linalg.norm((c2 - c1)[:2]), np.linalg.norm((c3 - c2)[:2])
    out = dict(height=height, edges=(float(e1), float(e2)), expected_edges=tuple(sorted(np.delete(size, axis), reverse=True)),
               height_error=float(height - size[axis]), face=None)
    for face, pitch in FACES.items():                     # the face whose up axis and mark side match
        R0 = _ry(pitch)
        up = int(np.argmax(np.abs(R0[2])))
        if up != axis or (np.sign(R0[2, up]) == MARK[up]) != mark_on_top:
            continue
        # corners of this face at yaw 0, centre at the origin: the same three corners
        m = np.asarray(MARK, float)
        m[up] = np.sign(R0[2, up])                        # the top end of the marked vertical edge
        k = [i for i in range(3) if i != up]
        k2 = max(k, key=lambda i: size[i])                # c1 -> c2 along the longest top edge
        k3 = min(k, key=lambda i: size[i])
        n2, n3 = m.copy(), m.copy()
        n2[k2] *= -1
        n3[k2] *= -1
        n3[k3] *= -1
        p1, p2, p3 = (R0 @ (half * v) for v in (m, n2, n3))
        yaw = math.atan2((c2 - c1)[1], (c2 - c1)[0]) - math.atan2((p2 - p1)[1], (p2 - p1)[0])
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        centre = np.r_[(c1 + c3)[:2] / 2, table_z + half[axis]]
        fit = max(np.linalg.norm(Rz @ p + centre - c) for p, c in ((p1, c1), (p2, c2), (p3, c3)))
        yaw_deg = (math.degrees(yaw) + 180.0) % 360.0 - 180.0
        out.update(x=float(centre[0]), y=float(centre[1]), yaw_deg=180.0 if yaw_deg == -180.0 else yaw_deg, face=face,
                   fit_mm=float(fit * 1e3))
    return out


def read_point(panda, robot, n=50):
    """Fingertip centre {E} in the table frame, averaged over n states (no command is sent)."""
    from pandaSim.control.primitive_executor import contact_pose

    q = np.mean([np.asarray(panda.get_state().q, float) for _ in range(n)], axis=0)
    return contact_pose(robot, q)[:3, 3]


def session(panda, robot, size, ask=input, out=print, table=True):
    """Guided measurement; ask(prompt) waits for the user, out(text) prints. Returns the result dict.
    table=False: no table points, the table height 0 of the model is used (a final pose after a run)."""
    table_z, tilt = 0.0, None
    if table:
        pts = []
        for k in range(3):
            ask(f"table point {k + 1} of 3: touch the table, then press Enter ")
            pts.append(read_point(panda, robot))
        table_z, tilt = plane(pts)
        out(f"table: height {table_z * 1e3:.1f} mm in the table frame of the model (expected 0), tilt {tilt:.2f} deg")
    corners = []
    for name in ("top corner of the vertical edge that carries the mark",
                 "other end of the longest top edge from there", "the corner next to that one"):
        ask(f"box: touch the {name}, then press Enter ")
        corners.append(read_point(panda, robot))
    top = ask("is the mark at the top of that edge? [y/n] ").strip().lower().startswith("y")
    r = box_from_top_corners(*corners, table_z, size, mark_on_top=top)
    if r["face"] is None:
        out(f"box: height {r['height'] * 1e3:.1f} mm: on its largest face, or the mark answer does not fit the height; "
            "no pose of run_box.py")
        return dict(r, table_z=table_z, tilt=tilt)
    out(f"box: centre ({r['x']:.3f}, {r['y']:.3f}) m, yaw {r['yaw_deg']:.1f} deg, face {r['face']}, "
        f"height {r['height'] * 1e3:.1f} mm; touched corners within {r['fit_mm']:.1f} mm of the fitted pose")
    out(f"     top edges {r['edges'][0] * 1e3:.1f} x {r['edges'][1] * 1e3:.1f} mm, config "
        f"{r['expected_edges'][0] * 1e3:.0f} x {r['expected_edges'][1] * 1e3:.0f} mm, height off by {r['height_error'] * 1e3:.1f} mm")
    out(f"run_box.py:  --start {r['x']:.3f},{r['y']:.3f},{r['yaw_deg']:.1f},{r['face']}")
    return dict(r, table_z=table_z, tilt=tilt)


def measured_error(r, run_dir, size):
    """Error of the measured pose r (box_from_top_corners) against the goal of a run folder: dict for measured.json."""
    import run_plan as rp
    from pandaSim.planning.sequencer import Box, pose_error

    d = np.load(Path(run_dir) / "expected.npz", allow_pickle=True)
    Tg = d["Tg"]
    out = dict(measured=f"{r['x']:.4f},{r['y']:.4f},{r['yaw_deg']:.2f},{r['face']}", goal=pose_text(Tg), fit_mm=r.get("fit_mm"))
    if r["face"] is None:
        return dict(out, face_ok=False)
    T = rp.box_pose(Box(tuple(size), 1.0, 0.5), dict(xy=[r["x"], r["y"]], rpy=[0.0, FACES[r["face"]], r["yaw_deg"]]))
    ep, er = pose_error(T, Tg, symmetric=False)
    yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]) - math.atan2(Tg[1, 0], Tg[0, 0]))
    return dict(out, face_ok=r["face"] == out["goal"].split(",")[3],
                error_mm=float(ep * 1e3), error_deg=float(math.degrees(er)), yaw_error_deg=float((yaw + 180) % 360 - 180),
                dx_mm=float((T[0, 3] - Tg[0, 3]) * 1e3), dy_mm=float((T[1, 3] - Tg[1, 3]) * 1e3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("host")
    ap.add_argument("--config", default=str(HERE / "configs" / "box_real.yaml"))
    ap.add_argument("--measure", metavar="RUN_DIR", help="final pose after a run: error against its goal -> measured.json")
    a = ap.parse_args(argv)
    import panda_py
    import spatialmath as sm
    from pandaSim.geometry.utils import create_virtual_panda

    cfg = yaml.safe_load(Path(a.config).read_text())
    robot = create_virtual_panda()
    robot.base = sm.SE3(*cfg["robot"]["base"])
    panda = panda_py.Panda(a.host)          # connection only: no controller is started, nothing is commanded
    r = session(panda, robot, cfg["box"]["size"], table=a.measure is None)
    if a.measure:
        m = measured_error(r, a.measure, cfg["box"]["size"])
        (Path(a.measure) / "measured.json").write_text(json.dumps(m, indent=1))
        print(f"against the goal: {m.get('error_mm', float('nan')):.1f} mm, {m.get('error_deg', float('nan')):.2f} deg "
              f"(face {'right' if m['face_ok'] else 'WRONG'}); written to {Path(a.measure) / 'measured.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
